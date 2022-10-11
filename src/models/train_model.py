import io
import os
import json
import pickle
import numpy as np
import torch
from torch import nn
from codecarbon import EmissionsTracker
import mlflow
import boto3
from transformer import Predictor

# Setting hyperparameters
print(os.getcwd())
with open('../../models/config.json') as config:
    hyperparams = json.load(config)

boto3.setup_default_session(profile_name=hyperparams['AWS_PROFILE'])


def download_s3(bucket_name, path_to_file):
    """Function to download dataset from s3 bucket to be used by the model."""
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket_name, Key=path_to_file)
    f = io.BytesIO(obj["Body"].read())
    return f

class Vocabulary():
    """This class defines the vocabulary that the NLP model will work with."""
    def __init__(self, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):
        self.token2idx = {}
        self.idx2token = []
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        if pad_token is not None:
            self.pad_index = self.add_token(pad_token)
        if unk_token is not None:
            self.unk_index = self.add_token(unk_token)
        if eos_token is not None:
            self.eos_index = self.add_token(eos_token)

    def add_token(self, token):
        """Function to add a new token."""
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def get_index(self, token):
        """Get index from token."""
        if isinstance(token, str):
            return self.token2idx.get(token, self.unk_index)
        return [self.token2idx.get(t, self.unk_index) for t in token]

    def __len__(self):
        """Return length of vocabulary."""
        return len(self.idx2token)

    def save(self, filename):
        """Save vocabulary into pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        """Load vocabulary from pickle file."""
        # with open(filename, 'rb') as f:
        #     self.__dict__.update(pickle.load(f))
        f = download_s3(hyperparams['BUCKET_NAME'], filename)
        self.__dict__.update(pickle.load(f))


def batch_generator(idata, target, batch_size, shuffle=True):
    """Generator of data batches to train the model."""
    nsamples = len(idata)
    if shuffle:
        perm = np.random.permutation(nsamples)
    else:
        perm = range(nsamples)

    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i+batch_size]
        if target is not None:
            yield idata[batch_idx], target[batch_idx]
        else:
            yield idata[batch_idx], None


def load_preprocessed_dataset(prefix):
    """Load preprocessed dataset from s3 bucket."""
    # Try loading precomputed vocabulary and preprocessed data files
    token_vocab = Vocabulary()
    token_vocab.load(f'{prefix}.vocab')
    data = []
    for part in ['train', 'valid']:
        file = download_s3(hyperparams['BUCKET_NAME'], f'{prefix}.{part}.npz')
        with np.load(file) as set_data:
            idata, target = set_data['idata'], set_data['target']
            data.append((idata, target))
            print(f'Number of samples ({part}): {len(target)}')
    print("Using precomputed vocabulary and data files")
    print(f'Vocabulary size: {len(token_vocab)}')
    return token_vocab, data


def train(model, criterion, optimizer, idata, target, batch_size, device, log=False):
    """Train Transformer model with the defined parameters."""
    model.train()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    for X, y in batch_generator(idata, target, batch_size, shuffle=True):
        # Get input and target sequences from batch
        X = torch.tensor(X, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # Training statistics
        total_loss += loss.item()
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()
        ntokens += y.numel()
        niterations += 1
        if niterations == 200 or niterations == 500 or niterations % 1000 == 0:
            print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, \
                accuracy={100*ncorrect/ntokens:.1f}, loss={total_loss/ntokens:.2f}')

    total_loss = total_loss / ntokens
    accuracy = 100 * ncorrect / ntokens
    if log:
        print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, \
            accuracy={accuracy:.1f}, loss={total_loss:.2f}')
    return accuracy, total_loss

def validate(model, criterion, idata, target, batch_size, device):
    """Function to validate Transformer model."""
    model.eval()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    y_pred = []
    with torch.no_grad():
        for X, y in batch_generator(idata, target, batch_size, shuffle=False):
            # Get input and target sequences from batch
            X = torch.tensor(X, dtype=torch.long, device=device)
            output = model(X)
            if target is not None:
                y = torch.tensor(y, dtype=torch.long, device=device)
                loss = criterion(output, y)
                total_loss += loss.item()
                ncorrect += (torch.max(output, 1)[1] == y).sum().item()
                ntokens += y.numel()
                niterations += 1
            else:
                pred = torch.max(output, 1)[1].detach().to('cpu').numpy()
                y_pred.append(pred)

    if target is not None:
        total_loss = total_loss / ntokens
        accuracy = 100 * ncorrect / ntokens
        return accuracy, total_loss
    return np.concatenate(y_pred)

# Create working dir (check this!!)
# pathlib.Path(WORKING_ROOT).mkdir(parents=True, exist_ok=True)

# Select device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Training on MPS")
else:
    device = torch.device('cpu')
    print("WARNING: Training without GPU can be very slow!")

# Change this according to config
vocab, data = load_preprocessed_dataset(hyperparams['preprocessed'])

# Load model
model = Predictor(len(vocab), hyperparams['embedding_dim'], \
    num_heads=hyperparams['num_heads']).to(device)

print(model)
for name, param in model.named_parameters():
    print(f'{name:20} {param.numel()} {list(param.shape)}')
print(f'TOTAL                {sum(p.numel() for p in model.parameters())}')

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(reduction='sum')

tracker = EmissionsTracker()
tracker.start()

with mlflow.start_run():
    train_accuracy = []
    wiki_accuracy = []
    valid_accuracy = []
    mlflow.log_param("num_heads", hyperparams['num_heads'])
    mlflow.log_param("epochs", hyperparams['epochs'])
    mlflow.log_param("embedding_dim", hyperparams['embedding_dim'])
    mlflow.log_param("window_size", hyperparams['window_size'])
    mlflow.log_param("batch_size", hyperparams['batch_size'])
    mlflow.log_param("dataset_version", hyperparams['DATASET_VERSION'])
    for epoch in range(hyperparams['epochs']):
        acc, epoch_loss = train(model, criterion, optimizer, data[0][0], data[0][1], \
            hyperparams['batch_size'], device, log=True)
        train_accuracy.append(acc)
        mlflow.log_metric("training_acc", acc)
        mlflow.log_metric("training_loss", epoch_loss)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={epoch_loss:.2f}')
        acc, valid_loss = validate(model, criterion, data[1][0], data[1][1], \
            hyperparams['batch_size'], device)
        wiki_accuracy.append(acc)
        mlflow.log_metric("valid_acc", acc)
        mlflow.log_metric("valid_loss", valid_loss)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, \
            valid loss={valid_loss:.2f} (wikipedia)')
    mlflow.pytorch.save_model(model, f"./artifacts/{hyperparams['modelname']}")

tracker.stop()
