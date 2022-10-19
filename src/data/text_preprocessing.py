from types import SimpleNamespace
from collections import Counter
import re
import pathlib
import pickle
import numpy as np

class Vocabulary:
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

    def get_token(self, index):
        """Get token from index."""
        return self.idx2token[index]

    def __len__(self):
        """Return length of vocabulary."""
        return len(self.idx2token)

    def save(self, filename):
        """Save vocabulary into pickle file."""
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, filename):
        """Load vocabulary from pickle file."""
        with open(filename, 'rb') as file:
            self.__dict__.update(pickle.load(file))

class Punctuation:
    """Class that manages punctuation."""
    html = re.compile(r'&apos;|&quot;')
    punctuation = re.compile(r'[^\w\s·]|_')
    spaces = re.compile(r'\s+')
    ela_geminada = re.compile(r'l · l')

    def strip(self, sent):
        """
        Remove all punctuation characters.
        """
        sent = self.html.sub(' ', sent)
        sent = self.punctuation.sub(' ', sent)
        sent = self.spaces.sub(' ', sent).strip()
        sent = self.ela_geminada.sub('l·l', sent)
        return sent

def remove_punctuation(input_path, output_path):
    """Remove punctuation for each line."""
    punc = Punctuation()
    with open(input_path, 'r', encoding='utf-8') as inpf, \
        open(output_path, 'w', encoding='utf-8') as outf:
        for line in inpf:
            line = punc.strip(line)
            print(line, file=outf)

def get_token_counter(file_path):
    """Count tokens in file."""
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                tokens = line.split()
                counter.update(tokens)
    return counter

def get_token_vocabulary(token_counter, cutoff=3, maxtokens=None, verbose=1, eos_token=None):
    """Define token vocabulary."""
    vocab = Vocabulary(eos_token=eos_token)
    total_count = sum(token_counter.values())
    in_vocab_count = 0

    for token, count in token_counter.most_common(maxtokens):
        if count >= cutoff:
            vocab.add_token(token)
            in_vocab_count += count

    if verbose:
        oov_count = total_count - in_vocab_count
        print('OOV ratio: %.2f%%.' % (100*oov_count / total_count))
    return vocab

def get_token_index(file_path, vocab, eos_token=None):
    """Get index from each token in a file."""
    index_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                if eos_token is not None:
                    line += ' ' + eos_token
                tokens = line.strip().split()
                index_list.append([vocab.get_index(token) for token in tokens])
    return index_list

def get_number_of_samples(idx_list, window_size):
    """Get number of samples in index list"""
    nsamples = 0
    for line in idx_list:
        if len(line) <= window_size // 2:
            continue
        nsamples += len(line)
    return nsamples

def get_data(idx_list, window_size, pad_index=0):
    """Get prepared windows to be embed into the model."""
    nsamples = get_number_of_samples(idx_list, window_size)
    winput = np.empty((nsamples, window_size - 1), dtype=np.int32)
    target = np.empty(nsamples, dtype=np.int32)
    left_window = window_size // 2
    right_window = window_size - left_window - 1
    sample = 0
    for line in idx_list:
        if len(line) <= window_size // 2:
            continue
        ext_line = [pad_index] * left_window + line + [pad_index] * right_window
        for i, token_id in enumerate(line):
            winput[sample] = ext_line[i:i + left_window] + \
                             ext_line[i + left_window + 1:i + window_size]
            target[sample] = token_id
            sample += 1
    assert nsamples == sample
    return winput, target

def prepare_dataset(params):
    """Prepared the dataset to be embed into the model."""
    dataset_prefix = params.dataset
    working_prefix = params.working
    cutoff = params.cutoff
    maxtokens = params.maxtokens
    window_size = params.window_size

    data = []
    for part in ['train', 'valid', 'test']:
        data_filename = f'{dataset_prefix}.{part}.tokens'
        data_filename_nopunct = f'{working_prefix}.{part}.tokens.nopunct'
        remove_punctuation(data_filename, data_filename_nopunct)

        if part == 'train':
            # Basic token statistics
            token_counter = get_token_counter(data_filename_nopunct)
            print(f'Number of Tokens: {sum(token_counter.values())}')
            print(f'Number of different Tokens: {len(token_counter)}')
            with open(f'{data_filename_nopunct}.dic', 'wb') as datafile:
                pickle.dump(token_counter, datafile)

            # Token vocabulary
            token_vocab = get_token_vocabulary(token_counter, cutoff=cutoff, maxtokens=maxtokens)
            token_vocab.save(f'{working_prefix}.vocab')
            print(f'Vocabulary size: {len(token_vocab)}')

        # Token indexes
        train_idx = get_token_index(data_filename_nopunct, token_vocab)
        print(f'Number of lines ({part}): {len(train_idx)}')

        # Get input and target arrays
        idata, target = get_data(train_idx, window_size)
        data.append((idata, target))
        print(f'Number of samples ({part}): {len(target)}')

        # Save numpy arrays
        np.savez(f'{working_prefix}.{part}.npz', idata=idata, target=target)
    return token_vocab, data

DATASET_ROOT = './raw'
WORKING_ROOT = './preprocessed'
DATASET_PREFIX = 'ca.wiki'

for DATASET_VERSION in ["ca-all"]:#os.listdir(DATASET_ROOT):
    # Create working dir
    pathlib.Path(f"{WORKING_ROOT}/{DATASET_VERSION}").mkdir(parents=True, exist_ok=True)

    params_def = SimpleNamespace(
        window_size=7,
        cutoff=3,
        maxtokens=100000,
        dataset=f'{DATASET_ROOT}/{DATASET_VERSION}/{DATASET_PREFIX}',
        working=f'{WORKING_ROOT}/{DATASET_VERSION}/{DATASET_PREFIX}',
    )

    print(f"Processing {DATASET_VERSION}")
    tok_vocab, prepared_data = prepare_dataset(params_def)

    print('token to index:')
    for word in ['raïm', 'intel·ligent']:
        ex_index = tok_vocab.get_index(word)
        print(f'{word} -> {ex_index}')

    print('\nindex to token:')
    for ex_index in [8428, 7466]:
        word = tok_vocab.get_token(ex_index)
        print(f'{ex_index} -> {word}')
