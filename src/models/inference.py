from transformer import Predictor
import os
import torch
import json
import pickle
import logging
import sys
import boto3
import io

handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
log = logging.getLogger(__name__)

def download_s3():
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket='crystal-gazers', Key='preprocessed/ca-100/ca.wiki.vocab')
    f = io.BytesIO(obj["Body"].read())
    return f

file = pickle.load(download_s3())
token2idx = file['token2idx']
idx2token = file['idx2token']
len_vocab = len(idx2token)
emb_size = 256
def model_fn(model_dir):
    model = Predictor(len_vocab, emb_size)
    device = torch.device("cpu")
    model_dir = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device).eval()
    log.info("Model loaded")
    return model

def input_fn(request_body, request_content_type):
    log.info("Starting input")
    assert request_content_type=='application/json'
    device = torch.device("cpu")
    data = json.loads(request_body)['input']
    tokens = []
    for x in data:
        if x not in token2idx:
            idx = token2idx['<unk>']
        else:
            idx = token2idx[x]
        tokens.append(idx)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    log.info("Input ready")
    return tokens

def predict_fn(input_object, model):
    log.info("Starting prediction")
    with torch.no_grad():
        prediction = model(input_object)
        prediction[0][1] = -10
    pred = torch.max(prediction, 1)[1].detach().to('cpu').numpy()
    pred = idx2token[pred[0]]
    log.info("Predictions ready")
    return pred

def output_fn(predictions, content_type):
    log.info("Starting output")
    assert content_type == 'application/json'
    res = {"prediction": predictions}
    log.info("Output ready")
    return json.dumps(res)