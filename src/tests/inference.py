import json
import logging
import os
import pickle
import sys

import boto3
import torch
from transformer import Predictor

boto3.setup_default_session(profile_name='cgmaria')
client = boto3.client('sagemaker-runtime')

handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)

log = logging.getLogger(__name__)

len_vocab = 100002
emb_size = 256
filename = 'ca.wiki.vocab'

with open(filename, 'rb') as f:
    file = pickle.load(f)

token2idx = file['token2idx']
idx2token = file['idx2token']

def model_fn(model_dir):
    '''Function to import the model'''
    model = Predictor(len_vocab, emb_size)
    device = torch.device("cpu")
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.to(device).eval()
    log.info("Model loaded")
    return model

def input_fn(request_body, request_content_type):
    '''Funtion to preprocess the input which will be sent to the model'''
    assert request_content_type == 'application/json'
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
    '''Function that passes the input to the model and makes the prediction'''
    with torch.no_grad():
        prediction = model(input_object)
    pred = torch.max(prediction, 1)[1].detach().to('cpu').numpy()
    pred = idx2token[pred[0]]
    log.info("Predictions ready")
    return pred

def output_fn(predictions, content_type):
    '''Function that preprocesses the output diven by preduct_fn() into the correct format'''
    assert content_type == 'application/json'
    res = predictions
    log.info("Output ready")
    return json.dumps(res)

def fetch(text):
    '''Function that makes a call to the API and returns a prediction'''
    try:
        #Sagemaker request
        data = {'input': text}
        body = str.encode(json.dumps(data))
        response = client.invoke_endpoint(
            EndpointName='pytorch-inference-2022-10-16-11-01-23-464',
            ContentType='application/json',
            Accept='application/json',
            Body=body
            )
        res_json = json.loads(response['Body'].read().decode("utf-8"))
        #print(json.load(response))
        return res_json, 1

    except Exception as err:
        return err, 0
    