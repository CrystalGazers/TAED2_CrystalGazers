import torch
from transformer import Predictor

PATH="./"

# Load model somehow
model = Predictor()
model.load_state_dict(torch.load(PATH))

# Import test dataset from amazon

