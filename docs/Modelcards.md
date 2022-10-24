---
language:
- ca
tags:
- wikipedia
- transformer
- text prediction
- natural language prediction
license:
- apache-2.0
datasets:
- [Viquipèdia](https://www.kaggle.com/datasets/jarfo1/viquipdia)
metrics:
- accuracy
- CrossEntropyLoss
- CO2 emission
---

# Crystal-Gazers
This universitary project aims to train a ML model with the Wikipedia dataset in order to predict the hidden word given a sequence of context words at both sides. The model will output a word for every input given. The model is executed an developped in [Kaggle.com](https://www.kaggle.com) .

## Model description

To archieve the desired prediction we use a Transformer model with two layers. A transformer is a model that adopts the mechanism of self-attention, deferentially weighting the significance of each part of the input data. The model is trained on text data which was originally created by humans. 

The model file gives the possibility to introduce multihead attention. At the end, there is an accuracy comparison based on the performance of both methods. 

The training of the model is based on Cross Entropy Loss and the parameters are updated by Adam optimizer, which is chosen to be the optimum empirically. The script is developped in Python using `torch` , `numpy` and `pandas` libraries among others.

The model was created by [Jose A.R. Fonollosa](https://www.kaggle.com/jarfo1)

The observed accuracy metric after training are the following:

## Dataset
The Wikipedia dataset is a collection of scraped Wikipedia pages. The dataset is defined in catalan language, thus the model is trained to recognize input exclusively in catalan. There are 2 versions of the dataset, `ca-2` and `ca-100`, with the same obtention method but the latter collecting more data from more articles. The splits used to train this model are the following:

|                         | train  | validation | test |
|-------------------------|-------:|-----------:|-----:|
| ca-2                    |11MB    |1.1MB       |1.1MB |
| ca-100                  |505MB   |1.1MB       |1.1MB |

The dataset can be found [here](https://www.kaggle.com/datasets/jarfo1/viquipdia).

## Results

---
- ca-2:
  - Single-head:
    - Accuracy: 35.6%
    - CrossEntropyLoss: 4.33
    - CO2 emissions: 0.00333
  - Multi-head:
    - Accuracy: 36.4%
    - CrossEntropyLoss: 4.29
    - CO2 emissions: 0.00342
- ca-100:
  - Single-head:
    - Accuracy: 46.4%
    - CrossEntropyLoss: 3.13
    - CO2 emissions: 0.1543
---
