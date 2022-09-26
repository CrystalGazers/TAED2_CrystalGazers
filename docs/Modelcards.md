# Crystal-Gazers
This universitary project aims to train a ML model with the Wikipedia dataset in order to predict the next word given a sequence of words. The model will output a word for every input given. The model is executed an developped in [Kaggle.com](https://www.kaggle.com) .

## Model description

To archieve the desired prediction we will use a Transformer model with two layers. A transformer is a model that adopts the mechanism of self-attention, deferentially weighting the significance of each part of the input data. The model will be trained on text data which was originally created by humans. 

The model file gives the possibility to introduce multihead attention. At the end, there will be an accuracy comparison based on the performance of both methods. 

The training of the model is based on Cross Entropy Loss and the parameters are updated by Adam optimizer, which is chosen to be the optimum empirically. The script is developped in Python using `torch` , `numpy` and `pandas` libraries among others.

The model were created by [Jose A.R. Fonollosa](https://www.kaggle.com/jarfo1)

The observed accuracy metric after training are the following:
Single-head: 39.27%
Multi-head: 40.34%

## Dataset
The Wikipedia dataset is a collection of scraped Wikipedia pages. The dataset is defined in catalan language, thus the model will be trained to recognize input exclusively in catalan.
The dataset can be found [here](https://www.kaggle.com/datasets/jarfo1/viquipdia).

---

language:
  - ca
  
tags:
  - prediction
  - NLP
  - text
  - transformer
  - CrossEntropy
  - text generation
  
license:  
  - apache-2.0
  
dataset:
  - Viquipèdia
  
metrics:
  - accuracy
  - CrossEntropyLoss
  - CO2 emission

---
