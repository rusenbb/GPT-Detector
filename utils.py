from transformers import AutoTokenizer, AutoModelForSequenceClassification
from os.path import exists
from os import mkdir
import numpy as np
from torch.nn import functional as F
import torch
from tqdm import tqdm

def get_model(model_name):
    if not exists("models"):
        mkdir("models")

    if not exists(f"models/{model_name}-model"):
        print("Model does not exist, downloading...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(f"./models/{model_name}-model")
        tokenizer.save_pretrained(f"./models/{model_name}-tokenizer")
    else:
        print("Model already exists, loading...")
        model = AutoModelForSequenceClassification.from_pretrained(f"./models/{model_name}-model")
        tokenizer = AutoTokenizer.from_pretrained(f"./models/{model_name}-tokenizer")

    return model, tokenizer


def model_predict(model, tokenizer, sentences):

    outputs_list = []
    # batch the sentences
    for i in tqdm(range(0, len(sentences), 100)):
        batch = sentences[i:i + 100]
        # Tokenize sentences
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        # Classify sentences
        with torch.no_grad():
            outputs = model(**inputs)
            outputs = F.softmax(outputs.logits, dim=1)
            outputs = outputs.tolist()
            outputs = [[round(x, 5) for x in output] for output in outputs]
            outputs_list.extend(outputs)
        
    # concatenate the outputs
    #outputs = torch.cat(outputs)

    # Tokenize sentences
    """inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    # Classify sentences
    outputs = model(**inputs)"""

    # softmax

    # turn into floats with 2 decimals
    

    return outputs_list


def get_label_output(predictions):
    # select the idmax of each prediction
    predictions = np.argmax(predictions, axis=1)
    return predictions