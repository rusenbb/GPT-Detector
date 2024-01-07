from transformers import AutoTokenizer, AutoModelForSequenceClassification
from os.path import exists
from os import mkdir
import numpy as np
from torch.nn import functional as F
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_name: str):
    """Get the model and tokenizer from the model name"""

    if not exists("Models"):
        mkdir("Models")

    if not exists(f"Models/{model_name}-model"):
        print(f"Model {model_name} does not exist, downloading...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(f"./Models/{model_name}-model")
        tokenizer.save_pretrained(f"./Models/{model_name}-tokenizer")
    else:
        print(f"Model {model_name} already exists, loading...")
        model = AutoModelForSequenceClassification.from_pretrained(f"./Models/{model_name}-model")
        tokenizer = AutoTokenizer.from_pretrained(f"./Models/{model_name}-tokenizer")

    return model, tokenizer


def model_predict(model, tokenizer, sentences):
    """
    Predict the labels of the sentences using the model and tokenizer
    Args:
        model: Model (transformers)
        tokenizer: Tokenizer (transformers tokenizer)
        sentences: Sentences to predict (ndarray)
    Returns:
        predictions: Predicted labels
    """

    outputs_list = []
    # batch the sentences
    for i in tqdm(range(0, len(sentences), 100)):
        batch = sentences[i:i + 100]
        # Tokenize sentences
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        # Classify sentences
        with torch.no_grad():
            outputs = model(**inputs) # get the logits
            labels = np.argmax(outputs.logits.to("cpu"), axis=1)
            labels = labels.tolist() # convert to list
            outputs_list.extend(labels) # add to outputs list


    return outputs_list


def eval_model(pred_labels:list, y_true:list, return_scores=False):
    """
    Evaluate the model by computing `precision`, `recall` and `F1-score`.
    Args:
        pred_labels: Predicted labels
        y_true: True labels
    Returns:
        conf_matrix: Confusion matrix
    Returns if return_scores is True:
        conf_matrix: Confusion matrix
        accuracy: Accuracy score
        recall: Recall score
        precision: Precision score
        f1: F1 score
    """
    #Evaluate the model by computing precision, recall and F1-score.
    accuracy = accuracy_score(y_true, pred_labels)
    recall = recall_score(y_true, pred_labels, average='macro')
    precision = precision_score(y_true, pred_labels, average='macro')
    f1 = f1_score(y_true, pred_labels, average='macro')
    
    conf_matrix = confusion_matrix(y_true, pred_labels)
    
    print('Accuracy: {:.2f}%'.format(accuracy*100))
    print('Recall: {:.2f}%'.format(recall*100))
    print('Precision: {:.2f}%'.format(precision*100))
    print('F1-score: {:.2f}%'.format(f1*100))
    if return_scores:
        return conf_matrix, (accuracy, recall, precision, f1)
    return conf_matrix