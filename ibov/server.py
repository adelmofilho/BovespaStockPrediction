import argparse
import json
import io
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler

from model import model_lstm, train, torch_data
from utils import load_config
from feature import create_lags, consolidate_features, create_delta_sign, label_train_test


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model...")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lstm(input_layer=model_info["input_layer"], 
                        hidden_layer=model_info["hidden_layer"], 
                        dropout=model_info["dropout"])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/csv':
        dados = pd.read_csv(io.StringIO(serialized_input_data.decode('utf-8'), sep=","))

        scaler = MinMaxScaler()
        dados[['close']] = scaler.fit_transform(dados[['close']])

        ibov_lags_df = create_lags(dados, window=window, var="close", index="date")
        ibov_delta_sign_df = create_delta_sign(ibov_lags_df, var="lags", index="date", window=window)
        master_table = consolidate_features(dados, "date", ibov_lags_df, ibov_delta_sign_df)
        
        train_loader, train_x_tensor, train_y_tensor = \
            torch_data(master_table, target="target", variables=variables, group_var="group", batch=50, group=None)

        return train_loader
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0

    result = model(input_data).detach().numpy() 

    return result