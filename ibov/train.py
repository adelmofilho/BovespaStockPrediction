import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import torch
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
from model import model_lstm, train
from utils import load_config
from request import label_train_test
from feature import create_lags, consolidate_features, create_delta_sign
from sklearn.preprocessing import MinMaxScaler

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
    model = IbovModel(input_layer=7)

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model


def loader(data_dir, config):

    config = load_config()
    filename = config.get("data").get("file")

    # Load data
    filepath = os.path.join(data_dir, filename)
    dados = pd.read_csv(filepath)

    # Data preparation
    dados = label_train_test(dados, split=test_split, split_valid=valid_split)
    
    # Feature Engineering
    scaler = MinMaxScaler()
    dados[['close']] = scaler.fit_transform(dados[['close']])

    ibov_lags_df = create_lags(ibovespa, window=window, var="close", index="date")
    ibov_delta_sign_df = create_delta_sign(ibov_lags_df, var="lags", index="date", window=window)
    master_table = consolidate_features(ibovespa, "date", ibov_lags_df, ibov_delta_sign_df)
    
    train_loader, train_x_tensor, train_y_tensor = \
        torch_data(master_table, target="target", variables=variables, group_var="group", batch=50, group="train")

    valid_loader, valid_x_tensor, valid_y_tensor = \
        torch_data(master_table, target="target", variables=variables, group_var="group", batch=50, group="valid")

    return train_loader, valid_loader


if __name__ == '__main__':

    # Set parameters
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', 
                        type=int, default=32, metavar='B',
                        help='Batch size for training dataset (default: 32)')

    parser.add_argument('--epochs', 
                        type=int, default=50, metavar='E',
                        help='Number of epochs to train (default: 50)')

    parser.add_argument('--seed',
                        type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Model Parameters
    parser.add_argument('--input-layer',
                        type=int, default=7, metavar='H',
                        help='Size of the input dimension (default: 7)')

    parser.add_argument('--hidden-layer',
                        type=int, default=50, metavar='H',
                        help='Size of the hidden dimension (default: 32)')

    parser.add_argument('--dropout',
                        type=float, default=0.25, metavar='D',
                        help='Dropout rate (default: 0.25)')

    # SageMaker Parameters
    parser.add_argument('--hosts', 
                        type=list, default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--current-host', 
                        type=str, default=os.environ['SM_CURRENT_HOST'])

    parser.add_argument('--model-dir', 
                        type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--data-dir', 
                        type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--config', 
                        type=str, default=os.environ['SM_CHANNEL_CONFIG'])

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load the training e validation data
    train_loader, valid_loader = loader(args.data_dir, args.config)

    # Build the model
    model = model_lstm(args.input_layer, args.hidden_layer, args.dropout)

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.L1Loss()

    train(model, train_loader, valid_loader, criterion, optimizer, args.epochs)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'seed': args.seed,
            'input_layer': args.input_layer,
            'hidden_layer': args.hidden_layer,
            'dropout': args.dropout
            }
        torch.save(model_info, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
