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
from model import IbovModel, train


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

def create_lags(data, window):

    y_list = list()
    x_list = list()

    for idx in range(data.shape[0]-window):
        y = [data[idx]]
        x = list(data[(idx+1):(window+idx+1)].values)
        y_list.append(y)
        x_list.append(x)
    
    return y_list, x_list


def loader(batch_size, training_dir, file):
    window = 7
    # Inicialization
    print("Get train data loader.")

    # Load data
    filepath = os.path.join(training_dir, file)
    dados_ibov = pd.read_csv(filepath)

    dados_ibov_train = dados_ibov.sort_values(by="date", ascending=False)[180:360].reset_index(drop="True")
    dados_ibov_valid = dados_ibov.sort_values(by="date", ascending=False)[0:180].reset_index(drop="True")

    trainY, trainX = create_lags(data = dados_ibov_train["close"], window = window)
    validY, validX = create_lags(data = dados_ibov_valid["close"], window = window)

    tensor_x = torch.Tensor(np.array(trainX))
    tensor_y = torch.Tensor(np.array(trainY))

    tensor_val_x = torch.Tensor(np.array(validX))
    tensor_val_y = torch.Tensor(np.array(validY))

    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

    val_dataset = torch.utils.data.TensorDataset(tensor_val_x,tensor_val_y)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

    return dataloader, val_dataloader

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

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load the training e validation data
    train_loader, valid_loader = \
        loader(args.batch_size, args.data_dir, file="train.csv")
    
    # Build the model
    model = IbovModel(args.input_layer, args.hidden_layer, args.dropout)

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

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
