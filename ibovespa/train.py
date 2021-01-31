import argparse
import json
import io
import os
import pickle
import sys
import sagemaker_containers
import torch
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
from utils import load_config
from model_training import train_model, read_feature_table, torch_data
from feature_engineering import Normalize



def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model...")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    input_layer = model_info["config"]["model_training"]["hyperparameters"]["input_layer"]
    hidden_layer = model_info["config"]["model_training"]["hyperparameters"]["hidden_layer"]
    dropout = model_info["config"]["model_training"]["hyperparameters"]["dropout"]
    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(input_layer=input_layer, 
                        hidden_layer=hidden_layer, 
                        dropout=dropout)

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    model.maximo = model_info["maximo"]
    model.minimo = model_info["minimo"]
    model.window = model_info["input_layer"]
    model.config = model_info["config"]
    print("Done loading model.")
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/csv':  

        dados = pd.read_csv(io.StringIO(serialized_input_data), sep=",")

        return dados
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    saida = np.array(prediction_output).flatten().tolist()

    return str(saida)


def predict_fn(input_data, model):
   
    scaler = Normalize()
    scaler.load_configs(maximo=model.maximo, minimo=model.minimo)

    train_loader, train_x_tensor, train_y_tensor = \
        torch_data(input_data, 
                   target="target", 
                   variables=model.config["feature"]["variables"], 
                   group_var="group", 
                   batch=50, 
                   group=None)

    model.eval()

    result = scaler.denormalize(model(train_x_tensor).detach().numpy())

    return result


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
    # parser.add_argument('--input-layer',
    #                     type=int, default=7, metavar='H',
    #                     help='Size of the input dimension (default: 7)')

    # parser.add_argument('--hidden-layer',
    #                     type=int, default=50, metavar='H',
    #                     help='Size of the hidden dimension (default: 32)')

    # parser.add_argument('--dropout',
    #                     type=float, default=0.25, metavar='D',
    #                     help='Dropout rate (default: 0.25)')

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

    parser.add_argument('--scaler', 
                        type=str, default=os.environ['SM_CHANNEL_SCALER'])

    args = parser.parse_args()

    # Read configurations

    config = load_config(os.path.join(args.config, "config.json"))

    with open(os.path.join(args.scaler, "scaler.json")) as f:
        scaler = json.load(f)

    # Read table of features

    filename = os.path.join(args.data_dir, "data.csv")
    target = config["model_training"]["target"]
    variables = config["model_training"]["variables"]

    feature_table = read_feature_table(filename, target, variables)

    # Train-Test split

    window = config["feature_engineering"]["window"]
    target = config["model_training"]["target"]
    variables = config["model_training"]["variables"]
    hyperparameters = config["model_training"]["hyperparameters"]

    model, tensores = train_model(feature_table, variables, window, hyperparameters)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            # 'batch_size': args.batch_size,
            # 'epochs': args.epochs,
            # 'seed': args.seed,
            # 'input_layer': args.input_layer,
            # 'hidden_layer': args.hidden_layer,
            # 'dropout': args.dropout,
            "maximo": scaler["maximo"],
            "minimo": scaler["minimo"],
            "config": config
            }
        torch.save(model_info, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
