import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from model import IbovModel, train

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def feature_eng(data):


def loader(batch_size, training_dir, file):

    # Inicialization
    print("Get train data loader.")

    # Load data
    filepath = os.path.join(training_dir, file)
    data = pd.read_csv(filepath, header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

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
                        type=int, default=32, metavar='H',
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

    parser.add_argument('--train-dir', 
                        type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--valid-dir', 
                        type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load the training e validation data
    train_loader = loader(args.batch_size, args.train_dir, file="train.csv")
    valid_loader = loader(args.batch_size, args.valid_dir, file="valid.csv")

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
