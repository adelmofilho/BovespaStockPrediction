import pandas as pd
from torch import nn
import pylab as pl
from IPython import display

class Ibovespa(nn.Module):

    def __init__(self, input_layer, hidden_layer=50, dropout=0.25):

        super(Ibovespa, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)
        
    def forward(self, input):

        x = self.fc1(input)
        x = self.dropout(x)
        output = self.fc2(x)

        return output


def train(model, trainData, validData, criterion, optimizer, epochs):

    loss_list = []
    valid_loss_list = []

    for epoch in range(epochs):
        
        # Restart model training status
        model.train()
        # Restart loss value
        loss_value = 0.0
        valid_loss_value = 0.0
        
        for batch in trainData:
            # Get training data
            batch_x, batch_y = batch
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()
            # Update Loss
            loss_value += loss.item() / len(batch_y)
            # Turn model to evaluation mode
            model.eval()
        # Predict over validation dataset
        for batch in validData:
            # Get validation data
            batch_x, batch_y = batch
            # Prediction
            outputs = model(batch_x)
            # Validation loss
            valid_loss = criterion(outputs, batch_y)
            # Update loss
            valid_loss_value += valid_loss.item() / len(batch_y)
        # Plot loss across epochs
        loss_list.append(loss_value)
        valid_loss_list.append(valid_loss_value)
        # print(valid_loss_value)
        pl.plot(loss_list, '-b', label="TrainLoss")
        pl.plot(valid_loss_list, '-r', label="ValidLoss")
        # if epoch == 1:
        #     pl.legend(loc='upper right')
        display.display(pl.gcf())
        display.clear_output(wait=True)
    print(f"training error: {loss_value}")
    print(f"validation error: {valid_loss_value}")
    model.eval()