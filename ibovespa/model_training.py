import pandas as pd
from torch import nn, manual_seed
import torch
import time
import os
import random
import numpy as np
from ast import literal_eval
from torch.nn import L1Loss
from torch.optim import Adam

def read_feature_table(path, target, variables):
    
    feature_table = pd.read_csv(path)
    for var in [target]+variables:
        feature_table[var] = feature_table[var].apply(literal_eval)
        
    return feature_table


def train_model(feature_table, variables, window, hyperparameters):

    train_loader, train_x_tensor, train_y_tensor = \
        torch_data(feature_table, target="target", variables=variables, group_var="group", batch=50, group="train")

    valid_loader, valid_x_tensor, valid_y_tensor = \
        torch_data(feature_table, target="target", variables=variables, group_var="group", batch=50, group="valid")

    test_loader, test_x_tensor, test_y_tensor = \
        torch_data(feature_table, target="target", variables=variables, group_var="group", batch=50, group="test")

    model = Model(input_layer=window, 
                  hidden_layer=hyperparameters["hidden_layer"], 
                  dropout=hyperparameters["dropout"])

    criterion = L1Loss()
    optimizer = Adam(model.parameters(), lr=hyperparameters["lr"])

    tensores = {"train_x_tensor": train_x_tensor,
                    "train_y_tensor": train_y_tensor,
                    "valid_x_tensor": valid_x_tensor,
                    "valid_y_tensor": valid_y_tensor,
                    "test_x_tensor": test_x_tensor,
                    "test_y_tensor": test_y_tensor}

    train(model, 
          train_loader, valid_loader, 
          criterion, optimizer, 
          epochs=hyperparameters["epochs"], 
          seed=42)

    return model, tensores


def torch_data(data, target, variables, group_var, batch, group):
    
    if group is not None:
        data  = data[data[group_var] == group].reset_index()
    
    x_tensor = torch.Tensor(data[variables].values.tolist())
    y_tensor = torch.Tensor(data[target])
    
    dataset = torch.utils.data.TensorDataset(x_tensor,y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch)
    
    return loader, x_tensor, y_tensor


# class model_fc2h(nn.Module):

#     def __init__(self, input_layer, hidden_layer=50, dropout=0.25):

#         super(model_fc2h, self).__init__()
        
#         self.dropout = nn.Dropout(dropout)
#         self.fc1 = nn.Linear(input_layer, hidden_layer)
#         self.fc2 = nn.Linear(input_layer, hidden_layer)
#         self.fc3 = nn.Linear(hidden_layer*2, 1)
        
#         self.tanh = nn.Tanh()
        
#     def forward(self, input):
       
#         x_lags = input[:,0] 
#         x_sign = input[:,1]
        
#         out_lags = self.fc1(x_lags)
#         out_lags = self.dropout(out_lags)

#         out_sign = self.fc2(x_sign)
#         out_sign = self.tanh(out_lags)


#         output = self.fc3(torch.cat((out_lags, out_sign), 1))

#         return output
    
    
# class model_lstm(nn.Module):

#     def __init__(self, input_layer, hidden_layer, dropout):

#         super(model_lstm, self).__init__()
        
#         self.hidden_layer = hidden_layer
        
#         self.hidden_cell = (torch.zeros(1,1,self.hidden_layer),
#                             torch.zeros(1,1,self.hidden_layer))
        
#         self.lstm = nn.LSTM(input_layer, hidden_layer)

#         self.linear = nn.Linear(hidden_layer, 1)
        
#     def forward(self, input):
       
#         x_lags = input
#         lstm_out, self.hidden_cell = self.lstm(x_lags.view(len(x_lags),1 , -1), self.hidden_cell)
      
#         output = self.linear(lstm_out)
#         return lstm_out[:,:,0]


# class model_fc1h(nn.Module):

#     def __init__(self, input_layer, hidden_layer=50, dropout=0.25):

#         super(model_fc1h, self).__init__()
        
#         self.dropout = nn.Dropout(dropout)
#         self.fc1 = nn.Linear(input_layer, hidden_layer)
#         self.fc2 = nn.Linear(hidden_layer, 1)
        
#     def forward(self, input):

#         x = self.fc1(input)
#         x = self.dropout(x)
#         output = self.fc2(x)

#         return output[:,:,0]


class Model(nn.Module):

    def __init__(self, input_layer, hidden_layer=50, dropout=0.25):

        super(Model, self).__init__()
        self.hidden_layer = hidden_layer
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer*2+7, 1)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer),
                            torch.zeros(1,1,self.hidden_layer))
        
        self.lstm = nn.LSTM(input_layer, hidden_layer)
        
    def forward(self, input):

        lags = input[:,0,:]
        delta_sign = input[:,1,:]
        weekday_vector = input[:,2,:]

        #lag_pct_IBOV = input[:,3,:]
        #lag_pct_IBOV = lag_pct_IBOV.view(len(input[:,3,:]),1,-1)[:,:,0]

        fc1_out = self.dropout(self.fc1(lags))

        lstm_out, self.hidden_cell = self.lstm(delta_sign.view(len(delta_sign),1 , -1), self.hidden_cell)
        
        ds = torch.cat((fc1_out,lstm_out[:,0,:], weekday_vector),1)
        output = self.fc2(ds)

        return output


def train(model, trainData, validData, criterion, optimizer, epochs, seed):

    manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)      

    for epoch in range(epochs):
        
        # Restart model training status
        model.train()
        train_loss = 0.0
        valid_loss = 0.0
        
        # Training model
        for batch in trainData:
            
            batch_x, batch_y = batch
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
            
        # Turn model to evaluation mode
        model.eval()

        # Evaluate model on validation dataset
        for batch in validData:

            batch_x, batch_y = batch
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            valid_loss += loss.item()

        # Log epoch results
        now = time.strftime("%H:%M:%S")
        rnd_ltrn = round(train_loss, 3)
        rnd_lvld = round(valid_loss, 3)
        print("{}, epoch: {}, train: {}, valid: {}".format(now, epoch, rnd_ltrn, rnd_lvld))
        
    # Set model to evaluation mode
    model.eval()