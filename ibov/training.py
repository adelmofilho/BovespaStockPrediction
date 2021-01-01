from torch import nn

class IbovModel(nn.Module):

    def __init__(self, window, hidden_layer=50, dropout=0.25):

        super(IbovModel, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(window, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)
        
    def forward(self, input):

        x = self.fc1(input)
        x = self.dropout(x)
        output = self.fc2(x)

        return output


def train(data, model, criterion, optimizer, epochs):

    for epoch in range(epochs):
        
        model.train()
        loss_value = 0.0
        
        for batch in data:

            batch_x, batch_y = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # print statistics
            loss_value += loss.item()
        print(loss_value)
    model.eval()