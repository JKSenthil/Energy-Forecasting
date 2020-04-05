import numpy as np
import torch
import torch.nn as nn

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class BasicMLP(nn.Module):
    """  
    Neural Network which only uses current weather to
    make immediate energy demand prediction, 1 time step
    """
    def __init__(self):
        super(BasicMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(11 * 154 + 10 * 154, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 154)
        )

    def forward(self, curr_weather):
        curr_weather = torch.from_numpy(curr_weather).float()
        return self.model(curr_weather)

class RNN(nn.Module):
    def __init__(self, n_out, n_layers=1, hidden_layer_size=192):
        super(RNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size

        # initialize neural network layers
        self.gru = nn.GRU(11, hidden_layer_size, num_layers=n_layers)
        self.dense1 = nn.Linear(hidden_layer_size + 154 * 10, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 256)
        self.dense4 = nn.Linear(256, n_out)

    def forward(self, inputs, curr_weather):
        """
        inputs will be of shape (sequence_length, batch_size, input_size)
        """
        # convert inputs from numpy to pytorch tensor
        inputs = torch.from_numpy(inputs).float()
        curr_weather = torch.from_numpy(curr_weather).float()

        # extract hidden state
        _, hidden_state = self.gru(inputs)
        hidden_state = hidden_state.view(1, 1, inputs.size()[1], self.hidden_layer_size)
        last_hidden_state = torch.squeeze(hidden_state[-1])
        
        # feed hidden state and curr_weather data into rest of network
        cat = torch.cat([last_hidden_state, curr_weather], axis=1)
        out = nn.ReLU()(self.dense1(cat))
        return self.dense4(nn.ReLU()(self.dense3(nn.ReLU()(self.dense2(out)))))
