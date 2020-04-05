import numpy as np
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, n_out, n_layers=2, hidden_layer_size=192):
        super(RNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size

        # initialize neural network layers
        self.gru = nn.GRU(92, hidden_layer_size, num_layers=n_layers)
        self.dense1 = nn.Linear(hidden_layer_size + 91, 512)
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
        hidden_state = hidden_state.view(2, 1, inputs.size()[1], self.hidden_layer_size)
        last_hidden_state = torch.squeeze(hidden_state[-1])
        
        # feed hidden state and curr_weather data into rest of network
        cat = torch.cat([last_hidden_state, curr_weather], axis=1)
        out = nn.ReLU()(self.dense1(cat))
        return self.dense4(nn.ReLU()(self.dense3(nn.ReLU()(self.dense2(out)))))
