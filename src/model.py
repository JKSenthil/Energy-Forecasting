import numpy as np
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, n_out, lr=1e-4, hidden_layer_size=256):
        super(RNN, self).__init__()

        # initialize neural network layers
        self.gru = nn.GRU(92, hidden_layer_size)
        self.dense1 = nn.Linear(hidden_layer_size + 91, 128)
        self.dense2 = nn.Linear(128, n_out)

    def forward(self, inputs, curr_weather):
        """
        inputs will be of shape (sequence_length, batch_size, input_size)
        """
        # convert inputs from numpy to pytorch tensor
        inputs = torch.from_numpy(inputs).float()
        curr_weather = torch.from_numpy(curr_weather).float()

        gru_output, _ = self.gru(inputs)
        gru_output = gru_output[-1,:,:]
        cat = torch.cat([gru_output, curr_weather], axis=1)
        out = nn.ReLU()(self.dense1(cat))
        return self.dense2(out)
