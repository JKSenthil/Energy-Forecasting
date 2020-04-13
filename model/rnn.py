import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, n_in=10, n_out=96, n_layers=1, hidden_layer_size=192):
        super(RNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size

        # initialize neural network layers
        self.gru = nn.GRU(n_in, hidden_layer_size, num_layers=n_layers)
        self.dense1 = nn.Linear(hidden_layer_size + n_in, 128)
        self.rest = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_out),
        )

    def forward(self, inputs, curr_weather):
        """
        inputs will be of shape (sequence_length, batch_size, input_size)
        """
        # extract hidden state
        _, hidden_state = self.gru(inputs)
        hidden_state = hidden_state.view(1, 1, inputs.size()[1], self.hidden_layer_size)
        last_hidden_state = torch.squeeze(hidden_state[-1])
        
        # feed hidden state and curr_weather data into rest of network
        cat = torch.cat([last_hidden_state, curr_weather], dim=1)
        out = self.dense1(cat)
        return self.rest(out)