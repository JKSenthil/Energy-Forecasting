import torch
import torch.nn as nn

class cRNN(nn.Module):
    """
    Implements a conditional Recurrent Neural Network as guided by 
    https://datascience.stackexchange.com/questions/17099/adding-features-to-time-series-model-lstm/17139#17139
    """
    def __init__(self, lag_size, curr_size, hidden_size):
        super(cRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dense1 = nn.Linear(lag_size, hidden_size)
        self.gru = nn.GRUCell(curr_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, 1)
    
    def forward(self, lag, curr):
        outputs = []
        h_t = self.dense1(lag) # embed lag information into hidden state
        for weather in torch.chunk(curr, curr.size(1), dim=1):
           h_t = self.gru(weather, h_t)
           prediction = self.dense2(h_t)
           outputs += [prediction]
        return torch.stack(outputs, 1).squeeze(2)

class cRNNv2(nn.Module):
    """
    TODO: unfinished
    Implements SOTA conditional Recurrent Neural Network
    based on https://project.inria.fr/aaltd19/files/2019/08/AALTD_19_vanDerLugt.pdf
    """
    def __init__(self, input_size, hidden_size):
        super(cRNNv2, self).__init__()

        self.encoder = nn.GRU(input_size, hidden_size)
        self.decoder = nn.GRU(hidden_size + hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        _, h_n = self.encoder(inputs)
        h_n = h_n[-1]
        cat = torch.cat([inputs, h_n], dim=1)
        output, _ = self.decoder(cat)
        pass

