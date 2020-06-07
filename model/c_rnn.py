import torch
import torch.nn as nn
import torch.nn.functional as F

class cRNN(nn.Module):
    """
    Implements a conditional Recurrent Neural Network as guided by 
    https://datascience.stackexchange.com/questions/17099/adding-features-to-time-series-model-lstm/17139#17139
    """
    def __init__(self, lag_size, latent_size, weather_size, gru_hiddensize=256):
        super(cRNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(lag_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size, gru_hiddensize)
        )
        self.gru = nn.GRUCell(weather_size, gru_hiddensize)
        self.dense3 = nn.Linear(gru_hiddensize, 1)
    
    def forward(self, lag, curr):
        outputs = []
        
        # embed lag information into hidden state
        h_t = self.encoder(lag)
        for weather in curr:
            h_t = self.gru(weather, h_t)
            prediction = self.dense3(h_t)
            outputs += [prediction]
        return torch.stack(outputs, 1).squeeze(2)

class cRNNv2(nn.Module):
    """
    Implements SOTA conditional Recurrent Neural Network
    based on https://project.inria.fr/aaltd19/files/2019/08/AALTD_19_vanDerLugt.pdf
    """
    def __init__(self, enc_input_size, dec_input_size, enc_hidden_size, dec_hidden_size):
        super(cRNNv2, self).__init__()

        self.encoder = nn.GRUCell(enc_input_size, enc_hidden_size)
        self.decoder = nn.GRUCell(enc_hidden_size + dec_input_size, dec_hidden_size)
        self.dense = nn.Linear(dec_hidden_size, 1)

    def forward(self, lag, curr):
        outputs = []
        h_t = None

        # encoder goes through lag data
        for demand in lag:
            if type(h_t) == type(None):
                h_t = self.encoder(demand)
            else:
                h_t = self.encoder(demand, h_t)

        h_tz = None
        for weather in curr:
            z = torch.cat([h_t, weather], dim=1)
            if type(h_tz) == type(None):
                h_tz = self.decoder(z)
            else:
                h_tz = self.decoder(z, h_tz)
            prediction = self.dense(h_tz)
            outputs += [prediction]
        return torch.stack(outputs, 1).squeeze(2)

