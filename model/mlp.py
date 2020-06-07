import torch
import torch.nn as nn

class BasicMLP(nn.Module):
    """  
    Neural Network which only uses current weather to
    make immediate energy demand prediction, 1 time step
    """
    def __init__(self, prev_len, curr_len, n_out=96):
        super(BasicMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(prev_len + curr_len, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, n_out)
        )

    def forward(self, prev_state, curr_weather):
        cat = torch.cat([prev_state, curr_weather], dim=1)
        return self.model(cat)