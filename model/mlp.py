import torch
import torch.nn as nn

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