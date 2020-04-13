import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 80),
            nn.LeakyReLU(),
            nn.Linear(80, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 80),
            nn.LeakyReLU(),
            nn.Linear(80, input_size)
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))

    def enc(self, inputs):
        return self.encoder(inputs)