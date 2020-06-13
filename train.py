import pickle
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from model.rnn import RNN
from model.c_rnn import cRNN, cRNNv2
from model.mlp import BasicMLP
from model.autoencoder import AutoEncoder
from model.data_loader import load_formatted_datav6

from experiments import EXPERIMENTS_DIR

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DemandWeatherDataset_Deprecated(Dataset):
    def __init__(self, x, y, z, flatten=False):
        self.flatten = flatten
        self.x = x
        if not flatten:
            self.y = y
        else:
            self.y = np.reshape(y, (y.shape[0], -1))
        self.z = z

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.flatten:
            return (self.x[idx,:], self.y[idx], self.z[idx, :])
        return (self.x[idx,:], self.y[idx,:,:], self.z[idx, :])

class DemandWeatherDataset(Dataset):
    def __init__(self, x, y, z, flatten=False):
        self.flatten = flatten
        self.x = x
        if not flatten:
            self.y = y
        else:
            self.y = np.reshape(y, (y.shape[0], -1))
        self.z = z

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.flatten:
            return (self.x[idx,:], self.y[idx], self.z[idx, :])
        return (self.x[idx,:], self.y[idx,:,:], self.z[idx, :])


def cut_data(data, percentage=0.5, first=False):
    n = len(data)
    if first:
        size = int(n * (1-percentage))
    else:
        size = int(n * percentage)
    if first:
        return data[size:,:]
    return data[:size,:]

def train_test_split(data, percentage=0.9):
    """
    Splits first x percent of data into train, the other test
    """
    n = len(data)
    train_size = int(n * percentage)
    return data[:train_size,:], data[train_size:,:]

def random_train_test_split(data, percentage=0.8):
    """
    Generates random train-test split of the data
    """
    n = len(data)
    np.random.shuffle(data) # shuffle data in place
    
    train_size = int(n * percentage)
    return data[:train_size], data[train_size:]

def train_basicMLP(model, optimizer, loss_function, x, y, z, num_epochs=1000, batch_size=32):
    # x = cut_data(x)
    # y = cut_data(y)
    # z = cut_data(z)

    x_train, x_test = train_test_split(x)
    y_train, y_test = train_test_split(y)
    z_train, z_test = train_test_split(z)

    train_dataset = DemandWeatherDataset(x_train, y_train, z_train, flatten=True)
    test_dataset = DemandWeatherDataset(x_test, y_test, z_test, flatten=True)
    train_gen = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_gen = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    for epoch in range(num_epochs):
        train_loss = 0
        i = 0
        for x, y, z in train_gen:
            x = x.float()
            y = y.float()
            z = z.float()

            optimizer.zero_grad()
            y_pred = model.forward(x, y)
            
            loss = loss_function(z*(_max - _min) + _min, y_pred*(_max - _min) + _min)
            loss.backward()
            optimizer.step()

            train_loss += loss

            print("Batch {} has loss {}".format(i + 1, loss))
            i += 1

        test_loss = 0
        for x, y, z in test_gen:
            x = x.float()
            y = y.float()
            z = z.float()

            test_loss += loss_function(z*(_max - _min) + _min, y_pred*(_max - _min) + _min)
        
        avg_train_loss = train_loss / (len(train_gen) * batch_size)
        avg_test_loss = test_loss / (len(test_gen) * batch_size)

        print("Epoch {} of {} done, train loss={}, test loss={}".format(epoch + 1, num_epochs, avg_train_loss, avg_test_loss))

def train_crnn_deprecated(model, optimizer, loss_function, data, num_epochs, batch_size, n_prev, n_out):
    train_gen = DataLoader(data, batch_size=32, shuffle=True, drop_last=True)

    for i in range(num_epochs):
        print("Starting epoch {}".format(i))

        # shuffles indicies to shuffle training data order
        np.random.shuffle(indicies)

        for j in range(num_batches):
            # initialize batch data array
            batch_lag = np.zeros((batch_size, lag_size))
            batch_curr = np.zeros((batch_size, n_out, len(data[0]) - 1))
            batch_Y = np.zeros((batch_size, n_out))
            for k in range(j*batch_size,(j+1)*batch_size):
                l = indicies[k]
                batch_lag[k % batch_size, :] = data[l-n_prev:l,:].flatten()
                batch_curr[k % batch_size, :, :] = data[l:l+n_out, :-1]
                batch_Y[k % batch_size, :] = data[l:l+n_out, -1]    
            
            batch_curr = np.swapaxes(batch_curr, 0, 1) # conform with rnn input requirements

            # convert inputs to torch tensors
            batch_lag = torch.from_numpy(batch_lag).float().to(device)
            batch_curr = torch.from_numpy(batch_curr).float().to(device)
            batch_Y = torch.from_numpy(batch_Y).float().to(device)

            # compute backprop for model
            optimizer.zero_grad()
            y_pred = model.forward(batch_lag, batch_curr)
            loss = loss_function((batch_Y * _max) + _min, (y_pred * _max) + _min)
            loss.backward()
            optimizer.step()
            
            print("Batch {} of {} done, loss={}".format(j+1, num_batches, loss))

def train_crnn(model, optimizer, loss_function, x, y, z, num_epochs=1000, batch_size=32):
    x = cut_data(x)
    y = cut_data(y)
    z = cut_data(z)
    
    x_train, x_test = train_test_split(x)
    y_train, y_test = train_test_split(y)
    z_train, z_test = train_test_split(z)

    train_dataset = DemandWeatherDataset(x_train, y_train, z_train)
    test_dataset = DemandWeatherDataset(x_test, y_test, z_test)
    train_gen = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_gen = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    for epoch in range(num_epochs):
        train_loss = 0
        for x, y, z in train_gen:
            x = x.float()
            # y = y.float().permute((1,0,2))
            z = z.float()

            optimizer.zero_grad()
            y_pred = model.forward(x, y)
            loss = loss_function((z * _max) + _min, (y_pred * _max) + _min)
            loss.backward()
            optimizer.step()

            train_loss += loss

        test_loss = 0
        for x, y, z in test_gen:
            x = x.float()
            y = y.float().permute((1,0,2))
            z = z.float()

            test_loss += loss_function((z * _max) + _min, (y_pred * _max) + _min)

        avg_train_loss = train_loss / (len(train_gen) * batch_size)
        avg_test_loss = test_loss / (len(test_gen) * batch_size)

        print("Epoch {} of {} done, train loss={}, test loss={}".format(epoch + 1, num_epochs, avg_train_loss, avg_test_loss))

def example_pass(model, x, y, z):
    x0 = np.expand_dims(x[-500, :], 0)
    y0 = np.expand_dims(y[-500, :, :].flatten(), 0)
    z0 = np.expand_dims(z[-500, :], 0)

    x0 = torch.from_numpy(x0).float()#.to(device)
    y0 = torch.from_numpy(y0).float()#.to(device)
    z0 = torch.from_numpy(z0).float()#.to(device)

    print(x0.size())
    print(y0.size())
    print(z0.size())

    output = model.forward(x0, y0)
    output = output.data.numpy()
    z0 = z0.data.numpy()
    for pred, real in zip(output, z0):
        print(pred*(_max - _min) + _min, real*(_max - _min) + _min)


# ================ TRAIN AUTOENCODER ======================== #
# data = load_formatted_datav3()
# # data = data[:,:-1] # drop demand data
# model = AutoEncoder(len(data[0]), 10)
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# train, test = random_train_test_split(data, percentage=0.85)
# loss_function = nn.MSELoss().to(device)
# train_autoencoder(model, optimizer, train, test, loss_function, 256)
# ============================================================ #

# ================ TRAIN RNN ======================== #
# data = load_formatted_datav3()
# model = RNN()
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_function = nn.MSELoss().to(device)
# train_rnn(model, optimizer, loss_function, data, 100, 128, n_prev=96*3)
# =================================================== #

# ================ TRAIN crnnV2 ======================== #
# num_days = 10/96
# past_wd, future_w, future_d, _max, _min = load_formatted_datav2()
# data = data[:,[-5,-1]]
# enc_weather_size = len(data[0]) - 1
# train_data, test_data = train_test_split(data, percentage=0.90)
# weather_ae = None #AutoEncoder(len(data[0]) - 1, 10)
# #weather_ae.load_state_dict(torch.load(EXPERIMENTS_DIR + "/weather_ae.pth"))
# #weather_ae.to(device)
# model = cRNNv2(1, enc_weather_size, 64, 32)
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_function = nn.MSELoss().to(device)
# train_crnnV2(model, weather_ae, optimizer, loss_function, train_data, test_data, 100, 128, 96 * 5, 96)
# =================================================== #

# ================ TRAIN finale ======================== #
x, y, z, _max, _min = load_formatted_datav6(False)
# model = cRNN(lag_size=96, latent_size=16, weather_size=3, gru_hiddensize=64)
model = BasicMLP(96, 3 * 12, n_out=12)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_function = nn.MSELoss().to(device)
# train_basicMLP(model, optimizer, loss_function, x, y, z, num_epochs=5)
example_pass(model, x, y, z)
# =================================================== $

# ================ TRAIN MLP ======================== #
# data, _max, _min = load_formatted_datav3()
# model = BasicMLP(len(data)*96, 96 * (len(data) - 1), 96 * (len(data) - 1))
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_function = nn.MSELoss().to(device)
# train_basicMLP(model, optimizer, loss_function, data, 10, 64)
# =================================================== #