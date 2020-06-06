import pickle
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from model.rnn import RNN
from model.c_rnn import cRNN, cRNNv2
from model.mlp import BasicMLP
from model.autoencoder import AutoEncoder
from model.data_loader import load_formatted_datav2, load_formatted_datav3, load_formatted_datav5

from experiments import EXPERIMENTS_DIR

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DemandWeatherDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        

def train_test_split(data, percentage=0.8):
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

def train_basicMLP(model, optimizer, loss_function, data, num_epochs, batch_size, n_prev=96*3, n_out=96):
    """
    model: BasicMLP model to train
    data: formatted data to work with
    num_epochs: number of times model trains on whole data
    batch_size: simultanous data model trains on (right value can speed up model convergence)
    n_out: number of timesteps to predict demand for
    """
    # initialize training infrastructure vars
    indicies = np.arange(n_prev, len(data)-n_out)
    num_batches = (len(data) // batch_size) - batch_size
    prev_size = len(data)*7*96
    for i in range(num_epochs):
        print("Starting epoch {}".format(i+1))
        
        # shuffles indicies to essentially shuffle training data order
        np.random.shuffle(indicies)

        for j in range(num_batches):
            # initialize batch data array
            batch_X = torch.zeros((batch_size, prev_size)).to(device)
            batch_curr_weather = torch.zeros((batch_size, n_out*(len(data) - 1))).to(device)
            batch_Y = np.zeros((batch_size, n_out))
            for k in range(j*batch_size,(j+1)*batch_size):
                l = indicies[k]
                _prev_data = torch.from_numpy(data[l-n_prev:l,:]).float().to(device)
                batch_X[k % batch_size, :] = _prev_data
                _curr_data = torch.from_numpy(data[l:l+n_out, :-1]).float().to(device)
                batch_curr_weather[k % batch_size, :] = _curr_data

                batch_Y[k % batch_size, :] = data[l:l+n_out, -1]

            batch_Y = torch.from_numpy(batch_Y).float().to(device)

            # compute backprop for model
            optimizer.zero_grad()
            y_pred = model(batch_X, batch_curr_weather)
            loss = loss_function((batch_Y * _max) + _min, (y_pred * _max) + _min)
            loss.backward()
            optimizer.step()
            
            print("Batch {} of {} done, loss={}".format(j+1, num_batches, loss))

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
x, y, z, _max, _min = load_formatted_datav5(False)
# =================================================== $

# ================ TRAIN MLP ======================== #
# data, _max, _min = load_formatted_datav3()
# model = BasicMLP(len(data)*96, 96 * (len(data) - 1), 96 * (len(data) - 1))
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_function = nn.MSELoss().to(device)
# train_basicMLP(model, optimizer, loss_function, data, 10, 64)
# =================================================== #