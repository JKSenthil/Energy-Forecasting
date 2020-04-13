import pickle
import numpy as np
import torch
import torch.nn as nn

from model.rnn import RNN
from model.mlp import BasicMLP
from model.autoencoder import AutoEncoder
from model.data_loader import load_formatted_datav3

from experiments import EXPERIMENTS_DIR

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_test_split(data, percentage=0.8):
    """
    Splits first x percent of data into train, the other test
    """
    n = len(data)
    train_size = int(n * percentage)
    return data[:train_size], data[train_size:]

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
    # load autoencoder models
    autoencoder1 = AutoEncoder(len(data[0]), 10)
    autoencoder1.load_state_dict(torch.load(EXPERIMENTS_DIR + "/ac.pth"))
    autoencoder2 = AutoEncoder(len(data[0])-1, 10)
    autoencoder2.load_state_dict(torch.load(EXPERIMENTS_DIR + "/ac_drop_demand.pth"))
    autoencoder1.to(device)
    autoencoder2.to(device)

    # initialize training infrastructure vars
    indicies = np.arange(n_prev, len(data)-n_out)
    num_batches = (len(data) // batch_size) - batch_size
    input_size = 10*3*96
    for i in range(num_epochs):
        print("Starting epoch {}".format(i+1))
        
        # shuffles indicies to essentially shuffle training data order
        np.random.shuffle(indicies)

        for j in range(num_batches):
            # initialize batch data array
            batch_X = torch.zeros((batch_size, input_size)).to(device)
            batch_curr_weather = torch.zeros((batch_size, n_out*10)).to(device)
            batch_Y = np.zeros((batch_size, 96))
            for k in range(j*batch_size,(j+1)*batch_size):
                l = indicies[k]
                _prev_data = torch.from_numpy(data[l-n_prev:l,:]).float().to(device)
                batch_X[k % batch_size, :] = torch.flatten(autoencoder1.enc(_prev_data))
                _curr_data = torch.from_numpy(data[l:l+n_out, :-1]).float().to(device)
                batch_curr_weather[k % batch_size, :] = torch.flatten(autoencoder2.enc(_curr_data))

                batch_Y[k % batch_size, :] = data[l:l+n_out, -1]

            batch_Y = torch.from_numpy(batch_Y).float().to(device)

            # compute backprop for model
            optimizer.zero_grad()
            y_pred = model(batch_X, batch_curr_weather)
            loss = loss_function(batch_Y, y_pred)
            loss.backward()
            optimizer.step()
            
            print("Batch {} of {} done, loss={}".format(j+1, num_batches, loss))

def train_rnn(model, optimizer, loss_function, data, num_epochs, batch_size, n_prev=96, n_out=96):
    # load autoencoder models
    autoencoder1 = AutoEncoder(len(data[0]), 10)
    autoencoder1.load_state_dict(torch.load(EXPERIMENTS_DIR + "/ac.pth"))
    autoencoder2 = AutoEncoder(len(data[0])-1, 10)
    autoencoder2.load_state_dict(torch.load(EXPERIMENTS_DIR + "/ac_drop_demand.pth"))
    autoencoder1.to(device)
    autoencoder2.to(device)

    # initialize training infrastructure vars
    indicies = np.arange(n_prev, len(data)-n_out)
    num_batches = (len(data) // batch_size) - batch_size
    input_size = len(data[0])
    for i in range(num_epochs):
        print("Starting epoch {}".format(i))

        # shuffles indicies to essentially shuffle training data order
        np.random.shuffle(indicies)

        for j in range(num_batches):
            # initialize batch data array
            batch_X = np.zeros((batch_size, n_prev, input_size))
            batch_Y = np.zeros((batch_size, n_out))
            batch_curr_weather = np.zeros((batch_size, len(data[0]) - 1))
            for k in range(j*batch_size,(j+1)*batch_size):
                l = indicies[k]
                batch_X[k % batch_size, :, :] = data[l-n_prev:l,:]
                batch_curr_weather[k % batch_size, :] = data[l, :-1]
                batch_Y[k % batch_size, :] = data[l:l+n_out, -1]    
            batch_X = np.swapaxes(batch_X, 0, 1) # conform with rnn input requirements

            # convert inputs to torch tensors
            batch_X = torch.from_numpy(batch_X).float().to(device)
            batch_curr_weather = torch.from_numpy(batch_curr_weather).float().to(device)
            batch_Y = torch.from_numpy(batch_Y).float().to(device)

            enc_batch_X = autoencoder1.enc(batch_X)
            enc_curr_weather = autoencoder2.enc(batch_curr_weather)

            # compute backprop for model
            optimizer.zero_grad()
            y_pred = model.forward(enc_batch_X, enc_curr_weather)
            loss = loss_function(batch_Y, y_pred)
            loss.backward()
            optimizer.step()
            
            print("Batch {} of {} done, loss={}".format(j+1, num_batches, loss))

def train_autoencoder(model, optimizer, train, test, loss_function, batch_size):
    indices = np.arange(0, len(train))
    num_batches = (len(train) // batch_size) - batch_size
    test = torch.from_numpy(test).float().to(device)

    i = 0 # epoch count
    prev_loss = 10000000
    last_saved = 0 # if last_saved is high, we can terminate training loop as model is not learning anymore
    while True:
        print("Starting epoch {}".format(i+1))
        
        # shuffles indices to shuffle training data order
        np.random.shuffle(indices)

        for j in range(num_batches):
            # initialize batch data array
            batch_X = train[indices[j*batch_size:(j+1)*batch_size],:]

            batch_X = torch.from_numpy(batch_X).float().to(device)

            # compute backprop for model
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_function(batch_X, y_pred)
            loss.backward()
            optimizer.step()
            
            print("Batch {} of {} done, loss={}".format(j+1, num_batches, loss))
        
        test_loss = loss_function(test, model(test)).item()
        print("Epoch {} done, test error is {}".format(i+1, test_loss))

        if test_loss < prev_loss:
            prev_loss = test_loss
            torch.save(model.state_dict(), EXPERIMENTS_DIR + "/ac_nodemand.pth")
            last_saved = 0
            print("Model saved!")
        else:
            last_saved += 1

        if last_saved >= 10:
            break

        i += 1
    
    print("Training ended, reached epoch {} with a best test loss of {}".format(i+1, prev_loss))

# ================ TRAIN AUTOENCODER ======================== #
# data = load_formatted_datav3()
# data = data[:,:-1] # drop demand data
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


# ================ TRAIN MLP ======================== #
data = load_formatted_datav3()
model = BasicMLP()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.MSELoss().to(device)
train_basicMLP(model, optimizer, loss_function, data, 10, 64)
# =================================================== #