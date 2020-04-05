import pickle
import numpy as np
import torch
import torch.nn as nn

from model import RNN

def load_formatted_data(filepath='./data/data.p'):
    data = pickle.load(open(filepath, "rb")) # opens our preprocessed data file stored as a pickle
    data = [df.to_numpy() for df in data] # convert from pandas dataframe to numpy
    formatted_data = np.zeros((len(data[0]), 1 + 9 * len(data) + 1)) # data to return to user

    # insert unix time and demand
    formatted_data[:, 0] = data[0][:,0] # unix
    formatted_data[:, -1] = data[0][:,-1] # demand

    # insert weather data from each city
    for t in range(len(data)):
        formatted_data[:, (t*9)+1:((t+1)*9)+1] = data[t][:, 1:-1] # inserts weather data to appropriate slot

    # normalize unix and demand data
    formatted_data[:, 0] = formatted_data[:, 0] % (1440 * 60) # converts to time of day
    formatted_data[:, 0] /= np.max(formatted_data[:, 0])
    
    return formatted_data

def train(model, optimizer, data, num_epochs, batch_size, n_prev, n_out):
    """
    model: RNN model to train
    data: formatted data to work with
    num_epochs: number of times model trains on whole data
    batch_size: simultanous data model trains on (right value can speed up model convergence)
    n_prev: how much previous timesteps model should observe before making prediction
            (should be same as arg to RNN model)
    """
    indicies = np.arange(n_prev, len(data) - n_out)
    num_batches = len(data) // batch_size
    input_size = len(data[0])
    for i in range(num_epochs):
        print("Starting epoch {}".format(i))

        # shuffles indicies to essentially shuffle training data order
        np.random.shuffle(indicies)

        for j in range(num_batches):
            # initialize batch data array
            batch_X = np.zeros((batch_size, n_prev, input_size))
            batch_Y = np.zeros((batch_size, n_out))
            for k in range(j*batch_size,(j+1)*batch_size):
                l = indicies[k]
                batch_X[k % batch_size, :, :] = data[l-n_prev:l,:]
                batch_Y[k % batch_size, :] = data[l:l+n_out, -1]
            batch_X = np.swapaxes(batch_X, 0, 1) # conform with rnn input requirements
            batch_X[:,:,-1] /= np.max(data[:, -1]) # normalize demand data

            batch_curr_weather = data[indicies[j*batch_size:(j+1)*batch_size],:-1]
            batch_Y = torch.from_numpy(batch_Y).float()

            # compute backprop for model
            optimizer.zero_grad()
            y_pred = model.forward(batch_X, batch_curr_weather)
            loss = torch.sum((batch_Y - y_pred) ** 2)/ n_out # measn squared error loss
            loss.backward()
            optimizer.step()
            
            print("Batch {} of {} done, loss={}".format(j+1, num_batches, loss))

model = RNN(n_out=154).float()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = load_formatted_data()

train(model, optimizer, data, 10, 32, 154, 154)
