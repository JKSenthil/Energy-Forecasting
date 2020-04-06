import pickle
import numpy as np

from . import DATAFILE_PATH

def load_formatted_data(filepath=DATAFILE_PATH):
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
    # formatted_data[:,-1] /= np.max(formatted_data[:, -1])

    return formatted_data

def load_formatted_datav2(filepath=DATAFILE_PATH):
    data = pickle.load(open(filepath, "rb")) # opens our preprocessed data file stored as a pickle
    data = [df.to_numpy() for df in data] # convert from pandas dataframe to numpy
    formatted_data = data[7] # use only one city

    # normalize unix and demand data
    formatted_data[:, 0] = formatted_data[:, 0] % (1440 * 60) # converts to time of day
    formatted_data[:, 0] /= np.max(formatted_data[:, 0])
    
    return formatted_data