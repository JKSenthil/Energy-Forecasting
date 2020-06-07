import os
import pickle
import numpy as np
import pandas as pd
import datetime
# from . import DATAFILE_PATH

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(MODEL_DIR, "..")
DATAFILE_PATH = os.path.join(ROOT_DIR, "data", "data_new.p")

# def load_formatted_data(filepath=DATAFILE_PATH):
#     data = pickle.load(open(filepath, "rb")) # opens our preprocessed data file stored as a pickle
#     data = [df.to_numpy() for df in data] # convert from pandas dataframe to numpy
#     formatted_data = np.zeros((len(data[0]), 1 + 9 * len(data) + 1)) # data to return to user

#     # insert unix time and demand
#     formatted_data[:, 0] = data[0][:,0] # unix
#     formatted_data[:, -1] = data[0][:,-1] # demand

#     # insert weather data from each city
#     for t in range(len(data)):
#         formatted_data[:, (t*9)+1:((t+1)*9)+1] = data[t][:, 1:-1] # inserts weather data to appropriate slot

#     # normalize unix and demand data
#     formatted_data[:, 0] = formatted_data[:, 0] % (1440 * 60) # converts to time of day
#     formatted_data[:, 0] /= np.max(formatted_data[:, 0])
#     # formatted_data[:,-1] /= np.max(formatted_data[:, -1])

#     return formatted_data

# def load_formatted_datav2(filepath = "/Users/ngarg11/Energy-Forecasting/data/data.p"):
#     data = pickle.load(open(filepath, "rb")) # opens our preprocessed data file stored as a pickle
#     data = [df.to_numpy() for df in data] # convert from pandas dataframe to numpy
#     formatted_data = data[7] # use only one city

#     # normalize unix and demand data
#     # formatted_data[:, 0] = formatted_data[:, 0] % (1440 * 60) # converts to time of day
#     # formatted_data[:, 0] /= np.max(formatted_data[:, 0])
    
#     # normalize demand data EXPERIMENT
#     # formatted_data[:, -1] = 1 / formatted_data[:, -1]
#     _min = np.min(formatted_data[:, -1])
#     _max = np.max(formatted_data[:, -1])
#     formatted_data[:, -1] -= _min
#     _max = np.max(formatted_data[:, -1])
#     formatted_data[:, -1] /= _max

#     return formatted_data[:, 1:], _max, _min

def load_formatted_datav3(filepath="/Users/ngarg11/Energy-Forecasting/data/data.p"):
    data = pickle.load(open(filepath, "rb")) # opens our preprocessed data file stored as a pickle
    data = [df.to_numpy() for df in data] # convert from pandas dataframe to numpy
    # formatted_data = np.zeros((len(data[0]), 1 + 9 * len(data) + 1)) # data to return to user

    start = 1
    years = 6
    months = 12
    days = 7
    timestamps = 96

    unix = 1
    #originally len(data)
    number_cities = 1
    #originally 9
    data_points = 2 
    demand = 1

    # print(len(data[0])) number of data points
    formatted_data = np.zeros((len(data[0]), unix + data_points * number_cities + demand + years + months + days + timestamps)) 
    #
    # insert unix time and demand
    formatted_data[:, 0] = data[0][:,0] # unix
    formatted_data[:, -1] = data[0][:,-1] # demand
    # print(data[0][0, -1])

    year_asc = {}
    month_asc = {}
    day_asc = {}
    timestamp_asc = {}

    start_year = 2016
    start_month = 1
    start_day = 0
    for y in range(0, years):
        year_asc[y + start_year] = start + y
    for m in range(0, months):
        month_asc[m + start_month] = start + years + m
    for d in range(0, days):
        day_asc[d + start_day] = start + years + months + d
    for t in range(0, timestamps):
        timestamp_asc[t] = start + years + months + days + t

    for time in range(len(data[0])):
        my_date = datetime.datetime.fromtimestamp(int(formatted_data[time, 0]))
        year = my_date.year
        month = my_date.month
        day = my_date.weekday()
        timestamp = int((formatted_data[time, 0] % (60 * 1440)) // 900)

        formatted_data[time, year_asc[year]] = 1
        formatted_data[time, month_asc[month]] = 1
        formatted_data[time, day_asc[day]] = 1
        formatted_data[time, timestamp_asc[timestamp]] = 1
        
     # insert weather data from each city
    # for t in range(len(data)):
    #     formatted_data[:, (t*9)+1 + years + months + days + timestamps :((t+1)*9)+1 + years + months + days + timestamps] = data[t][:, 1:-1] # inserts weather data to appropriate slot

    # print(data[0][0,:])

    #putting all the weather/demand data after the time
    
    columns_to_add = [-8, -6]
    formatted_data[:, (0*data_points)+1 + years + months + days + timestamps :((0+1)*data_points)+1 + years + months + days + timestamps] = data[7][:, columns_to_add] # inserts weather data to appropriate slot



    # normalize unix and demand data
    # formatted_data[:, 0] = formatted_data[:, 0] % (1440 * 60) # converts to time of day
    # formatted_data[:, 0] = (formatted_data[:, 0]  - np.min(formatted_data[:,0]))/ np.max(formatted_data[:, 0])
    # formatted_data[:,-1] /= np.max(formatted_data[:, -1])
    
    _min = np.min(formatted_data[:, -1])
    _max = np.max(formatted_data[:, -1])
    formatted_data[:, -1] -= _min
    _max = np.max(formatted_data[:, -1])
    formatted_data[:, -1] /= _max
    

    # print(data[0][0,:])
    # print(data[7][0, -9:-1])
    # print(formatted_data[0][124])

    #Data is in right form now, with timestamp info, then weather info (only the 2 indices), then demand
    x = np.zeros((132724 - 96, 122))
    y = np.zeros((132724 - 96, 123)) 

    #depends on the 96
    total_days = 1381
    z = np.zeros((total_days, 96))

    counter = 0
    days = 0
    while counter < len(formatted_data) - 150:

        x[counter:counter + 96][:] = np.concatenate((formatted_data[counter:(counter+96), 1:-3], formatted_data[counter:(counter+96), -1:]), axis = 1)
        
        y[counter:counter + 96][:] = formatted_data[counter + 96:(counter+192), 1:-1]
        z[days][:] = formatted_data[counter + 96:(counter+192), -1]
        counter += 96
        days += 1
 
    return x, y, z, _max, _min



def load_formatted_datav4(version, filepath="/Users/ngarg11/Energy-Forecasting/data/correctData.p"):
    data = pickle.load(open(filepath, "rb")) # opens our preprocessed data file stored as a pickle
    new1_data = [df.to_numpy() for df in data] # convert from pandas dataframe to numpy
    start = 1
    years = 6
    months = 12
    days = 7
    timestamps = 96

    unix = 1
    #originally len(data)
    number_cities = 1
    #originally 9
    data_points = 2 
    demand = 1
    observations = len(data[0])
    # print(len(data[0])) number of data points
    formatted_data = np.zeros((observations, unix + data_points * number_cities + demand + years + months + days + timestamps)) 
    #
    # insert unix time and demand
    formatted_data[:, 0] = new1_data[0][:,0] # unix
    formatted_data[:, -1] = new1_data[0][:,-1] # demand
    # print(data[0][0, -1])

    year_asc = {}
    month_asc = {}
    day_asc = {}
    timestamp_asc = {}

    start_year = 2016
    start_month = 1
    start_day = 0
    for y in range(0, years):
        year_asc[y + start_year] = start + y
    for m in range(0, months):
        month_asc[m + start_month] = start + years + m
    for d in range(0, days):
        day_asc[d + start_day] = start + years + months + d
    for t in range(0, timestamps):
        timestamp_asc[t] = start + years + months + days + t

    for time in range(len(data[0])):
        my_date = datetime.datetime.fromtimestamp(int(formatted_data[time, 0]))
        year = my_date.year
        month = my_date.month
        day = my_date.weekday()
        timestamp = int((formatted_data[time, 0] % (60 * 1440)) // 900)

        formatted_data[time, year_asc[year]] = 1
        formatted_data[time, month_asc[month]] = 1
        formatted_data[time, day_asc[day]] = 1
        formatted_data[time, timestamp_asc[timestamp]] = 1
        
     # insert weather data from each city
    # for t in range(len(data)):
    #     formatted_data[:, (t*9)+1 + years + months + days + timestamps :((t+1)*9)+1 + years + months + days + timestamps] = data[t][:, 1:-1] # inserts weather data to appropriate slot


    #putting all the weather/demand data after the time
    
    #equals number of columns to add
    columns_to_add = [-9, -7]
    formatted_data[:, (0*data_points)+1 + years + months + days + timestamps :((0+1)*data_points)+1 + years + months + days + timestamps] = new1_data[7][:, columns_to_add] # inserts weather data to appropriate slot



    # normalize unix and demand data
    # formatted_data[:, 0] = formatted_data[:, 0] % (1440 * 60) # converts to time of day
    # formatted_data[:, 0] = (formatted_data[:, 0]  - np.min(formatted_data[:,0]))/ np.max(formatted_data[:, 0])
    # formatted_data[:,-1] /= np.max(formatted_data[:, -1])
    
    _min = np.min(formatted_data[:, -1])
    _max = np.max(formatted_data[:, -1])
    formatted_data[:, -1] -= _min
    _max = np.max(formatted_data[:, -1])
    formatted_data[:, -1] /= _max
    

    # formatted_data = np.zeros((len(data[0]), unix + data_points * number_cities + demand + years + months + days + timestamps)) 
    print(len(data[0]))

    lookback = 96
    lookahead = 96

    #only weather and demand
    if version:
        total_days = ( observations- 96)//96
        x = np.zeros((total_days, (data_points * lookback * number_cities) + lookahead))
        y = np.zeros((total_days, data_points * lookahead * number_cities)) 
        z = np.zeros((total_days, 96))
        counter = lookback
        days = 0

        while counter < (len(formatted_data) - max(lookback, lookahead)):
            x[days][:] = formatted_data[days*lookback: (days + 1) * lookback, -3:].reshape(3*lookback)
            y[days][:] = formatted_data[(days+1)*lookahead: (days + 2) * lookahead, -3:-1].reshape(2 * lookahead)
            z[days][:] = formatted_data[(days+1)*lookahead: (days + 2) * lookahead, -1]
            counter += 96
            days += 1

    #demand
    else:
        total_days = (observations - 96)//96
        x = np.zeros((total_days, 2 * lookback + lookahead))
        y = np.zeros((total_days, 2 * lookahead)) 
        z = np.zeros((total_days, 96))
        counter = lookback
        days = 0

        while counter < len(formatted_data) - max(lookback, lookahead) + 100:
            x[days][:] = formatted_data[days*lookback: (days + 1) * lookback, -1:].reshape(3*lookback)
            y[days][:] = formatted_data[(days+1)*lookahead: (days + 2) * lookahead, -3:-1].reshape(2 * lookaheadg)
            z[days][:] = formatted_data[(days+1)*lookahead: (days + 2) * lookahead, -1]
            counter += 96
            days += 1

    print(data[7].head(30))
    print(y[0])
    # print(y[0])
    # print(z[0])
    return x, y, z, _max, _min


    #, _max, _min # drop unix

def load_formatted_datav5(version, filepath=DATAFILE_PATH):
    data = pickle.load(open(filepath, "rb")) # opens our preprocessed data file stored as a pickle
    # data[7].to_excel("test.xlsx")
    new1_data = [df.to_numpy() for df in data] # convert from pandas dataframe to numpy

    unix = 1
    number_cities = 1
    data_points = 3
    demand = 1
    observations = len(data[0])
        
    formatted_data = np.zeros((observations, data_points * number_cities + demand))
    
    # insert unix time and demand
    # formatted_data[:, 0] = new1_data[0][:,0] # unix
    formatted_data[:, -1] = new1_data[0][:,-1] # demand
        
     # normalize unix and demand data
    # formatted_data[:, 0] = formatted_data[:, 0] % (1440 * 60) # converts to time of day
    # formatted_data[:, 0] = (formatted_data[:, 0]  - np.min(formatted_data[:,0]))/ np.max(formatted_data[:, 0])
    _min = np.min(formatted_data[:, -1])
    _max = np.max(formatted_data[:, -1])
    formatted_data[:, -1] = (formatted_data[:, -1] - _min) / _max

    #equals number of columns to add for just one city right now
    columns_to_add = [-8, -6, -5]
    formatted_data[:, (0*data_points) :((0+1)*data_points)] = new1_data[7][:, columns_to_add] # inserts weather data to appropriate slot

    lookback = 96
    lookahead = 4
    day_timestamps = 96

    #only weather, and demand
    if version:
        total_days = (observations- day_timestamps)//day_timestamps
        x = np.zeros((total_days, ((data_points * number_cities + demand) * lookback )))
        y = np.zeros((total_days, (data_points * lookahead * number_cities) )) 
        z = np.zeros((total_days, lookahead))
        counter = lookback
        days = 1

        while counter < (len(formatted_data) - max(lookback, lookahead)*2):
            # print(((unix + data_points * number_cities + demand) *lookback))
            # print(len(formatted_data[0][:]))
            x[days - 1][:] = formatted_data[ (days - 1) * lookback: (days) * lookback, :].reshape((data_points * number_cities + demand) *lookback)
            y[days - 1][:] = formatted_data[ (days)*lookahead: (days + 1) * lookahead, 0:-1].reshape((data_points * number_cities) * lookahead)
            z[days - 1][:] = formatted_data[   (days)*lookahead: (days + 1) * lookahead, -1]
            counter += day_timestamps
            days += 1

        x = np.reshape(x, (x.shape[0], -1, 4))
        y = np.reshape(y, (y.shape[0], -1, 3))

        return x, y, z, _max, _min

    #demand
    else:
        total_days = (observations - day_timestamps)//day_timestamps
        x = np.zeros((total_days,lookback * number_cities))
        y = np.zeros((total_days, data_points * number_cities * lookahead)) 
        z = np.zeros((total_days, lookahead))
        counter = lookback
        days = 1

        while counter < (len(formatted_data) - max(lookback, lookahead)*2):
            x[days-1][:] = formatted_data[ (days - 1) * lookback: (days) * lookback, -1].reshape(demand *lookback)
            y[days-1][:] = formatted_data[ (days)*lookahead: (days + 1) * lookahead, 0:-1].reshape((data_points * number_cities) * lookahead)
            z[days-1][:] = formatted_data[   (days)*lookahead: (days + 1) * lookahead, -1]
            counter += 1 #day_timestamps
            days += 1
        
        y = np.reshape(y, (y.shape[0], -1, 3))
        return x, y, z, _max, _min   


    # print(formatted_data[ 0:96, -1])
    # print(data[7].head(30))
        
    # print(y[0])
    # print(y[0])
    # print(z[0])

if __name__ == "__main__":
    # print(load_formatted_datav3("/Users/ngarg11/Energy-Forecasting/data/data.p")) 
    print(load_formatted_datav5(False))
