import pickle
import numpy as np
import pandas as pd
import datetime
# from . import DATAFILE_PATH

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
    
    columns_to_add = [-9, -7]
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


    #, _max, _min # drop unix

print(load_formatted_datav3("/Users/ngarg11/Energy-Forecasting/data/data.p"))
