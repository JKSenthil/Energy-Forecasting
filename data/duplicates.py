import pandas as pd
import pickle
unpickled_old= pd.read_pickle("/Users/ngarg11/Energy-Forecasting/data/data.p")
unpickled_new= pd.read_pickle("/Users/ngarg11/Energy-Forecasting/data/new_data.p")
for x in range(len(unpickled_new)):
    indexNames = unpickled_new[x][(unpickled_new[x]['Unix'].isin(unpickled_old[x]['Unix']))].index
    unpickled_new[x].drop(indexNames, inplace=True)
pickle.dump(unpickled_new, open("new_data_no_duplicates.p", "wb"))