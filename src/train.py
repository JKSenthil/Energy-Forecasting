import pickle
import numpy as np
import pandas as pd


data = pickle.load(open("./data/data.p", "rb"))

# print(data[0]["Unix"])
a = 0

index2city = {
    'Bhagalpur': 0,
    'Bhojpur' : 1,
    'East Champaran' : 2,
    'Kishanganj' : 3,
    'Munger' : 4,
    'Muzzafarpur' : 5,
    'Nalanda' : 6,
    'Patna' : 7,
    'Rohtas' : 8,
    'Vaishali' : 9
}

# data[0] -> dataframe, data[0]["Unix"]
data[0]["Unix"] = data[0]["Unix"] % 1440
columns = ['CloudCoveragePercent', 'SurfaceTemperatureCelsius', 'SurfaceDewpointTemperatureCelsius',
         'RelativeHumidityPercent', 'SurfaceAirPressureKilopascals', 'ApparentTemperatureCelsius',
         'WindChillTemperatureCelsius', 'WindSpeedKph', 'WindDirectionDegrees', 'Demand']

# Extract Patna data for now


processed_data = data[index2city['Patna']].to_numpy()
X = processed_data[:,1:-1]
y = processed_data[:,-1]

length = len(X)
x_groups = []
y_groups = []
for i in range(10, length):
    x_data = [X[(i-10):(i)]]
    # y_data = [y[i:(i+11)]]
    x_groups.append(x_data)
    # y_groups.append(y_data)
y = y[10:]

print(len(x_groups))
print(len(y))

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

model.fit(x_groups, y, epochs=50, batch_size=100, shuffle=False)

#Input arrays should have the same number of samples as target arrays. Found 1 input samples and 132714 target samples.
 