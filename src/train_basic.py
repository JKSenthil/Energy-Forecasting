import pickle
import numpy as np

data = pickle.load(open("./data/data.p", "rb"))

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

patna_data = data[index2city['Patna']]
patna_data["Unix"] = patna_data["Unix"] % 1440
patna_data["Unix"] /= max(patna_data["Unix"])
patna_data = patna_data.to_numpy()
y = patna_data[:,-1]
X = patna_data

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128))
model.add(Dense(154))
model.compile(optimizer='adam', loss='mse')

X = X.reshape((X.shape[0], 1, X.shape[1]))
# fit model
model.fit(X, y, epochs=5, verbose=1, shuffle=False)