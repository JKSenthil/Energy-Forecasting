import pickle
import numpy as np
import pandas as pd
from pandas import read_csv
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import concat
from pandas import DataFrame


everything = pickle.load(open("../data/data.p", "rb"))


data = everything[0]
data = data[['Unix', 'Demand','CloudCoveragePercent', 'SurfaceTemperatureCelsius', 'SurfaceDewpointTemperatureCelsius',
         'RelativeHumidityPercent', 'SurfaceAirPressureKilopascals', 'ApparentTemperatureCelsius',
         'WindChillTemperatureCelsius', 'WindSpeedKph', 'WindDirectionDegrees']]

data['Unix'] = pd.to_datetime(data['Unix'],unit='s')

x = pd.concat([data, everything[1].drop(columns = ['Unix', 'Demand']), everything[2].drop(columns = ['Unix', 'Demand']), 
               everything[3].drop(columns = ['Unix', 'Demand']), 
               everything[4].drop(columns = ['Unix', 'Demand']), everything[5].drop(columns = ['Unix', 'Demand']), 
               everything[6].drop(columns = ['Unix', 'Demand']), everything[7].drop(columns = ['Unix', 'Demand'])], axis = 1)
print(x.shape)
x.to_csv('raw.csv', index=False)
dataset = read_csv('raw.csv', index_col=0)
dataset.index.name = 'Unix'
print(dataset.head(5))
dataset.to_csv('pollution.csv')


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

dataset = read_csv('pollution.csv', header=0, index_col=0)
print(dataset.shape)
values = dataset.values
values = values.astype('float32')
n_hours = 360
n_features = 72
look_ahead = 154
reframed = series_to_supervised(values, n_hours, look_ahead)

#Using Multilayer Perceptron
#10 * 360 * data points
arr = reframed.to_numpy()
print(arr.shape)
y_value = []
for x in range(1, 155):
  y_value.append(-x * 10 + 5140)
n_input = 4987

X = np.delete(arr, y_value, axis = 1)
Y = arr[:, y_value]

model = Sequential()
model.add(Dense(100, activation='relu', input_dim = n_input - 1))
model.add(Dense(154))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, Y, epochs=5, verbose=0)

n = 9000
test = X[n, :]
test = test.reshape(1, n_input - 1)
actual_y = Y[n]
print(actual_y)
yhat = model.predict(test, verbose=0)
print(yhat)