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


data = pickle.load(open("../data/data.p", "rb"))

data = data[0]
data = data[['Unix', 'Demand','CloudCoveragePercent', 'SurfaceTemperatureCelsius', 'SurfaceDewpointTemperatureCelsius',
         'RelativeHumidityPercent', 'SurfaceAirPressureKilopascals', 'ApparentTemperatureCelsius',
         'WindChillTemperatureCelsius', 'WindSpeedKph', 'WindDirectionDegrees']]

data['Unix'] = pd.to_datetime(data['Unix'],unit='s')
data.to_csv('raw.csv', index=False)
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
values = dataset.values
values = values.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
scaled = values
n_hours = 360
n_features = 10
reframed = series_to_supervised(scaled, n_hours, 1)
reframed.drop(reframed.columns[[-9, -8, -7, -6, -5, -4, -3, -2, -1]], axis=1, inplace=True)
print(reframed.head())

#Using Multilayer Perceptron
#10 * 360 * data points
arr = reframed.to_numpy()
Y = arr[:, 3600]
arr = arr[:, :3600]
X = np.reshape(arr, (-1, n_hours, n_features))

n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, Y, epochs=5, verbose=0)


# # print(data[0]["Unix"])
# # a = 0

# # index2city = {
# #     'Bhagalpur': 0,
# #     'Bhojpur' : 1,
# #     'East Champaran' : 2,
# #     'Kishanganj' : 3,
# #     'Munger' : 4,
# #     'Muzzafarpur' : 5,
# #     'Nalanda' : 6,
# #     'Patna' : 7,
# #     'Rohtas' : 8,
# #     'Vaishali' : 9
# # }

# # # data[0] -> dataframe, data[0]["Unix"]
# # data[0]["Unix"] = data[0]["Unix"] % 1440
# # columns = ['CloudCoveragePercent', 'SurfaceTemperatureCelsius', 'SurfaceDewpointTemperatureCelsius',
# #          'RelativeHumidityPercent', 'SurfaceAirPressureKilopascals', 'ApparentTemperatureCelsius',
# #          'WindChillTemperatureCelsius', 'WindSpeedKph', 'WindDirectionDegrees', 'Demand']

# # # Extract Patna data for now


# # processed_data = data[index2city['Patna']].to_numpy()
# # X = processed_data[:,1:-1]
# # y = processed_data[:,-1]

# # length = len(X)
# # x_groups = []
# # y_groups = []
# # for i in range(10, length):
# #     x_data = [X[(i-10):(i)]]
# #     # y_data = [y[i:(i+11)]]
# #     x_groups.append(x_data)
# #     # y_groups.append(y_data)
# # y = y[10:]

# # print(len(x_groups))
# # print(len(y))

# # import tensorflow
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense

# # model = Sequential()
# # model.add(LSTM(50))
# # model.add(Dense(1))
# # model.compile(loss='mae', optimizer='adam')

# # model.fit(x_groups, y, epochs=50, batch_size=100, shuffle=False)

# # #Input arrays should have the same number of samples as target arrays. Found 1 input samples and 132714 target samples.
 