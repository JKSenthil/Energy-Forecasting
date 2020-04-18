import numpy as np
import pandas as pd
import pickle

cols_historical = ['DateHrGmt','CloudCoveragePercent', 'SurfaceTemperatureCelsius', 'SurfaceDewpointTemperatureCelsius',
         'RelativeHumidityPercent', 'SurfaceAirPressureKilopascals', 'ApparentTemperatureCelsius',
         'WindChillTemperatureCelsius', 'WindSpeedKph', 'WindDirectionDegrees']

cols_recent = ['validTimeUtc','cloudCover', 'temperature', 'temperatureDewPoint',
         'relativeHumidity', 'pressureMeanSeaLevel', 'temperatureFeelsLike',
         'temperatureWindChill', 'windSpeed', 'windDirection']

cols_ultimate = ['Unix','CloudCoveragePercent', 'SurfaceTemperatureCelsius', 'SurfaceDewpointTemperatureCelsius',
         'RelativeHumidityPercent', 'SurfaceAirPressureKilopascals', 'ApparentTemperatureCelsius',
         'WindChillTemperatureCelsius', 'WindSpeedKph', 'WindDirectionDegrees']

cols_ultimate_to_norm = cols_ultimate[1:]

Bhagalpur_recent = pd.read_excel('./data/recent/Bhagalpur_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Bhojpur_recent = pd.read_excel('./data/recent/Bhojpur_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
East_Champaran_recent = pd.read_excel('./data/recent/East_Champaran_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Kishanganj_recent = pd.read_excel('./data/recent/Kishanganj_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Munger_recent = pd.read_excel('./data/recent/Munger_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Muzzafarpur_recent = pd.read_excel('./data/recent/Muzzafarpur_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Nalanda_recent = pd.read_excel('./data/recent/Nalanda_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Patna_recent = pd.read_excel('./data/recent/Patna_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Rohtas_recent = pd.read_excel('./data/recent/Rohtas_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)
Vaishali_recent = pd.read_excel('./data/recent/Vaishali_Weather_Jan12020_till_15April2020.xlsx', usecols = cols_recent)


recent = [Bhagalpur_recent, Bhojpur_recent, East_Champaran_recent, 
           Kishanganj_recent, Munger_recent, Muzzafarpur_recent,
           Nalanda_recent, Patna_recent, Rohtas_recent, Vaishali_recent]


recent_new = [None] * 10

for j in range(len(recent_new)):
  recent_new[j] = recent[j][:][:]
  recent_new[j] = recent[j].reindex(columns=cols_recent)
  recent_new[j].columns = cols_ultimate
  recent_new[j][cols_ultimate_to_norm] = recent_new[j][cols_ultimate_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

x_final = [None] * 10

for x in range(len(x_final)):
  x_final[x] = recent_new[x]


for y in range(len(x_final)):
  x_final[y] = pd.DataFrame(x_final[y]).to_numpy()
  x_final[y] = np.repeat(x_final[y], repeats=4, axis=0)
  
for z in range(359176):
    if z % 4 == 0:
      x_final[y][z][0] = x_final[y][z][0]
    elif z % 4 == 1:
      x_final[y][z][0] += 900
    elif z % 4 == 2:
      x_final[y][z][0] += 1800
    elif z % 4 == 3:
      x_final[y][z][0] += 2700

BH_DataNew = pd.read_csv('./data/BH_DataNew.csv', engine='python')

BH_DataNew['Date'] = pd.to_datetime((BH_DataNew['Date']))
BH_DataNew['Date'] = ( BH_DataNew['Date']  - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
BH_DataNew['Date'] = BH_DataNew['Date'] + 900 * (BH_DataNew['Block'] - 23)
BH_DataNew = BH_DataNew.rename(columns={"Date": "Unix"})
BH_DataNew = BH_DataNew.drop(columns=['Block'])

indices = [None] * 359176
everything = [None] * 10

for i, x in enumerate(x_final):
  df = pd.DataFrame(data=x, index= indices, columns= cols_ultimate)
  final_df = pd.merge(df, BH_DataNew, on='Unix', how='inner')
  everything[i] = final_df

pickle.dump(everything, open("new_data.p", "wb"))