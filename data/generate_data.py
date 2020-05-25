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

BH_DataNew = pd.read_csv('BH_DataNew.csv', engine='python')
Bhagalpur_historical = pd.read_csv('historical/Bhagalpur_1_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Bhagalpur_recent = pd.read_excel('recent/Bhagalpur_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Bhojpur_historical = pd.read_csv('historical/Bhojpur_3_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Bhojpur_recent = pd.read_excel('recent/Bhojpur_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
East_Champaran_historical = pd.read_csv('historical/East_Champaran_8_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
East_Champaran_recent = pd.read_excel('recent/East_Champaran_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Kishanganj_historical = pd.read_csv('historical/Kishanganj_7_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Kishanganj_recent = pd.read_excel('recent/Kishanganj_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Munger_historical = pd.read_csv('historical/Munger_2_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Munger_recent = pd.read_excel('recent/Munger_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Muzzafarpur_historical = pd.read_csv('historical/Muzzafarpur_9_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Muzzafarpur_recent = pd.read_excel('recent/Muzzafarpur_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Nalanda_historical = pd.read_csv('historical/Nalanda_5_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Nalanda_recent = pd.read_excel('recent/Nalanda_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Patna_historical = pd.read_csv('historical/Patna_6_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Patna_recent = pd.read_excel('recent/Patna_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Rohtas_historical = pd.read_csv('historical/Rohtas_4_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Rohtas_recent = pd.read_excel('recent/Rohtas_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Vaishali_historical = pd.read_csv('historical/Vaishali_10_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Vaishali_recent = pd.read_excel('recent/Vaishali_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)

historical = [Bhagalpur_historical, Bhojpur_historical, East_Champaran_historical, 
           Kishanganj_historical, Munger_historical, Muzzafarpur_historical,
           Nalanda_historical, Patna_historical, Rohtas_historical, Vaishali_historical]
recent = [Bhagalpur_recent, Bhojpur_recent, East_Champaran_recent, 
           Kishanganj_recent, Munger_recent, Muzzafarpur_recent,
           Nalanda_recent, Patna_recent, Rohtas_recent, Vaishali_recent]


recent_new = [None] * 10
historical_new = [None] * 10
for i in range(len(historical_new)):
  historical_new[i] = historical[i][:][:]
  historical_new[i] = historical_new[i][cols_historical]
  historical_new[i].columns = cols_ultimate
  historical_new[i]['Unix'] = pd.to_datetime((historical_new[i]['Unix']))
  historical_new[i]['Unix'] = ( historical_new[i]['Unix']  - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  
  # historical_new[i][cols_ultimate_to_norm] = historical_new[i][cols_ultimate_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
  
for j in range(len(recent_new)):
  recent_new[j] = recent[j][:][:]
  recent_new[j] = recent[j][cols_recent]
  recent_new[j].columns = cols_ultimate
  # recent_new[j][cols_ultimate_to_norm] = recent_new[j][cols_ultimate_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

x_final = [None] * 10

for x in range(len(x_final)):
  x_final[x] = historical_new[x]
  # x_final[x] = historical_new[x].append(recent_new[x])


for y in range(len(x_final)):
  x_final[y] = pd.DataFrame(np.repeat(x_final[y].values,4,axis=0))
  x_final[y].columns = cols_ultimate
  for index, row in x_final[y].iterrows():
    if index % 4 == 1:
      x_final[y].at[index,'Unix'] += 900
    elif index % 4 == 2:
      x_final[y].at[index,'Unix'] += 1800
    elif index % 4 == 3:
      x_final[y].at[index,'Unix'] += 2700


# x_final[7].to_excel("weather_correct.xlsx")  


BH_DataNew = pd.read_csv('BH_DataNew.csv', engine='python')

BH_DataNew['Date'] = pd.to_datetime((BH_DataNew['Date']))
BH_DataNew['Date'] = (BH_DataNew['Date']  - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
BH_DataNew['Date'] = BH_DataNew['Date'] + 900 * (BH_DataNew['Block'] - 23)
BH_DataNew = BH_DataNew.rename(columns={"Date": "Unix"})
BH_DataNew = BH_DataNew.drop(columns=['Block'])

# BH_DataNew.to_excel("demand_correct.xlsx")  


everything = [None] * 10

for i, x in enumerate(x_final):
  final_df = pd.merge(x, BH_DataNew, on='Unix', how='inner')
  everything[i] = final_df

everything[7].to_excel("output.xlsx")  
pickle.dump(everything, open("data.p", "wb"))