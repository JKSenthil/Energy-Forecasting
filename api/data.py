import numpy as np
import pandas as pd
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

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

BH_DataNew = pd.read_csv('/content/gdrive/Shared drives/Weather_2/BH_DataNew.csv', engine='python')
Bhagalpur_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Bhagalpur_1_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Bhagalpur_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Bhagalpur_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Bhojpur_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Bhojpur_3_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Bhojpur_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Bhojpur_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
East_Champaran_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/East_Champaran_8_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
East_Champaran_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/East_Champaran_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Kishanganj_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Kishanganj_7_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Kishanganj_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Kishanganj_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Munger_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Munger_2_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Munger_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Munger_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Muzzafarpur_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Muzzafarpur_9_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Muzzafarpur_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Muzzafarpur_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Nalanda_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Nalanda_5_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Nalanda_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Nalanda_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Patna_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Patna_6_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Patna_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Patna_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Rohtas_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Rohtas_4_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Rohtas_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Rohtas_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)
Vaishali_historical = pd.read_csv('/content/gdrive/Shared drives/Weather_2/historical/Vaishali_10_10-31-2009_11-01-2019_hourly.csv', usecols= cols_historical, engine='python')
Vaishali_recent = pd.read_excel('/content/gdrive/Shared drives/Weather_2/recent/Vaishali_Weather_Jan12020_till_29March2020.xlsx', usecols = cols_recent)

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
  historical_new[i].columns = cols_ultimate
  historical_new[i]['Unix'] = pd.to_datetime((historical_new[i]['Unix']))
  historical_new[i]['Unix'] = ( historical_new[i]['Unix']  - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  
  historical_new[i][cols_ultimate_to_norm] = historical_new[i][cols_ultimate_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
  
for j in range(len(recent_new)):
  recent_new[j] = recent[j][:][:]
  recent_new[j] = recent[j].reindex(columns=cols_recent)
  recent_new[j].columns = cols_ultimate
  recent_new[j][cols_ultimate_to_norm] = recent_new[j][cols_ultimate_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))



x_final = [None] * 10

for x in range(len(x_final)):
  x_final[x] = historical_new[x].append(recent_new[x])


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

BH_DataNew = pd.read_csv('/content/gdrive/Shared drives/Weather_2/BH_DataNew.csv', engine='python')

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
 
print(everything)
