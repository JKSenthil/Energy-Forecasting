import numpy as np
import pandas as pd
import pickle
import xlrd
import openpyxl
import ftplib
import os
import time
from dateutil import parser
from datetime import datetime


cols_recent = ['validTimeUtc','cloudCover', 'temperature', 'temperatureDewPoint',
         'relativeHumidity', 'pressureMeanSeaLevel', 'temperatureFeelsLike',
         'temperatureWindChill', 'windSpeed', 'windDirection']

cols_ultimate = ['Unix','CloudCoveragePercent', 'SurfaceTemperatureCelsius', 'SurfaceDewpointTemperatureCelsius',
         'RelativeHumidityPercent', 'SurfaceAirPressureKilopascals', 'ApparentTemperatureCelsius',
         'WindChillTemperatureCelsius', 'WindSpeedKph', 'WindDirectionDegrees']

cols_ultimate_to_norm = cols_ultimate[1:]

## FTP host name and credentials
ftp = ftplib.FTP('ftp.mercadosetrm.com', 'mercados_weather_ibm@mercadosetrm.com','IBM@2020')

## Go to the required directory



filename = datetime.today().strftime('%Y%m') + "09"

month = {	'01':'January',
		'02':'February',
		'03':'March',
		'04':'April',
		'05':'May',
		'06':'June',
		'07':'Jul',
		'08':'August',
		'09':'September',
		'10':'October',
		'11':'November',
		'12':'December'		}


ftp.cwd("15minForecast/Bihar/" + datetime.today().strftime('%Y_') + month[datetime.today().strftime('%m') ] + "/" + filename)

names = ftp.nlst()

for name in names:
    file = open(name, "wb")
    ftp.retrbinary('RETR '+ name, file.write)

Bhagalpur_historical = pd.read_csv('Bhagalpur_latest_update.csv', usecols= cols_recent, engine='python')
Bhojpur_historical = pd.read_csv('Bhojpur_latest_update.csv', usecols= cols_recent, engine='python')
East_Champaran_historical = pd.read_csv('East_Champaran_latest_update.csv', usecols= cols_recent, engine='python')
Kishanganj_historical = pd.read_csv('Kishanganj_latest_update.csv', usecols= cols_recent, engine='python')
Munger_historical = pd.read_csv('Munger_latest_update.csv', usecols= cols_recent, engine='python')
Muzaffarpur_historical = pd.read_csv('Muzaffarpur_latest_update.csv', usecols= cols_recent, engine='python')
Nalanda_historical = pd.read_csv('Nalanda_latest_update.csv', usecols= cols_recent, engine='python')
Patna_historical = pd.read_csv('Patna_latest_update.csv', usecols= cols_recent, engine='python')
Rohtas_historical = pd.read_csv('Rohtas_latest_update.csv', usecols= cols_recent, engine='python')
Vaishali_historical = pd.read_csv('Vaishali_latest_update.csv', usecols= cols_recent, engine='python')


recent = historical = [Bhagalpur_historical, Bhojpur_historical, East_Champaran_historical, 
           Kishanganj_historical, Munger_historical, Muzaffarpur_historical,
           Nalanda_historical, Patna_historical, Rohtas_historical, Vaishali_historical]

recent_new = [None] * 10



counter = 0 
for j in range(len(recent_new)):
  recent_new[j] = recent[j][:][:]
  recent_new[j] = recent[j][cols_recent]
  recent_new[j].columns = cols_ultimate
  counter +=1
  recent_new[j][['SurfaceAirPressureKilopascals']] = recent_new[j][['SurfaceAirPressureKilopascals']]/10.0
  # recent_new[j][cols_ultimate_to_norm] = recent_new[j][cols_ultimate_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

ftp = ftplib.FTP('ftp.mercadosetrm.in', 'bihar_scada_ext@mercadosetrm.in','bihar_scada@30')


file = open('Demand_data_bihar.csv', "wb")
ftp.retrbinary('RETR '+ 'Demand_data_bihar.csv', file.write)

BH_DataNew = pd.read_csv('Demand_data_bihar.csv', engine='python')

data = pd.concat([recent_new[0], recent_new[1], recent_new[2], recent_new[3], recent_new[4], recent_new[5], recent_new[6],
recent_new[7], recent_new[8]]).groupby(level=0).mean()

demand = BH_DataNew[['ACTUAL']]
weather = data[['SurfaceDewpointTemperatureCelsius', 'SurfaceAirPressureKilopascals','ApparentTemperatureCelsius']]

demand = demand.to_numpy()
weather = weather.to_numpy()

relevant_demand = demand[-96:].flatten()
insert_weather = weather.flatten()
print(relevant_demand, insert_weather)