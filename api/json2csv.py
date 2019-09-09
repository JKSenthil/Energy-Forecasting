import json
import pandas as pd

from dateutil import tz
from datetime import datetime

FROM_ZONE = tz.gettz('UTC')
TO_ZONE = tz.gettz('Asia/Calcutta')

def hourlyjson2csv(city, data):
    date = []
    icon = []
    precip_intensity = []
    precip_probability = []
    temp = []
    apparent_temp = []
    dew_point = []
    humidity = []
    wind_speed = []
    wind_gust = []
    wind_bearing = []
    cloud_cover = []
    uv_index = []
    visibility = []

    for day in data[city]:
        for hour_data in data[city][day]['hourly']['data']:
            ts = int(hour_data['time'])
            utc = datetime.utcfromtimestamp(ts)
            utc = utc.replace(tzinfo=FROM_ZONE)
            ist = utc.astimezone(TO_ZONE)
            date.append(ist)

            icon.append(hour_data.get('icon'))
            precip_intensity.append(hour_data.get('precipIntensity'))
            precip_probability.append(hour_data.get('precipProbability'))
            temp.append(hour_data.get('temperature'))
            apparent_temp.append(hour_data.get('apparentTemperature'))
            dew_point.append(hour_data.get('dewPoint'))
            humidity.append(hour_data.get('humidity'))
            wind_speed.append(hour_data.get('windSpeed'))
            wind_bearing.append(hour_data.get('windBearing'))
            wind_gust.append(hour_data.get('windGust'))
            cloud_cover.append(hour_data.get('cloudCover'))
            uv_index.append(hour_data.get('uvIndex'))
            visibility.append(hour_data.get('visibility'))
    
    d = {
        "Date": date, 
        "Icon": icon, 
        "Precipitation Intensity": precip_intensity, 
        "Precipitation Probability": precip_probability, 
        "Temperature": temp, 
        "Apparent Temperature": apparent_temp,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "Wind Gust": wind_gust,
        "Wind Bearing": wind_bearing,
        "Cloud Cover": cloud_cover,
        "UV Index": uv_index,
        "Visibility": visibility
        }
            
    df = pd.DataFrame(d)
    df.to_csv(city + '_Weather.csv')


with open('data/data.json') as json_file:
    data = json.load(json_file)

for city in data:
    hourlyjson2csv(city, data)