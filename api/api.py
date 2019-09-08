import json
import requests
import numpy as np

KEY = "c9691deebbbe8f7538f597fb79afc319"
START_TIME = 1388514600
END_TIME = 1567708200
SECONDS_PER_DAY = 86400

cities = {}
cities["Lucknow"] = {"lat": 26.7617, "long": 80.8857}
cities["Unnao"] = {"lat": 26.5393, "long": 80.4878}
cities["Noida"] = {"lat": 28.3590, "long": 77.5508}

data = {}

for city in cities:
    data[city] = {}
    for time in range(START_TIME, END_TIME, SECONDS_PER_DAY):
        url = "https://api.darksky.net/forecast/{}/{},{},{}".format(KEY, cities[city]["lat"], cities[city]["long"], time)
        response_data = requests.get(url)
        data[city][time] = json.loads(response_data.text)

with open('data.json', 'w') as f:
    json.dump(data, f)