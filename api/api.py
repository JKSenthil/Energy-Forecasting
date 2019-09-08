import json
import requests
import numpy as np
import pandas as pd

KEY = "c9691deebbbe8f7538f597fb79afc319"

cities = {}
cities["Lucknow"] = {"lat": 26.7617, "long": 80.8857}
cities["Unnao"] = {"lat": 26.5393, "long": 80.4878}
cities["Noida"] = {"lat": 28.3590, "long": 77.5508}

data = {}

for city in cities:
    for time in range(1388514600, 1567708200 + 86400, 86400):
        url = "https://api.darksky.net/forecast/{}/{},{},{}".format(KEY, cities[city]["lat"], cities[city]["long"], time)


url = "https://api.darksky.net/forecast/{}/{},{},{}".format(KEY, lat, long, time)

data = requests.get(url)
tru = json.loads(data.text)
