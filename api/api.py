import json
import requests
import numpy as np
import pandas as pd

#lat = "26.8467"
#long = "80.9462"
lat = "28.3670"
long = "79.4304"
time = "1391230800"
#time = 1389939200
#?exclude=hourly,daily

key = "c9691deebbbe8f7538f597fb79afc319"
url = "https://api.darksky.net/forecast/{}/{},{},{}".format(key, lat, long, time)

data = requests.get(url)
tru = json.loads(data.text)
print(tru.keys())
print(tru['currently'])

#print(data)