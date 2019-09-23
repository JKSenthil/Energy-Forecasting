import pandas as pd
import xgboost as xgb

# read in data
dtrain = xgb.DMatrix('../data/Lucknow_Weather.csv')
dtest = xgb.DMatrix('../data/Lucknow_Weather.csv')

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)

# make prediction
preds = bst.predict(dtest)


