
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/submit backups/submit_v_6.csv',
                                sep='|',header=0,index_col=None)

# remove unusable features
data = data.drop(['content', 'ppid', 'npid', 'pharmForm', 'category', 'manufacturer', 'group',
'min_unitPrice','max_unitPrice', 'unit_price',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)

data = data.drop(['unit_ST', 'unit_G', 'unit_M', 'unit_KG', 'unit_ML', 'unit_P', 'unit_L', 'unit_CM',
'campaignIndex_B', 'campaignIndex_C', 'campaignIndex_A',
'salesIndex_52', 'salesIndex_40', 'salesIndex_44', 'salesIndex_53'], axis=1)

data = data.convert_objects(convert_numeric=True)
data = data.sort_values(['lineID'], axis=0, ascending=True)

IDs = data['lineID']
data = data.drop(['day', 'pid', 'lineID', 'vnum'], axis=1)

xgbr = pickle.load(open("F:/Business Informatics/Data Mining 2/The Cup/Regression/tweedie.dat", "rb"))

data['predicted_revenue'] = xgbr.predict(data)
data['predicted_revenue'][data['predicted_revenue'] < 0] = 0

fig, ax = plt.subplots()
ax.scatter(data['predicted_revenue'], IDs)
