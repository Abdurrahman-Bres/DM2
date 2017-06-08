
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_25.csv',
                                sep='|',header=0,index_col=None)

# remove unusable features
data = data.drop(['content','click','basket','weekday', 'ppid', 'npid','pcpdiff',
'day_productCluster', 'min_unitPrice','max_unitPrice', 'unit_price', 'isWeekend', 'isCPna',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)

data = data.drop(['unit_st', 'unit_g', 'unit_m', 'unit_kg', 'unit_ml', 'unit_p', 'unit_l', 'unit_cm',
'campaignIndex_b', 'campaignIndex_c', 'campaignIndex_a', 'campaignIndex_unknown',
'salesIndex_52', 'salesIndex_40', 'salesIndex_44', 'salesIndex_53',
'new pharmForm_pharmFormA', 'new pharmForm_pharmFormB', 'new pharmForm_pharmFormC', 'new pharmForm_pharmFormD',
'new group_groupA', 'new group_groupB', 'new group_groupC', 'new group_groupD', 'action_rank'
], axis=1)

data = data.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue', 'vnum'], axis=1)


xgbr = pickle.load(open("F:/Business Informatics/Data Mining 2/The Cup/Regression/tweedie.dat", "rb"))

for x in xgbr.feature_importances_:
    print(x)

