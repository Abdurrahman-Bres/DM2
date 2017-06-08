
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV

data = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_25.csv',
                                sep='|',header=0,index_col=None)

# remove unusable features
data = data.drop(['content','click','basket','weekday', 'ppid', 'npid','pcpdiff',
'day_productCluster', 'min_unitPrice','max_unitPrice', 'unit_price',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)

data = data.drop(['unit_st', 'unit_g', 'unit_m', 'unit_kg', 'unit_ml', 'unit_p', 'unit_l', 'unit_cm',
'campaignIndex_b', 'campaignIndex_c', 'campaignIndex_a', 'campaignIndex_unknown',
'salesIndex_52', 'salesIndex_40', 'salesIndex_44', 'salesIndex_53'], axis=1)

train = data[(data.day < 50) & (data.day > 45)]

train = train.convert_objects(convert_numeric=True)

train = train.sort_values(['lineID'], axis=0, ascending=True)

# set label
label = train['revenue']
train = train.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue'], axis=1)

params = {  
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.1,
    'colsample_bytree': 0.5,
    'subsample': 0.5,
    'gamma': 0,
    'min_child_weight': 1,
    'scale_pos_weight': 1,
    'objective': 'reg:linear'
    #'objective': 'reg:gamma'
    #'objective': 'reg:tweedie'
}


cv_params = {'max_depth': [3,5,7,9], 'min_child_weight': [1,3,5]}

optimized_GBM = GridSearchCV(estimator = XGBRegressor(**params), param_grid = cv_params, scoring = 'neg_mean_squared_error') 

optimized_GBM.fit(train, label)

optimized_GBM.grid_scores_