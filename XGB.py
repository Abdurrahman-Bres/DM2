
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
'day_productCluster', 'min_unitPrice','max_unitPrice', 'unit_price',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)

data = data.drop(['unit_st', 'unit_g', 'unit_m', 'unit_kg', 'unit_ml', 'unit_p', 'unit_l', 'unit_cm',
'campaignIndex_b', 'campaignIndex_c', 'campaignIndex_a', 'campaignIndex_unknown',
'salesIndex_52', 'salesIndex_40', 'salesIndex_44', 'salesIndex_53'], axis=1)

train = data[(data.day < 46)]
test = data[(data.day >= 46)] # 46

train = train.convert_objects(convert_numeric=True)
test = test.convert_objects(convert_numeric=True)

train = train.sort_values(['lineID'], axis=0, ascending=True)
test = test.sort_values(['lineID'], axis=0, ascending=True)

# set label
label = train['qty']
train = train.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue'], axis=1)

# set test label
ori_test = test
testLabel = test['qty']
test = test.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue'], axis=1)

params = {  
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'colsample_bytree': 0.5,
    'subsample': 0.5,
    'gamma': 0,
    'min_child_weight': 1,
    'scale_pos_weight': 1,
    'objective': 'reg:linear'
}

xgbr = XGBRegressor(**params)

xgbr.fit(train, label)

# post processing
ori_test['predicted_qty'] = xgbr.predict(test)
# make all negatives zeros
ori_test['predicted_qty'][ori_test['predicted_qty'] < 0] = 0
# round quantities
ori_test['predicted_qty'] = ori_test['predicted_qty'].round()

# get predicted revenue
ori_test['predicted_revenue'] = ori_test['predicted_qty'] * ori_test['price']

qty_mse = np.sqrt(mean_squared_error(testLabel,ori_test['predicted_qty']))

revenue_mse = np.sqrt(mean_squared_error(ori_test['revenue'], ori_test['predicted_revenue']))

fig, ax = plt.subplots()

# Plot the prediction versus real:
ax.scatter(ori_test['predicted_qty'], testLabel)

ax.set_xlabel('predicted')
ax.set_ylabel('real')

xgb.plot_importance(xgbr,max_num_features=30)
