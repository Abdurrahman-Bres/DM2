
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

dataTrain = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/full_v_27.csv',
                                sep='|',header=0,index_col=None)

#dataTest = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_25.csv',
#                                sep='|',header=0,index_col=None)

dataSubmit = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/submit backups/submit_v_6.csv',
                                sep='|',header=0,index_col=None)

# remove unusable features
dataTrain = dataTrain.drop(['content','click','basket','weekday', 'ppid', 'npid','pcpdiff',
'day_productCluster', 'min_unitPrice','max_unitPrice', 'unit_price', 'isWeekend', 'isCPna',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)

dataSubmit = dataSubmit.drop(['content', 'ppid', 'npid',
'min_unitPrice','max_unitPrice', 'unit_price',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)

dataTrain = dataTrain.drop(['unit_st', 'unit_g', 'unit_m', 'unit_kg', 'unit_ml', 'unit_p', 'unit_l', 'unit_cm',
'campaignIndex_b', 'campaignIndex_c', 'campaignIndex_a', 'campaignIndex_unknown',
'salesIndex_52', 'salesIndex_40', 'salesIndex_44', 'salesIndex_53',
'new pharmForm_pharmFormA', 'new pharmForm_pharmFormB', 'new pharmForm_pharmFormC', 'new pharmForm_pharmFormD',
'new group_groupA', 'new group_groupB', 'new group_groupC', 'new group_groupD', 'action_rank',
'new content_contentA', 'new content_contentB', 'new content_contentC', 'new content_contentD',
'new manufacturer_manufacturerA', 'new manufacturer_manufacturerB', 'new manufacturer_manufacturerC', 'new manufacturer_manufacturerD',
'new category_categoryA', 'new category_categoryB', 'new category_categoryC', 'new category_categoryD'
], axis=1)

dataSubmit = dataSubmit.drop(['unit_st', 'unit_g', 'unit_m', 'unit_kg', 'unit_ml', 'unit_p', 'unit_l', 'unit_cm',
'campaignIndex_b', 'campaignIndex_c', 'campaignIndex_a', 'campaignIndex_unknown',
'salesIndex_52', 'salesIndex_40', 'salesIndex_44', 'salesIndex_53',
'new pharmForm_pharmFormA', 'new pharmForm_pharmFormB', 'new pharmForm_pharmFormC', 'new pharmForm_pharmFormD',
'new group_groupA', 'new group_groupB', 'new group_groupC', 'new group_groupD', 'action_rank',
'new content_contentA', 'new content_contentB', 'new content_contentC', 'new content_contentD',
'new manufacturer_manufacturerA', 'new manufacturer_manufacturerB', 'new manufacturer_manufacturerC', 'new manufacturer_manufacturerD',
'new category_categoryA', 'new category_categoryB', 'new category_categoryC', 'new category_categoryD'
], axis=1)

#################
#dataTest = dataTest.drop(['content','click','basket','weekday', 'ppid', 'npid','pcpdiff',
#'day_productCluster', 'min_unitPrice','max_unitPrice', 'unit_price', 'isWeekend', 'isCPna',
#'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)
#
#dataTest = dataTest.drop(['unit_st', 'unit_g', 'unit_m', 'unit_kg', 'unit_ml', 'unit_p', 'unit_l', 'unit_cm',
#'campaignIndex_b', 'campaignIndex_c', 'campaignIndex_a', 'campaignIndex_unknown',
#'salesIndex_52', 'salesIndex_40', 'salesIndex_44', 'salesIndex_53',
#'new pharmForm_pharmFormA', 'new pharmForm_pharmFormB', 'new pharmForm_pharmFormC', 'new pharmForm_pharmFormD',
#'new group_groupA', 'new group_groupB', 'new group_groupC', 'new group_groupD', 
#], axis=1)
#################

# train and test
train = dataTrain #[(dataTrain.day < 46)]
#test = dataTest[(dataTrain.day >= 46)]

train = train.convert_objects(convert_numeric=True)
#test = test.convert_objects(convert_numeric=True)
dataSubmit = dataSubmit.convert_objects(convert_numeric=True)

train = train.sort_values(['lineID'], axis=0, ascending=True)
#test = test.sort_values(['lineID'], axis=0, ascending=True)
dataSubmit = dataSubmit.sort_values(['lineID'], axis=0, ascending=True)

# set label
ori_train = train
label = train['revenue']
train = train.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue', 'vnum'], axis=1)

# set test label
#ori_test = test
#testLabel = test['revenue']
#test = test.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue', 'vnum'], axis=1)

# submission
ori_dataSubmit = dataSubmit
dataSubmit = dataSubmit.drop(['day', 'pid', 'lineID', 'vnum'], axis=1)

# order columns
train = train.reindex_axis(sorted(train.columns), axis=1)
#test = test.reindex_axis(sorted(train.columns), axis=1)
dataSubmit = dataSubmit.reindex_axis(sorted(train.columns), axis=1)

params = {  
    'max_delta_step': 1,
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.01,
    'colsample_bytree': 1,
    'subsample': 1,
    'gamma': 0,
    'min_child_weight': 1,
    'scale_pos_weight': 1,
    #'objective': 'count:poisson'
    #'objective': 'reg:linear'
    #'objective': 'reg:gamma'
    'objective': 'reg:tweedie'
}

xgbr = XGBRegressor(**params)
xgbr.fit(train, label)

# load saved model
# xx = pickle.load(open("F:/Business Informatics/Data Mining 2/The Cup/Regression/tweedie.dat", "rb"))

ori_train['predicted_revenue'] = xgbr.predict(train)
ori_train['predicted_revenue'][ori_train['predicted_revenue'] < 0] = 0
         
ori_dataSubmit['revenue'] = xgbr.predict(dataSubmit)
ori_dataSubmit['revenue'][ori_dataSubmit['revenue'] < 0] = 0

# train error
train_mse = np.sqrt(mean_squared_error(label ,ori_train['predicted_revenue']))

#ori_test['predicted_revenue'] = xgbr.predict(test)
#ori_test['predicted_revenue'][ori_test['predicted_revenue'] < 0] = 0

# test error
#revenue_mse = np.sqrt(mean_squared_error(testLabel,ori_test['predicted_revenue']))

fig, ax = plt.subplots()

# Plot the prediction versus real:
ax.scatter(ori_train['predicted_revenue'], label)

ax.set_xlabel('predicted')
ax.set_ylabel('real')

xgb.plot_importance(xgbr,max_num_features=30)

# save model to disk
#pickle.dump(xgbr, open("F:/Business Informatics/Data Mining 2/The Cup/Regression/final2_tweedie.dat", "wb"))
