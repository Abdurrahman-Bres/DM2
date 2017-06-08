
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_21.csv',
                                sep='|',header=0,index_col=None)

data = data.drop(['content','click','basket','weekday',
'day_productCluster', 'min_unitPrice','max_unitPrice', 'unit_price',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice','meanPrice'], axis=1)

# remove features
data = data.drop(['new pharmForm_pharmFormC', 'new category_categoryC', 
'unit_st', 'pcpdiff', 'unit_g', 'campaignIndex_c', 'unit_m', 'unit_kg', 
'campaignIndex_b', 'unit_l', 'unit_cm', 'new category_categoryF', 
'new pharmForm_pharmFormF', 'salesIndex_52', 'new group_groupF', 'new content_contentF',
'new manufacturer_manufacturerF', 'new content_contentA', 'new content_contentB'
, 'new content_contentC', 'salesIndex_44', 'campaignIndex_unknown', 'unit_ml'
, 'unit_p', 'new category_categoryA' , 'availability_4', 'new pharmForm_pharmFormA',
'availability_3', 'new pharmForm_pharmFormB'],axis=1)

# only revenue
train = data[(data.day < 46) & (data.order == 1)]
test = data[(data.day >= 46) & (data.order == 1)]

train = train.convert_objects(convert_numeric=True)
test = test.convert_objects(convert_numeric=True)

# set label
label = train['revenue']
train = train.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue', 'productCluster'],axis=1)

# set test label
testLabel = test['revenue']
test = test.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue', 'productCluster'],axis=1)

lr = LinearRegression(fit_intercept=True)

lr.fit(train, label)

mse = np.sqrt(mean_squared_error(testLabel, lr.predict(test)))

fig, ax = plt.subplots()

# Plot the prediction versus real:
ax.scatter(lr.predict(test), testLabel)

ax.set_xlabel('predicted')
ax.set_ylabel('real')


