
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

data = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_25.csv',
                                sep='|',header=0,index_col=None)

# remove unusable features
data = data.drop(['content','click','basket','weekday', 'ppid', 'npid','pcpdiff',
'day_productCluster', 'min_unitPrice','max_unitPrice', 'unit_price',
'competitor_unitPrice', 'mean_unitPrice', 'maxPrice', 'minPrice', 'meanPrice', 'productCluster'], axis=1)

train = data[(data.day < 46)]
test = data[(data.day >= 46)]

train = train.convert_objects(convert_numeric=True)
test = test.convert_objects(convert_numeric=True)

# set label
label = train['qty']
train = train.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue'],axis=1)

# set test label
testLabel = test['qty']
test = test.drop(['day', 'order', 'pid', 'lineID', 'qty', 'revenue'],axis=1)

params = {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(train, label)

mse = np.sqrt(mean_squared_error(testLabel, clf.predict(test)))

fig, ax = plt.subplots()

# Plot the prediction versus real:
ax.scatter(clf.predict(test), testLabel)

ax.set_xlabel('predicted')
ax.set_ylabel('real')