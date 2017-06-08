
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.model_selection import StratifiedShuffleSplit


dataset = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/dataTwoMonth_Clean.csv',
                                sep='|',header=0,index_col=None,
                                dtype='unicode')

dataset['isComp'] = np.where((dataset['isComp'] == 'False'),0, 1)

dataset['day'] = dataset['day'].apply(pd.to_numeric)

train = dataset[(dataset['day'] <= 61)]
test = dataset[(dataset['day'] > 61)]

del dataset

train_order = train[(train['order'] == '1')]
train_zero = train[(train['order'] == '0')]

train = pd.concat([train_order.sample(50000),train_zero.sample(50000)])

test = test.sample(30000,random_state=9)

#dataset = dataset.drop(['pid', 'manufacturer', 'group', 'content',
#                        'pharmForm', 'genericProduct', 'category', 
#                        'rrp', 'day', 'adFlag', 'competitorPrice', 
#                        'click', 'basket', 'order', 'price', 'revenue'],axis=1)

#
#dataset = dataset.drop(['upid', 'upidCon', 'upidMan',
#                        'availability_nan', 'campaignIndex_nan',
#                        'salesIndex_nan', 'unit_nan',
#                        'new pharmForm_nan', 'new category_nan',
#                        'new content_nan', 'new group_nan',
#                        'PCP_ratio','orderPid', 'basketPid', 'clickPid', 'revenuePid'], axis=1)


train.to_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/sample v 6/training.csv',sep='|',index=False)
test.to_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/sample v 6/testing.csv',sep='|',index=False)

