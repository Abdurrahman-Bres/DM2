
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr


dataset = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_5.csv',
                                sep='|',header=0,index_col=None,
                                dtype='unicode')

# get feature values
#data = dataset.groupby('content').count() #[dataset.day < 2]

dataset['day'] = dataset['day'].apply(pd.to_numeric)
data = dataset[['manufacturer','revenue']] #[dataset.day < 2]
data['revenue'] = data['revenue'].apply(pd.to_numeric)
data = data[data.revenue != 0]
del dataset

#data2 = dataset[dataset.revenue != 0].groupby('content').sum()

col_to_transform = ['manufacturer']
p = pd.get_dummies(data, columns = col_to_transform, dummy_na=True).corr(method='pearson')['revenue']
#


# Spearman rank-order correlation
#print(spearmanr(data.revenue, data.group))



#print(data)


