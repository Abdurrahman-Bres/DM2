import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

#print("Missing values per column:")
#print(data.apply(num_missing, axis=0))

# count missing values
def num_missing(x):
  return sum(x.isnull())

#items = pd.DataFrame.from_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/items.csv',sep='|',header=0,index_col=None)
#train = pd.DataFrame.from_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/train.csv',sep='|',header=0,index_col=None)
#
#dataset = pd.merge(items, train, on=['pid','pid'], how='inner')

dataset = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_16.csv',
                                sep='|',header=0,index_col=None)

print(num_missing(dataset.group))


#ax = sns.countplot(x="click", data=train.groupby('click').filter(lambda x: x == 1), palette="Reds_d");
#ax = sns.countplot(x="click", data=train, palette="Reds_d");
#ax.show()

#plt.scatter(train.click[train.click == 1],train.lineID[train.click == 1],alpha=0.9,c="g")
#plt.scatter(train.basket[train.basket == 1],train.lineID[train.basket == 1],alpha=0.6,c="b")
#plt.scatter(train.order[train.order == 1],train.lineID[train.order == 1],alpha=0.2,c="r")
#
#plt.show()

#train['revenue'].hist(bins=50,hue=True)

# revenue zero 2050913
# 705090 = 2756003

# count counts only the values, no nulls
#print(train[train.competitorPrice == 0].count())
#print("\n")
#print(train[train.competitorPrice.isnull()].count())


#for name, value in items.iteritems():
#    print(name,'  ',num_missing(items[name]))
#    
#
#for name, value in train.iteritems():
#    print(name,'  ',num_missing(train[name]))

# get rid of categorical values

#cols_to_transform = [ 'group', 'content', 'unit', 'pharmForm', 'salesIndex','campaignIndex', 'availability']
#df_with_dummies = pd.get_dummies(dataset, columns = cols_to_transform, dummy_na=True)

