
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

items = pd.DataFrame.from_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/items.csv',sep='|',header=0,index_col=None)
train = pd.DataFrame.from_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/train.csv',sep='|',header=0,index_col=None)

items_cols = ['group', 'content', 'unit', 'pharmForm', 'campaignIndex', 'category']


# fill missing values
items['pharmForm'].fillna('unknown',inplace=True)
items['category'].fillna('unknown',inplace=True)
items['campaignIndex'].fillna('unknown',inplace=True)

train['competitorPrice'].fillna(0,inplace=True) # 0 means unknown or 0

# make all feaure values in smallcase
for col in items_cols:
    items[col] = items[col].map(lambda x: str(x).lower())
    

# join tables
dataset = pd.merge(items, train, on='pid', how='inner')

dataset.to_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_1.csv',sep='|')

