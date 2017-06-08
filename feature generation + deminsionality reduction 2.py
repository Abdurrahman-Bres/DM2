
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

dataset = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_25.csv',
                                sep='|',header=0,index_col=None)

#new5fs = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/train_v4.csv',
#                      sep=',',header=0,index_col=None)

#original_items = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/items.csv',
#                                sep='|',header=0,index_col=None)

dataset = dataset.convert_objects(convert_numeric=True)
#new5fs = new5fs.convert_objects(convert_numeric=True)

# ignore first 20 days
#dataset = dataset[(dataset.day > 20)].drop(['manufacturer', 'group', 'pharmForm', 'category', 'pcpratio'], axis=1)

#feature generation

# max/min/mean price per product functions ^^# October & November^^ v 10

def get_max_price(data, pid):
    data = data[(data.day < 64)]
    maxPrice = data['price'][(data['pid'] == pid)].max()
    return maxPrice
    
def get_min_price(data, pid):
    data = data[(data.day < 64)]
    minPrice = data['price'][(data['pid'] == pid)].min()
    return minPrice
    
def get_mean_price(data, pid):
    data = data[(data.day < 64)]  
    meanPrice = data['price'][(data['pid'] == pid)].mean()
    return meanPrice

# SLOW ################################################################

# get max prices
#temp = []
#for pid in dataset['pid'].unique():
#    temp.append({'pid':pid,'max price':get_max_price(dataset, pid)})
#
#max_prices = pd.DataFrame(temp)
#dataset = pd.merge(dataset, max_prices, on=['pid','pid'], how='left')
#
## get min prices
#temp = []
#for pid in dataset['pid'].unique():
#    temp.append({'pid':pid,'min price':get_min_price(dataset, pid)})
#
#min_prices = pd.DataFrame(temp)
#dataset = pd.merge(dataset, min_prices, on=['pid','pid'], how='left')
#
## get mean prices
#temp = []
#for pid in dataset['pid'].unique():
#    temp.append({'pid':pid,'mean price':get_mean_price(dataset, pid)})
#
#mean_prices = pd.DataFrame(temp)
#dataset = pd.merge(dataset, mean_prices, on=['pid','pid'], how='left')

# SLOW ################################################################

# NO NEED FOR 64

#tempData = pd.pivot_table(dataset, index = 'pid', values = 'price', aggfunc = max)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'maxPrice'
#dataset = dataset.drop(['maxPrice'],axis=1)
#dataset = pd.merge(dataset, series_to_frame, how='left', left_on='pid', right_index=True)
#
#tempData = pd.pivot_table(dataset, index = 'pid', values = 'price', aggfunc = min)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'minPrice'
#dataset = dataset.drop(['minPrice'],axis=1)
#dataset = pd.merge(dataset, series_to_frame, how='left', left_on='pid', right_index=True)
#
#tempData = pd.pivot_table(dataset, index = 'pid', values = 'price', aggfunc = np.mean)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'meanPrice'
#dataset = dataset.drop(['meanPrice'],axis=1)
#dataset = pd.merge(dataset, series_to_frame, how='left', left_on='pid', right_index=True)

# get price differencies
#dataset['minPriceDiff'] = dataset['price'] - dataset['minPrice']
#dataset['maxPriceDiff'] = dataset['price'] - dataset['maxPrice']
#dataset['meanPriceDiff'] = dataset['price'] - dataset['meanPrice']

# create similar product clusters

# make all feaure values in smallcase
#for col in original_items:
#    original_items[col] = original_items[col].map(lambda x: str(x).lower())
#    
#original_items.pid = original_items.pid.astype(np.int64)
#
#original_items['productCluster'] = original_items['group'].map(str) + '_' + original_items['pharmForm'].map(str) + '_' + original_items['unit'].map(str)  + '_' + original_items['salesIndex'].map(str) 
#
#dataset = pd.merge(dataset, original_items[['pid','productCluster']], how='left', on='pid')

#unique_Clusters = original_items['productCluster'].unique()

# get total content

#for col in original_items:
#    original_items[col] = original_items[col].map(lambda x: str(x).lower())
#
#original_items['content2'] = original_items['content'].replace('x','*',regex=True)
#original_items['content2'] = original_items['content2'].replace('pak','1')
#original_items['content2'] = original_items['content2'].replace('l   125','1')
#
#for index, row in original_items.iterrows():
#    original_items.set_value(index,'units_count',pd.eval(row['content2']))

#original_items.pid = original_items.pid.astype(np.int64)

#dataset = pd.merge(dataset, original_items[['pid','units_count']], how='left', on='pid')

# get unit price
#dataset['unit_price'] = dataset['price'] / dataset['units_count']

# get min unit price per day
#tempData = pd.pivot_table(dataset, index = ['day','productCluster'], values = 'unit_price', aggfunc = min)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'min_unitPrice'
#series_to_frame.reset_index(level=1, inplace=True)                
#dataset = pd.merge(dataset, series_to_frame, how='left', on=['productCluster','day'])

# get max unit price per day
#tempData = pd.pivot_table(dataset, index = ['day','productCluster'], values = 'unit_price', aggfunc = max)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'max_unitPrice'
#series_to_frame.reset_index(level=0, inplace=True) 
#series_to_frame.reset_index(level=1, inplace=True)                
#dataset = pd.merge(dataset, series_to_frame, how='left', on=['productCluster','day'])

# get mean unit price per day
#tempData = pd.pivot_table(dataset, index = ['day','productCluster'], values = 'unit_price', aggfunc = np.mean)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'mean_unitPrice'
#series_to_frame.reset_index(level=0, inplace=True) 
#series_to_frame.reset_index(level=1, inplace=True)                
#dataset = pd.merge(dataset, series_to_frame, how='left', on=['productCluster','day'])

#get competitor unit price
#dataset['competitor_unitPrice'] = dataset['competitorPrice'] / dataset['units_count']

#get unit price differencies
#dataset['minUnitPriceDiff'] = dataset['unit_price'] - dataset['min_unitPrice']
#dataset['maxUnitPriceDiff'] = dataset['unit_price'] - dataset['max_unitPrice']
#dataset['meanUnitPriceDiff'] = dataset['unit_price'] - dataset['mean_unitPrice']
#dataset['competitor_UnitPriceDiff'] = dataset['unit_price'] - dataset['competitor_unitPrice']

#add is less than competitor unit price
#dataset['is_lessthan_competitorUnitPrice'] = (dataset['unit_price'] < dataset['competitor_unitPrice'])
#dataset.is_lessthan_competitorUnitPrice = dataset.is_lessthan_competitorUnitPrice.astype(int)

#rank rows based on day & productCluster per day
#dataset = dataset.sort_values(['lineID'], axis=0, ascending=True)
#dataset['action_rank'] = dataset.groupby(['day','productCluster'])['day'].rank(method='first')

#deviation per product price
#tempData = pd.pivot_table(dataset, index = 'pid', values = 'price', aggfunc = np.std)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'price_std'
#dataset = dataset.drop(['price_std'],axis=1)
#dataset = pd.merge(dataset, series_to_frame, how='left', left_on='pid', right_index=True)

#deviation per product unit price
#tempData = pd.pivot_table(dataset, index = 'pid', values = 'unit_price', aggfunc = np.std)
#series_to_frame = pd.DataFrame(tempData)
#series_to_frame.columns.values[0] = 'unit_price_std'
#dataset = dataset.drop(['unit_price_std'],axis=1)
#dataset = pd.merge(dataset, series_to_frame, how='left', left_on='pid', right_index=True)

#fill caused missing values
#dataset['price_std'].fillna(0,inplace=True)
#dataset['unit_price_std'].fillna(0,inplace=True)

#dataset = dataset.sort_values(['lineID'], axis=0, ascending=True)
#dataset['ppid'] = dataset['pid'].shift(1)
#dataset['npid'] = dataset['pid'].shift(-1)

#add is less than previous pid
#dataset['is_lessthan_prvpid'] = (dataset['pid'] < dataset['ppid'])
#dataset.is_lessthan_prvpid = dataset.is_lessthan_prvpid.astype(int)

#add is less than next pid
#dataset['is_lessthan_nxtpid'] = (dataset['pid'] < dataset['npid'])
#dataset.is_lessthan_nxtpid = dataset.is_lessthan_nxtpid.astype(int)

#join Boyan new 5 features
#new5fs = new5fs.drop(['pid'],axis=1)
#dataset = pd.merge(dataset, new5fs, how='inner', on='lineID')

#dataset.to_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_25.csv',sep='|',index=False)



