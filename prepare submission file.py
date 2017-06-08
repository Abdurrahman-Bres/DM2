
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

# load previous submit
submit = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/submit backups/submit_v_1.csv',
                                sep='|',header=0,index_col=None)

# load items
items = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/items.csv',
                                sep='|',header=0,index_col=None)

# load train
train = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/train.csv',
                                sep='|',header=0,index_col=None)

# load class
Class = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/class.csv',
                                sep='|',header=0,index_col=None)

# make all feaure values smallcase in items
items_cols = ['group', 'content', 'unit', 'pharmForm', 'campaignIndex']
for col in items_cols:
    items[col] = items[col].map(lambda x: str(x).lower())


dataset  = pd.merge(items, train, on='pid', how='inner')
submit  = pd.merge(items, Class, on='pid', how='inner')

# ignore first 20 days
dataset = dataset[(dataset.day > 20)]

dataset = dataset.convert_objects(convert_numeric=True)
submit = submit.convert_objects(convert_numeric=True)

#feature generation

# max/min/mean price per product ON 4 months

fourMonths = pd.concat([dataset, submit])

tempData = pd.pivot_table(fourMonths, index = 'pid', values = 'price', aggfunc = max)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'maxPrice'
submit = pd.merge(submit, series_to_frame, how='left', left_on='pid', right_index=True)

tempData = pd.pivot_table(fourMonths, index = 'pid', values = 'price', aggfunc = min)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'minPrice'
submit = pd.merge(submit, series_to_frame, how='left', left_on='pid', right_index=True)

tempData = pd.pivot_table(fourMonths, index = 'pid', values = 'price', aggfunc = np.mean)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'meanPrice'
submit = pd.merge(submit, series_to_frame, how='left', left_on='pid', right_index=True)

# get price differencies
submit['minPriceDiff'] = submit['price'] - submit['minPrice']
submit['maxPriceDiff'] = submit['price'] - submit['maxPrice']
submit['meanPriceDiff'] = submit['price'] - submit['meanPrice']

# add product clusters
items['productCluster'] = items['group'].map(str) + '_' + items['pharmForm'].map(str) + '_' + items['unit'].map(str)  + '_' + items['salesIndex'].map(str) 
submit = pd.merge(submit, items[['pid','productCluster']], how='left', on='pid')

#unique_Clusters = original_items['productCluster'].unique()

# get total content
items['content2'] = items['content'].replace('x','*',regex=True)
items['content2'] = items['content2'].replace('pak','1')
items['content2'] = items['content2'].replace('l   125','1')

for index, row in items.iterrows():
    items.set_value(index,'units_count',pd.eval(row['content2']))

submit = pd.merge(submit, items[['pid','units_count']], how='left', on='pid')

# get unit price
submit['unit_price'] = submit['price'] / submit['units_count']

# get min unit price per day
tempData = pd.pivot_table(submit, index = ['day','productCluster'], values = 'unit_price', aggfunc = min)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'min_unitPrice'
series_to_frame.reset_index(level=0, inplace=True) 
series_to_frame.reset_index(level=1, inplace=True)                
submit = pd.merge(submit, series_to_frame, how='left', on=['productCluster','day'])

# get max unit price per day
tempData = pd.pivot_table(submit, index = ['day','productCluster'], values = 'unit_price', aggfunc = max)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'max_unitPrice'
series_to_frame.reset_index(level=0, inplace=True) 
series_to_frame.reset_index(level=1, inplace=True)                
submit = pd.merge(submit, series_to_frame, how='left', on=['productCluster','day'])

# get mean unit price per day
tempData = pd.pivot_table(submit, index = ['day','productCluster'], values = 'unit_price', aggfunc = np.mean)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'mean_unitPrice'
series_to_frame.reset_index(level=0, inplace=True) 
series_to_frame.reset_index(level=1, inplace=True)                
submit = pd.merge(submit, series_to_frame, how='left', on=['productCluster','day'])

#get competitor unit price
submit['competitor_unitPrice'] = submit['competitorPrice'] / submit['units_count']

#get unit price differencies
submit['minUnitPriceDiff'] = submit['unit_price'] - submit['min_unitPrice']
submit['maxUnitPriceDiff'] = submit['unit_price'] - submit['max_unitPrice']
submit['meanUnitPriceDiff'] = submit['unit_price'] - submit['mean_unitPrice']
submit['competitor_UnitPriceDiff'] = submit['unit_price'] - submit['competitor_unitPrice']

#add is less than competitor unit price
submit['is_lessthan_competitorUnitPrice'] = (submit['unit_price'] < submit['competitor_unitPrice'])
submit.is_lessthan_competitorUnitPrice = submit.is_lessthan_competitorUnitPrice.astype(int)

# v 1

#rank rows based on day & productCluster per day
submit = submit.sort_values(['lineID'], axis=0, ascending=True)
submit['action_rank'] = submit.groupby(['day','productCluster'])['day'].rank(method='first')

#deviation per product price on 4 months
tempData = pd.pivot_table(fourMonths, index = 'pid', values = 'price', aggfunc = np.std)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'price_std'
submit = pd.merge(submit, series_to_frame, how='left', left_on='pid', right_index=True)

#deviation per product unit price on 4 months
fourMonths = pd.merge(fourMonths, items[['pid','units_count']], how='left', on='pid')
fourMonths['unit_price'] = fourMonths['price'] / fourMonths['units_count']
tempData = pd.pivot_table(fourMonths, index = 'pid', values = 'unit_price', aggfunc = np.std)
series_to_frame = pd.DataFrame(tempData)
series_to_frame.columns.values[0] = 'unit_price_std'
submit = pd.merge(submit, series_to_frame, how='left', left_on='pid', right_index=True)

# fill missing values with 0

submit = submit.sort_values(['lineID'], axis=0, ascending=True)
submit['ppid'] = submit['pid'].shift(1)
submit['npid'] = submit['pid'].shift(-1)

#add is less than previous pid
submit['is_lessthan_prvpid'] = (submit['pid'] < submit['ppid'])
submit.is_lessthan_prvpid = submit.is_lessthan_prvpid.astype(int)

#add is less than next pid
submit['is_lessthan_nxtpid'] = (submit['pid'] < submit['npid'])
submit.is_lessthan_nxtpid = submit.is_lessthan_nxtpid.astype(int)

# v 2

# assign new bins
# done externally

# get dummies
col_to_transform = ['unit','salesIndex',  'campaignIndex', 'availability',
                    'new group', 'new content', 'new category', 'new pharmForm', 'new manufacturer']

submit = pd.get_dummies(submit, columns = col_to_transform, dummy_na=False)

#join with Boyan new features
#dataset = pd.merge(dataset, new5fs, how='inner', on='lineID')

submit.to_csv('F:/Business Informatics/Data Mining 2/The Cup/submit backups/submit_v_5.csv',sep='|',index=False)



