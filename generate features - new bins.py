
import pandas as pd
import numpy as np

dataset = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/submit backups/submit_v_3.csv',
                                sep='|',header=0,index_col=None,
                                dtype='unicode')

items = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/items.csv',
                                sep='|',header=0,index_col=None)

items = items.drop(['genericProduct','salesIndex','rrp'], axis=1)

# fill missing values
items['pharmForm'].fillna('unknown',inplace=True)
items['category'].fillna('unknown',inplace=True)
items['campaignIndex'].fillna('unknown',inplace=True)

for col in items:
    items[col] = items[col].map(lambda x: str(x).lower())
    
items.pid = items.pid.astype(np.int64)
dataset.pid = dataset.pid.astype(np.int64)

dataset  = pd.merge(dataset, items, on='pid', how='inner')

groups = pd.read_csv('F:\Business Informatics\Data Mining 2\The Cup\Feature Engineering/less groups.csv',
                                sep=',',header=0,index_col=None)

contents = pd.read_csv('F:\Business Informatics\Data Mining 2\The Cup\Feature Engineering/less contents.csv',
                                sep=',',header=0,index_col=None)

categories = pd.read_csv('F:\Business Informatics\Data Mining 2\The Cup\Feature Engineering/less categories.csv',
                                sep=',',header=0,index_col=None)

pharmForms = pd.read_csv('F:\Business Informatics\Data Mining 2\The Cup\Feature Engineering/less pharmForms.csv',
                                sep=',',header=0,index_col=None)

manufacturers = pd.read_csv('F:\Business Informatics\Data Mining 2\The Cup\Feature Engineering/less manufacturers.csv',
                                sep=',',header=0,index_col=None)

groupA = groups['group name'][groups['new group'] == 'group A']
groupB = groups['group name'][groups['new group'] == 'group B']
groupC = groups['group name'][groups['new group'] == 'group C']
groupD = groups['group name'][groups['new group'] == 'group D']

contentA = contents['content name'][contents['new content'] == 'content A']
contentB = contents['content name'][contents['new content'] == 'content B']
contentC = contents['content name'][contents['new content'] == 'content C']
contentD = contents['content name'][contents['new content'] == 'content D']

categoryA = categories['category name'][categories['new category'] == 'category A']
categoryB = categories['category name'][categories['new category'] == 'category B']
categoryC = categories['category name'][categories['new category'] == 'category C']
categoryD = categories['category name'][categories['new category'] == 'category D']

pharmFormA = pharmForms['pharmForm name'][pharmForms['new pharmForm'] == 'pharmForm A']
pharmFormB = pharmForms['pharmForm name'][pharmForms['new pharmForm'] == 'pharmForm B']
pharmFormC = pharmForms['pharmForm name'][pharmForms['new pharmForm'] == 'pharmForm C']
pharmFormD = pharmForms['pharmForm name'][pharmForms['new pharmForm'] == 'pharmForm D']

manufacturerA = manufacturers['manufacturer name'][manufacturers['new manufacturer'] == 'manufacturer A']
manufacturerB = manufacturers['manufacturer name'][manufacturers['new manufacturer'] == 'manufacturer B']
manufacturerC = manufacturers['manufacturer name'][manufacturers['new manufacturer'] == 'manufacturer C']
manufacturerD = manufacturers['manufacturer name'][manufacturers['new manufacturer'] == 'manufacturer D']

dataset['new group'] = 'groupF'
dataset['new group'] = np.where((('group_'+ dataset['group']).isin(groupA)),'groupA',dataset['new group'])
dataset['new group'] = np.where((('group_'+ dataset['group']).isin(groupB)),'groupB',dataset['new group'])
dataset['new group'] = np.where((('group_'+ dataset['group']).isin(groupC)),'groupC',dataset['new group'])
dataset['new group'] = np.where((('group_'+ dataset['group']).isin(groupD)),'groupD',dataset['new group'])

dataset['new content'] = 'contentF'
dataset['new content'] = np.where((('content_'+ dataset['content']).isin(contentA)),'contentA',dataset['new content'])
dataset['new content'] = np.where((('content_'+ dataset['content']).isin(contentB)),'contentB',dataset['new content'])
dataset['new content'] = np.where((('content_'+ dataset['content']).isin(contentC)),'contentC',dataset['new content'])
dataset['new content'] = np.where((('content_'+ dataset['content']).isin(contentD)),'contentD',dataset['new content'])

dataset['new category'] = 'categoryF'
dataset['new category'] = np.where((('category_'+ dataset['category'].map(str)).isin(categoryA)),'categoryA',dataset['new category'])
dataset['new category'] = np.where((('category_'+ dataset['category'].map(str)).isin(categoryB)),'categoryB',dataset['new category'])
dataset['new category'] = np.where((('category_'+ dataset['category'].map(str)).isin(categoryC)),'categoryC',dataset['new category'])
dataset['new category'] = np.where((('category_'+ dataset['category'].map(str)).isin(categoryD)),'categoryD',dataset['new category'])

dataset['new pharmForm'] = 'pharmFormF'
dataset['new pharmForm'] = np.where((('pharmForm_'+ dataset['pharmForm']).isin(pharmFormA)),'pharmFormA',dataset['new pharmForm'])
dataset['new pharmForm'] = np.where((('pharmForm_'+ dataset['pharmForm']).isin(pharmFormB)),'pharmFormB',dataset['new pharmForm'])
dataset['new pharmForm'] = np.where((('pharmForm_'+ dataset['pharmForm']).isin(pharmFormC)),'pharmFormC',dataset['new pharmForm'])
dataset['new pharmForm'] = np.where((('pharmForm_'+ dataset['pharmForm']).isin(pharmFormD)),'pharmFormD',dataset['new pharmForm'])

dataset['new manufacturer'] = 'manufacturerF'
dataset['new manufacturer'] = np.where((('manufacturer_'+ dataset['manufacturer'].map(str)).isin(manufacturerA)),'manufacturerA',dataset['new manufacturer'])
dataset['new manufacturer'] = np.where((('manufacturer_'+ dataset['manufacturer'].map(str)).isin(manufacturerB)),'manufacturerB',dataset['new manufacturer'])
dataset['new manufacturer'] = np.where((('manufacturer_'+ dataset['manufacturer'].map(str)).isin(manufacturerC)),'manufacturerC',dataset['new manufacturer'])
dataset['new manufacturer'] = np.where((('manufacturer_'+ dataset['manufacturer'].map(str)).isin(manufacturerD)),'manufacturerD',dataset['new manufacturer'])

dataset.to_csv('F:/Business Informatics/Data Mining 2/The Cup/submit backups/submit_v_4.csv',sep='|',index=False)
