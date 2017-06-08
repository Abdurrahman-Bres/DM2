
import pandas as pd

train = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/train.csv',
                                sep='|',header=0,index_col=None,
                                dtype='unicode')

items = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/items.csv',
                                sep='|',header=0,index_col=None,
                                dtype='unicode')


test = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/class.csv',
                                sep='|',header=0,index_col=None,
                                dtype='unicode')


# join tables
data = pd.merge(items, train, on='pid', how='inner')


dataTest = pd.merge(items, test, on='pid', how='inner')

