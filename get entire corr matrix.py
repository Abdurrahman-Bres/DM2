
import pandas as pd

dataset = pd.read_csv('F:/Business Informatics/Data Mining 2/The Cup/DMC_2017_task/joint_v_23.csv',
                                sep='|',header=0,index_col=None,
                                dtype='unicode')

dataset = dataset.convert_objects(convert_numeric=True)

p = dataset['action_rank'].corr(method='pearson')['revenue']

