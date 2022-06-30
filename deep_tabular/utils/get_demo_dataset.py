""" get_demo_dataset.py
    Utilities for splitting yeast data into upstream and downstream tasks
    Developed for Tabular-Transfer-Learning project
    March 2022
    Data link: http://mulan.sourceforge.net/datasets-mlc.html
"""
import numpy as np
import scipy
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
data, meta = scipy.io.arff.loadarff('data/yeast/yeast.arff')
df = pd.DataFrame(data)

target_columns = [col for col in df.columns if 'Class' in col]
non_target_columns = [col for col in df.columns if 'Class' not in col]

le = preprocessing.LabelEncoder()
Y = df[target_columns].apply(le.fit_transform)
print(Y)

X = df[non_target_columns]

downstream_target_index = 5
downstream_target = target_columns[downstream_target_index]
target_columns.pop(downstream_target_index)

X_upstream, X_downstream, Y_upstream, Y_downstream = train_test_split(X, Y, test_size=0.2, random_state=0)
Y_downstream = Y_downstream[downstream_target]
Y_upstream = Y_upstream[target_columns]

X_upstream.to_csv('data/yeast_upstream/N.csv', index = False)
Y_upstream.to_csv('data/yeast_upstream/y.csv', index = False)

X_downstream.to_csv('data/yeast_downstream/N.csv', index = False)
Y_downstream.to_csv('data/yeast_downstream/y.csv', index = False)

#0.628 with TL
#0.578 with no TL
