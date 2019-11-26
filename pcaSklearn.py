import operator
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# from mlxtend.evaluate import bias_variance_decomp

warnings.filterwarnings('ignore')

filenames = ['cat1.csv', 'cat1_correct_r1.csv', 'cat1_r1.csv',
             'cat2.csv', 'cat2_correct_r1.csv',
             'cat2_correct_r3.csv', 'cat2_misclassified_r1.csv', 'cat2_misclassified_r3.csv', 'cat2_r1.csv', 'cat2_r3.csv',

             'cat3.csv', 'cat3_correct_r1.csv',
             'cat3_correct_r3.csv', 'cat3_misclassified_r1.csv', 'cat3_misclassified_r3.csv', 'cat3_r1.csv', 'cat3_r3.csv',

             'cat4.csv', 'cat4_correct_r1.csv',
             'cat4_correct_r3.csv', 'cat4_misclassified_r1.csv', 'cat4_misclassified_r3.csv', 'cat4_r1.csv', 'cat4_r3.csv'
             ]


def loadDataset(filename, crossVal=False):
    dataset = pd.read_csv(filename)
    if crossVal:
        y = dataset['class']
        x = dataset['spectrometric_redshift']
        x = x.to_numpy()
    else:
        if len(dataset.columns) == 38:
            # drop all cols after class column
            dataset = dataset.drop(dataset.columns[31:], axis=1)
            y = dataset['class']
            dataset = dataset.drop(dataset.columns[13:16], axis=1)  # drop fuv
            x = dataset.drop(dataset.columns[0:7], axis=1)
        else:  # 31 cols, ie, all catalog 4 csvs
            # drop all after class
            dataset = dataset.drop(dataset.columns[30:], axis=1)
            y = dataset['class']
            dataset = dataset.drop(dataset.columns[13:15], axis=1)
            x = dataset.drop(dataset.columns[0:7], axis=1)
        sc = MinMaxScaler(feature_range=(0, 1))
        x = sc.fit_transform(x)  # scale x vals

    # y = y.to_numpy()  # convert y to numpy array
    return x, y


pca = PCA(n_components=2)
for filename in filenames:
    totSetx, totSety = loadDataset(filename)
    # print(type(x))
    principalComponents = pca.fit_transform(totSetx)

    principalDf = pd.DataFrame(data = principalComponents
             , columns=['principal component 1', 'principal component 2'])
             
    finalDf = pd.concat([principalDf, totSety], axis=1)
    newFileName = 'pca_'+filename
    finalDf.to_csv(newFileName, index=False)
    print (newFileName)

