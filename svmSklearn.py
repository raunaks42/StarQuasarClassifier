import operator
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')


def loadDataset(filename, crossVal=False):
    dataset = pd.read_csv(filename)
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

    y = y.to_numpy()  # convert y to numpy array
    return x, y


def main():
    inittime = time.time()

    filenames = ['cat2.csv', 'cat2_r1.csv', 'cat2_r3.csv']  #will not work for files which have only 1 class
    for filename in filenames:
        if filename.endswith('.csv'):
            totSetx, totSety = loadDataset(filename)
            trainingSetx, testSetx, trainingSety, testSety = train_test_split(
                totSetx, totSety, test_size=0.25, random_state=69)
            print('Test file: ', filename)
            print('Test set size: ' + repr(len(testSetx)))

            svclassifier = SVC(kernel='linear')
            svclassifier.fit(trainingSetx, trainingSety)

            y_pred = svclassifier.predict(testSetx)

            print(f1_score(testSety, y_pred, average='weighted')*100, end='\n')

    print('Total time elapsed:', time.time()-inittime)


main()
