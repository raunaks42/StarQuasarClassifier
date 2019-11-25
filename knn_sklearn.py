import operator
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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
    model = KNeighborsClassifier(n_neighbors=5)
    filenames = ['cat1.csv', 'cat1_correct_r1.csv',                       'cat1_r1.csv',
                
                'cat2.csv', 'cat2_correct_r1.csv',
                'cat2_correct_r3.csv', 'cat2_misclassified_r1.csv', 'cat2_misclassified_r3.csv', 'cat2_r1.csv', 'cat2_r3.csv',
                
                'cat3.csv', 'cat3_correct_r1.csv',
                'cat3_correct_r3.csv', 'cat3_misclassified_r1.csv', 'cat3_misclassified_r3.csv', 'cat3_r1.csv', 'cat3_r3.csv',
        
                'cat4.csv', 'cat4_correct_r1.csv',
                'cat4_correct_r3.csv', 'cat4_misclassified_r1.csv', 'cat4_misclassified_r3.csv', 'cat4_r1.csv', 'cat4_r3.csv'
                 ]

    for filename in filenames:
        start = time.time()
        if filename.endswith('.csv'):
            totSetx, totSety = loadDataset(filename)
            trainingSetx, testSetx, trainingSety, testSety = train_test_split(
                totSetx, totSety, test_size=0.25, random_state=69)
            print('Test file: ', filename)
            print('Test set size: ' + repr(len(testSetx)))

            # svclassifier = SVC(kernel='linear')
            # svclassifier.fit(trainingSetx, trainingSety)
            model.fit(trainingSetx, trainingSety)

            # y_pred = svclassifier.predict(testSetx)
            y_pred = model.predict(testSetx)

            print('accuracy = ',f1_score(testSety, y_pred, average='weighted')*100, end='\n')

            # print('Current time elapsed:', time.time()-inittime, '\n\n')
            stop = time.time()

            duration = stop - start
            
            print("Duration for this file: ", duration, '\n')



    print('Total time elapsed:', time.time()-inittime)


main()
