import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os
import time
import pandas as pd

def loadDataset(filename, crossVal=False):
        dataset=pd.read_csv(filename)
        if crossVal:
            y=dataset['class']
            x=dataset['spectrometric_redshift']
            x = x.to_numpy()
        else:
            if len(dataset.columns)==38:
                dataset=dataset.drop(dataset.columns[31:],axis=1) #drop all cols after class column
                y=dataset['class']
                dataset=dataset.drop(dataset.columns[13:16],axis=1) #drop fuv
                x=dataset.drop(dataset.columns[0:7],axis=1)
            else:   #31 cols, ie, all catalog 4 csvs
                dataset=dataset.drop(dataset.columns[30:],axis=1) #drop all after class
                y=dataset['class']
                dataset=dataset.drop(dataset.columns[13:15],axis=1)
                x=dataset.drop(dataset.columns[0:7],axis=1)
            sc = MinMaxScaler(feature_range=(0, 1))
            x = sc.fit_transform(x) #scale x vals

        y = y.to_numpy()    #convert y to numpy array
        return x,y

results=open('results.txt','a') #append to results file
print('\ne100 b10 36/40/1\n',file=results)
direc=os.fsencode('.')  #current directory
cnt=24  #32 csvs
totinittime=time.time()
for File in os.listdir(direc):  #each file in directory
    filename=os.fsdecode(File)  #get filename
    if filename.endswith('.csv'):   #only if csv
        inittime=time.time()
        dataset = np.loadtxt(filename, delimiter=',',skiprows=1)    #load into 2d numpy array, skip first row containing column labels
        if dataset.shape[1]==38:    #for csvs containing 38 columns after fix
            X = np.concatenate((dataset[:,7:13],dataset[:,16:31]),axis=1)   #all columns except class and predicted columns
            y = dataset[:,14]   #class column
        else:   #for csvs containing 31 cols
            X = np.concatenate((dataset[:,7:13],dataset[:,15:30]),axis=1) 
            y = dataset[:,13]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 69)    #train test split with specified random seed to get same output every time

        sc_X = StandardScaler() #scale cols to similar ranges
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        model = Sequential()
        model.add(Dense(25, input_dim=21, activation='relu')) #add input layer of 36 or 29 nodes and hidden layer of 12 nodes, with relu activation func
        model.add(Dense(8, activation='relu'))  #add second hidden layer with relu
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))   #add output layer with sigmoid for clear output
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   #compile model with loss function as binary crossentropy (like squared error for linear reg) used for binary classification problems, optimiser as adam (adaptive moment estimation) (like gradient descent for linear reg
        model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)  #1 epoch is 1 run through whole dataset, batch size is number of training examples considered at a time for training, all training examples considered batch by batch for each epoch
        y_pred = model.predict_classes(X_test)  #predict class for test x
        acc = f1_score(y_test , y_pred, average='weighted')    #check accuracy by comparing predicted y and test y
        
        totSetx,totSety=loadDataset(filename,True)
        trainingSetx,testSetx,trainingSety,testSety=train_test_split(totSetx,totSety, test_size = 0.25, random_state = 69)
        predictedRedShift=[]
        for i in testSetx:
            if i>=0.004:
                predictedRedShift.append(1)
            else:
                predictedRedShift.append(0)
        crossvalfscore=f1_score(y_pred,predictedRedShift,average='weighted')*100.0
        with open('times.txt','a+') as f:
            f.write( 'ann' + ',' + repr(len(X_test)) + ',' + repr(time.time()-inittime) + ',' + repr(crossvalfscore) +'\n' )
        print('done',cnt-1)   #number of files left
        cnt-=1
        #print(filename,acc*100,file=results)    #put into results file
print(time.time()-totinittime)
results.close()
