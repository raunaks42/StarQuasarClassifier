import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score as acc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

results=open('results.txt','a') #append to results file
print('\ne100 b10 36/40/1\n',file=results)
direc=os.fsencode('.')  #current directory
cnt=32  #32 csvs
for File in os.listdir(direc):  #each file in directory
    filename=os.fsdecode(File)  #get filename
    if filename.endswith('.csv'):   #only if csv
        dataset = np.loadtxt(filename, delimiter=',',skiprows=1)    #load into 2d numpy array, skip first row containing column labels
        if dataset.shape[1]==38:    #for csvs containing 38 columns after fix
            X = np.concatenate((dataset[:,0:14],dataset[:,15:37]),axis=1)   #all columns except class and predicted columns
            y = dataset[:,14]   #class column
        else:   #for csvs containing 31 cols
            X = np.concatenate((dataset[:,0:13],dataset[:,14:30]),axis=1)
            y = dataset[:,13]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    #train test split with specified random seed to get same output every time

        sc_X = StandardScaler() #scale cols to similar ranges
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        model = Sequential()
        model.add(Dense(40, input_dim=(36 if dataset.shape[1]==38 else 29), activation='relu')) #add input layer of 36 or 29 nodes and hidden layer of 12 nodes, with relu activation func
        #model.add(Dense(8, activation='relu'))  #add second hidden layer with relu
        model.add(Dense(1, activation='sigmoid'))   #add output layer with sigmoid for clear output
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   #compile model with loss function as binary crossentropy (like squared error for linear reg) used for binary classification problems, optimiser as adam (adaptive moment estimation) (like gradient descent for linear reg
        model.fit(X_train, y_train, epochs=100, batch_size=10)  #1 epoch is 1 run through whole dataset, batch size is number of training examples considered at a time for training, all training examples considered batch by batch for each epoch
        y_pred = model.predict_classes(X_test)  #predict class for test x
        acc = acc_score(y_test , y_pred)    #check accuracy by comparing predicted y and test y
        print('done',cnt-1)   #number of files left
        cnt-=1
        print(filename,acc*100,file=results)    #put into results file
results.close()
