import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from concurrent.futures import ProcessPoolExecutor
import operator
import time
import os
import warnings
warnings.filterwarnings('ignore')

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

def getNeighbors(trainingSetx, trainingSety, testInstance, k):
    distances = []
    futures=[]
    with ProcessPoolExecutor(8) as executor:
        for x in range(len(trainingSetx)):
            futures.append(executor.submit(manhattanDistance, testInstance,trainingSetx[x]) )
    for i in futures:
        distances.append(( trainingSety[x] , i.result() ))
    #for x in range(len(trainingSetx)):
        #distances.append( ( trainingSety[x] , manhattanDistance(testInstance,trainingSetx[x]) ) ) # tuple ( training example y, distance from test example ) appended in distances

    distances.sort(key=operator.itemgetter(1))  #sort by distance ascending
    neighbors=[x[0] for x in distances[:k]]    #for best k, remove distance values, ie, append training example y to neighbours

    return neighbors

def manhattanDistance(instance1, instance2):
	return np.sum(np.absolute(instance1-instance2)) #element wise subtraction, element wise absolute, sum of elements

def getResponse(neighbors):
	classVotes = [0,0]  #votes of class 0 and class 1
	for x in range(len(neighbors)):
		response = neighbors[x]  #y value of neighbor training example
		classVotes[response] += 1
	return int(classVotes[1]>classVotes[0]) #1 if more votes from label 1, else 0

def kNN(Setx, Sety, trainingSetx, trainingSety, k):
    predictions=[]
    #futures=[]
    #with ProcessPoolExecutor() as executor:
        #for x in range(len(Setx)):
            #futures.append(executor.submit(getResponse, getNeighbors(trainingSetx, trainingSety, Setx[x], k) ) )
    #for i in futures:
        #predictions.append(i.result())
    for x in range(len(Setx)):	#runs the number of times the amount of values in the test data set
        neighbors = getNeighbors(trainingSetx, trainingSety, Setx[x], k)    #returns all nearest y values
        result = getResponse(neighbors) #returns majority vote
        predictions.append(result)

    return f1_score(Sety.tolist(),predictions,average='weighted')*100.0 , predictions    #sety.tolist() converts numpy array to list

def runkNN(filename):
    output=''
    #testSetx,testSety=loadDataset(filename) #load test data
    totSetx,totSety=loadDataset(filename)
    trainingSetx,testSetx,trainingSety,testSety=train_test_split(totSetx,totSety, test_size = 0.25, random_state = 69)
    output+='Test file: '+filename+'\n'+'Test size: '+repr(len(testSetx))+'\n'

    #cvSetx, reportSetx, cvSety, reportSety = train_test_split(testSetx,testSety, test_size = 0.5, random_state = 0) #split half into set for finding best k and set for reporting final fscore
    '''
    bestk=1
    bestacc=0
    for k in range(1,32,2): #k=1,3,...,31
        accuracy = kNN(cvSetx, cvSety, trainingSetx, trainingSety, k)
        if accuracy>bestacc:
            bestk=k
            bestacc=accuracy
        print("k=",k,"\tf1 score=",accuracy,'%')
    '''
    accuracy, predictedModel = kNN(testSetx, testSety, trainingSetx, trainingSety, 5)    #report final fscore for best k
    output+='Reported f1 score: '+repr(accuracy)+'%\n'

    totSetx,totSety=loadDataset(filename,True)
    trainingSetx,testSetx,trainingSety,testSety=train_test_split(totSetx,totSety, test_size = 0.25, random_state = 69)
    predictedRedShift=[]
    for i in testSetx:
        if i>=0.004:
            predictedRedShift.append(1)
        else:
            predictedRedShift.append(0)
    #accuracy, predictedRedShift = kNN(testSetx, testSety, trainingSetx, trainingSety, 5)
    output+='Crossvalidated acc='+repr(f1_score(predictedModel,predictedRedShift,average='weighted')*100.0)+'%\n'
    print(output)

def main():
    direc=os.fsencode('.')  #encode current directory

    inittime=time.time()
    #filenames=[]
    #filenames=['cat2.csv','cat2_r1.csv','cat2_r2.csv','cat2_r3.csv']
    #for filename in filenames:
    for File in os.listdir(direc):  #each file in directory
        filename=os.fsdecode(File)  #get filename
        if filename.endswith('.csv'):
            #filenames.append(filename)
            runkNN(filename)
    #with ProcessPoolExecutor() as executor:
        #executor.map(runkNN,filenames)

    print('Total time elapsed:', time.time()-inittime)

main()

