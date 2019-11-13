import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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
	for x in range(len(trainingSetx)):
		distances.append( ( trainingSety[x] , manhattanDistance(testInstance,trainingSetx[x]) ) ) # tuple ( training example y, distance from test example ) appended in distances
	
	distances.sort(key=operator.itemgetter(1))  #sort by distance ascending
	neighbors=[x[0] for x in distances[:k]]    #for best k, remove distance values, ie, append training example y to neighbours
	
	return neighbors

def manhattanDistance(instance1, instance2):
	return np.sum(np.absolute(instance1-instance2)) #element wise subtraction, element wise absolute, sum of elements

def getResponse(neighbors):
	classVotes = {0:0,1:0}  #votes of class 0 and class 1
	for x in range(len(neighbors)):
		response = neighbors[x]  #y value of neighbor training example
		classVotes[response] += 1
	return int(classVotes[1]>classVotes[0]) #1 if more votes from label 1, else 0
		
def kNN(Setx, Sety, trainingSetx, trainingSety, k):
    predictions=[]
    for x in range(len(Setx)):	#runs the number of times the amount of values in the test data set
        neighbors = getNeighbors(trainingSetx, trainingSety, Setx[x], k)    #returns all nearest y values
        result = getResponse(neighbors) #returns majority vote
        predictions.append(result)
        
    return f1_score(Sety.tolist(),predictions)*100.0 , predictions    #sety.tolist() converts numpy array to list

def main():
    direc=os.fsencode('.')  #encode current directory
    
    inittime=time.time()
    
    #filenames=['cat1.csv','cat1_r1.csv','cat1_r2.csv','cat1_r3.csv']
    #for filename in filenames:
    for File in os.listdir(direc):  #each file in directory
        filename=os.fsdecode(File)  #get filename
        if filename.endswith('.csv'):
            #testSetx,testSety=loadDataset(filename) #load test data
            totSetx,totSety=loadDataset(filename)
            trainingSetx,testSetx,trainingSety,testSety=train_test_split(totSetx,totSety, test_size = 0.25, random_state = 69)
            print('Test file: ',filename)
            print ('Test set size: ' + repr(len(testSetx)))
            
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
            print('Reported f1 score: ' + repr(accuracy) + '%')
            
            totSetx,totSety=loadDataset(filename,True)
            trainingSetx,testSetx,trainingSety,testSety=train_test_split(totSetx,totSety, test_size = 0.25, random_state = 69)
            accuracy, predictedRedShift = kNN(testSetx, testSety, trainingSetx, trainingSety, 5)
            print('Crossvalidated acc=',f1_score(predictedModel,predictedRedShift)*100.0,'%\n')
            
            
    print('Total time elapsed:', time.time()-inittime)
	    
main()
