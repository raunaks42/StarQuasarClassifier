import csv
import random
import math
import operator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import time
import numpy as np

'''
def loadDataset(filename, split, trainingSet=[], testSet=[]):
	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)

		dataset = (pd.DataFrame(list(lines)))[:][1:]    #remove column labels
		dataset.drop(dataset.columns[0:2],axis=1).drop(dataset.columns[13],axis=1).drop(dataset.columns[15:],axis=1)
		
		sc_X = MinMaxScaler(feature_range=(0, 1))
		dataset = sc_X.fit_transform(dataset)
		# print("\n\n\n",dataset[1],"\n\n\n")
		for x in range(len(dataset) - 1):
			# print(dataset[x],"\n")
			for y in range(12): #to use all the data values upto the 37th column in the Dataset csv
				dataset[x][y] = float(dataset[x][y])
			if random.random() < split:
				trainingSet.append(dataset[x])	#the set containing the data points that we'll use to train the algorithm
			else:
				testSet.append(dataset[x])	#the set containing the data points that we'll use to test the algorithm
'''

def loadDataset(filename):
	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)
        
		dataset = ( pd.DataFrame(list(lines)) )[:][1:]    #remove column labels (first row)
		if len(dataset.columns)==38:
		    dataset=dataset.drop(dataset.columns[15:],axis=1) #drop all cols after class column
		    dataset=dataset.drop(dataset.columns[13],axis=1) #drop fuv
		    y=dataset[14]
		else:
		    dataset=dataset.drop(dataset.columns[14:],axis=1)
		    y=dataset[13]
		    
		dataset=dataset.drop(dataset.columns[0:2],axis=1) #drop id cols
		x=dataset.drop(dataset.columns[11],axis=1) #drop class col (after previous drops, its col index is 11
		
		sc = MinMaxScaler(feature_range=(0, 1))
		x = sc.fit_transform(x)
		y = y.to_numpy()
		
		return x,y
		#for x in range(len(dataset)):
		    #Set.append(dataset[x])

def getNeighbors(trainingSetx, trainingSety, testInstance, k):
	#getting the nearest neighbours
	#testInstance contains the xth (the entire row of values {list of len 38}) value of the testDataSet, so we'll be getting the nearest neighbours of that data point(list), in the test data set, wrt to the trainDataSet
	distances = []
	for x in range(len(trainingSetx)):
		dist = manhattanDistance(testInstance, trainingSetx[x])
		distances.append( ( (trainingSetx[x],trainingSety[x]) , dist ) )
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def manhattanDistance(instance1, instance2):
	#instance 1: the testInstance 
	#instance 2: the xth value of the training data set 
	#length 3: the total number of columns-1: 38
	
	distance=np.sum(np.absolute(instance1-instance2))
	return distance
	#return math.sqrt(distance)

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSety, predictions):
	correct = 0
	for x in range(len(testSety)):
		if testSety[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testSety))) * 100.0
	
def kNN(Setx, Sety, trainingSetx, trainingSety, k):
    predictions=[]
    for x in range(len(Setx)):	#runs the number of times the amount of values in the test data set
        neighbors = getNeighbors(trainingSetx, trainingSety, Setx[x], k)	#trainingset->list of lists, testset[x] -> 1 list, basically the entire row 
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(Sety, predictions)
    return accuracy

def main():
    # prepare data
    direc=os.fsencode('.')
    trainingSetx,trainingSety=loadDataset('cat1.csv')
    print('Train file: cat1.csv')
    print ('Train set size: ' + repr(len(trainingSetx)),'\n\n')
    inittime=time.time()
    #filenames=['cat1_r1.csv','cat1_r2.csv','cat1_r3.csv']
    for File in os.listdir(direc):  #each file in directory
        filename=os.fsdecode(File)  #get filename
    #for filename in filenames:
        if filename.endswith('.csv'):
            testSetx,testSety=loadDataset(filename)
            #split = 0.67
            print('Test file: ',filename)
            print ('Test set size: ' + repr(len(testSetx)))
            cvSetx, reportSetx, cvSety, reportSety = train_test_split(testSetx,testSety, test_size = 0.5, random_state = 0)
            
            bestk=1
            bestacc=0
            for k in range(1,32,2):
                accuracy = kNN(cvSetx, cvSety, trainingSetx, trainingSety, k)
                if accuracy>bestacc:
                    bestk=k
                    bestacc=accuracy
                print("k=",k,"\taccuracy=",accuracy,'%')
            
            print("Best k=", bestk,"\n")
            accuracy = kNN(reportSetx,reportSety, trainingSetx, trainingSety, bestk)
            print('Reported accuracy: ' + repr(accuracy) + '%\n\n')
            print('Time elapsed:', time.time()-inittime)
	    
main()
