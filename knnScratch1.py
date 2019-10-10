import csv
import random
import math
import operator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def loadDataset(filename, split, trainingSet=[], testSet=[]):
	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)

		dataset = pd.DataFrame(dataset)
		dataset=dataset.drop(dataset.columns[14], axis=1)
		
		sc_X = MinMaxScaler(feature_range=(0, 1))
		dataset = sc_X.fit_transform(dataset)
		# print("\n\n\n",dataset[1],"\n\n\n")
		for x in range(len(dataset) - 1):
			# print(dataset[x],"\n")
			for y in range(37): #to use all the data values upto the 37th column in the Dataset csv
				dataset[x][y] = float(dataset[x][y])
			if random.random() < split:
				trainingSet.append(dataset[x])	#the set containing the data points that we'll use to train the algorithm
			else:
				testSet.append(dataset[x])	#the set containing the data points that we'll use to test the algorithm

def getNeighbors(trainingSet, testInstance, k):
	#getting the nearest neighbours
	#testInstance contains the xth (the entire row of values {list of len 38}) value of the testDataSet, so we'll be getting the nearest neighbours of that data point(list), in the test data set, wrt to the trainDataSet
	distances = []
	length = len(testInstance) - 1	# length = 37, beacuse there are a total of 38 columns
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def euclideanDistance(instance1, instance2, length):
	#instance 1: the testInstance 
	#instance 2: the xth value of the training data set 
	#length 3: the total number of columns-1: 38
	distance = 0
	for x in range(length):
		# print(type(instance1[x]))
		distance += pow(instance1[x] - instance2[x], 2)
		# print(distance)
	return math.sqrt(distance)

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('cat2_correct_r1_test.csv', split, trainingSet, testSet)
	print ('Train set: ' + repr(len(trainingSet)))
	print ('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 8
	for x in range(len(testSet)):	#runs the number of times the amount of values in the test data set
		neighbors = getNeighbors(trainingSet, testSet[x], k)	#trainingset->list of lists, testset[x] -> 1 list, basically the entire row 
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()