# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:44:21 2017

@author: BE231
"""
import scipy
import matplotlib
import numpy
import random
import string

'''
How it works?
-The first funtion is determineChar() consists of 2 main functions
--1 getWvectors
--- loops over each char in the alphabetic order
--- creates a linear perceptron between each char's images (on one side) and the rest of letters
--- returns a dict of each char as key and it's W vector as value

--2 getProbableLetter
--- gets the test image X values
--- apply all W vectors and returns 1 char with the highest value

'''
#read image pixels into 1 Dimension numpy array
def getImageXValues(imageName):
    arr = numpy.append(scipy.misc.imread(imageName+".jpg").reshape(144,),1)
    return arr

#initialize W vector W1 = 1 all the rest = 0
def initializeW():
    arr = numpy.zeros(145)
    arr[0] = 1
    return arr

#check all training data for misclassified 
def getMisclassified(trainingData, W):
    misclassified = []
    for X in trainingData:
        if (numpy.dot(W.transpose(),X[:-1])*X[-1]) <= 0:#tn is added at the end of X
            misclassified.append(X)
    return misclassified

#gets W0 and all training data X with last value = 1 (it's our char), =-1 for all others
#keeps updating W vectors until char is lineraly seperated
def updatingEquation(trainingData, Wold):
    W = Wold
    misclassified = getMisclassified(trainingData,Wold)
    while (len(misclassified)>0):
        missX = random.choice(misclassified)
        W = numpy.add(W,(.05*missX[:-1]*missX[-1]))
        misclassified = getMisclassified(trainingData,W)
    return W

#for specific char get that char t = 1 and all others = -1 at the end of X vector
#create appropriate training set to each char where X[-1] = 1 and rest chars X[-1] = -1
def readImagesValues(charCor):
    trainingData = []
    for char in string.ascii_lowercase:
        for num in range(1,8):
            X = getImageXValues("A1"+char+str(num))
            if(char == charCor):
                X = numpy.append(X,1)
            else:
                X = numpy.append(X,-1)
            trainingData.append(X)
    return trainingData

#for each char read images and train perceptron to get final W
def perceptron(char):
    trainingData = readImagesValues(char)
    Wfinal = updatingEquation(trainingData,initializeW())
    return Wfinal

#calls perceptorn() on each char to get Wfinal and get all in 1 dict
def getWvectors():
    Wdict = {}
    for char in string.ascii_lowercase:
        Wdict[char] = perceptron(char)     
    return Wdict

#for every test image compute values with against each char W to get the most preferable one
def getTestScores(Wdict):
    scores = {}
    for char in string.ascii_lowercase:
        scores[char] = 0
    for char in string.ascii_lowercase:
        for num in range(8,10):
            X = getImageXValues("A1"+char+str(num))#read test image
            #get most probable letter and check with the correct one
            bestScore = -10000000000
            predictedChar = ''
            for W in Wdict:
                if(numpy.dot(Wdict[W].transpose(),X)>bestScore):
                    bestScore = numpy.dot(Wdict[W].transpose(),X)
                    predictedChar = W
            if (predictedChar == char):
                scores[char] += 1
    return scores

#return scores for each char
def getTestResults():
    Wdict = getWvectors()
    scores = getTestScores(Wdict)
    return scores


scores = getTestResults()
matplotlib.pyplot.bar(range(len(scores)), scores.values(), align='center')
matplotlib.pyplot.xticks(range(len(scores)), scores.keys())
#matplotlib.pyplot.show()
matplotlib.pyplot.savefig('Accuracy'+'.jpg')
