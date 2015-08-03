#coding:utf-8
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#用已经识别的文本构建词库
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

#标识在词库中出现的单词
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec
#分类器训练函数
def trianNB0(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix)
	numWords=len(trainMatrix[0])
	pAbusive=sum(trainCategory)
	p0Num=zeros(numWords);p1Num=zeros(numWords)
	p0Denom=0.0;p1Denom=0.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	return p0Vect,p1Vect,pAbusive
#改进的方法是：将p0num p1num改成1，总数为2，并取对数函数，将值变大；
#
#
#
