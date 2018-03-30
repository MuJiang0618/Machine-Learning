# -*- coding:'utf-8' -*-
from numpy import *
def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)    #返回不重复出现的所有词汇的列表

def setOfWords2Vec(vocabList, inputSet):  #inPutSet为某个贴子所包含的词汇,是一个list
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
          returnVec[vocabList.index(word) ] = 1  #如果inputSet中的单词出现在vocabular中，就把输出文档向量中该单词的索引置为1，表示出现了
        else:
          print('the word: %s is not in my vocabulary!' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):   #trainCategory依次对应tainMatrix中每个文件是否为侮辱性文件
    numTrainDocs = len(trainMatrix)
    numWords = len( trainMatrix[0] )
    pAbusive = sum(trainCategory) / float(numTrainDocs)  #侮辱性文件出现的概率
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0 , p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  #遍历判断某文件是否为侮辱性文件
            p1Num += trainMatrix[i]    #求所有侮辱性文档向量的和
            p1Denom += sum( trainMatrix[i] )    #求所有侮辱性文档中侮辱性词汇的出现次数和
        else:
            p0Num += trainMatrix[i]   #与上两句相反
            p0Denom += sum(trainMatrix[i] )   
 
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive       
