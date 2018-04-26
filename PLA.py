#2018/04/26
from sklearn.datasets.samples_generator import *
import matplotlib.pyplot as plt
from numpy import *

def createTrainDataSet():    #样本含2个特征,第一个1对应w0
    trainData=[     [1, 1, 4],  
                    [1, 2, 3],                         
                    [1, -2, 2],   
                    [1, 0, 1],   
                    [1, 1, 2],
                    [1, -1., 3.2], 
                    [1, -1.2, 2.5] ]
    label= [1, 1, -1, -1, -1, 1, -1]  
    return trainData, label

def presee_distri():
    x, y = createTrainDataSet()
    xcord1 = [] ; ycord1 = [] ; xcord2 = [] ; ycord2 = []
    m = len(x)
    for i in range(m):
        if y[i] == 1:
            xcord1.append(x[i][1])
            ycord1.append(x[i][2])

        else:
            xcord2.append(x[i][1])
            ycord2.append(x[i][2])
    plt.scatter(xcord1, ycord1, color='r')
    plt.scatter(xcord2, ycord2, color='g')
    plt.legend()
    plt.show()   #预览数据分布特点

def sigmoid(score):
    if score > 0:
        return 1
    if score < 0:
        return -1

    return 0

def createTestDataSet():#数据样本  
    testData = [   [1, 1, 1],  
                   [1, 2, 0],   
                   [1, 2, 4],   
                   [1, 1, 3]]
    return testData

def PLA():
    data, label = createTrainDataSet()
    data = array(data)
    m, n = data.shape
    #weights = ones(n)  #weights为行向量nd
    weights = array([4.0, 4.0, 4.0])
    isComplete = True
    while isComplete:
        isComplete = False
        for i in range(m): #遍历每个样本
            score = sum(weights * data[i, :])
            if sigmoid(score) != label[i]:
                isComplete = True
                weights +=  label[i] * data[i]

    return weights

def plotBestFit():
    weights = PLA()
    x, y = createTrainDataSet()
    xcord1 = [] ; ycord1 = [] ; xcord2 = [] ; ycord2 = []
    m = len(x)
    for i in range(m):
        if y[i] == 1:
            xcord1.append(x[i][1])
            ycord1.append(x[i][2])

        else:
            xcord2.append(x[i][1])
            ycord2.append(x[i][2])
    plt.scatter(xcord1, ycord1, color='r')
    plt.scatter(xcord2, ycord2, color='g')

    x_pred = arange(-2, 2, 0.2)
    y_pred = (-weights[0] - weights[1] * x_pred) / weights[2]
    plt.plot(x_pred, y_pred)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

    return 0

plotBestFit()