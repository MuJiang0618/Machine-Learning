#2018/04/23
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():  #载入数据
    dataMat = [] ; labelMat = []
    fr = open(r'C:\Users\Administrator\Desktop\logistic testSet.txt', 'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append( [1.0, float(lineArr[0]), float(lineArr[1] )]) #在首位添加1.0是为了方便矩阵计算
        labelMat.append( int(lineArr[2] ))
    return dataMat, labelMat

def sigmoid(inX):         #得到标签概率
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):  #梯度下降法
    dataMatrix = mat(dataMatIn)  #输入的数据集本来是ndlist，这里将其矩阵化
    labelMat = mat(classLabels).transpose()   #转置为列矩阵
    m, n = shape(dataMatrix)
    alpha = 0.001   #步长
    maxCycles = 500  #最大循环次shu
    weights = ones( (n, 1) )  # 初始化所有参数为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   #h是一个列向量
        error = (labelMat - h)
        weights += alpha * dataMatrix.transpose() * error
    #因为每一次迭代都要计算包含所有数据的向量的运算，遍历了所有数据，复杂度较高
    return weights  #返回3个参数最佳拟合值

def random_gradAscent(dataMatrix, classLabels):  #随机梯度法
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = ones((n))
    for i in range(m):
        h = sigmoid( sum(dataMatrix[i] * weights) )   #这里h是一个数值而非向量
        error = classLabels[i] - h
        weights += alpha * error * dataMatrix[i]
    
    return weights


def plotBestFit(weights):
    weights = array(weights)
    dataSet, labelSet = loadDataSet()
    dataArr = array(dataSet)
    n = dataArr.shape[0]
    xcord0 = [] ; ycord0 = [] ; xcord1 = [] ; ycord1 = []

    for i in range(n):
        if labelSet[i] == 1:
            xcord1.append(dataArr[i, 1]) ; ycord1.append(dataArr[i, 2])
        else:
            xcord0.append(dataArr[i, 1]) ; ycord0.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, s =30, c='red', marker='s')
    ax.scatter(xcord1, ycord1, s =30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] *x) / weights[2]  #  w0x0 + w1x1 + w2x2 = 0,分数=0是两类样本的分界线
    ax.plot(x, y)
    plt.legend()
    plt.xlabel('X1') ; plt.ylabel('X2')
    plt.show()

def plt_fun():
    data_x, data_y = loadDataSet()
    weights = gradAscent(data_x, data_y)   #这里用的是全批量梯度下降
    plotBestFit(weights)