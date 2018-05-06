#最小二乘法线性回归
from numpy import *
import matplotlib.pyplot as plt

def LoadDataSet(filename = r'C:\Users\Administrator\Desktop\regression.txt'):
    fr = open(filename)
    num_feat = len(fr.readline().strip().split('\t')) - 1
    data_list = [[1.0, 0.067732]]; label_list = [3.176513]
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        temp_list = []
        for i in range(num_feat):
            temp_list.append(float(cur_line[i]) )

        data_list.append(temp_list)
        label_list.append(float(cur_line[-1]))
    fr.close()

    return data_list, label_list

def StdRegression(dataSet, labelSet):
    xMat = mat(dataSet) ; yMat = mat(labelSet).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0 :    #考虑是否可逆
        print('This matrix is not reversible')
        return 
    print(xTx)
    weights = xTx.I * (xMat.T * yMat)
    return weights

def Plt_fun(dataSet, labelSet):
    xList = [0.067732]
    for item in dataSet:
        xList.append(item[1])

    ws = StdRegression(dataSet, labelSet)
    xList, yList = Plt_StrightLine(dataSet, ws)
    plt.plot(xList, yList, label = 'PredictLine')
    plt.scatter(xArr, labelSet, label = 'skitscat', color='k', s=8, marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression')
    plt.legend()
    plt.show()

    return

def Plt_StrightLine(dataSet,weights):
    xMat = mat(dataSet)
    yMat = xMat * weights
    yMat = yMat.T
    yList = list(yMat)
    xList = dataSet[:,1]

    return xList, yList

dataSet, label_list = LoadDataSet()
weights = StdRegression(dataSet, label_list)
Plt_fun(dataSet, weights)
