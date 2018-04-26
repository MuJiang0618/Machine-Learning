'''
数据标准化
迭代n次:
    设置最小误差lowestError为正无穷
    对weights的每个分量:
        增大或缩小x:
            得到新的weights
            计算新weights下的误差
            如果误差<当前最小误差lowestError:
                更新lowestError
                更新weights   
'''

from numpy import *
import matplotlib.pyplot as plt

def Forward_regression(xArr, yArr, step = 0.01, num_iter = 100):
    x_Mat = mat(xArr) ; y_Mat = mat(yArr).T
    m, n = x_Mat.shape
    y_mean = mean(y_Mat, 0)
    y_Mat = y_Mat - y_mean
    x_Mat = regularize(x_Mat)

    return_weights = zeros((num_iter, n)) 
    weights = zeros((n, 1)) ; best_weights = weights.copy() 
    for i in range(num_iter):
        Lowest_error = 1000000
        for feat in range(n):
            for diff in [-1, 1]:
                temp_weigths = weights.copy()  #一次遍历修改一次weights,每次遍历结束前对weights的尝试都是在原来的weights上
                temp_weigths[feat] += step * diff
                y_pred = x_Mat * temp_weigths   #y_pred为列向量
                bias = Calcu_Error(y_pred.A, y_Mat.A)
                if bias < Lowest_error:
                    best_weights = temp_weigths
                    Lowest_error = bias
        weights = best_weights.copy()
        return_weights[i,:] = weights.T
    return return_weights

def regularize(xMat):  # 按列进行规范化
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # 计算平均值然后减去它
    inVar = var(inMat, 0)  # 计算除以Xi的方差
    inMat = (inMat - inMeans) / inVar
    return inMat

def LoadDataSet(filename = r'C:\Users\Administrator\Desktop\regression.txt'):
    fr = open(filename)
    num_feat = len(fr.readline().strip().split('\t')) - 1
    data_list = []; label_list = []
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        temp_list = []
        for i in range(num_feat):
            temp_list.append(float(cur_line[i]) )
        data_list.append(temp_list)
        label_list.append(float(cur_line[-1]))
    fr.close()
    return data_list, label_list

def Calcu_Error(Mat1, Mat2):
    return ((Mat1 - Mat2) **2).sum()


xArr, yArr = LoadDataSet(r'C:\Users\Administrator\Desktop\abalone.txt')
result = Forward_regression(xArr, yArr)
print(result)