#2018/04/03
# -*- coding:utf-8 -*-
"""
对每个特征:
    对每个特征的诸多特征值(多步):
        对inequal:
            用以feat、feat_value、inequal为参数的树模型进行分类
            计算分类后的加权误差
选择加权误差最小的树桩作为该弱分类器的树模型

以同一训练数据集构建多个弱分类器,赋予每个弱分类器权重，并不断更新数据权重D,D会影响后面的弱分类器树模型的参数选择
"""    
from numpy import *

def loadSimpData():  #载入训练数据集
    dataArr = array([ [1. , 2.1],
                   [2. , 1.1],
                   [1.3, 1.],
                   [1. , 1.],
                   [2. , 1.] ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]   #为什么以1和-1作为标签而不是1和0？
    return dataArr, classLabels

def predict(dataArray, feat, thresh_Val, inequal):   #给定参数的树桩对训练数据的预测
    m = dataArray.shape[0]
    predicted_result = mat(ones((m, 1)))
    if inequal == 'lt':
        predicted_result[nonzero(dataArray[:, feat] <= thresh_Val)[0]] = -1

    else:
        predicted_result[nonzero(dataArray[:, feat] > thresh_Val)[0]] = -1        

    return predicted_result

def buildStump(dataArr, classLabels, D):  #用不同参数的树桩来预测训练样本,选择加权错误率最小的具有特定参数的树桩作为弱分类器树模型
    Min_error = inf
    m, n =dataArr.shape
    best_Stump = {}
    for feat in range(n):
        feat_Min = min(dataArr[:, feat]) ; feat_Max = max(dataArr[:, feat])
        feat_range = feat_Max - feat_Min
        num_step = 10.0   #步长如何设置才能达到最好的分类效果？
        size_step = feat_range / num_step
        for step in range(-1, int(num_step)+1):  #为什么从-1开始？
            for inequal in ['lt', 'gt']:   #名称的由来？
                thresh_Val = feat_Min + step * size_step
                predicted_Val = predict(dataArr, feat, thresh_Val, inequal)  #给予树桩不同的参数
                errorArr = ones((m, 1))
                errorArr[nonzero(predicted_Val == mat(classLabels).T)[0]] = 0  #改成-1怎么样？这样就也考虑了正确分类的奖励
                weightError = D.T * mat(errorArr)  #上次被分错的点的权重更大,如果这次又分错,惩罚就越大,所以这是在选择能够最好地将上次错分的点分对的弱分类器
                if weightError < Min_error:                
                    Min_error = weightError
                    best_feat = feat
                    best_threshVal = thresh_Val
                    best_inequal = inequal
                    best_prediction = predicted_Val.copy()

    best_Stump['feat'] = best_feat ; best_Stump['thresh_Val'] = best_threshVal; best_Stump['inequal'] = best_inequal 

    return best_Stump, best_prediction, Min_error   #返回best_prediction是因为更新D要用到上个弱分类器的预测与真实值的比较

def adaBoostTrain(dataArr, classLabels, num_classifier):  #训练多个弱分类器
    classifier_list = []
    m = dataArr.shape[0]
    final_prediction = zeros((m, 1))
    D = mat(ones((m, 1)) / m )  #一开始所有数据点的权重都为均值——1/m
    aggClassEst = mat(zeros((m, 1)))
    for i in range(num_classifier):
        stump, prediction, Min_error = buildStump(dataArr, classLabels, D)
        #print('D:', D.T)
        Alpha = float(0.5 * log((1 - Min_error) / max(Min_error, 1e-16)) )     #max(Min_error, 1e-16)))确保没有错误时不会发生除0溢出
        stump['Alpha'] = Alpha
        classifier_list.append(stump)
        #print('classEst:', prediction.T)
        #接下来更新D
        expon = multiply(-1 * Alpha * prediction, mat(classLabels).T)   #这就是为什么标签值为1或-1的原因
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += Alpha * prediction
        #print('aggClassEst:', aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, mat(ones((m, 1)) ) )
        errorRate = aggErrors.sum() / m
        #print('totle errorRate:', errorRate,'\n')
        if errorRate == 0:   #若训练了n个弱分类器后可以完全正确分类训练样本,则停止训练
            break

    return classifier_list

def adaBoostClassifier(dataIn, classifier_list):
    dataMat = mat(dataIn)
    m = dataMat.shape[0]
    result = mat(zeros((m, 1)))
    aggClassEst = mat(zeros((m, 1)))
    for classifier in classifier_list:
        classEst = predict(dataMat, classifier['feat'], classifier['thresh_Val'], classifier['inequal']) #一个弱分类器对各测试数据点作出的分类预测
        aggClassEst += classEst  * classifier['Alpha']  #叠加每个弱分类器的预测结果X它的可信度
        print('aggClassEst:', aggClassEst)
    print('\n')

    return sign(aggClassEst)

def test():
    dataArr, classLabels = loadSimpData()
    classifier_list = adaBoostTrain(dataArr, classLabels, 30)
    result = adaBoostClassifier([[5, 5]], classifier_list)
    print(result)