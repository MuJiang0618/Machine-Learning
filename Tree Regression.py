from numpy import *
import matplotlib.pyplot as plt

def LoadDataSet(filename = r'C:\Users\Administrator\Desktop\RegTree.txt'):
    fr = open(filename)
    data_list = []
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        fltline = list(map(float, cur_line))
        data_list.append(fltline)
    return data_list

def CreateTree(dataMat):
    split_feat, split_val = ChooseSplit(dataMat)
    if split_feat == None:
        return split_val
    
    #每个特征在之后的划分中仍然可能起作用，所以每次划分的数据集就是总数据集，不用在每次划分后删改
    Tree = {}
    Tree['split_feat'] = split_feat
    Tree['split_val'] = split_val
    left_dataMat, right_dataMat = SplitDataSet(dataMat, split_feat, split_val)
    Tree['left'] = CreateTree(left_dataMat)
    Tree['right'] = CreateTree(right_dataMat)
    return Tree

def SplitDataSet(dataMat, split_feat, split_val):
    left_dataMat = dataMat[nonzero( dataMat[:, split_feat] < split_val)[0], :]
    right_dataMat = dataMat[nonzero( dataMat[:, split_feat] >= split_val)[0], :]
    return left_dataMat, right_dataMat

def ChooseSplit(dataMat, ops = (1,4)):
    m, n = dataMat.shape
    num_Iter = 30
    best_feat = 0; best_value = 0
    totle_error = CalError(dataMat)
    LowestError = CalError(dataMat)
    for feat in range(n-1):             #寻找所有特征的所有取值中最好的切分特征与特征值
        for splitval in set(dataMat[:, feat]):
            left, right= SplitDataSet(dataMat, feat, splitval)
            Error = CalError(left) + CalError(right)
            if Error < LowestError:
                LowestError = Error
                best_feat = feat
                best_value = splitval

    if (totle_error - LowestError) < ops[0]:   #预剪枝:切分后减少的误差较小则合并为叶节点
        leaf_val = LeafNode(dataMat)
        return None, leaf_val

    return best_feat, best_value

def LeafNode(dataMat):
    return mean(dataMat[:, -1])   #取叶节点中所有数据点标签的均值作为叶节点标签

def CalError(dataSet):         #计算左右子数据集的方差
    return var(dataSet[:, -1]) * dataSet.shape[0]   #均方差×数据点个数

DataSet = array(LoadDataSet())
dict_ = CreateTree(DataSet)
print(dict_)