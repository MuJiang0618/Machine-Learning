#2018/05/06
#690个样本，14个特征,类别标签0、1

from numpy import *
import urllib.request
from sklearn import tree

def loadDataSet():
    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    raw_data = urllib.request.urlopen(url)
    dataSet = loadtxt(raw_data, delimiter=" ")

    return dataSet

def binSplit(dataSet, feat, val):
    subSetL = dataSet[nonzero(dataSet[:, feat] <= val)[0], :]
    subSetR = dataSet[nonzero(dataSet[:, feat] > val)[0], :]
    return subSetL, subSetR

def leaf_cls(dataSet):
    y = dataSet[:, -1]
    labels = {}
    for i in range(len(y)):
        if y[i] not in labels.keys():
            labels[y[i]] = 0
        labels[y[i]] += 1

    max_label = 1 ; major_label = 0
    for label in labels.keys():
        if labels[label] > max_label:
            max_label = labels[label]
            major_label = label

    return major_label

def leaf_reg(dataSet):
    return mean(dataSet)

def err_reg(dataSet):
    n = len(dataSet)
    return n * var(dataSet)

def err_cls(sub_y):
    labels = {}
    for i in range(len(sub_y)):
        if sub_y[i] not in labels.keys():
            labels[sub_y[i]] = 0
        labels[sub_y[i]] += 1
    gini = 1
    for j in labels.keys():
        prob = labels[j] / len(sub_y)
        gini -= prob**2

    return gini

def chooseBestSplit(dataSet, ops=(0.02, 5), leafNode=leaf_cls, err_cal=err_cls):
    if sum(dataSet[:, -1]) == 0:    #如果该子数据集所有样本同分类
        return None, 0
    elif sum(dataSet[:, -1]) == len(dataSet[:, -1]):
        return None, 1

    lowest_error = err_cal(dataSet[:, -1]) ;best_splitFeat = -1 
    for feat in range(len(dataSet[0])-1):
        for val in set(dataSet[:, feat]):
            subSetL, subSetR = binSplit(dataSet, feat, val)
            err = err_cal(subSetL[:, -1]) + err_cal(subSetR[:, -1])
            if err < lowest_error:
                lowest_error = err
                best_splitVal = val
                best_splitFeat = feat

    org_err = err_cal(dataSet[:, -1])
    tol_err = ops[0] ; tol_num = ops[1]
    if (org_err - lowest_error) < tol_err:
        return None, leafNode(dataSet)

    subSetL, subSetR = binSplit(dataSet, best_splitFeat, best_splitVal)
    if len(subSetL) + len(subSetR) < tol_num:
        return None, leafNode(dataSet)

    return best_splitFeat, best_splitVal

def creat_Tree(dataSet, max_depth=5, leafNode=leaf_cls):
    best_splitFeat, best_splitVal = chooseBestSplit(dataSet)
    if best_splitFeat == None:
        return best_splitVal

    if max_depth == 0:
        return leafNode(dataSet[:, -1])

    tree_dict = {}
    subSetL, subSetR = binSplit(dataSet, best_splitFeat, best_splitVal)
    tree_dict['feat'] = best_splitFeat; tree_dict['Val'] = best_splitVal
    tree_dict['left'] = creat_Tree(subSetL, max_depth-1, leafNode) ; tree_dict['right'] = creat_Tree(subSetR, max_depth-1, leafNode)
    return tree_dict

def main():
    dataSet = loadDataSet()
    tree = creat_Tree(dataSet)
    print(tree)

if __name__ == '__main__':
    main() 