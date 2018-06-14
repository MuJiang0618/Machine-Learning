#2018/05/14
from numpy import *
import urllib.request

def loadOnlineDataSet(url):
    raw_data = urllib.request.urlopen(url)
    dataSet = loadtxt(raw_data, delimiter=" ")
    return dataSet

def loadLocalData(route):
    fr = open(route)
    vector = []
    for line in fr.readlines():
        temp_list = line.strip().split()
        vector.append(list(map(float, temp_list)))

    return array(vector)

def makeDataSet():
    X, Y = make_regression(500, 7, 7)
    X_normalized = preprocessing.normalize(X, norm='l2') #'l1'(绝对值之和)、'l2'范式(几何距离)
    return c_[X_normalized, Y]

def binSplit(dataSet, feat, val):
    subSetL = dataSet[nonzero(dataSet[:, feat] <= val)[0], :]
    subSetR = dataSet[nonzero(dataSet[:, feat] > val)[0], :]

    return subSetL, subSetR

def leaf_cls(Leaf_dataSet):
    y = Leaf_dataSet[:, -1]
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

def leaf_reg(Leaf_dataSet):
    return mean(Leaf_dataSet[:, -1])

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

def chooseBestSplit(dataSet, prune_tol=(0.0, 0), leafNode=leaf_cls, err_cal=err_cls):
    if sum(dataSet[:, -1]) == 0:    #如果该子数据集所有样本同分类
        return None, 0
    elif sum(dataSet[:, -1]) == len(dataSet[:, -1]):
        return None, 1

    lowest_error = err_cal(dataSet[:, -1]) ;best_splitFeat = -1 ; best_splitVal = 0.0
    for feat in range(len(dataSet[0])-1):
        for val in set(dataSet[:, feat]):
            subSetL, subSetR = binSplit(dataSet, feat, val)
            err = err_cal(subSetL[:, -1]) + err_cal(subSetR[:, -1])
            if err < lowest_error:
                lowest_error = err
                best_splitVal = val
                best_splitFeat = feat

    org_err = err_cal(dataSet[:, -1])
    tol_err = prune_tol[0] ; tol_num = prune_tol[1]
    if (org_err - lowest_error) < tol_err:
        return None, leafNode(dataSet)

    subSetL, subSetR, a, b = binSplit(dataSet, best_splitFeat, best_splitVal)
    if len(subSetL) + len(subSetR) < tol_num:
        return None, leafNode(dataSet)

    return best_splitFeat, best_splitVal

def g_predict(tree, test_data):
    if not type(tree) == dict:
        return tree

    else:
        feat_index = tree['feat'] ; split_val = tree['val']
        if test_data[feat_index] < split_val:
            return g_predict(tree['left'], test_data)
        else:
            return g_predict(tree['right'], test_data)

def get_residuals(trees, samples):
    scores = []
    for i in samples[:, :-1]:
        score = 0.0
        for tree in trees:
            score += g_predict(tree, i)
        scores.append(score)
    residuals = samples[:, -1] - scores

    return residuals

def creat_Tree(dataSet, residuals, leafNode, err_cal, max_depth = 6):    #拟合残差
    dataSet[:, -1] = residuals      #因为要拟合的是残差，所以每个样本的值改为残差

    if max_depth == 0:
        return leafNode(dataSet)

    best_splitFeat, best_splitVal = chooseBestSplit(dataSet, (20,10) )
    if best_splitFeat == None:
        return best_splitVal

    tree_dict = {}
    subSetL, subSetR = binSplit(dataSet, best_splitFeat, best_splitVal)
    tree_dict['feat'] = best_splitFeat; tree_dict['val'] = best_splitVal
    tree_dict['left'] = creat_Tree(subSetL, leafNode, err_cls, max_depth-1) 
    tree_dict['right'] = creat_Tree(subSetR, leafNode, err_cls, max_depth-1)

    return tree_dict

def creat_GBDT(dataSet, n_tree):
    '''
    先只考虑根节点以残差划分
    后面的节点就按CART的均方差划分
    '''
    tree_dict =[]
    first_tree = creat_Tree(dataSet, dataSet[:, -1], leaf_reg, err_reg)
    
    for i in range(n_tree - 1):
        residuals = get_residuals(tree_dict, dataSet)
        new_tree = creat_Tree(dataSet, residuals, leaf_reg, err_reg)
        tree_dict.append(new_tree)

    return tree_dict

def test():
    dataSet = loadLocalData(r'C:\Users\Administrator\Desktop\ML dataSet\regression.txt')[:, 1:]
    trees = creat_GBDT(dataSet[:180], 6)
    error = get_residuals(trees, dataSet[20:])
    sqare_error = (error **2).sum()
    print('The sqare error is %d' %sqare_error)

test()