#2018/05/13
from numpy import *
import urllib.request
from sklearn import tree

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
    return mean(Leaf_dataSet)

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
    tol_err = prune_tol[0] ; tol_num = prune_tol[1]
    if (org_err - lowest_error) < tol_err:
        return None, leafNode(dataSet)

    subSetL, subSetR = binSplit(dataSet, best_splitFeat, best_splitVal)
    if len(subSetL) + len(subSetR) < tol_num:
        return None, leafNode(dataSet)

    return best_splitFeat, best_splitVal

def g_predict(tree, test_data):
    if not type(tree) == dict:
        return tree

    else:
        feat_index = tree['feat'] ; split_val = tree['Val']
        if test_data[feat_index] < split_val:
            return g_predict(tree['left'], test_data)
        else:
            return g_predict(tree['right'], test_data)

def creat_Tree(dataSet, leafNode, err_cal, max_depth=5):
    if max_depth == 0:
        return leafNode(dataSet)
    best_splitFeat, best_splitVal = chooseBestSplit(dataSet, (-1,0))
    if best_splitFeat == None:
        return best_splitVal

    tree_dict = {}
    subSetL, subSetR = binSplit(dataSet, best_splitFeat, best_splitVal)
    tree_dict['feat'] = best_splitFeat; tree_dict['Val'] = best_splitVal
    tree_dict['left'] = creat_Tree(subSetL, leafNode, err_cls, max_depth-1) ; tree_dict['right'] = creat_Tree(subSetR, leafNode, err_cls, max_depth-1)
    return tree_dict

def get_Alpha(dataSet, tree_list):
    m = dataSet.shape[0] ; sn = zeros(m)
    for i in range(m):
        for tree in tree_list:
            sn[i] += g_predict(tree, dataSet[i]) * tree['Alpha']
        sn[i] /= len(tree_list)    #取平均

    yn = dataSet[:, -1]
    residuals = yn - sn

    weights = zeros(m)

def get_residuals(tree_list, dataSet):
    m = dataSet.shape[0]
    sn = zeros(m)
    for i in range(m):
        for tree in tree_list:
            sn[i] += g_predict(tree, dataSet[i])
        sn[i] /= m   #取平均值
    return dataSet[:, -1] - sn

def creat_GBDT(dataSet, residuals, leafNode, depth=6):
    '''
    先只考虑根节点以残差划分
    后面的节点就按CART的均方差划分
    '''
    if depth == 0:          #到达最大深度，直接返回叶节点
        return leafNode(dataSet)

    n_feat = len(dataSet[0]) - 1
    best_feat = -1 ; best_val = -1 ; Min_err = 1e10
    for feat in range(n_feat):
        uni_featVal = set(dataSet[:, feat])    #获取所有特征值,set()会自动排序
        mean_val = []
        for i in range(len(uni_featVal)-1):
            mean_val.append((uni_featVal[i] + uni_featVal[i+1]) / 2)   #获取所有特征值的二分点
        for point in mean_val:
            subSetL, subSetR = binSplit(dataSet, feat, point)
            R_err = sum((mean(subSetR[:, -1]) - residuals) **2)
            L_err = sum((mean(subSetL[:, -1]) - residuals) **2)
            if (L_err + R_err) < Min_err:
                best_feat = feat; best_val = point
                Min_err = L_err + R_err

    subSetL, subSetR = binSplit(dataSet, best_feat, best_val)
    tree = {}
    tree['feat'] = best_feat ; tree['val'] = best_val
    tree['left'] = creat_Tree(subSetL, leaf_reg, err_reg, depth-1) 
    tree['right'] = creat_Tree(subSetR, leaf_reg, err_reg, depth-1) 

def trainWeakClf(dataSet, n_clf=10, tree_depth=5):
    org_tree = creat_Tree(dataSet, leaf_reg, err_reg, 5)
    tree_list = [] ; tree_list.append(org_tree) ; iter_residuals = []
    for i in range(n_clf):
        residuals = get_residuals(tree_list, dataSet)
        iter_residuals.append(residuals)
        new_tree = creat_GBDT(dataSet, residuals, leaf_reg)
        new_tree['Alpha'] = 1 / n_clf   #这里我暂时先设置为平均投票制，本来是要用get_Alpha()，但是我不会
        tree_list.append(new_tree)
    return tree_list, iter_residuals

def test():
    dataSet = loadLocalData(r'C:\Users\Administrator\Desktop\dataSet\horseTraining.txt')
    clf_list, iter_residuals = trainWeakClf(dataSet, 10)
    print(clf_list,'——————————————————————————————', iter_residuals)