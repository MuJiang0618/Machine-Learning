import numpy as np
from math import log
import pdb

def Load_DataSet(filename = r'C:\Users\Administrator\Desktop\credit.txt'):
    DataSet = [] ; Label_list = []
    fr = open(filename)
    for line in fr.readlines():
        new_line = line.strip().split('\t')
        line_list = []
        for i in new_line:
            line_list.append(i)
        DataSet.append(line_list)
    Label_list = ['年龄', '有工作', '有房子', '信贷情况']

    return DataSet, Label_list

def Cal_Entropy(DataSet):
    #用字典保存各列别的出现次数
    label_dic = {}
    for line in DataSet:
        label = line[-1]
        if label not in label_dic:
            label_dic[label] = 0
        label_dic[label] += 1

    #计算总熵
    total_entropy = 0.0
    num_data = len(DataSet)
    for item in label_dic:
        prob = label_dic[item] / num_data 
        total_entropy -= prob * log(prob, 2)

    return total_entropy

def Split_DataSet(DataSet, feat_index, feat_value):
    Sub_DataSet = []
    for line in DataSet:
        if line[feat_index] == feat_value:
            temp_list = []
            temp_list.extend( line[: feat_index])
            temp_list.extend( line[feat_index +1:])
            Sub_DataSet.append(temp_list)

    return Sub_DataSet

def Choose_best_feat(DataSet):
    #遍历每个特征
    #   遍历每个特征的划分子集计算熵增益
    num_feat = len(DataSet[0]) - 1
    basic_entropy = Cal_Entropy(DataSet)   #父集的熵
    best_feat = -1 ; largest_info_gain = 0.0
    for i in range(num_feat):
        #获取每个特征的取值范围以求得该特征下划分子集的熵增益和
        feat_range = [example[i] for example in DataSet]
        feat_range = list(set(feat_range))
        sub_entropy = 0.0
        for value in feat_range:
            split_dataSet = Split_DataSet(DataSet, i, value)
            prob = len(split_dataSet) / float(len(DataSet))
            sub_entropy += prob * Cal_Entropy(split_dataSet)
        InfoGain = basic_entropy - sub_entropy
        if InfoGain > largest_info_gain:
            largest_info_gain = InfoGain
            best_feat = i

    return best_feat

def BuildDecisionTree(DataSet, Label_list):
    is_same_list = [example[-1] for example in DataSet]
    if is_same_list.count(is_same_list[0]) == len(DataSet):  #所有数据的标签一致
        return DataSet[0][-1]

    best_feat_index = Choose_best_feat(DataSet)
    best_feat = Label_list[best_feat_index]
    tree_dic = {best_feat : {}}
    feat_range = [example[best_feat_index] for example in DataSet]
    feat_range = list(set(feat_range))

    del(Label_list[best_feat_index])
    for value in feat_range:
        sub_label_list = Label_list[:]
        split_dataset = Split_DataSet(DataSet, best_feat_index, value)
        tree_dic[best_feat][value] = BuildDecisionTree(split_dataset, sub_label_list)

    return tree_dic