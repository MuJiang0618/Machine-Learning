import numpy as np
import collections
import pdb

def Load_TraingData():
    fr = open(r'C:\Users\Administrator\Desktop\data.txt','r')
    lines = fr.readlines()
    rows = len(lines)    #获取训练集数据个数
    #获取每个数据的列数
    columns = len(lines[0].split())
    TrainingSet = np.zeros((rows, columns))
    labels_list = []
    for i in range(rows):
        temp = lines[i].split()
        for q in range(columns) :
            TrainingSet[i][q] = temp[q]
        labels_list.append(temp[-1])
    return TrainingSet, labels_list

def classifier(inX, Training_set, Labels, k):
    #先求出训练集数据的个数
    #print(Training_set)
    num_Train = Training_set.shape[0]
    dist_list = []
    for i in range(num_Train):
        dist = abs(sum(inX - Training_set[i][:-1]))   #把距离化为绝对值
        dist_list.append(int(dist))
    #找出dist_list中距离预测点最近的k个数据,并提取这些数据的类别
    Labels_list = [] ; j = 0
    while j < k:
        closest = dist_list.index(min(dist_list))
        Labels_list.append(Training_set[closest][-1] )
        dist_list[closest] = 10e8   #标签被选出来后，化为无穷大值防止再次被识别为距离最近的点
        j += 1
    #统计Labels_list中出现次数最多的label
    most_label = Get_most_label(Labels_list) 
    print('the prediction is %d' % most_label)
    return most_label

def Get_most_label(list_):   #求出现次数最多的标签
    dic_ = collections.Counter(list_)
    a = list(dic_.values())
    a.sort(reverse = True)
    for m in dic_.items():
        if m[1] == a[0]:
            return m[0]

def test():
    TrainingSet, labels_list = Load_TraingData()
    inX = np.array([14488, 7.153469, 1.673904])
    classifier(inX, TrainingSet, labels_list, 5)
