#2018/03/30
# -*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName=r'C:\Users\Administrator\Desktop\testSet2.txt'): 
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = list(map(float, curLine) ) #python3中map返回迭代器,所以这里要list()
        dataMat.append(fltLine)
    fr.close()
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataSet, k):
    n = dataSet.shape[1] # 列的数量
    centroids = mat(zeros((k,n)) ) # 创建k个质心矩阵
    for j in range(n): # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:,j])    # 最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)    # 范围 = 最大值 - 最小值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))    # 为k个质心赋予在数据集范围内的随机值
    return centroids

def kMeans(dataSet, k, dist_count=distEclud, create_Cent=randCent):
    m, n = dataSet.shape
    cent_Mat = create_Cent(dataSet, k)   #创建质心簇矩阵
    clusterAssment = zeros((m, 2))  #每个数据点与质心的关系,第一列表示所属质心，第二列表示该数据点与所属质心的平方距离
    change_flag = True
    while change_flag:
        change_flag = False
        for data_point in range(m):
            min_dist = inf ; closest_cent = -1
            for cent in range(k):
                dist = distEclud(dataSet[data_point, :], cent_Mat[cent, :])
                if dist < min_dist:
                    min_dist = dist ; closest_cent = cent
            if clusterAssment[data_point, 0] != closest_cent:   #直到所有数据点所属簇不变为止
                change_flag = True
            clusterAssment[data_point, :] = closest_cent, dist**2

    #每次遍历后更新质心
        for cent in range(k):
            owned_point = dataSet[nonzero(clusterAssment[:, 0] == cent)[0] ]
            cent_Mat[cent, :] = mean(owned_point, axis= 0)

    return cent_Mat, clusterAssment

def Bi_Kmeans(dataSet, k, dist_cal=distEclud):
    m, n = dataSet.shape
    clusterAssment = zeros((m, 2))  #一开始所有数据点默认簇都是0
    centroid0 = mean(dataSet, axis= 0).tolist()[0]
    cent_list = [centroid0]   #为什么要用列表存储簇而不是矩阵，ndarray? 因为列表方便后续的簇划分后添加新簇的质心
    for j in range(m):
        clusterAssment[j, 1] = dist_cal(dataSet[j, :], mat(centroid0)) **2   #平方距离

    while len(cent_list) < k:
        lowest_sse = 10e7
        for cent in range(len(cent_list)):
            pt_in_currentCluster = dataSet[nonzero(clusterAssment[:, 0] == cent)[0], :]   #找到属于当前待划分簇的数据点
            split_centMat, split_clusterAssment = kMeans(pt_in_currentCluster, 2)
            sse_split = sum(split_clusterAssment[:, 1])
            sse_notsplit = sum(clusterAssment[nonzero(clusterAssment[:, 0] != cent)[0], 1])
            if (sse_split + sse_notsplit) <  lowest_sse:
                best_splitCent = cent
                best_centMat = split_centMat
                best_clusterAssment = split_clusterAssment.copy()  #质心矩阵没有copy而这个要copy?  因为
                lowest_sse = sse_notsplit + sse_split

        ##一个簇被划分为2个子簇后,子簇0替代被划分簇的簇号及被划分簇在cent_list中的位置,子簇1簇号为簇列表长度,作为簇列表中最后的簇
        best_clusterAssment[nonzero(best_clusterAssment[:, 0] == 0)[0], 0] = best_splitCent      #更新簇号
        best_clusterAssment[nonzero(best_clusterAssment[:, 0] == 1)[0], 0] = len(cent_list)
        cent_list[best_splitCent] = best_centMat[0].tolist()[0]  
        cent_list.append(best_centMat[1].tolist()[0])    #更新簇列表位置
        clusterAssment[nonzero(clusterAssment[:, 0] == best_splitCent)[0], :] = best_clusterAssment

    return mat(cent_list), clusterAssment

def plot_result():
    dataSet = mat(loadDataSet())
    cent_Mat, clusterAssment = Bi_Kmeans(dataSet, 3)
    #先获取质心坐标
    cent_x = cent_Mat[:, 0].tolist()
    cent_y = cent_Mat[:, 1].tolist()
    data_point = []
    for i in range(3):
        data_point.append(dataSet[nonzero(clusterAssment[:, 0] == i)[0], :].tolist())  #3维list
    cent_xlist = [] ; cent_ylist = []
    for x in cent_x:
        cent_xlist.append(x[0])

    for y in cent_y:
        cent_ylist.append(y[0])

    data_xlist = [] ; data_ylist = []
    for cent  in data_point:
        temp_xlist = [] ; temp_ylist = []
        for q in cent:
            temp_xlist.append(q[0])
            temp_ylist.append(q[1])
        data_xlist.append(temp_xlist)
        data_ylist.append(temp_ylist)

    plt.scatter(data_xlist[0], data_ylist[0], label='1', color='g', s=25, marker='^')
    plt.scatter(data_xlist[1], data_ylist[1], label='2', color='r', s=25, marker='v')
    plt.scatter(data_xlist[2], data_ylist[2], label='3', color='b', s=25, marker='*')
    plt.scatter(cent_xlist, cent_ylist, label='Cents', color='k', s=80, marker='X')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('BiKmeans')
    plt.legend()
    plt.show()