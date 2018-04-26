#-*- coding:utf-8 -*-
"""
构建含有不重复的所有训练词汇的词袋:
    计算词袋中每个词汇在正常句子/不正常句子中出现的概率
    
"""
from numpy import *
from tkinter import *
import re

def loadDataSet():   #载入用于训练贝叶斯分类器的数据
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec    

def createVocabList(dataSet):  #将训练数据构建为不含重复词汇的词袋
    """
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):   #将测试数据中的词汇映射到词袋中的词汇
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)# [0,0......]

    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def _trainNB0(trainMatrix, trainCategory):
    """
    训练数据原版
    :param trainMatrix: 文件单词矩阵 [[1,0,1,1,1....],[],[]...]
    :param trainCategory: 文件对应的类别[0,1,1,0....]，列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
    :return:
    """
    # 文件数
    numTrainDocs = len(trainMatrix)
    # 单词数
    numWords = len(trainMatrix[0])
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数，
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)  
    # 构造单词出现次数列表

    # 本来‘=’两边都是zeros(),但为了避免p(w0|0)*p(w1|0)*p(w2|0)其中1个为0导致整个结果为0
    p0Num = ones(numWords)  #改动前  p0Num = zeros(numWords)
    p1Num = ones(numWords)  #  p1Num = zeros(numWords)

    # 整个数据集单词出现总数
    p0Denom = 2.0  # ‘=’两边本来也都是0.0,改动理由与line 64同理
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 计算词袋中的每个词汇出现在侮辱性文档的次数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #trainMatrix是文档矩阵,每个行向量为1个文档,元素值表示是否为侮辱性词汇
            p1Denom += sum(trainMatrix[i])  #统计侮辱贴中辱骂词的数量
        else:
        # 计算词袋中的每个词汇出现在正常文档的次数
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 每个单词在1类别下出现次数的占比
    p1Vect = log(p1Num / p1Denom)    #改动前 pVect = pNum / pDenom
    # 每个单词在0类别下出现次数的占比
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive   #pVect为ndarray行向量

def trial0():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMatrix = []
    for line in listOPosts:
        trainMatrix.append(setOfWords2Vec(myVocabList, line))

    p0V, p1V, pAb = _trainNB0(trainMatrix, listClasses)
    print(myVocabList,'————————————————————————————————————————')
    print('p0V:',p0V)
    print('————————————————————————————————————————')
    print('p1V:',p1V)

def ClassifyNB(post_toClassify, p0V, p1V, myVocabList, pClass0):
    mirror_post = array(setOfWords2Vec(myVocabList, post_toClassify))
    p0 = sum(mirror_post * p0V) + log(pClass0)
    p1 = sum(mirror_post * p1V) + log(1 - pClass0)
    if p0 >= p1:   # 如果2个概率均为0，即该帖子所含词汇不存在于词袋中，则默认为是正常帖子
        #print('Normal' + '\n')
        return 'Normal'
    else:
        #print('Watch your word!' + '\n')
        return 'Abusive!'

def trial1():
    post_toClassify = ['you','are','stupid']
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    trainMatrix = []
    for line in postingList:
        trainMatrix.append(setOfWords2Vec(myVocabList, line))

    p0V, p1V, pClass1 = _trainNB0(trainMatrix, classVec)
    result = ClassifyNB(post_toClassify, p0V, p1V, myVocabList, pClass1)
    #print(result)

def pred_func():
    postingList, classVec = loadDataSet()
    vocabSet = createVocabList(postingList)
    trainMatrix = []
    for item in postingList:
        trainMatrix.append(setOfWords2Vec(vocabSet, item))

    pattern = re.compile(r'\W*')   #去掉空格、逗号等非单词字符
    input_data = entry.get()
    process_inputData = pattern.split(input_data)
    processed_inputData = [token.lower() for token in process_inputData if len(token) > 0] 
    #print(processed_inputData)
    p0Vect, p1Vect, pAbusive = _trainNB0(trainMatrix, classVec)
    result = ClassifyNB(processed_inputData, p0Vect, p1Vect, vocabSet, 1-pAbusive)
    status_label['text'] = result
    return


root = Tk()
root.wm_title('Naive Bayes')

label0 = Label(root, text='input')
label0.pack()
status_label = Label(root, text="")
status_label.pack()

entry = Entry(root)
entry.pack()

button = Button(root, text='Predict', command= pred_func)
button.pack()

label_1 = Label(root, text="")
label_1.pack()

root.mainloop()