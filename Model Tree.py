#在树回归的基础上修改叶节点模型、误差计算模型
def modelLeaf(dataSet):
    """
    Desc:
        当数据不再需要切分的时候，生成叶节点的模型。
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回得到的 回归系数ws
    """
    ws, X, Y = linearSolve(dataSet)
    return ws

# 计算线性模型的误差值
def modelErr(dataSet):
    """
    Desc:
        在给定数据集上计算误差。
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

 # helper function used in two places
def linearSolve(dataSet):
    """
    Desc:
        将数据集格式化
    Args:
        dataSet -- 输入数据
    Returns:
        ws -- 执行线性回归的回归系数 
        X -- 格式化自变量X
        Y -- 格式化目标变量Y
    """
    m, n = shape(dataSet)  
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1: n] = dataSet[:, 0: n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

____________________________________________________________________________________________________________________

#利用模型树进行预测
from numpy import *

def Is_Tree(obj):
    return (type(obj).__name__ == 'dict')

def Reg_Tree(leaf_value, in_Data):   #树回归处理叶节点的函数
    return float(leaf_value)

def Model_Tree(leaf_weights, in_Data):   #模型树处理叶节点的函数
    n = in_Data.shape[1]
    X = ones((1, n+1))
    X[:, 1: n+1] = in_Data
    return (X * leaf_value)

def ModelTree_forecast(tree, inData, leaf_model= Model_Tree):  #输入数据，进行预测
    if not Is_Tree(tree):
        return leaf_model(inData)

    if inData[tree['split_feat']] > tree['split_value']:   #递归寻找输入值对应的叶节点
        if Is_Tree(tree['right']):
            return ModelTree_forecast(tree['right'], in_Data, leaf_model)
        else:
            return leaf_model(tree['right'], in_Data)

    else:
        if Is_Tree(tree['left']):
            return ModelTree_forecast(tree['left'], in_Data, leaf_model)
        else:
            return leaf_model(tree['left'], in_Data)

def Test(tree, test_set):
    m = test_set.shape[0]
    yHat = zeros((m, 1))
    for i in range(m):
        yHat[i, 0] = ModelTree_forecast(tree, test_set[i], leaf_model)

    return yHat