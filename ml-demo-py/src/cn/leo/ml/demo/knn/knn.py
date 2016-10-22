#coding=utf-8


'''
k-近邻算法:
计算输入数据到样本的距离, 按从小到大排序, 取前k个数据 计算每个类别出现的次数, 次数最多的即为输入数据的类别

距离计算方法:
输入数据每个特征与样本数据每个特征 差的平方和开平方

'''


import numpy

def knnAlgo(inputX, dataSet, labels, k):
    '''
    input    输入数据
    dataSet  样本数据集合(矩阵)
    labels   样本的类别
    k
    '''
    
    dataSetSize = dataSet.shape[0] # 样本个数
    #print('样本个数: %d' % dataSetSize)
    
    #差的平方和开平方
    diff = numpy.tile(inputX, (dataSetSize, 1)) - dataSet 
    distances = ((diff**2).sum(axis=1))**0.5 # 到样本的距离
    #print(distances)
    
    sortDistIndex = distances.argsort(); # 距离排序 sortDistIndex中保存 distances中数据的索引值
    #print(sortDistIndex)
    
    # 取前k个值 计算每个类别出现的次数
    classCount = {}
    for i in range(k):
        label = labels[sortDistIndex[i]]
        classCount[label] = classCount.get(label, 0) + 1
    
    rs = None
    v = 0
    for k in classCount.keys():
        if(classCount.get(k) >= v):
            v = classCount.get(k)
            rs = k
         
    return rs
    

    



