import numpy as np
import time
import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
aggdatadir = "E:/Research/NILM/UKDALE/house2/mains_ap.dat"  # "E:/Research/NILM/REDD/house_3/low_freq/channel_1.txt"  "E:/Research/NILM/UKDALE/house2/mains_ap.dat"
labeldir =  "E:/Research/NILM/UKDALE/house2/label/channel_1_13.txt"   # "E:/Research/NILM/REDD/house_3/low_label/channel_1_16.txt"   "E:/Research/NILM/UKDALE/house2/label/channel_1_15.txt"

# 窗口长度和功率阈值(mad阈值) by lcy
SWWidth = 5
thresPower = 500

# 返回输入数据的平均绝对偏差 by lcy
def getMetric(data,mode=0):
    """

    :param data:  type:np.array
    :return: MAD-metric
    """
    if mode == 0:
        return abs(data-data.mean()).mean()
    else:
        return data.var()

# 数据矩阵，将数据分为5行（窗口长度），相邻行的数据有一位移位。这样每次操作一列对应的就是一个窗口中的数据 by lcy
def getdataMatrix(data):
    dataMatrix = np.zeros((SWWidth,len(data)-SWWidth+1))
    for i in range(SWWidth):
        dataMatrix[i,:] = data[i:len(data)-SWWidth+1+i]
    return dataMatrix


def getEventsIndex(dataMatrix):
    """
    meanRow存的是所有窗口的均值，
    mad存的是所有窗口的平均绝对偏差，
    eflag存的是mad大于阈值的窗口的索引
    然后结合窗口，生成所有事件窗口的前后索引(窗口长度是预设的)
    by lcy
    """
    meanRow = dataMatrix.mean(axis=0)
    MAD = abs(dataMatrix - meanRow).mean(axis=0)
    eflag = np.where(MAD >= thresPower)[0]
    eventsIndex = []
    preindex = -6
    for startindex in eflag:
        if startindex - preindex < SWWidth:
            continue
        eventsIndex.append([startindex,startindex+SWWidth-1])
        preindex = startindex
    return np.array(eventsIndex)

# 返回与事件（开始与结束时间指定的）交并比最大的标签的下标以及对应的交并比
def getTruth(starttime,endtime,labels):
    """
    :param starttime:
    :param endtime:
    :param labels:
    :return: 对应gtbox在labels中的下标，对应的iou
    """
    t = np.arange(len(labels))
    # n1, n2中存的分别是开始时间大于事件结束时间和结束时间小于事件开始时间的标签的索引 by lcy
    labelstime = labels[:,:2]
    n1 = np.where( (labelstime[:,0] - endtime) >=0 )[0]
    n2 = np.where( (labelstime[:,1] - starttime) <=0 )[0]
    # t是标签索引，从中过滤掉n1, n2中的索引，剩下的就是和事件存在交集的标签的索引 by lcy
    t = np.setdiff1d(t,n1)
    t = np.setdiff1d(t,n2)
    if len(t) == 0:
        return np.array([-1,0])
    ious = []
    # 获得所有候选标签与事件的iou，从中选择最大的 by lcy
    for i in t:
        ious.append(getIOU(starttime,endtime,labels[i][0],labels[i][1]))
    return np.array([t[np.argmax(ious)],max(ious)])


def getIOU(starttime,endtime,event1,event2):
    inleft = max(starttime,event1)
    inright = min(endtime,event2)
    outleft = min(starttime,event1)
    outright = max(endtime,event2)
    return (inright-inleft)/(outright-outleft)


time.clock()
labels = np.loadtxt(labeldir)
labels = labels[np.where(labels[:, 4] == 0)[0]]
print(labels.shape)
print('load labels access.  ' + str(time.clock()))
aggdata = np.loadtxt(aggdatadir)
print('load data access.  ' + str(time.clock()))
dataMatrix = getdataMatrix(aggdata[:,1])
events = getEventsIndex(dataMatrix)  # 每个事件：开始下标 结束下标
print('检测事件总数：'+str(len(events)))

#到目前为止，我们所获得的就是所有事件的开始下标和结束下标，结束下标就是开始下标+窗口长度-1
#这里的下标就是事件开始与结束在总线序列上的索引

gtbox = []
for e in events:
    # 首先获得事件的开始和结束时间戳，然后
    starttime = aggdata[int(e[0])][0]
    endtime = aggdata[int(e[1])][0]
    gtbox.append(getTruth(starttime, endtime, labels))

# 到这里，我们已经得到了与所有事件以及与其交并比最大的标签
# 如果一个事件没有标签与其存在交集，那么其标签下标就是-1，也就是假阳性，剩下的就是真阳性，然后我们还知道所有标签数量（相当于知道了fn假阴性）
# 然后就可以得到精度、召回率、F1了
# 那么，我是不是可以说，这个算法干的事就是事件检测。。。也没干啥识别的事，最多
# by lcy


gtbox = np.array(gtbox)
events = np.c_[events, gtbox]  # 每个事件：开始下标 结束下标 对应真实事件下标  对应iou
detectedtpindex = np.where(events[:,2]!=-1)[0]
detectedtp = events[detectedtpindex]
detectedfpindex = np.where(events[:,2]==-1)[0]
detectedfp = events[detectedfpindex]
np.savetxt('H:/detectedall.txt',events)
np.savetxt('H:/detectedtp.txt', detectedtp)
np.savetxt('H:/detectedfp.txt', detectedfp)
print('检测到的所有事件：'+str(events.shape))
print('检测到的tp' + str(detectedtp.shape))
print('检测到的fp' + str(detectedfp.shape))
print('真实事件个数：' + str(len(labels)))
recall = len(detectedtp)/len(labels)
precision = len(detectedtp)/len(events)
print('精度:'+str(precision))
print('召回率:'+str(recall))
print('F1:' + str(precision*recall*2/(recall+precision)))
print('LA' + str(detectedtp[:,-1].mean()))


