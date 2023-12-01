import numpy as np
import time
import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from matplotlib.ticker import MultipleLocator

aggdatadir = "E:/Research/NILM/REDD/house_3/low_freq/channel_1.txt"  # "E:/Research/NILM/REDD/house_3/low_freq/channel_1.txt"  "E:/Research/NILM/UKDALE/house2/mains_ap.dat"
channeldir = "E:/Research/NILM/REDD/house_3/low_freq/channel_10.txt"  # "E:/Research/NILM/UKDALE/house2/channel_8.dat"  "E:/Research/NILM/REDD/house_3/low_freq/channel_16.txt"
labeldir = "E:/Research/NILM/REDD/house_3/low_label/channel_1_10.txt"   # "E:/Research/NILM/REDD/house_3/low_label/channel_1_16.txt"   "E:/Research/NILM/UKDALE/house2/label/channel_1_15.txt"
labelthres = 5  # REDD：5  UKDALE：10
type = 10
exectime= 1
# 阈值设置 by lcy
varthres = 50000
if type == 14:
    # wash-dryer
    varthres = 50000
elif type == 16:
    # microwave
    varthres = 1500
elif type == 10:
    # furnace
    varthres = 1500
elif type == 8:
    # kettle
    varthres = 300000
elif type == 9:
    # rice cooker
    varthres = 5000
elif type == 13:
    varthres = 10000
elif type == 15:
    # microwave
    varthres = 50000

# 不是很清楚这个是干嘛的，难不成是聚类的中心点，具体的值是功率波动？
def getcenters(type):
    if type == 14 :
        return np.array([-2191.29000000000,-2113.44666666667,-2017.03666666667,-1728.09000000000,-1104.91000000000,-976.433333333334,-895.693333333333,-627.016666666667,927.866666666667,1118.05666666667,1745.50000000000,2269.61000000000,2345.61666666667,2419.40333333333])
    elif type == 16:
        return np.array([-2118.20666666667,-1734.18666666667,-1108.32666666667,-913.623333333333,-632.963333333334,-381.670000000000,-113.736666666667,119.583333333333,281.696666666667,389.153333333333,918.326666666667,1118.33000000000,1749.35333333333,2218.32666666667,2295.69000000000,2366.53333333333])
    elif type == 10:
        return np.array([-2118.81333333333,-1735.04000000000,-1108.94000000000,-914.883333333333,-635.333333333334,-383.796666666667,-112.713333333333,99.7133333333336,404.926666666667,692.963333333333,918.326666666667,1117.70333333333,1749.35333333333,2199.09666666667,2275.29000000000,2349.85000000000])
    elif type == 8:
        return np.array([-2946.69333333333,-2815.21666666667,-2217.27666666667,-2032.69333333333,-1899.36000000000,-1761.22666666667,-1256.19000000000,98.5466666666666,535.846666666667,1732.29666666667,1873.64000000000,2027.87000000000,2218.78000000000,2824.79666666667,2980.60333333333])
    elif type == 9:
        return np.array([-2118.81333333333,-1735.04000000000,-1108.94000000000,-914.883333333333,-635.333333333334,-383.796666666667,-112.713333333333,99.7133333333336,404.926666666667,692.963333333333,918.326666666667,1117.70333333333,1749.35333333333,2199.09666666667,2275.29000000000,2349.85000000000])
    elif type == 13:
        return np.array([-2118.81333333333,-1735.04000000000,-1108.94000000000,-914.883333333333,-635.333333333334,-383.796666666667,-112.713333333333,99.7133333333336,404.926666666667,692.963333333333,918.326666666667,1117.70333333333,1749.35333333333,2199.09666666667,2275.29000000000,2349.85000000000])
    elif type == 15:
        return np.array([-2946.69333333333,-2815.21666666667,-2217.27666666667,-2032.69333333333,-1899.36000000000,-1761.22666666667,-1256.19000000000,98.5466666666666,535.846666666667,1532.29666666667,1873.64000000000,2027.87000000000,2218.78000000000,2824.79666666667,2980.60333333333])
    else:
        return np.array([])

#
def getMetric(data,mode=0):
    """
    :param data: 待统计数组
    :return: 功率均值，方差均值，滑动窗口总数
    """
    if len(data) < 3:
        if mode == 0:
            exit(5)
        else:
            return np.array([]),np.array([])
    # 首先生成三个数组，它们同一位置构成的一列就是一个窗口的数据的索引
    # 然后得到datamatrix，它的一列就是一个窗口的总线功率数据
    # 然后得到所有窗口的均值以及方差，最后返回功率均值总和、方差总和、滑动窗口总数
    # by lcy
    a = np.arange(len(data) - 2)
    b = a + 1
    c = a + 2
    datamatrix = np.array((data[a], data[b], data[c]))
    avglist = datamatrix.mean(axis=0)
    varlist = datamatrix.var(axis=0)
    if mode == 0:
        return avglist.sum(),varlist.sum(),len(a)
    else:
        return avglist,varlist


# 返回所有窗口的平均功率均值、平均方差
def getAggMetric():
    """
    output:
        233.9569554696737
        1048.490399196951
        0.1465908
    return:
        233.9569554696737
        1048.490399196951
    """
    aggdata = np.loadtxt(aggdatadir)
    aggdata = aggdata[:, 1]
    time.clock()
    avgsum,varsum,datalen = getMetric(aggdata)
    print(avgsum/datalen)
    print(varsum/datalen)
    time2 = time.clock()
    print(time2)
    return avgsum/datalen,varsum/datalen

def getEventMetic():
    """
    getMetric mode==0:
    1349645715.6275997 284622.80656049907

    getMetric mode==1:
    1381347723.8
    94.37
    674822857.8138015
    (65586,)
    1696375920.388889
    2.22222222221818e-05
    142311.4032802495
    (65586,)
    """
    avgs = np.array([])
    vars= np.array([])
    data_dirs = ['E:/NILM/REDD/NILM924_10/','E:/NILM/REDD/NILM103_14/','E:/NILM/REDD/NILM1022_16/','E:/NILM/UKDALE/NILM1210_H28/','E:/NILM/UKDALE/NILM1226_H29/','E:/NILM/UKDALE/NILM1206_H213/','E:/NILM/UKDALE/NILM1205_H215/']
    for data_dir in data_dirs:
        ids = np.loadtxt(data_dir+'ImageSets/Main/trainval2.txt',dtype=int)  # 设置部分
        plot_diff = True
        for id_ in ids:
            anno = ET.parse(data_dir + 'Annotations/' + str(id_) + '.xml')
            for obj in anno.findall('object'):
                bndbox_anno = obj.find('bndbox')
                box = [int(bndbox_anno.find(tag).text)   # xml自身已经从0开始
                    for tag in ('xmin', 'xmax')]
                data = np.loadtxt(data_dir + 'JPEGImages/' + str(id_) + '.txt')
                img_short = data[max(box[0]-2,0):min(box[1]+3,1024)]  # -2 +3
                avg_,var_ = getMetric(img_short,1)
                avgs = np.append(avgs,avg_)
                vars = np.append(vars,var_)
    print(avgs.max())
    print(avgs.min())
    print(avgs.mean())
    print(avgs.shape)
    print(vars.max())
    print(vars.min())
    print(vars.mean())
    print(vars.shape)

def mylog(power):
    if power > 0:
        return math.log(power)
    else:
        return -math.log(-power)

#  生成所有事件的开始索引，结束索引，功率幅度，也就是两个稳态之间的地方，以及这两个稳态的功率差
def getEvent(data):
    # 首先生成所有窗口的均值和方差
    events = []
    a = np.arange(len(data) - 3)
    b = a + 1
    c = a + 2
    datamatrix = np.array((data[a], data[b], data[c]))
    avglist = datamatrix.mean(axis=0)
    varlist = datamatrix.var(axis=0)
    powerpresteady = 0
    steady = 1
    eventstart = 0
    eventend = 0
    # 对于所有的窗口：
    for i,(p,v) in enumerate(zip(avglist,varlist)):
        # 如果当前窗口是非稳态，且前一个窗口是稳态，则记录成事件开始下标
        if v > varthres: # 非稳态
            if steady == 1:
                powerpresteady = avglist[i-1]
                steady = 0
                eventstart = i
            else: # steady == 0
                continue
        # 如果当前窗口是稳态，且前一个窗口是非稳态，则记录成一次完整的事件（事件开始，事件结束）
        else:  #稳态
            if steady == 1:
                continue
            else: # steady == 0 #重新从非稳态进入稳态
                steady = 1
                eventend = i+3
                powernextsteady = avglist[i+1]
                #捕捉到一次完整的事件：前稳态后稳态都找到了
                events.append([eventstart,eventend,powernextsteady-powerpresteady])
    print(events)
    return np.array(events)

# 首先生成所有事件
# 然后找到和对应的centers最接近的那个，所以所谓的类编号就是centers中的索引了
def getEventClass(aggdata,centers):
    """
    :param aggdata:
    :return: 每个事件：开始下标 结束下标 △P 所属类簇
    """
    events = getEvent(aggdata[:, 1])
    np.savetxt('H:/subclus.txt', events)
    if exectime == 0:
        print('事件个数:'+str(len(events)))
        exit(136)
    eventclass = []
    for e in events:
        ctemp = centers - e[2]
        eventclass.append(abs(ctemp).argmin())
    events = np.c_[events, eventclass]
    return events


def getIOU(starttime,endtime,event1,event2):
    inleft = max(starttime,event1)
    inright = min(endtime,event2)
    outleft = min(starttime,event1)
    outright = max(endtime,event2)
    return (inright-inleft)/(outright-outleft)

def getTruth(starttime,endtime,labels):
    """
    :param starttime:
    :param endtime:
    :param labels:
    :return: 对应gtbox在labels中的下标，对应的iou
    """
    t = np.arange(len(labels))
    labelstime = labels[:,:2]
    n1 = np.where( (labelstime[:,0] - endtime) >=0 )[0]
    n2 = np.where( (labelstime[:,1] - starttime) <=0 )[0]
    t = np.setdiff1d(t,n1)
    t = np.setdiff1d(t,n2)
    if len(t) == 0:
        return np.array([-1,0])
    ious = []
    for i in t:
        ious.append(getIOU(starttime,endtime,labels[i][0],labels[i][1]))
    return np.array([t[np.argmax(ious)],max(ious)])

def getbigcountnumber(data):  #求一个整数数列的众数
    data = data[np.where(data>=0)[0]]
    return np.argmax(np.bincount(data))


def getChannelTimes(data):  # 求支线没有缺漏的时间列表
    innertimelist = []
    subdata = data[1:,0] - data[:-1,0]
    breaktimepoint = np.where(subdata>50)[0]
    startindex = 0
    for bp in breaktimepoint:
        endindex = bp
        innertimelist.append([data[startindex,0],data[endindex,0]])
        startindex = bp+1
    return np.array(innertimelist)


def getInnerIndex(events,innertime,aggdata):  # 过滤检测到的事件：求在innertime内部的事件（支线时间连续无缺失）
    innerindex = []
    for i,e in enumerate(events):
        starttime = aggdata[int(e[0]),0]
        endtime = aggdata[int(e[1]), 0]
        t = np.arange(len(innertime))
        n1 = np.where((innertime[:, 0] - endtime) >= 0)[0]
        n2 = np.where((innertime[:, 1] - starttime) <= 0)[0]
        t = np.setdiff1d(t, n1)
        t = np.setdiff1d(t, n2)
        if len(t) == 0:
            continue
        else:
            innerindex.append(i)
    return np.array(innerindex)


def detect():
    time.clock()
    labels = np.loadtxt(labeldir)
    labels = labels[np.where(labels[:,4]==0)[0]]
    print(labels.shape)
    print('load labels access.  '+str(time.clock()))
    aggdata = np.loadtxt(aggdatadir)
    print('load data access.  '+str(time.clock()))
    channel = np.loadtxt(channeldir)
    print('load channel access.  '+str(time.clock()))
    innertime = getChannelTimes(channel)
    print('get timeindex.  '+str(time.clock()))
    events = getEventClass(aggdata,getcenters(type))
    print('get events access.  '+str(time.clock()))
    events = events[getInnerIndex(events,innertime,aggdata)]
    print('filter access.  ' + str(time.clock()))
    # print(events)  # 每个事件：开始下标 结束下标 △P 所属类簇
    print('检测事件总数：'+str(len(events)))
    gtbox = []
    for e in events:
        starttime = aggdata[int(e[0])][0]
        endtime = aggdata[int(e[1])][0]
        gtbox.append(getTruth(starttime,endtime,labels))
    gtbox = np.array(gtbox)
    events = np.c_[events, gtbox]  # 每个事件：开始下标 结束下标 △P 所属类簇 对应真实事件下标  对应iou
    detectedmatch = events[np.where(events[:,4] != -1)[0]]  # 分类前所有能匹配到标签的事件
    print('匹配标签事件数:'+str(detectedmatch.shape))
    clusdict = dict()  # 类簇标签计数
    clusset = set()  # 有效类簇
    for count,d in enumerate(detectedmatch):
        if count > len(detectedmatch)/2:
            break
        if int(d[3]) in clusdict:
            clusdict[int(d[3])] += 1
        else:
            clusdict[int(d[3])] = 1
    for ckey in clusdict.keys():
        if clusdict[ckey] >= labelthres:  # 大于5个标签有的才算
            clusset.add(ckey)  #
    print('-----')
    print(clusdict)
    print(clusset)
    detectedallindex = np.array([],dtype=int)
    for classid in clusset:
        detectedallindex = np.append(detectedallindex,np.where(events[:,3] == classid)[0])
    print(detectedallindex)
    detectedall = events[detectedallindex]  # 每个事件：开始下标 结束下标 △P 所属类簇 对应真实事件下标 对应iou
    detectedall = np.c_[detectedall,labels[detectedall[:,4].astype(int),3]]  # 每个事件：开始下标 结束下标 △P 所属类簇 对应真实事件下标 对应iou 对应真实类别
    detectedall[np.where(detectedall[:,4]==-1)[0],6] = -1  # 对应真实事件下标为-1的，将对应真实类别设置为-1(因为在上一行，对应真实事件下标为-1的，将取到labels[-1]的类别，即为labels的最后一个但是这里应该是没有的意思)
    detectedall = np.c_[detectedall, np.zeros(len(detectedall))]  # 每个事件：开始下标 结束下标 △P 所属类簇 对应真实事件下标 对应iou 对应真实类别 0(接下来修改为预测类别)
    for clus in clusset:
        detected_clus = detectedall[np.where(detectedall[:,3]==clus)[0]]
        pclass = getbigcountnumber(detected_clus[:,6].astype(int))
        detectedall[np.where(detectedall[:,3]==clus)[0],7] = pclass
    # detectedall # 每个事件：开始下标 结束下标 △P 所属类簇 对应真实事件下标 对应iou 对应真实类别 预测类别
    detectedall = detectedall[np.where(detectedall[:,7]!=-1)[0]]
    detectedtpindex = np.where(detectedall[:,6]==detectedall[:,7])[0]
    detectedtp = detectedall[detectedtpindex]
    detectedfpindex = np.where(detectedall[:,6]!=detectedall[:,7])[0]
    detectedfp = detectedall[detectedfpindex]
    np.savetxt('H:/detectedall.txt',detectedall)
    np.savetxt('H:/detectedtp.txt', detectedtp)
    np.savetxt('H:/detectedfp.txt', detectedfp)
    print('检测到的所有事件：'+str(detectedall.shape))
    print('检测到的tp' + str(detectedtp.shape))
    print('检测到的fp' + str(detectedfp.shape))
    print('真实事件个数：' + str(len(labels)))
    recall = len(detectedtp)/len(labels)
    precision = len(detectedtp)/len(detectedall)
    print('精度:'+str(precision))
    print('召回率:'+str(recall))
    print('F1:' + str(precision*recall*2/(recall+precision)))
    print('LA' + str(detectedtp[:,-3].mean()))


def plotevent(event,ax,mode=0):
    event[:,1] = event[:,1]-event[:,1].min()
    if event[:,1].max() > 1500 or event[:,1].max() < 50 or len(event[:,1]) > 8:
        return
    y_data = event[:,1]
    if mode == 1:
        y_data = event[:6, 1]
    x_data = np.arange(0,len(y_data))
    if mode == 0:
        ax.plot(x_data,y_data,color='royalblue',linewidth = 0.5)
    else:
        ax.plot(x_data,y_data,color='royalblue')
    print(int(event[0,0]))
    print(int(event[-1,0]))

def resultAnalysis():
    res = np.loadtxt('H:/研究生/毕业论文/对比论文/2017方差对数检测结果/10电暖炉/detectedtp.txt')
    res = res[np.where(res[:, -2] == 4)[0]]
    resfp = np.loadtxt('H:/研究生/毕业论文/对比论文/2017方差对数检测结果/10电暖炉/detectedfp.txt')
    resfp = resfp[np.where(resfp[:,-1]==4)[0]]
    aggdata = np.loadtxt(aggdatadir)

    # figure分成3行3列, 取得第一个子图的句柄, 第一个子图跨度为1行3列, 起点是表格(0, 0)
    ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
    # ax1.set_title('I', fontsize=15, pad=-65, color='red', alpha=0.5)
    ax1.xaxis.set_major_locator(MultipleLocator(1))  # 将x主刻度标签设置为1的倍数
    # figure分成3行3列, 取得第二个子图的句柄, 第二个子图跨度为1行3列, 起点是表格(1, 0)
    ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1, rowspan=1)
    ax2.xaxis.set_major_locator(MultipleLocator(1))  # 将x主刻度标签设置为1的倍数
    # ax2.set_title('II', fontsize=15, pad=-65, color='red', alpha=0.5)

    matrix_res = []
    matrix_fpres = []
    for event_id in range(1,len(res)+1):
        event = aggdata[int(res[event_id-1,0]):int(res[event_id-1,1])+1]
        plotevent(event,ax1,1)
        matrix_res.append(event[:6,1])
    matrix_res = np.array(matrix_res)
    np.savetxt('D:/new.csv', matrix_res, delimiter=',',fmt='%.2f')
    print('-----------')
    for event_id in range(1,len(resfp)+1):
        event = aggdata[int(resfp[event_id-1,0]):int(resfp[event_id-1,1])+1]
        plotevent(event,ax2)
        matrix_fpres.append(event[:6,1])
    matrix_fpres = np.array(matrix_fpres)
    np.savetxt('D:/new2.csv', matrix_fpres, delimiter=',',fmt='%.2f')
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    plt.show()


def ploteventaround():
    plt.rcParams['figure.figsize'] = (12.0, 2.5)  # 单位是inches
    # starttime = [1303048534,1303174133,1303324085,1303415941]  #  furnace-II
    # endtime = [1303048538,1303174137,1303324089,1303415945]    #  furnace-II
    # starttime = [1303178177,1302977299]  #  furnace-II
    # endtime = [1303178182,1302977487]  #  furnace-II
    starttime = [1302956387,1302987742,1303057110]  #  16-2
    endtime = [1302956391,1302987746,1303057115]  #  16-2
    aggdata = np.loadtxt(aggdatadir)
    for i in range(len(starttime)):
        startindex = np.where(aggdata[:,0]==starttime[i])[0][0]
        endindex = np.where(aggdata[:, 0] == endtime[i])[0][0]-1
        y_data = aggdata[startindex:endindex+1, 1]
        x_data = np.arange(100, 100+len(y_data))

        startindex2 = max(startindex-100,0)
        endindex2 = endindex + 100
        y_data2 = aggdata[startindex2:endindex2+1, 1]
        x_data2 = np.arange(0, len(y_data2))

        plt.plot(x_data2, y_data2,color='royalblue')
        plt.plot(x_data, y_data,color='red')
        plt.show()
        np.savetxt('C:/Users/qwe63/Desktop/画图数据'+str(i)+'.txt',y_data2)




if __name__ =='__main__':
    # detect()
    resultAnalysis()
    # ploteventaround()









