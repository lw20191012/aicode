# # import numpy as np
# #
# #
# # # 加载数据使用
# # def load_data_set():
# #     data_mat = []
# #     label_mat = []
# #     fr = open('testSet.txt')  # 数据为（100,3）
# #     for line in fr.readlines():
# #         lines = line.strip().split()
# #         # 把数据转换成列表字符串形式，strip只能删除开头和结尾的字符，默认删除两边的空白符，例如：/n, /r, /t, ' '
# #         # split切片，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
# #         data_mat.append([1.0, float(lines[0]), float(lines[1])])  # 此时data的维度还是（100,3）
# #         label_mat.append(int(lines[2]))  # 标签维度为1行100列
# #     return data_mat, label_mat
# #
# #
# # # sigmoid函数
# # def sigmoid(inx):
# #     return 1.0 / (1 + np.exp(- inx))
# #
# #
# # # 梯度提升算法
# # def grad_ascent(data_mat_in, class_labels):
# #     datamattrix = np.mat(data_mat_in)  # 转换为numpy可以处理的数据类型
# #     labelmat = np.mat(class_labels).transpose()  # 100行1列
# #     m, n = np.shape(datamattrix)  # 得到数组的维度100,3
# #     print(m, n)
# #     alpha = 0.001
# #     maxcycles = 500
# #     weights = np.ones((n, 1))  # 权值为（3,1）的，初始为1
# #     for k in range(maxcycles):
# #         # 在这里会出错，在python3中的矩阵运算为mat1.dot(mat2)或者是mat1@mat2
# #         h = sigmoid(datamattrix.dot(weights))  # 此时为矩阵计算（100,3）*（3，1）因此会得到（100,1）维度的矩阵
# #         error = (labelmat - h)  # 计算差值，这应该是最小二乘法啊？，但是为什么会是梯度，晚点博客详解
# #         weights = weights + alpha * (datamattrix.transpose() @ error)  # 矩阵相乘，和上面一样的错误
# #         # 这一句是这段代码最不好理解的，，其实很简单，首先要明确，weights的维度为（3,1），而alpha为数值常数，
# #         # datamattrix的维度为（100,3）的，经过转置以后为（3,100），而error为（100,1）
# #         # 因此datamattrix.transpose() * error的维度就是（3,1）的，从数据意义解释一下，因为error是训练值和真实值
# #         # 的差，此时有100行一列，即每个样本对应一个误差值，然后和原始数据相乘的意义就是梯度了，因为梯度的基本
# #         # 形式为：datamattrix.transpose() * error或者是datamattrix.transpose() * (labelmat - h)
# #     return weights
# #
# #
# # # 画图
# # def plotbestfit(weights):
# #     import matplotlib.pyplot as plt
# #     datamat, labelmat = load_data_set()
# #     dataarr = np.array(datamat)
# #     n = np.shape(datamat)[0]
# #     xcord1 = [];
# #     ycord1 = []
# #     xcord2 = [];
# #     ycord2 = []
# #     for i in range(n):
# #         if int(labelmat[i]) == 1:
# #             xcord1.append(dataarr[i, 1]);
# #             ycord1.append(dataarr[i, 2])
# #         else:
# #             xcord2.append(dataarr[i, 1]);
# #             ycord2.append(dataarr[i, 2])
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111)
# #     ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
# #     ax.scatter(xcord2, ycord2, s=30, c='blue')
# #     x = np.arange(-3.0, 3.0, 0.1)
# #     # 画出直线，weights[0]*1.0+weights[1]*x+weights[2]*y=0
# #     y = (-weights[0] - weights[1] * x) / weights[2]
# #     ax.plot(x, y.transpose())  # 出错，原因是维度不对，需要转置一下
# #     plt.xlabel('X1')
# #     plt.ylabel('X2')
# #     plt.show()
# #
# # data_mat_in, class_labels=load_data_set()
# # weights=grad_ascent(data_mat_in, class_labels)
# # plotbestfit(weights)
#
#
#
#
# import numpy as np
# from matplotlib import pyplot as plt
# from statistics import mean
# #读取数据
#
# #数据分割
# data_x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# data_y=[2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]
# # x_ave,y_ave = data_x.mean(),data_y.mean()
# x_ave,y_ave = mean(data_x),mean(data_y)
# xysum,xsum,ysum,sum = 0,0,0,0
# dic = dict(zip(data_x,data_y))
# for x,y in dic.items():
#     x_t,y_t = x-x_ave,y-y_ave
#     xysum,xsum,ysum = xysum+x_t*y_t,xsum+x_t**2,ysum+y_t**2
# sum = xysum/((xsum*ysum)**0.5)
# print(sum**2)
#
# #y=kx+b
# #参数预定义
# lr=0.001
# b=1
# k=0
# epochs=3000 #轮次
# data_len=float(len(data_x))
#
#
# def error_num(b,k):
#     sum=0
#     for i in range(0,len(data_x)):
#         sum+=(data_y-(k*data_x+b))**2
#     return sum/float(len(data_x))
#
#
# def Gradient_Descent(b,k,lr,epochs):
#     for i in range(epochs):
#         b_grad=0
#         k_grad=0
#         for j in range(0,len(data_x)):
#             b_grad,k_grad=b_grad-(1/data_len)*(data_y[j]-k*data_x[j]-b),k_grad-(1/data_len)*(data_x)*(data_y[j]-k*data_x[j]-b)
#         b,k=b-(lr*b_grad),k-(lr*k_grad)
#         if i%1000 == 0:
#             print("epochs:",i)
#             print('b:',b,'k:',k)
#             print('error:',error_num(b,k))
#             plt.plot(data_x,data_y,'b')
#             plt.plot(data_x,k*data_x+b,'r')
#             plt.show()
#     return b,k
#
# Gradient_Descent(b,k,lr,epochs)
#

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import MultipleLocator
import warnings


def normalization(x):
    mu = mean(x, axis=0)
    sigma = std(x, axis=0)
    return (x - mu) / sigma


# def loadDataSet():
#     f = open('Exam/train/x.txt')
#     data = []
#     for line in f:
#         lineArr = line.strip().split()
#         data.append(float(lineArr[0]))
#     f.close()
#     f = open('Exam/train/y.txt')
#     label = []
#     for line in f:
#         lineArr = line.strip().split()
#         label.append(int(float(lineArr[0])))
#     f.close()
#     # data归一化 添加1
#     data = normalization(data)
#     data1 = []
#     for i in data:
#         data1.append([1, i[0], i[1]])
#     return data1, label


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def gradDescent(data, label):
    dataMat = mat(data)
    # labelMat = mat(label).transpose()
    labelMat = mat(label)
    n, m = shape(dataMat)  # n samples, m features
    print('n',n)
    print('m',m)
    # theta = mat([[1], [-1], [1]])
    theta = mat(np.ones((m, 1)))
    alpha = 0.001
    maxCycle = 100
    # maxCycle = 10000
    episilon = 0.01
    h = sigmoid(dataMat * theta)
    error = h - labelMat
    precost = (-1) * (labelMat.transpose() * log(h) + (ones(
        (n, 1)) - labelMat).transpose() * log(ones((n, 1)) - h))

    plt.ion()
    xs = [0, 0]
    ys = [0, precost[0, 0]]
    for k in range(maxCycle):

        theta = theta + alpha * (dataMat.transpose() * (-error))
        h = sigmoid(dataMat * theta)
        error = h - labelMat
        cost = (-1) * (labelMat.transpose() * log(h) + (ones(
            (n, 1)) - labelMat).transpose() * log(ones((n, 1)) - h))

        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = k + 1
        ys[1] = cost[0, 0]
        fig = plt.figure(1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = plt.subplot(121)

        ax.set_title("cost", fontsize=8)
        ax.plot(xs, ys)
        plotResult(data, label, theta, fig)
        plt.pause(0.01)
        if abs(precost - cost) < episilon:
            break
        precost = cost

    print(k)
    return theta


def plotResult(data, label, theta, fig):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    xcord1 = []
    ycord1 = []
    xcord0 = []
    ycord0 = []
    for i in range(shape(data)[0]):
        if (label[i]) == 0:
            xcord0.append(dataMat[i, 1])
            ycord0.append(dataMat[i, 2])
        else:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = plt.subplot(122)
    plt.cla()
    ax.scatter(xcord0, ycord0, s=30, c='red', marker='s')
    ax.scatter(xcord1, ycord1, s=30, c='green')
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(2)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    x = arange(-3, 3, 0.1)
    y = (-theta[1, 0] * x - theta[0, 0]) / theta[2, 0]
    ax.plot(x, y)


def main():
    # data1, label = loadDataSet()
    data1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    label = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]
    theta = gradDescent(data1, label)
    print(theta)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
