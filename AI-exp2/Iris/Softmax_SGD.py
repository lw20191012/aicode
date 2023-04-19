from numpy import *
import matplotlib.pyplot as plt
import random

# 标准化
def normalization(x):
    mu = mean(x, axis=0)
    sigma = std(x, axis=0)
    # print('x:',x)
    # print('mu:',mu)
    # print('sigma:',sigma)
    return (x - mu) / sigma

# softmax函数
def softmax(mat):
    expsum = sum(exp(mat), axis=1)
    # # 测试
    # print('expsum:',expsum)
    # print('exp(mat):',exp(mat))
    return exp(mat) / expsum


# 概率矩阵，标签矩阵，具体类别序号
def calError(h, label, type):
    # 120*1的全0矩阵
    ans = zeros((shape(h)[0], 1))
    # # 测试
    # print("--------ans--------")
    # print('ans:',ans)
    print("--------type--------")
    print('type:',type)

    # 遍历120个样本对应的类别序号
    for i in range(shape(h)[0]):
        # 对应标签 对应 类别序号,直接1减去得到的概率矩阵该样本在该类别的概率
        if label[i, 0] == type:
            ans[i, 0] = 1 - h[i, type]
        # 对应的标签不是该类别序号，直接对该概率取负值
        else:
            ans[i, 0] = -h[i, type]

    # 测试
    print("---------ans----------")
    print('ans:',ans)

    return ans


# 加载训练的数据
def loadTrainSet():
    # 加载Iris的训练数据中的特征向量x,放入data列表中
    f = open('../testSet/Iris/train/x.txt')
    data = []
    for line in f:
        lineArr = line.strip().split()
        data.append([float(lineArr[0]), float(lineArr[1])])
    f.close()

    # 加载Iris的训练数据中的标签y，放入label列表中
    f = open('../testSet/Iris/train/y.txt')
    label = []
    for line in f:
        lineArr = line.strip().split()
        label.append(int(float(lineArr[0])))
    f.close()
    # 对特征向量进行标准化然后放入trainData中
    data = normalization(data)
    trainData = []
    for i in data:
        # 加入偏置项1
        trainData.append([1, i[0], i[1]])
    return trainData, label

# 加载测试的数据
def loadTestSet():
    # 加载Iris的测试数据中的特征向量x,放入data列表中
    f = open('../testSet/Iris/test/x.txt')
    data = []
    for line in f:
        lineArr = line.strip().split()
        data.append([float(lineArr[0]), float(lineArr[1])])
    f.close()

    # 加载Iris的测试数据中的标签y,放入label列表中
    f = open('../testSet/Iris/test/y.txt')
    label = []
    for line in f:
        lineArr = line.strip().split()
        label.append(int(float(lineArr[0])))
    f.close()

    # 对特征向量进行标准化然后放入testData中
    data = normalization(data)
    testData = []
    for i in data:
        testData.append([1, i[0], i[1]])
    return testData, label


# 对测试样本数据进行预测
def getTestPredict(theta, testdata):
    testdataMat = mat(testdata)
    # n*4
    htest = testdataMat * theta
    # 测试
    print("--------htest--------")
    print('htest:', htest)
    # 找每一个样本的对应行里面最大的
    label = argmax(htest, axis=1)
    # 测试
    print("--------label--------")
    print('prelabel:', label)
    return label


def getCorrectRate(predictlabel, testlabel):
    correctnum = 0
    for i in range(shape(testlabel)[0]):
        if predictlabel[i, 0] == testlabel[i, 0]:
            correctnum = correctnum + 1
    return correctnum / shape(testlabel)[0]


def calLikelihood(h, labelMat):
    ans = 0
    # print("-------------------------")
    # print('shape(h)[0]:',shape(h)[0])
    # 循环样本个数
    for i in range(shape(h)[0]):
        # 将每一个样本的真实label对应的序号的概率值进行相加
        ans = ans + log(h[i, labelMat[i, 0]])
        # # 测试
        # print('h[i, labelMat[i, 0]]:',i,h[i, labelMat[i, 0]])
    return ans


def stocSoftmaxRegression(data, label):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    n, m = shape(dataMat)  # n samples, m features
    theta = zeros((m, 4))  # m features, 3 tpyes + 1
    alpha = 0.001
    maxCycle = 5
    episilon = 1e-7
    preLikelihood = 0.0
    for k in range(maxCycle):
        h = softmax(dataMat * theta)
        likelihood = calLikelihood(h, labelMat)
        if abs(likelihood - preLikelihood) < episilon:
            break
        preLikelihood = likelihood
        # choose one sample only
        rand = random.randint(0, n - 1)
        for i in range(shape(h)[1]):
            if labelMat[rand, 0] == i:
                delta = alpha * (1 - h[rand, i]) * dataMat[rand]
            else:
                delta = alpha * (-h[rand, i]) * dataMat[rand]
            theta[:, i] = theta[:, i] + delta
    print(k)
    return theta


def plotBoundary(para0, para1, para2, name):
    x = arange(-3, 3, 0.1)
    y = (-para1 * x - para0) / para2
    plt.plot(x, y, label=name)
    plt.legend()


def plotBestFit(fig, data, label, theta, name, subplot):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    xcord0 = [];
    ycord0 = []
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(shape(data)[0]):
        if label[i] == 0:
            xcord0.append(dataMat[i, 1])
            ycord0.append(dataMat[i, 2])
        elif label[i] == 1:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
        elif label[i] == 2:
            xcord2.append(dataMat[i, 1])
            ycord2.append(dataMat[i, 2])

    ax = fig.add_subplot(subplot)
    ax.set_title(name, fontsize=8)

    ax.scatter(xcord0, ycord0, s=30, c='red')
    ax.scatter(xcord1, ycord1, s=30, c='green')
    ax.scatter(xcord2, ycord2, s=30, c='blue')

    plotBoundary(theta[0, 0] - theta[0, 1], theta[1, 0] - theta[1, 1], theta[2, 0] - theta[2, 1], "red-green")
    plotBoundary(theta[0, 0] - theta[0, 2], theta[1, 0] - theta[1, 2], theta[2, 0] - theta[2, 2], "red-blue")
    plotBoundary(theta[0, 1] - theta[0, 2], theta[1, 1] - theta[1, 2], theta[2, 1] - theta[2, 2], "green-blue")


def main():
    fig = plt.figure()
    # 加载训练数据集
    trainData, label = loadTrainSet()
    # 加载测试数据集
    testdata, testlabel = loadTestSet()

    # # 测试
    print("----------trainData----------")
    print('trainData:',trainData)
    print('label:',label)
    print("----------testData----------")
    print('testData:',testdata)
    print('testlabel:',testlabel)

    # # # # # # # # # # # #
    # 随机梯度下降
    # # # # # # # # # # # #
    print("stocSoftmaxRegression:")
    print("theta")
    theta2 = stocSoftmaxRegression(trainData, label)

    print("---------result theta2----------")
    print('theta2=',theta2)

    # 划分测试数据集
    print("to TestDataSet:")
    predictlabel1 = getTestPredict(theta2, testdata)
    print("accuracy:")
    print(getCorrectRate(predictlabel1, mat(testlabel).transpose()))
    plotBestFit(fig, testdata, testlabel, theta2, "stocSoftmax, ToTestDataSet", 121)

    # 划分训练数据集
    print("to TrainDataSet:")
    predictlabel2 = getTestPredict(theta2, trainData)
    print("accuracy:")
    print(getCorrectRate(predictlabel2, mat(label).transpose()))
    plotBestFit(fig, trainData, label, theta2, "stocSoftmax, toTrainDataSet", 122)

    plt.show()


if __name__ == '__main__':
    main()