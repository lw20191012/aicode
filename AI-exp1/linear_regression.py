import time

import numpy as np
import matplotlib.pyplot as plt

# 构建数据集
data = []
year = []
price = []

year_file = 'testSet/x.txt'
price_file = 'testSet/y.txt'

with open(year_file, 'r') as f:
    for lines in f:
        print('lines:', lines)
        line = lines.strip('\n').split(' ')
        print('line:', line)
        year.append(int(line[0]))

with open(price_file, 'r') as f:
    for lines in f:
        print('lines:', lines)
        line = lines.strip('\n').split(' ')
        print('line:', line)
        price.append(float(line[0]))

# print('year:',year)
# print('price:',price)

# year = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# price = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]
for i in range(len(year)):
    data.append([year[i], price[i]])
print(data)
data = np.array(data)
print(data)
# print(len(data))

# 损失列表
loss_list = []


# 根据当前的 w,b,参数计算均方差损失
def mse(b, w, data):
    # 记录当前的w,b下的总误差
    TotalError = 0
    for i in range(0, len(data)):
        # x = data[i, 0]
        x = i
        y = data[i, 1]
        TotalError += (y - (w * x + b)) ** 2
    return TotalError / float(len(data))


# def change(b, w, data, a):
#     b_change = 0
#     w_change = 0
#     size = float(len(data))
#     for i in range(0, len(data)):
#         x = data[i, 0]
#         y = data[i, 1]
#         # 计算梯度
#         b_change += (2 / size) * ((w * x + b) - y)
#         w_change += (2 / size) * x * ((w * x + b) - y)
#         # 根据梯度更新权重和偏置
#     b -= a * b_change
#     w -= a * w_change
#     return [b, w]


# 梯度下降法
def gradient_descent():
    b = 1
    w = 1
    a = 0.00001
    iterations = 100

    for num in range(iterations):
        b_change = 0
        w_change = 0
        size = float(len(data))
        for i in range(0, len(data)):
            # x = data[i, 0]
            x = i
            y = data[i, 1]
            # 计算梯度
            b_change += (2 / size) * ((w * x + b) - y)
            w_change += (2 / size) * x * ((w * x + b) - y)
        # 根据梯度更新权重和偏置
        b -= a * b_change
        w -= a * w_change

        # # [b, w] = change(b, w, data, a)
        # print('[b, w]:',[b, w])

        # 计算本次的损失并将损失值加入损失列表
        loss = mse(b, w, data)
        loss_list.append(loss)

        # # 动态显示loss的变化
        # plt.title("loss_")
        # plt.xlabel("iterator times")
        # plt.ylabel("loss value")
        # plt.legend(labels=['loss'], loc='best')
        # plt.pause(0.01)
        # # 清除上一次显示
        # plt.cla()
        # plt.plot(loss_list, c='r')


        # # 绘制回归图像
        # # 对应的拟合直线的数据
        # y = []
        # for i in range(len(data)):
        #     # y.append(w * year[i] + b)
        #     y.append(w * i + b)
        # # y2 = w * year + b
        # # print("2014年预测房价为:" + str(w * (year[len(year) - 1] + 1) + b))
        # print(str(year[len(year) - 1] + 1) + "年预测房价为:" + str(w * len(data) + b))
        # plt.title("Fit the line graph")  # 标题名
        # plt.scatter(year, price, label='Original Data', s=10)  # 设置为散点图
        # plt.plot(year, y, color='Red', label='Fitting Line', linewidth=2)
        # plt.xlabel('year')
        # plt.ylabel('price')
        # plt.legend()
        # plt.show()

        # plt.clf()
        # plt.pause(0.1)

        # print('iteration:[%s] | loss:[%s] | w:[%s] | b:[%s]' % (num, loss, w, b))
    # print([b,w])
    # 返回最后一次迭代得到的b,w值
    return [b, w]


# 梯度下降
[b, w] = gradient_descent()

# 绘制原始数据的散点图
plt.title("Year-Price")  # 标题名
plt.scatter(year, price, s=10)  # 设置为散点图
plt.xlabel("year")  # x轴的标题
plt.ylabel("price")  # y轴的标题
plt.show()  # 绘制出来

# 绘制损失函数图像
plt.title("Loss Function")  # 标题名
plt.plot(loss_list)
plt.xlabel('Interation')
plt.ylabel('Loss Value')
plt.show()

# 绘制回归图像
# 对应的拟合直线的数据
y = []
for i in range(len(data)):
    # y.append(w * year[i] + b)
    y.append(w * i + b)
# y2 = w * year + b
# print("2014年预测房价为:" + str(w * (year[len(year) - 1] + 1) + b))
print(str(year[len(year) - 1] + 1) + "年预测房价为:" + str(w * len(data) + b))
plt.title("Fit the line graph")  # 标题名
plt.scatter(year, price, label='Original Data', s=10)  # 设置为散点图
plt.plot(year, y, color='Red', label='Fitting Line', linewidth=2)
plt.xlabel('year')
plt.ylabel('price')
plt.legend()
plt.show()
