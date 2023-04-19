# ###############################################
# #  线性回归
# #   1、代数求解
# #   2、梯度下降求解
# #   3、绘制动图
# ##############################################
# import numpy as np
# import matplotlib.pyplot as plt
# import datetime
# # 设置显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
# plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
#
#
#
# # 从txt中读取数据
# x_arr=np.loadtxt('testSet/x.txt').reshape(-1,1)
# y_arr=np.loadtxt('testSet/y.txt').reshape(-1,1)
#
# # # 测试
# # print('x_arr1:',x_arr)
# # print('y_arr1:',y_arr)
#
# # 由于年份本身并没有强烈的语意，因此此处对其数值进行-2000的操作便于后续计算
# x_arr=x_arr-2000
#
# # # 测试
# # print('x_arr2:',x_arr)
# # print('y_arr2:',y_arr)
#
#
#
#
# """
# (1)求解解析解
#
# """
#
# x_mat=np.mat(x_arr) # 转变成矩阵
# y_mat=np.mat(y_arr)
#
# mat=np.ones((x_mat.shape[0],1))
# x_mat=np.c_[x_mat,mat] # 增加偏置项：[2011,1][2012,1]
#
# xTx=x_mat.transpose(1,0)*x_mat # 内积
# theta=xTx.I*x_mat.transpose(1,0)*y_mat # (xTx)^-1*xT*y
#
# # 预测2014年房价的解析解
# theta=np.asarray(theta)
# pre_0=theta[0]*14+theta[1]
# print("解析解得到的线性拟合公式为： y=",theta[0][0],'*x +',theta[1][0])
# print("解析解预测2014年房价为: ",pre_0[0])
#
# """
# (2)梯度下降法求解
#
#
# """
# lr=0.001
# maxiter=200
# theta_list=np.array([0,0])
# theta1=[0,1]
# def H(theta,x):
#     # 预测
#     return theta[0]*x+theta[1]
#
# def cost(x,y,theta):
#     # 计算损失
#     n=x.shape[0]
#     return np.square(y-theta[0]*x-theta[1]).sum()/(2*n)
#
# def GRAD(x,y,theta,lr,n):
#     """
#     梯度下降法求解线性回归问题
#     :param x: 输入值
#     :param y: 真实的输出值
#     :param theta: 学习目标
#     :param lr: 学习率
#     :param n: 样本数量
#     """
#     hx=H(theta,x)
#     d0=np.sum((hx-y)*x)/n
#     d1=np.sum(hx-y)/n
#     theta[0]=theta[0]-lr*d0
#     theta[1]=theta[1]-lr*d1
#     return theta
#
# #迭代
# iter=0
# cost_list=[]
# prec=2
# while iter <maxiter:
#     iter+=1
#     theta1=GRAD(x_arr,y_arr,theta1,lr,x_arr.shape[0])
#     c=cost(x_arr,y_arr,theta1)
#     cost_list.append(c)
#     theta_list=np.vstack((theta_list,np.array(theta1)))
#     # 动态显示loss的变化
#     plt.title("损失函数变化")
#     plt.xlabel("iterator times")
#     plt.ylabel("loss value")
#     plt.legend(labels=['loss'], loc='best')
#     plt.pause(0.01)
#     # 清除上一次显示
#     plt.cla()
#     plt.plot(cost_list,c='r')
#
#
#
# theta_list=np.array(theta_list)
#
#
# '''绘制动图'''
# from matplotlib.animation import FuncAnimation
# import matplotlib.pyplot as plt
# fig,ax=plt.subplots()# 画布申请
# #--------------初始状态---------------
# atext_anti=plt.text(-1,9,'',fontsize=15)
# btext_anti=plt.text(-1,10,'',fontsize=15)
# ln,=plt.plot([],[],'red')
# def init():
#     ax.set_xlim(np.min(x_arr)-2,2+np.max(x_arr))
#     ax.set_ylim(np.min(y_arr)-2,2+np.max(y_arr))
#     return ln,
#
# def update(frame):
#     x=x_arr
#     y=frame[0]*x+frame[1]
#     ln.set_data(x,y)
#     atext_anti.set_text('theta0=%.3f' % frame[0])
#     btext_anti.set_text('theta1=%.3f' % frame[1])
#     return ln,
#
# # 静态图
# plt.title("Linear Regression--线性回归")
# ax.set_ylabel("房价")
# ax.set_xlabel("年份（-2000）")
#
# ax.scatter(x_arr,y_arr)
# ax.scatter(np.array(14),np.array(12.32),c='orange')
# ax.plot(x_arr,theta[0]*x_arr+theta[1],ls='--',c='c')
# plt.legend(labels=['梯度下降法拟合直线','解析解拟合直线'], loc='best')
#
# # 动态图
# ani=FuncAnimation(fig,update,frames=theta_list,init_func=init)
# plt.show()
#
#
#
#
# # GRAD预测的值
# print(theta1)
# print("GRAD预测的2014年的房价：",(theta1[0]*14+theta[1])[0])
#
# # 最后绘制总的图
# plt.cla()
# plt.title("线性回归")
# plt.ylabel("房价")
# plt.xlabel("年份（-2000）")
#
# plt.scatter(x_arr,y_arr)
# plt.scatter(np.array(14),np.array(12.32),c='orange')
# plt.plot(x_arr,theta[0]*x_arr+theta[1],ls='--',c='c')
# plt.plot(x_arr,theta1[0]*x_arr+theta1[1],c='r')
# plt.scatter(np.array(14),np.array((theta1[0]*14+theta[1])[0]),c='r',marker='*')
# plt.legend(labels=['解析解拟合直线','梯度下降法拟合直线','原始输入','解析解预测结果','梯度下降预测结果'],loc='best')
#
# filename="梯度下降-"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.jpg'
# plt.savefig(filename)
# plt.show()

print("测试git功能")

###############################################
#  线性回归
#   1、代数求解
#   2、梯度下降求解
#   3、绘制动图
##############################################
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 从txt中读取数据,将读出来的列表转换为1列
x_arr = np.loadtxt('testSet/x.txt').reshape(-1, 1)
y_arr = np.loadtxt('testSet/y.txt').reshape(-1, 1)

# # 测试
# print('x_arr1:',x_arr)
# print('y_arr1:',y_arr)

# 由于年份本身并没有强烈的语意，因此此处对其数值进行-2000的操作便于后续计算
x_arr = x_arr - 2000

# # 测试
# print('x_arr2:',x_arr)
# print('y_arr2:',y_arr)


"""
(1)求解解析解

"""

x_mat = np.mat(x_arr)  # 转变成矩阵
y_mat = np.mat(y_arr)

# # 测试
# print('x_mat:',x_mat)
# print('y_mat:',y_mat)

# 构造14*1元素全为1的矩阵
mat = np.ones((x_mat.shape[0], 1))
x_mat = np.c_[x_mat, mat]  # 增加偏置项：[2011,1][2012,1]

# X求内积
xTx = x_mat.transpose(1, 0) * x_mat  # 内积
# X内积的逆*X内积*y
theta = xTx.I * x_mat.transpose(1, 0) * y_mat  # (xTx)^-1*xT*y

# # 测试
# print('theta:', theta)

# 预测2014年房价的解析解
theta = np.asarray(theta)

# # 测试
# print('theta:', theta)

pre_0 = theta[0] * 14 + theta[1]

# # 测试
# print('pre_0:', pre_0)


print("解析解得到的线性拟合公式为： y=", theta[0][0], '*x +', theta[1][0])
print("解析解预测2014年房价为: ", pre_0[0])

"""
(2)梯度下降法求解
"""
# 学习率，最大迭代次数，初始斜率，初始偏置
lr = 0.001
maxiter = 100
theta_list = np.array([0, 0])
theta1 = [0, 1]


def H(theta, x):
    # 预测
    return theta[0] * x + theta[1]


# 计算损失函数
def cost(x, y, theta):
    # 计算损失
    n = x.shape[0]
    return np.square(y - theta[0] * x - theta[1]).sum() / (2 * n)


def GRAD(x, y, theta, lr, n):
    """
    梯度下降法求解线性回归问题
    :param x: 输入值
    :param y: 真实的输出值
    :param theta: 学习目标
    :param lr: 学习率
    :param n: 样本数量
    """
    hx = H(theta, x)
    d0 = np.sum((hx - y) * x) / n
    d1 = np.sum(hx - y) / n
    theta[0] = theta[0] - lr * d0
    theta[1] = theta[1] - lr * d1
    return theta


# 迭代
iter = 0
# 损失列表
cost_list = []
# prec = 2
while iter < maxiter:
    iter += 1
    theta1 = GRAD(x_arr, y_arr, theta1, lr, x_arr.shape[0])
    c = cost(x_arr, y_arr, theta1)
    cost_list.append(c)
    # 将theta1加入theta_list
    theta_list = np.vstack((theta_list, np.array(theta1)))
    # 动态显示loss的变化
    plt.title("损失函数变化")
    plt.xlabel("iterator times")
    plt.ylabel("loss value")
    plt.legend(labels=['loss'], loc='best')
    plt.pause(0.01)
    # 清除上一次显示
    plt.cla()
    plt.plot(cost_list, c='r')

theta_list = np.array(theta_list)

'''绘制动图'''
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # 画布申请
# --------------初始状态---------------
atext_anti = plt.text(-1, 9, '', fontsize=15)
btext_anti = plt.text(-1, 10, '', fontsize=15)
ln, = plt.plot([], [], 'red')


def init():
    ax.set_xlim(np.min(x_arr) - 2, 2 + np.max(x_arr))
    ax.set_ylim(np.min(y_arr) - 2, 2 + np.max(y_arr))
    return ln,


def update(frame):
    x = x_arr
    y = frame[0] * x + frame[1]
    ln.set_data(x, y)
    atext_anti.set_text('theta0=%.3f' % frame[0])
    btext_anti.set_text('theta1=%.3f' % frame[1])
    return ln,


# 静态图
plt.title("Linear Regression--线性回归")
ax.set_ylabel("房价")
ax.set_xlabel("年份（-2000）")

ax.scatter(x_arr, y_arr)
ax.scatter(np.array(14), np.array(12.32), c='orange')
ax.plot(x_arr, theta[0] * x_arr + theta[1], ls='--', c='c')
plt.legend(labels=['梯度下降法拟合直线', '解析解拟合直线'], loc='best')

# 动态图
ani = FuncAnimation(fig, update, frames=theta_list, init_func=init)
plt.show()

# GRAD预测的值
print(theta1)
print("GRAD预测的2014年的房价：", (theta1[0] * 14 + theta[1])[0])

# 最后绘制总的图
plt.cla()
plt.title("线性回归")
plt.ylabel("房价")
plt.xlabel("年份（-2000）")

plt.scatter(x_arr, y_arr)
plt.scatter(np.array(14), np.array(12.32), c='orange')
plt.plot(x_arr, theta[0] * x_arr + theta[1], ls='--', c='c')
plt.plot(x_arr, theta1[0] * x_arr + theta1[1], c='r')
plt.scatter(np.array(14), np.array((theta1[0] * 14 + theta[1])[0]), c='r', marker='*')
plt.legend(labels=['解析解拟合直线', '梯度下降法拟合直线', '原始输入', '解析解预测结果', '梯度下降预测结果'],
           loc='best')

filename = "梯度下降-" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jpg'
plt.savefig(filename)
plt.show()
