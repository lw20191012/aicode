import numpy as np
import matplotlib.pyplot as plt

# 读取数据集
data = np.loadtxt('gmm/GMM3.txt', delimiter='\t', skiprows=1)
X = data[:, 1:]  # 特征矩阵
y = data[:, 0]  # 类别标签

# 将数据集分为不同类别的样本
class_0 = X[y == 0]
class_1 = X[y == 1]
class_2 = X[y == 2]

# 求每个类别的均值向量和协方差矩阵
mean_0 = np.mean(class_0, axis=0)
mean_1 = np.mean(class_1, axis=0)
mean_2 = np.mean(class_2, axis=0)

cov_0 = np.cov(class_0.T)
cov_1 = np.cov(class_1.T)
cov_2 = np.cov(class_2.T)

# 计算类别的先验概率
prior_0 = len(class_0) / len(X)
prior_1 = len(class_1) / len(X)
prior_2 = len(class_2) / len(X)

# 定义高斯分布密度函数
def gaussian(x, mean, cov):
    n = len(x)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    x_mean = (x - mean).reshape((n, 1))
    return 1.0 / (np.power(2 * np.pi, n / 2) * np.power(det, 0.5)) * \
           np.exp(-0.5 * (x_mean.T @ inv @ x_mean))[0][0]

# 定义分类函数
def classify(x):
    g_0 = np.log(prior_0) + np.log(gaussian(x, mean_0, cov_0))
    g_1 = np.log(prior_1) + np.log(gaussian(x, mean_1, cov_1))
    g_2 = np.log(prior_2) + np.log(gaussian(x, mean_2, cov_2))
    if g_0 >= g_1 and g_0 >= g_2:
        return 0
    elif g_1 >= g_0 and g_1 >= g_2:
        return 1
    else:
        return 2

# 对测试数据进行分类
y_pred = np.apply_along_axis(classify, 1, X)

# 计算分类正确率
accuracy = np.mean(y_pred == y)

# 绘制分类曲线
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = np.apply_along_axis(classify, 1, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian Naive Bayes')
plt.show()

#
# # 绘制分类曲线
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Gaussian Naive Bayes')
# plt.show()