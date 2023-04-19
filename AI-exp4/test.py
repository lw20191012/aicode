import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



# 读取数据集
data = pd.read_csv('gmm/GMM3.txt', delimiter='\t')
# data = pd.read_csv('gmm/GMM4.txt', delimiter='\t')
# data = pd.read_csv('gmm/GMM6.txt', delimiter='\t')
# data = pd.read_csv('gmm/GMM8.txt', delimiter='\t')
# data = pd.read_csv('gmm/testData.txt', delimiter='\t')


X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
# # 测试
# print("X_train:"+str(X_train))
# print("y_train:"+str(y_train))
# print("X_test:"+str(X_test))
# print("y_test:"+str(y_test))

# 设置交叉验证次数
num_folds = 5
# 定义交叉验证的分数计算方法
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# 建立模型
model = GaussianNB()
# 交叉验证得到分数
scores = cross_val_score(model, X, y, cv=kfold)
print("5倍交叉验证得到的分类正确率：", np.mean(scores))


# 训练模型
model = GaussianNB()
model.fit(X_train, y_train)

# 预测结果
y_pred = model.predict(X_test)

# 计算分类正确率
accuracy = accuracy_score(y_test, y_pred)
print('分类正确率：', accuracy)

# 绘制分类曲线
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian Naive Bayes')
plt.show()
