import numpy as np
import matplotlib.pyplot as plt


# sigmoid 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# softmax 激活函数
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


# 初始化权重和偏置，[2,4,4,1]
def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # 网络层数
    # 1,2,3
    for l in range(1, L):
        # print("l=",l)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l - 1], layer_dims[l]) * 0.01
        parameters['b' + str(l)] = np.zeros((1, layer_dims[l]))
    print('parameters:',parameters)
    return parameters


# 前向传播
def forward_propagation(X, parameters):
    L = len(parameters) // 2  # 网络层数3层
    print("L1:"+str(L))
    A = X

    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        # 点乘加偏置之后，激活
        A = sigmoid(np.dot(A_prev, W) + b)
        # 暂存中间的一个权值
        parameters['A' + str(l)] = A
    # 最后进行softmax
    AL = softmax(np.dot(A, parameters['W' + str(L)]) + parameters['b' + str(L)])
    parameters['A' + str(L)] = AL
    print("------AL&parameters-------")
    print('AL:',AL)
    print('parameters:',parameters)
    return AL, parameters


# 计算交叉熵损失
def compute_cost(AL, Y):
    m = Y.shape[0]  # 样本数量
    cost = - np.sum(Y * np.log(AL)) / m

    return cost


# 反向传播
def backward_propagation(AL, Y, parameters):
    # print("-----LOSS-----")
    # print("AL",AL)
    # print("Y",Y)
    m = Y.shape[0]  # 样本数量
    # print("len(parameters):"+str(len(parameters)))
    # 3层
    L = len(parameters) // 3  # 网络层数
    # 测试
    print("L2:"+str(L))
    dAL = AL - Y
    # # 测试
    # print("dAL:"+str(dAL))
    grads = {}
    grads['dW' + str(L)] = np.dot(parameters['A' + str(L - 1)].T, dAL)
    grads['db' + str(L)] = np.sum(dAL, axis=0, keepdims=True)
    # 测试
    print("grads:",grads)

    dA_prev = dAL
    # 测试
    print("dA_prev:", dA_prev)
    for l in reversed(range(1, L)):

        dZ = np.dot(dA_prev, parameters['W' + str(l + 1)].T) * sigmoid(
            parameters['A' + str(l)] * (1 - parameters['A' + str(l)]))
        grads['dW' + str(l)] = np.dot(parameters['A' + str(l - 1)].T, dZ)
        grads['db' + str(l)] = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ

    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # 网络层数

    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters


def predict(X, y, params):
    AL, _ = forward_propagation(X, params)
    predictions = np.argmax(AL, axis=1)
    accuracy = np.mean(predictions == y)

    return accuracy


def train_model(X, Y, layer_dims, learning_rate=0.01, num_iterations=5000, print_cost=False):
    np.random.seed(1)
    # 损失列表
    costs = []

    # 初始化参数，权重和偏置，3层
    parameters = initialize_parameters(layer_dims)

    # 迭代训练，迭代5000次
    for i in range(num_iterations):
        # 前向传播，输入X，参数矩阵，返回 本轮次 训练结果AL 和 参数caches
        AL, caches = forward_propagation(X, parameters)

        # 计算损失，交叉熵损失
        cost = compute_cost(AL, Y)

        # 反向传播
        grads = backward_propagation(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 打印损失
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # 绘制损失下降曲线
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # 绘制分类边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], parameters)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.show()

    # 计算并报告分类正确率
    Y_pred = predict(X, parameters)
    accuracy = np.mean(Y_pred == Y) * 100
    print("Accuracy: {}%".format(accuracy))

    return parameters



# 数据集
X = np.array([[51.50, 69.00], [41.00, 76.00], [33.00, 76.50], [42.00, 59.50], [30.00, 64.00], [34.00, 71.50], [41.00, 63.50], [20.00, 65.50], [16.00, 72.50]])
Y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])

# FNN模型训练
parameters = train_model(X, Y, layer_dims=[2, 4, 4, 1], learning_rate=0.1, num_iterations=10000, print_cost=True)
