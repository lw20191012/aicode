import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义前向传播函数
def forward_propagation(X, parameters):
    A = X
    L = len(parameters) // 2 # 神经网络的层数
    # 测试
    print("网络层数L1:",L)

    for l in range(1, L):
        # 测试
        print("l:",l)
        Z = np.dot(parameters["W" + str(l)], A) + parameters["b" + str(l)]
        A = sigmoid(Z)
        # 另加，可除
        # 暂存中间的一个权值
        parameters['A' + str(l)] = A
        parameters['Z' + str(l)] = A
    # X前向的结果
    ZL = np.dot(parameters["W" + str(L)], A) + parameters["b" + str(L)]
    # 测试
    print("ZL:",ZL)
    AL = sigmoid(ZL)

    # 另加，可除
    # 暂存中间的一个权值
    parameters['A' + str(L)] = AL
    parameters['Z' + str(L)] = ZL
    return AL

# 定义损失函数,交叉熵
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    return cost

# 定义反向传播函数
def backward_propagation(X, Y, AL, parameters):
    grads = {}
    L = len(parameters) // 4

    m = Y.shape[1]
    dZL = AL - Y
    # 测试
    print("AL1:",AL)
    print("Y:",Y)
    print("网络层数L2:", L)
    print("parameters:",parameters)
    print("dZL:",dZL)

    grads["dW" + str(L)] = 1/m * np.dot(dZL, np.transpose(parameters["A" + str(L-1)]))
    grads["db" + str(L)] = 1/m * np.sum(dZL, axis=1, keepdims=True)
    dA = np.dot(np.transpose(parameters["W" + str(L)]), dZL)
    for l in reversed(range(1, L)):
        dZ = np.multiply(dA, sigmoid(parameters["Z" + str(l)]))
        grads["dW" + str(l)] = 1/m * np.dot(dZ, np.transpose(parameters["A" + str(l)]))
        grads["db" + str(l)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(np.transpose(parameters["W" + str(l)]), dZ)
    return grads

# 定义参数初始化函数
def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    # 测试
    print("初始参数")
    print("parameters:", parameters)
    return parameters

# 定义模型训练函数
def train_model(X, Y, layer_dims, num_iterations=1000, learning_rate=0.01, print_cost=False):
    np.random.seed(42)
    costs = []
    # 设置初始参数和偏置
    parameters = initialize_parameters(layer_dims)
    # 进行1000次训练
    for i in range(num_iterations):
        # 前向
        AL = forward_propagation(X, parameters)
        # 测试
        print("AL:",AL)
        cost = compute_cost(AL, Y)

        grads = backward_propagation(X, Y, AL, parameters)
        for l in range(1, len(layer_dims)):
            # 测试
            print("parameters:",parameters["W" + str(l)])
            print("learning_rate:",learning_rate)
            print("dW:",grads["dW" + str(l)])
            parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Iteration {}: Cost = {}".format(i, cost))
    plt.plot(costs)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Cost")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters

def predict(X, Y, parameters):
    AL = forward_propagation(X, parameters)
    predictions = np.round(AL)
    accuracy = np.mean(predictions == Y)
    return predictions, accuracy

# 准备训练集和测试集数据
X_train = np.array([[51.50, 69.00], [41.00, 76.00], [33.00, 76.50], [42.00, 59.50], [30.00, 64.00], [34.00, 71.50], [41.00, 63.50], [20.00, 65.50], [16.00, 72.50]])
Y_train = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0]).reshape(1, -1)
X_test = np.array([[30.00, 64.00], [34.00, 71.50], [42.00, 83.50], [36.50, 53.00], [39.00, 71.50]])
Y_test = np.array([1, 1, 1, 0, 0]).reshape(1, -1)

# 设置神经网络的层数和每层节点数
layer_dims = [2, 4, 1]

# 训练模型并预测测试集
# parameters = train_model(X_train.T, Y_train.T, layer_dims, num_iterations=5000, learning_rate=0.05, print_cost=True)
parameters = train_model(X_train.T, Y_train, layer_dims, num_iterations=5000, learning_rate=0.05, print_cost=True)
predictions, accuracy = predict(X_test.T, Y_test.T, parameters)
print("Test set accuracy: {}".format(accuracy))

# 绘制分类边界
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = forward_propagation(np.c_[xx.ravel(), yy.ravel()].T, parameters)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train.ravel(), cmap=plt.cm.Spectral)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('FNN classification')
plt.show()