import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from cffi import model


# 定义模型
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, activation):
        super(FNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        # 定义模型结构
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_hidden = nn.ModuleList()
        for i in range(num_hidden_layers - 1):
            self.fc_hidden.append(nn.Linear(hidden_size, hidden_size))
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.activation(self.fc_input(x))
        for i in range(self.num_hidden_layers - 1):
            x = self.activation(self.fc_hidden[i](x))
        x = self.fc_output(x)
        return x


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 训练模型
num_epochs = 1000
loss_list = []
for epoch in range(num_epochs):
    inputs = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(Y_train).long()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 绘制损失函数下降曲线
plt.plot(loss_list)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 计算模型在测试集上的准确率
with torch.no_grad():
    inputs = torch.from_numpy(X_test).float()
    labels = torch.from_numpy(Y_test).long()
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    print('Test Accuracy: {:.2f}%'.format(correct / total * 100))

# 绘制分类边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = np.argmax(model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()).data.numpy(), axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, edgecolor='k')
plt.title('Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
