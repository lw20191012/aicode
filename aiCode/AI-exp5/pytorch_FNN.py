# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.utils.data import DataLoader
#
#
# # 定义数据集类
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#
# # 定义 FNN 模型类
# class FNN(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim):
#         super().__init__()
#         self.layers = []
#         prev_dim = input_dim
#         for hidden_dim in hidden_dims:
#             self.layers.append(nn.Linear(prev_dim, hidden_dim))
#             self.layers.append(nn.ReLU())
#             prev_dim = hidden_dim
#         self.layers.append(nn.Linear(prev_dim, output_dim))
#         self.model = nn.Sequential(*self.layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# # 定义绘制分类边界函数
# def plot_decision_boundary(model, X, y):
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
#     Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).detach().numpy()
#     Z = np.argmax(Z, axis=1)
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
#
#
# # 定义主函数
# if __name__ == '__main__':
#     # 准备数据
#     X = [[51.50, 69.00], [41.00, 76.00], [33.00, 76.50], [42.00, 59.50], [30.00, 64.00], [34.00, 71.50], [41.00, 63.50],
#          [20.00, 65.50], [16.00, 72.50]]
#     y = [1, 1, 1, 1, 1, 1, 0, 0, 0]
#     dataset = MyDataset(X, y)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#     # 定义模型
#     input_dim = 2
#     hidden_dims = [10, 10]
#     output_dim = 2
#     model = FNN(input_dim, hidden_dims, output_dim)
#
#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     # 训练模型
#     num_epochs = 200
#     losses = []
#     for epoch in range(num_epochs):
#         for i, (inputs, labels) in enumerate(dataloader):
#             # 向前传播和计算损失
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             # 反向传播和更新参数
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # 记录损失值
#             losses.append(loss.item())
#
#         # 输出每个 epoch 的损失值
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#
#     # 绘制损失下降曲线
#     plt.plot(losses)
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.show()
#
#     # 测试模型并输出分类正确率
#     X_test = torch.tensor(X, dtype=torch.float32)
#     y_test = torch.tensor(y, dtype=torch.long)
#     y_pred = torch.argmax(model(X_test), axis=1)
#     accuracy = (y_pred == y_test).sum().item() / len(y_test)
#     print(f'Classification accuracy: {accuracy:.4f}')
#
#     # 绘制分类边界
#     plt.figure()
#     plot_decision_boundary(model, X_test.numpy(), y_test.numpy())
#     plt.show()
#


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 创建数据集
X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建模型
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

# 训练模型
model = FNN(input_size=2, hidden_size=10, output_size=2, num_hidden_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 50
losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())

    # 在训练集和测试集上计算损失值和分类正确率
    with torch.no_grad():
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_acc = (torch.argmax(train_outputs, axis=1) == y_train).float().mean().item()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_acc = (torch.argmax(test_outputs, axis=1) == y_test).float().mean().item()

    train_accs.append(train_acc)
    test_losses.append(test_loss.item())
    test_accs.append(test_acc)

    # 输出每个 epoch 的损失值和分类正确率
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss.item():.4f}')
