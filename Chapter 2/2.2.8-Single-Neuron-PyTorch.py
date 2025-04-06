import torch
import torch.nn as nn
import numpy as np
import random

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 参数设置
n_sample = 1000  # 训练样本数量
batch_size = 32  # 批大小
learning_rate = 0.1  # 学习率
n_epoch = 10  # 训练轮数

# 生成训练数据
def generate_data(n_samples):
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, (n_samples, 2))
    y = np.max(x, axis=1)
    return torch.FloatTensor(x), torch.FloatTensor(y)

# 单神经元模型
class SingleNeuron(nn.Module):
    def __init__(self, input_size):
        super(SingleNeuron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

# 训练函数
def train_model(model, x, y, learning_rate, n_epoch, batch_size):
    # 使用MSE损失函数
    criterion = nn.MSELoss()
    # 使用SGD优化器
    # 添加 momentum=0.9：加速收敛并减少震荡
    # 添加 weight_decay=0.0001：L2正则化，防止过拟合
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    
    # 将数据转换为DataLoader
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    for epoch in range(n_epoch):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.view(-1, 1))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练信息
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{n_epoch}], Loss: {avg_loss:.4f}')
        # 将张量移到CPU后再转换为numpy
        weights = model.linear.weight.data.cpu().numpy()
        bias = model.linear.bias.data.cpu().numpy()
        print(f'Weights: {weights}, Bias: {bias}')

# 主程序
if __name__ == "__main__":
    # 检查是否有可用的MPS设备（M1 Mac的GPU）
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 生成数据
    x, y = generate_data(n_sample)
    x, y = x.to(device), y.to(device)
    
    # 创建模型
    model = SingleNeuron(input_size=2).to(device)
    
    # 训练模型
    train_model(model, x, y, learning_rate, n_epoch, batch_size)
    
    # 测试模型
    test_input = torch.FloatTensor([[0.3, 0.7], [0.8, 0.2]]).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(test_input)
    
    print("\n测试结果:")
    test_input_cpu = test_input.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    for i, (x, pred) in enumerate(zip(test_input_cpu, predictions_cpu)):
        print(f"输入: {x}, 预测值: {pred[0]:.4f}, 真实值: {max(x):.4f}") 