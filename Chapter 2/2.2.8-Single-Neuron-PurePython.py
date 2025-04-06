import numpy as np
import random

# 设置随机种子以确保结果可复现
np.random.seed(42)

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
    return x, y

# 单神经元模型
class SingleNeuron:
    def __init__(self, input_size):
        # 初始化权重和偏置
        rng = np.random.default_rng(42)
        self.weights = rng.standard_normal(input_size)
        self.bias = rng.standard_normal(1)
        
    def forward(self, x):
        # 前向传播
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, x, y_true, y_pred):
        # 计算梯度
        error = y_pred - y_true
        # 明确使用x参数计算梯度
        dw = np.dot(x.T, error) / len(x)  # 使用x的转置计算权重梯度
        db = np.mean(error)
        return dw, db
    
    def update(self, dw, db, learning_rate):
        # 更新参数
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

# 训练函数
def train_model(model, x, y, learning_rate, n_epoch, batch_size):
    n_samples = len(x)
    n_batches = n_samples // batch_size
    
    for epoch in range(n_epoch):
        total_loss = 0
        # 打乱数据
        indices = np.random.permutation(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        for i in range(n_batches):
            # 获取当前批次
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = x_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            
            # 前向传播
            y_pred = model.forward(batch_x)
            
            # 计算损失（MSE）
            loss = np.mean((y_pred - batch_y) ** 2)
            total_loss += loss
            
            # 反向传播
            dw, db = model.backward(batch_x, batch_y, y_pred)
            
            # 更新参数
            model.update(dw, db, learning_rate)
        
        # 打印训练信息
        avg_loss = total_loss / n_batches
        print(f'Epoch [{epoch+1}/{n_epoch}], Loss: {avg_loss:.4f}')
        print(f'Weights: {model.weights}, Bias: {model.bias}')

# 主程序
if __name__ == "__main__":
    # 生成数据
    x, y = generate_data(n_sample)
    
    # 创建模型
    model = SingleNeuron(input_size=2)
    
    # 训练模型
    train_model(model, x, y, learning_rate, n_epoch, batch_size)
    
    # 测试模型
    test_input = np.array([[0.3, 0.7], [0.8, 0.2]])
    predictions = model.forward(test_input)
    print("\n测试结果:")
    for i, (x, pred) in enumerate(zip(test_input, predictions)):
        print(f"输入: {x}, 预测值: {pred:.4f}, 真实值: {max(x):.4f}") 