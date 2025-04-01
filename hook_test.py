import torch
import torch.nn as nn

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc1(x)
        return x

# 创建模型实例
model = SimpleNet()

# 定义钩子函数
def hook_fn(module, input, output):
    print(f"模块名称: {module.__class__.__name__}")
    print("输入形状:", [i.shape for i in input])
    print("输出形状:", output.shape)

# 注册钩子到想要监控的层，这里以conv1为例
handle = model.conv1.register_forward_hook(hook_fn)

# 创建一个随机输入
input_tensor = torch.randn(1, 3, 32, 32)

# 进行前向传播
output = model(input_tensor)

# 移除钩子
handle.remove()