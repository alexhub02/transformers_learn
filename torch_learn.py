import torch

x = torch.arange(0,6).reshape(2, 3)
y = torch.arange(0, 6).reshape(2, 3).transpose(0, 1)
print(x)
print(y)
print(torch.matmul(x, y))