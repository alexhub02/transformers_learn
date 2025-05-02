import numpy as np
import torch
import os


data_root = os.path.join(os.getcwd(), "data")
a = torch.arange(1, 25).reshape(2, 3, 4)
a.numpy().tofile(os.path.join(data_root, "a.bin"))
a_read = np.fromfile(os.path.join(data_root, "a.bin"), dtype=np.int64)
print(a)

b = a.transpose(0,2)
print(b)
b.numpy().tofile(os.path.join(data_root, "b.bin"))
b_read = np.fromfile(os.path.join(data_root, "b.bin"), dtype=np.int64)
print(b_read)
