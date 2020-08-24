import torch

# Initialize

t1 = torch.tensor([[1,2], [3,4]])
print(t1,'\n')

t1 = torch.tensor([[1,2], [3,4]], dtype=torch.float32)
print(t1,'\n')

# Specify run on GPU(CUDA)

t2 = torch.tensor([[1,2],[3,4]], dtype=torch.float32,
                  device="cuda")
print(t2,'\n')

# Specify run on CPU

t2 = torch.tensor([[1,2],[3,4]], dtype=torch.float32,
                  device="cpu")
print(t2,'\n')

# 怎麼簡單判斷要用 cpu 還是 gpu

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
t3 = torch.tensor([[1,2],[3,4]], dtype=torch.float32,
                  device=device)
print(t3,'\n')

# 反向傳播 backpropagation

t4 = torch.tensor([[1,2],[3,4]], dtype=torch.float32,
                  device=device, requires_grad=True)
print(t4,'\n','\n',t4.dtype,'\n','\n',t4.shape,'\n','\n',t4.requires_grad)

# Common Initailize Methods

t5 = torch.empty((4,4))
print(t5,'\n')

t5 = torch.zeros((4,4))
print(t5,'\n')

t5 = torch.rand((4,4))
print(t5,'\n')

t5 = torch.ones((4,4))
print(t5,'\n')

t5 = torch.eye(4,4) # create an indentiy matrix, 對角線元素為1，其餘元素為0
print(t5,'\n')

t5 = torch.arange(-5,5,0.3)
print(t5,'\n')

t5 = torch.linspace(-1,1,10)
print(t5,'\n')

t5 = torch.diag(torch.linspace(-1,1,5)) # diagonal matrix
print(t5,'\n')

# How to initialize and convert tensors to other types (int, float, double)

tensor = torch.arange(4)
print(tensor.bool()) # boolean
print(tensor.short()) # int 16
print(tensor.long()) # int 64 (Important)
print(tensor.half()) # float 16
print(tensor.float()) # float 32 (Important)
print(tensor.double(),'\n') # float 64

# Array to Tensor conversion and vice-versa
import numpy as np

np_array = np.zeros((5, 5))
print(np_array)
tensor = torch.from_numpy(np_array)
np_array_again = (
    tensor.numpy()
)  # np_array_again will be same as np_array (perhaps with numerical round offs)
print(np_array_again)

# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #

# 用法如同 numpy

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# -- Addition --
z2 = torch.add(x, y)  # This is one way
z = x + y  # This is another way. This is my preferred way, simple and clean.
print(z2)
print(z)
