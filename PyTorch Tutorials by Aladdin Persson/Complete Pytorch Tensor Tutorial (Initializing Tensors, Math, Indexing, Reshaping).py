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

# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# -- Addition --
z1 = torch.empty(3)
torch.add(x, y, out=z1)  # This is one way
z2 = torch.add(x, y)  # This is another way
z = x + y  # This is my preferred way, simple and clean.

# -- Subtraction --
z = x - y  # We can do similarly as the preferred way of addition

# -- Division (A bit clunky) --
z = torch.true_divide(x, y)  # Will do element wise division if of equal shape

# -- Inplace Operations --
t = torch.zeros(3)

t.add_(x)  # Whenever we have operation followed by _ it will mutate the tensor in place
t += x  # Also inplace: t = t + x is not inplace, bit confusing.

# -- Exponentiation (Element wise if vector or matrices) --
z = x.pow(2)  # z = [1, 4, 9]
z = x ** 2  # z = [1, 4, 9]


# -- Simple Comparison --
z = x > 0  # Returns [True, True, True]
z = x < 0  # Returns [False, False, False]

# -- Matrix Multiplication --
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  # Matrix multiplication of x1 and x2, out shape: 2x3
x3 = x1.mm(x2)  # Similar as line above

# -- Matrix Exponentiation --
matrix_exp = torch.rand(5, 5)
print(
    matrix_exp.matrix_power(3)
)  # is same as matrix_exp (mm) matrix_exp (mm) matrix_exp

# -- Element wise Multiplication --
z = x * y  # z = [9, 16, 21] = [1*9, 2*8, 3*7]

# -- Dot product --
z = torch.dot(x, y)  # Dot product, in this case z = 1*9 + 2*8 + 3*7

# -- Batch Matrix Multiplication --
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # Will be shape: (b x n x p)

# -- Example of broadcasting --
x1 = torch.rand((5, 5))
x2 = torch.ones((1, 5))
z = (
    x1 - x2
)  # Shape of z is 5x5: How? The 1x5 vector (x2) is subtracted for each row in the 5x5 (x1)
z = (
    x1 ** x2
)  # Shape of z is 5x5: How? Broadcasting! Element wise exponentiation for every row

# Other useful tensor operations
sum_x = torch.sum(
    x, dim=0
)  # Sum of x across dim=0 (which is the only dim in our case), sum_x = 6
values, indices = torch.max(x, dim=0)  # Can also do x.max(dim=0)
values, indices = torch.min(x, dim=0)  # Can also do x.min(dim=0)
abs_x = torch.abs(x)  # Returns x where abs function has been applied to every element
z = torch.argmax(x, dim=0)  # Gets index of the maximum value
z = torch.argmin(x, dim=0)  # Gets index of the minimum value
mean_x = torch.mean(x.float(), dim=0)  # mean requires x to be float
z = torch.eq(x, y)  # Element wise comparison, in this case z = [False, False, False]
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0)
# All values < 0 set to 0 and values > 0 unchanged (this is exactly ReLU function)
# If you want to values over max_val to be clamped, do torch.clamp(x, min=min_val, max=max_val)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # True/False values
z = torch.any(x)  # will return True, can also do x.any() instead of torch.any(x)
z = torch.all(
    x
)  # will return False (since not all are True), can also do x.all() instead of torch.all()

# ============================================================= #
#                        Tensor Indexing                        #
# ============================================================= #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# Get first examples features
print(x[0].shape)  # shape [25], this is same as doing x[0,:]

# Get the first feature for all examples
print(x[:, 0].shape)  # shape [10]

# For example: Want to access third example in the batch and the first ten features
print(x[2, 0:10].shape)  # shape: [10]

# For example we can use this to, assign certain elements
x[0, 0] = 100

# Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])  # x[indices] = [2, 5, 8]

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])  # Gets second row fourth column and first row first column

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])  # will be [0, 1, 9]
print(x[x.remainder(2) == 0])  # will be [0, 2, 4, 6, 8]

# Useful operations for indexing
print(
    torch.where(x > 5, x, x * 2)
)  # gives [0, 2, 4, 6, 8, 10, 6, 7, 8, 9], all values x > 5 yield x, else x*2
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()  # x = [0, 1, 2, 3, 4]
print(
    x.ndimension()
)  # The number of dimensions, in this case 1. if x.shape is 5x5x5 ndim would be 3
x = torch.arange(10)
print(
    x.numel()
)  # The number of elements in x (in this case it's trivial because it's just a vector)

# ============================================================= #
#                        Tensor Reshaping                       #
# ============================================================= #

x = torch.arange(9)

# Let's say we want to reshape it to be 3x3
x_3x3 = x.view(3, 3)

# We can also do (view and reshape are very similar)
# and the differences are in simple terms (I'm no expert at this),
# is that view acts on contiguous tensors meaning if the
# tensor is stored contiguously in memory or not, whereas
# for reshape it doesn't matter because it will copy the
# tensor to make it contiguously stored, which might come
# with some performance loss.
x_3x3 = x.reshape(3, 3)

# If we for example do:
y = x_3x3.t()
print(
    y.is_contiguous()
)  # This will return False and if we try to use view now, it won't work!
# y.view(9) would cause an error, reshape however won't

# This is because in memory it was stored [0, 1, 2, ... 8], whereas now it's [0, 3, 6, 1, 4, 7, 2, 5, 8]
# The jump is no longer 1 in memory for one element jump (matrices are stored as a contiguous block, and
# using pointers to construct these matrices). This is a bit complicated and I need to explore this more
# as well, at least you know it's a problem to be cautious of! A solution is to do the following
print(y.contiguous().view(9))  # Calling .contiguous() before view and it works

# Moving on to another operation, let's say we want to add two tensors dimensions togethor
x1 = torch.rand(2, 5)
x2 = torch.rand(2, 5)
print(torch.cat((x1, x2), dim=0).shape)  # Shape: 4x5
print(torch.cat((x1, x2), dim=1).shape)  # Shape 2x10

# Let's say we want to unroll x1 into one long vector with 10 elements, we can do:
z = x1.view(-1)  # And -1 will unroll everything

# If we instead have an additional dimension and we wish to keep those as is we can do:
batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(
    batch, -1
)  # And z.shape would be 64x10, this is very useful stuff and is used all the time

# Let's say we want to switch x axis so that instead of 64x2x5 we have 64x5x2
# I.e we want dimension 0 to stay, dimension 1 to become dimension 2, dimension 2 to become dimension 1
# Basically you tell permute where you want the new dimensions to be, torch.transpose is a special case
# of permute (why?)
z = x.permute(0, 2, 1)

# Splits x last dimension into chunks of 2 (since 5 is not integer div by 2) the last dimension
# will be smaller, so it will split it into two tensors: 64x2x3 and 64x2x2
z = torch.chunk(x, chunks=2, dim=1)
print(z[0].shape)
print(z[1].shape)

# Let's say we want to add an additional dimension
x = torch.arange(
    10
)  # Shape is [10], let's say we want to add an additional so we have 1x10
print(x.unsqueeze(0).shape)  # 1x10
print(x.unsqueeze(1).shape)  # 10x1

# Let's say we have x which is 1x1x10 and we want to remove a dim so we have 1x10
x = torch.arange(10).unsqueeze(0).unsqueeze(1)

# Perhaps unsurprisingly
z = x.squeeze(1)  # can also do .squeeze(0) both returns 1x10

# That was some essential Tensor operations, hopefully you found it useful!
