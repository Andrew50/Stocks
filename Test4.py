
from soft_dtw_cuda import SoftDTW
import torch

# Create the sequences
batch_size, len_x, len_y, dims = 8, 15, 12, 5
x = torch.rand((batch_size, len_x, dims), requires_grad=True)
y = torch.rand((batch_size, len_y, dims))
# Transfer tensors to the GPU
x = x.cuda()
y = y.cuda()

# Create the "criterion" object
sdtw = SoftDTW(use_cuda=True, gamma=0.1)

# Compute the loss value
loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()
print(loss)

# Aggregate and call backward()
loss.mean().backward()



#from soft_dtw_cuda import 

## Transfer data from CPU to GPU
#data = list(range(4))
#query = list(range(9))
#data_gpu = cp.asarray(data)  # data is your time series data

## Define a GPU kernel for DTW calculation
#@cp.fuse()
#def dtw_kernel(data_gpu, query):
#    # Implement DTW calculations here

#    return cp.zeros(9,9,9)

## Launch the GPU kernel

#results_gpu = dtw_kernel(data_gpu, query)

## Transfer results from GPU to CPU
#results_cpu = cp.asnumpy(results_gpu)

## Process results or perform other tasks

##from dtw_gpu.cudadtw import dtw_1D_jit2 as dtw#from numba import jit
##from Test import Data
#### Define a simple Python function
###def add(a, b):
###    result = a + b
###    return result
##df = Data('NFLX')
##df.np(50,True)

##y = df.np[0]
##df = Data('AAPL')
##df.np(50,True)
##for x in df.np:
##    print(dtw(x,y))

### Use the @jit decorator to compile the function with Numba
##@jit
##def jit_add(a, b):
##    result = a + b
##    return result

### Test the original Python function
##x = 10
##y = 20
##print("Python function result:", add(x, y))

### Test the Numba-compiled function
##print("Numba-compiled function result:", jit_add(x, y))