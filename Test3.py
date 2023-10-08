# from locale import normalize
# from Screener import Screener as screener
# from multiprocessing.pool import Pool
# from Data import Data as data
# import numpy as np
# import datetime
# from Screener import Screener as screener
# from scipy.spatial.distance import euclidean, cityblock
# from sfastdtw import sfastdtw
# import time
# from Test import Data,Get
# import os
# import numpy as np
# from sklearn import preprocessing
# import mplfinance as mpf
# import pyts

# from pyts.approximation import SymbolicAggregateApproximation


# if __name__ == '__main__':
# 	ticker = 'JBL' #input('input ticker: ')
# 	dt = '2023-10-03' #input('input date: ')
# 	bars = 10 #int(input('input bars: '))
from soft_dtw_cuda.soft_dtw_cuda import SoftDTW, fit
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

# Aggregate and call backward()
loss.mean().backward()



