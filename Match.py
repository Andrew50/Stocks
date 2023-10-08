from locale import normalize
from Screener import Screener as screener
from multiprocessing.pool import Pool
from Data import Data as data
import numpy as np
import datetime
from Screener import Screener as screener
from scipy.spatial.distance import euclidean, cityblock
from sfastdtw import sfastdtw
import time
from Test import Data
from discordwebhook import Discord
import numpy as np
from sklearn import preprocessing
import mplfinance as mpf
import pyts
from pyts.approximation import SymbolicAggregateApproximation
from pyts.metrics import dtw
import pyts.approximation as sax


import numpy as np
#import cupy as cp

# Define your time series data
# time_series1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
# time_series2 = np.array([2, 3, 4, 5, 6], dtype=np.float32)

# # Transfer data to GPU
# time_series1_gpu = cp.array(time_series1)
# time_series2_gpu = cp.array(time_series2)

# Function to compute DTW using CuPy on GPU


# Calculate DTW using GPU
# dtw_distance = gpu_dtw(time_series1_gpu, time_series2_gpu)

# print(f"GPU-accelerated DTW Distance: {dtw_distance}")
'''import numpy as np
import cupy as cp
import numba
from numba import cuda

@cuda.jit
def dtw_kernel(query, reference, cost_matrix):
    i, j = cuda.grid(2)
    if i < cost_matrix.shape[0] and j < cost_matrix.shape[1]:
        # Calculate the absolute difference between query[i] and reference[j]
        diff = cp.abs(query[i] - reference[j])
        # Compute the local cost (Euclidean distance)
        local_cost = diff ** 2
        # Update the cost_matrix with the minimum cost path
        if i == 0 and j == 0:
            cost_matrix[i, j] = local_cost
        elif i == 0:
            cost_matrix[i, j] = local_cost + cost_matrix[i, j - 1]
        elif j == 0:
            cost_matrix[i, j] = local_cost + cost_matrix[i - 1, j]
        else:
            cost_matrix[i, j] = local_cost + min(
                cost_matrix[i - 1, j],
                cost_matrix[i, j - 1],
                cost_matrix[i - 1, j - 1]
            )

def dtw(query, reference):
    query = cp.array(query)
    reference = cp.array(reference)
    n, m = len(query), len(reference)
    
    # Initialize the cost matrix with zeros
    cost_matrix = cp.zeros((n, m), dtype=cp.float32)

    # Define the thread grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (m + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    dtw_kernel[blocks_per_grid, threads_per_block](query, reference, cost_matrix)

    # Return the DTW distance
    return cp.sqrt(cost_matrix[-1, -1])

# Example usage
# query_sequence = [1.0, 2.0, 3.0, 4.0]
# reference_sequence = [2.0, 3.0, 3.0, 4.0]
# distance = dtw(query_sequence, reference_sequence)
# print(f"DTW Distance: {distance}")'''
			
class Match:
	def gpu_dtw(x, y):
    # Calculate pairwise distances between elements of x and y on the GPU
		distance_matrix = cp.abs(x[:, None] - y[None, :])
    
		# Initialize the accumulated cost matrix
		accumulated_cost = cp.empty_like(distance_matrix)
    
		# Initialize the first row and column of the accumulated cost matrix
		accumulated_cost[0, 0] = distance_matrix[0, 0]
    
		for i in range(1, x.shape[0]):
			accumulated_cost[i, 0] = distance_matrix[i, 0] + accumulated_cost[i-1, 0]
    
		for j in range(1, y.shape[0]):
			accumulated_cost[0, j] = distance_matrix[0, j] + accumulated_cost[0, j-1]
    
		# Calculate the accumulated cost matrix on the GPU
		for i in range(1, x.shape[0]):
			for j in range(1, y.shape[0]):
				accumulated_cost[i, j] = distance_matrix[i, j] + cp.min([accumulated_cost[i-1, j], accumulated_cost[i, j-1], accumulated_cost[i-1, j-1]])
    
		# Return the DTW distance
		return accumulated_cost[-1, -1]
	
	def fetch(ticker,bars=10,dt = None):
		tf = 'd'
		if dt != None:
			df = Data(ticker,tf,dt,bars = bars)
		else:
			df = Data(ticker,tf)
		df.np(bars,True)
		return df

	def worker(bar):
		df1, y = bar
		lis = []
		#print(f'{df1.np[0].shape} , {y.shape}')
		for x in df1.np:
			lis.append(gpu_dtw(x,y))
		setattr(df1,'scores',lis)
		return df1
	
	def match(ticker,dt,bars,dfs):
		y = Match.fetch(ticker,bars,dt).np[00]
		print(y)
		arglist = [[x,y] for x in dfs]
		dfs = data.pool(Match.worker,arglist)
		return dfs
	
	def initiate(ticker, dt, bars): 
		ticker_list = screener.get('full')[:500]
		dfs = data.pool(Match.fetch,ticker_list)
		start = datetime.datetime.now()
		dfs = Match.match(ticker,dt,bars,dfs)
		scores = []
		for df in dfs:
			lis = df.get_scores()
			scores += lis
		scores.sort(key=lambda x: x[2])
		print(f'completed in {datetime.datetime.now() - start}')
		return scores[:20]
		for ticker,index,score in scores[:20]:
			print(f'{ticker} {Data(ticker).df.index[index]}')
if __name__ == '__main__':
	ticker = 'JBL' #input('input ticker: ')
	dt = '2023-10-03' #input('input date: ')
	bars = 10 #int(input('input bars: '))
	y = Match.fetch(ticker,bars,dt).np
	if True:
		ticker_list = screener.get('full')[:1000]
		dfs = data.pool(Match.fetch,ticker_list)
		start = datetime.datetime.now()
		dfs = Match.match(ticker,dt,bars,dfs)
		scores = []
		for df in dfs:
			lis = df.get_scores()
			scores += lis
		scores.sort(key=lambda x: x[2])
		print(f'completed in {datetime.datetime.now() - start}')
		for ticker,index,score in scores[:20]:
			print(f'{ticker} {Data(ticker).df.index[index]}')
		

						#lis.append(pyts.metrics.dtw(x,y))
				#lis.append(sax(x, y))
				#lis.append( dtw(x, y, method='sakoechiba', options={'window_size': 0.5}))
