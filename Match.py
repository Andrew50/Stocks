from locale import normalize
from multiprocessing.pool import Pool
from Data import Data as data
import numpy as np
import datetime
from Screener import Screener as screener
import time
from Test import Data
from discordwebhook import Discord
import numpy as np
from sklearn import preprocessing
import mplfinance as mpf
import torch

#from soft_dtw_cuda.soft_dtw_cuda import SoftDTW

## Create the sequences
#batch_size, len_x, len_y, dims = 8, 15, 12, 5
#x = torch.rand((batch_size, len_x, dims), requires_grad=True)
#y = torch.rand((batch_size, len_y, dims))
## Transfer tensors to the GPU
#x = x.cuda()
#y = y.cuda()

## Create the "criterion" object
#sdtw = SoftDTW(use_cuda=True, gamma=0.1)

## Compute the loss value
#loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()

## Aggregate and call backward()
#loss.mean().backward()
from Dtw import dtw as dtw
			
class Match:
	
	def fetch(ticker,bars=10,dt = None):
		
		tf = 'd'
		if dt != None:
			df = Data(ticker,tf,dt,bars = bars+1)
		else:
			df = Data(ticker,tf)
		df.np(bars,True)
		return df

	def worker(bar):
		df1, y = bar
		lis = []
		#print(f'{df1.np[0].shape} , {y.shape}')
		for x in df1.np:
			print(f'{x.shape} , {y.shape}')
			distance, path = dtw(x, y)
			lis.append(distance)
		setattr(df1,'scores',lis)
		return df1
	
	def match(ticker,dt,bars,dfs):
		y = Match.fetch(ticker,bars,dt).np[0]
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
	print(y)
	if True:
		ticker_list = screener.get('full')[:10]
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
