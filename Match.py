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
from Test import Data as data
DF = Data.DF
from discordwebhook import Discord

import numpy as np
from sklearn import preprocessing
import mplfinance as mpf
import pyts

from pyts.approximation import SymbolicAggregateApproximation
from pyts.metrics import dtw

			
class Match:

	def worker(bar):
		df1, df2 = bar
		lis = []
		try:
			y = df2.np
			for x in df1.np:
				try:
					lis.append( dtw(x, y, method='sakoechiba', options={'window_size': 0.5}))
				except:
					pass
		except:
			pass
		df1.scores = lis
		
		return df1
		#rint('DTW distance:', dtw_distance)
		# x, y,ticker,bars, secondColumn = bar
		# partitions = bars//2
		# returns = []
		
		# for i in range(bars,x.shape[0],partitions):
		# 	try:
		# 		df = x[i-bars:i]		
		# 		df = np.column_stack((df, secondColumn))
		# 		distance = sfastdtw(df,y,1,euclidean)
		# 		returns.append([ticker,i,distance])
		# 	except: pass
		# return returns 


	
	def match(ticker,dt,bars,dfs):
		y = Match.fetch(ticker,dt,bars)
		y.preload_np(bars,True)
		arglist = [[x,y] for x in dfs]
		dfs = data.pool(Match.worker,arglist)
		#secondColumn = np.arange(bars)
		#arglist = [[x,y,ticker,bars, secondColumn] for x,ticker in x_list]
		
		return dfs

	def fetch(ticker,dt = None,bars = 0):
		tf = 'd'
		df = DF(ticker,tf,dt,bars = bars)
		
		if len(df) < 5: raise IndexError
		df.preload_np(bars,True)
		
		return df

if __name__ == '__main__':
	ticker_list = screener.get('full')[:20]
	

	dfs = data.pool(Match.fetch,ticker_list)
	ticker = 'SMCI' #input('input ticker: ')
	dt = '2023-05-23' #input('input date: ')
	bars = 50 #int(input('input bars: '))
	
	start = datetime.datetime.now()
	dfs = Match.match(ticker,dt,bars,dfs)
	scores = []
	print(dfs)
	for df in dfs: scores += df.scores
	
		
	scores.sort(key=lambda x: x[2])

	print(f'completed in {datetime.datetime.now() - start}')
	[print(f'{ticker} {data.get(ticker).index[index]}') for ticker,index,score in scores[:50]]
	discord = Discord(url='https://discord.com/api/webhooks/1160026016555732992/g4idM0ycWJ8mfrtI7Lxr3Hwt4lyzLR7l-8zWPAAY8Zv3wRUdKhveXrrY8tTK2-O3BAgW')
	discord.post()