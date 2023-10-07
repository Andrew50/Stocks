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
from Test import Data, Get
from Test import Data as data
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
		y = df2
		try:
			for x in df1.np:
				try:
					df1.scores.append( dtw(x, y, method='sakoechiba', options={'window_size': 0.5}))
				except:
					pass
		except:
			pass
			
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

	

	def fetch(ticker,dt = None,bars = 0):
		df = Get(ticker,dt = dt, bars= bars)
		if len(df) < 5: raise IndexError
		df.preload_np(bars)
		return df
	
	def match(ticker,dt,bars,x_list):
		y = Match.fetch(ticker,dt,bars)
		arglist = [[x,y] for x in x_list]
		dfs = data.pool(Match.worker,arglist)
		#secondColumn = np.arange(bars)
		#arglist = [[x,y,ticker,bars, secondColumn] for x,ticker in x_list]
		#scores = Pool().map(Match.worker,arglist)
		#scores = data.pool(Match.worker,arglist)
		sc = []
		for lis in scores:
			for s in lis:
				sc.append(s)
		return sc

	def fetch(ticker,dt = None,bars = 0):
		try: 
			df = data.get(ticker,bars = bars)
			if len(df) < 5: raise IndexError
			df = df.iloc[:,3]
			x = df.to_numpy()
			d = np.zeros((df.shape[0]-1))
			for i in range(len(d)): #add ohlc
				d[i] = x[i+1]/x[i] - 1
			partitions = bars//2
			returns = []
			for i in range(bars,d.shape[0],partitions):
				try:
					df = x[i-bars:i]		
					distance = sfastdtw(df,y,1,euclidean)
					returns.append([ticker,i,distance])
				except:
					pass
		except:
			d = np.zeros((1))
			ticker = 'failed'
		return d, ticker
	def test():
		x_list = data.pool(Match.fetch, ['JBL'])
		ticker = 'JBL'
		dt = '10/3/2023'
		bars = 30
		score = Match.match(ticker, dt, bars, x_list)
if __name__ == '__main__':
	ticker_list = screener.get('full')[:50]
	x_list = data.pool(Match.fetch,ticker_list)
	ticker = 'SMCI' #input('input ticker: ')
	dt = '2023-05-23' #input('input date: ')
	bars = 50 #int(input('input bars: '))
	start = datetime.datetime.now()
	dfs = Match.match(ticker,dt,bars,x_list)
	scores = []
	for df in dfs: scores += df.scores_table()
	
		
	scores.sort(key=lambda x: x[2])

	print(f'completed in {datetime.datetime.now() - start}')
	[print(f'{ticker} {data.get(ticker).index[index]}') for ticker,index,score in scores[:50]]
	discord = Discord(url='https://discord.com/api/webhooks/1160026016555732992/g4idM0ycWJ8mfrtI7Lxr3Hwt4lyzLR7l-8zWPAAY8Zv3wRUdKhveXrrY8tTK2-O3BAgW')
	discord.post()