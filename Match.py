from locale import normalize
from Screener import Screener as screener
from multiprocessing.pool import Pool
from Data import Data as data
import numpy as np
import datetime
from Screener import Screener as screener
from scipy.spatial.distance import euclidean
from sfastdtw import sfastdtw
import time
from Test import Data,Get


import numpy as np
from sklearn import preprocessing
import pyts

from pyts.approximation import SymbolicAggregateApproximation


class Match:

	def worker(bar):
		x, y,ticker,bars, = bar
		x = preprocessing.normalize([x])
		t = preprocessing.normalize([y])
		

		# Perform SAX on the normalized data
		#sax_chart_1 = pyts.approximation.SymbolicAggregateApproximation(n_bins=8).transform(normalized_chart_1)
		#sax_chart_2 = pyts.approximation.SymbolicAggregateApproximation(n_bins=8).transform(normalized_chart_2)
		
		transformer = SymbolicAggregateApproximation()
		x = transformer.transform(x)
		y =  transformer.transform(y)

		# Calculate the DTW distance between the two SAX representations
		dtw_distance = pyts.metrics.dtw(x,t)
		

		# Print the DTW distance
		print('DTW distance:', dtw_distance)

		return [ticker,0,dtw_distance]

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

	def match(ticker,dt,bars,x_list):
		y,_ = Match.fetch(ticker,dt,bars)
		arglist = [[x,y,ticker,bars] for x,ticker in x_list]
		scores = data.pool(Match.worker,arglist)
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
	ticker_list = screener.get('full')[:8000]
	x_list = data.pool(Match.fetch,ticker_list)
	
	ticker = 'SMCI' #input('input ticker: ')
	dt = '22023-05-23' #input('input date: ')
	bars = 50 #int(input('input bars: '))
	start = datetime.datetime.now()
	scores = Match.match(ticker,dt,bars,x_list)

		print(f'completed in {datetime.datetime.now() - start}')
		scores.sort(key=lambda x: x[2])
		print(scores[:10])
		[print(f'{ticker} {data.get(ticker).index[index]}') for ticker,index,score in scores[:50]]