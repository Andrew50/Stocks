from Screener import Screener as screener
from multiprocessing.pool import Pool
from Data import Data as data
import numpy as np
import datetime
from Screener import Screener as screener
from scipy.spatial.distance import euclidean, cityblock
from sfastdtw import sfastdtw
import time
class Match:

	def worker(bar):
		x, y,ticker,bars, secondColumn = bar
		partitions = bars//2
		returns = []
		
		for i in range(bars,x.shape[0],partitions):
			try:
				df = x[i-bars:i]		
				df = np.column_stack((df, secondColumn))	
				distance = sfastdtw(df,y,1,cityblock)
				returns.append([ticker,i,distance])
			except TimeoutError: pass
		return returns 

	def match(ticker,dt,bars,x_list):
		y,_ = Match.fetch(ticker,dt,bars)
		y = np.column_stack((y, np.arange(bars-1)))
		secondColumn = np.arange(bars)
		arglist = [[x,y,ticker,bars, secondColumn] for x,ticker in x_list]
		#scores = Pool().map(Match.worker,arglist)
		scores = data.pool(Match.worker,arglist)
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
	
	while True:
		ticker = input('input ticker: ')
		dt = input('input date: ')
		bars = int(input('input bars: '))
		start = datetime.datetime.now()
		scores = Match.match(ticker,dt,bars,x_list)

		print(f'completed in {datetime.datetime.now() - start}')
		scores.sort(key=lambda x: x[2])
		
		[print(f'{ticker} {data.get(ticker).index[index]}') for ticker,index,score in scores[:50]]