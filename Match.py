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

			
class Match:
	
	def fetch(ticker,bars=50,dt = None):
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
		for x in df1.np:
			lis.append(sfastdtw(x,y,1,euclidean))
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
	for ticker,index,score in scores[:20]:
		print(f'{ticker} {Data(ticker).df.index[index]}')
		

						#lis.append(pyts.metrics.dtw(x,y))
				#lis.append(sax(x, y))
				#lis.append( dtw(x, y, method='sakoechiba', options={'window_size': 0.5}))
