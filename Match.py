import pathlib, io, shutil, os, math, random, PIL
import pandas as pd
import mplfinance as mpf
import PySimpleGUI as sg
from Data import Data as data
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Screener import Screener as screener
from multiprocessing.pool import Pool
import time, random
import PySimpleGUI as sg




class Train:


	def add_setup(ticker1,dt1,ticker2, dt2, val):
		try: df = pd.read_feather('C:/Stocks/match_training_data.feather')
		except: df = pd.DataFrame()

		add = pd.DataFrame({'ticker1':[ticker1], 'dt1':[dt1], 'ticker2':[ticker2], 'dt2':[dt2], 'val':[val]})

		df = pd.concat([df,add]).reset_index(drop = True)
		df.to_feather('C:/Stocks/match_training_data.feather')

		print(df)


	def run(self):
		with Pool(int(data.get_config('Data cpu_cores'))) as self.pool:
			self.chart_height = data.get_config('Trainer image_box_height')
			self.chart_width = data.get_config('Trainer image_box_width')/2
			sg.theme('DarkGrey')
			self.data = []
			self.i = 0
			self.full_ticker_list = screener.get('full')
			self.init = True
			self.preload_amount = 10
			self.preload(self)
			self.update(self)
			while True:
				self.event, self.values = self.window.read()

				bar = self.data[self.i]
				Train.add_setup(bar[0].ticker,bar[0].dt,bar[1].ticker,bar[1].dt,self.event)
				self.i += 1
				self.preload(self)
				self.update(self)

				

				

			


	def preload(self):

		while len(self.data) < self.i + self.preload_amount: self.data.append([Sample(self.pool),Sample(self.pool)])
		print(self.data)
	def update(self):
		graph1 = sg.Graph(canvas_size = (self.chart_width, self.chart_height), graph_bottom_left = (0, 0), graph_top_right = (self.chart_width, 
				self.chart_height), key = '-chart1-', background_color = 'grey')
		graph2 = sg.Graph(canvas_size = (self.chart_width, self.chart_height), graph_bottom_left = (0, 0), graph_top_right = (self.chart_width, 
				self.chart_height), key = '-chart2-', background_color = 'grey')
		if self.init:
			self.init = False
			layout = [[graph1,graph2],[sg.Button('Yes'),sg.Button('No')]]
			self.window = sg.Window("Match Trainer",layout,margins = (10,10), scaling = data.get_config('Trainer ui_scale'), finalize = True)
			
		self.window['-chart1-'].draw_image(data=self.data[self.i][0].image.get().getvalue(), location=(0, self.chart_height))
		self.window['-chart2-'].draw_image(data=self.data[self.i][1].image.get().getvalue(), location=(0, self.chart_height))

		self.window.maximize()

		

class Sample:

	def plot(df):


		_, axlist = mpf.plot(df, type = 'candle', volume=True, style = mpf.make_mpf_style(marketcolors = mpf.make_marketcolors(up = 'g', down = 'r')), warn_too_much_data=100000, returnfig = True, figratio = (data.get_config('Trainer chart_aspect_ratio')/2,1),
		figscale = data.get_config('Trainer chart_size'), tight_layout = True,axisoff=True)
		ax = axlist[0]
		ax.set_yscale('log')
		ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
		buf = io.BytesIO()
		plt.savefig(buf, bbox_inches='tight',dpi = data.get_config('Trainer chart_dpi'))
		buf.seek(0)
		return buf

	def __init__(self,pool):
		while True:
			try:
				sample_size = 50
				full_ticker_list = screener.get('full')
				self.ticker = full_ticker_list[random.randint(20,len(full_ticker_list) - 1)]
				df = data.get(self.ticker)
				index = random.randint(0,len(df)-1)
		

				self.df = df[index-sample_size:index]
				self.dt = df.index[-1]
				#print(self.df)
				self.image = pool.apply_async(Sample.plot,[self.df])
			except: pass
			else: break


if __name__ == '__main__':
	Train.run(Train)

#class Match:



#	def train():

#		while True






	
	
#	def match(ticker, tf = 'd', dt = None):
#		start_time = time.time()
#		for i in range(1):
#			df = data.get(ticker, tf, dt, bars=40)
#			print(df)
#		print("--- %s seconds ---" % (time.time() - start_time))
		
#if __name__ == "__main__":

#	Match.match('IOT', 'd', '2023-06-02')