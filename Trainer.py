



from Data import Data as data
import matplotlib.pyplot as plt
import random
import mplfinance as mpf
import pandas as pd
import PySimpleGUI as sg
from multiprocessing.pool import Pool
from PIL import Image
import time
import sys
import pathlib
import datetime
import io
import matplotlib.ticker as mticker
import shutil
import os
from tensorflow.keras.models import load_model
import math
import sys
from Screener import Screener as screener



class Trainer:

	def run(self):
		with Pool(6) as self.pool:
			self.menu_list = ['Trainer','Validator','Tuner','Tester','Manual']
			self.full_setup_list = data.get_setups_list()
			self.chart_edge_offset = data.get_config('Trainer chart_edge_offset')
			self.current_menu = 'Trainer'
			self.trainer_cutoff = 60
			self.selected_trainer_index = 0
			self.current_setup = self.full_setup_list[0]
			self.current_tf = self.current_setup.split('_')[0]
			self.full_ticker_list = pd.read_feather('C:/Stocks/sync/files/current_scan.feather')['Ticker'].to_list()
			self.init = True
			self.update(self)
			while True:
				self.event, self.values = self.window.read()
				if self.event in self.menu_list:  ##
					self.current_menu = self.event
					self.init = True
					self.window.close()
					self.update(self)

				elif self.current_menu == 'Trainer':
					if self.event == 'Skip No' or self.event == 'Use No' :
						self.save_training_data(self)
						self.i+= 1
						self.update(self)
						self.preload(self)
					elif self.event == 'right_button' or self.event == 'left_button' or self.event == '-chart-' or self.event == 'center_button':
						self.click(self)
					elif self.event in self.tf_list:##
						self.current_tf = self.event
						self.init = True
						self.update(self)
					else:
						self.log(self)

				elif self.current_menu == 'Validator' or self.current_menu == 'Tuner':
					if self.event == 'right_button' or self.event == 'left_button':
						if self.event == 'right_button': val = 1
						else: val = 0
						bar = self.chart_info[self.i]
						data.add_setup(bar[1],bar[0].index[-1],self.current_seutp,0,val)
					self.i += 1
					self.update(self)
					self.preload(self)


				elif self.event == 'Enter':
					ticker = self.window.values['-input_ticker-']
					datetime = self.window.values['-input_datetime-']
					setup = self.window.values['-input_setup-']
					if self.current_menu == 'Manual': data.add_setup(ticker,datetime,setup,1,1)
					else:
						df = data.get(ticker,setup.split('_')[0],datetime)
						self.window['-score-'].update(data.score([df],model_type = self.current_setup))

						
	def save_training_data(self):
		df1 = self.chart_info[self.i][0]
		ticker = self.chart_info[self.i][1]
		ii = 0
		for s in self.current_setup_list:
			df = pd.DataFrame()
			if self.event == 'Use No':
				df['datetime'] = df1.index
				df['ticker'] = ticker
				df['value'] = 0
				df['required'] = 0
				df = df[self.cutoff:]
			for bar in self.current_setups:
				if bar[1] == s:
					self.setup_count_stats[ii] += 1
					index = bar[3]
					df.iat[index,2] = 1
					if index <= self.trainer_cutoff:
						df2 = pd.DataFrame({ 'ticker':[ticker],'datetime':[df1.index[index]], 'value':[1], 'required':[0]})
						df = pd.concat([df,df2]).reset_index(drop = True)
			add = df[['ticker','datetime','value','required']]
			ident = Data.identify()
			path = 'C:/Stocks/sync/database/' + ident + '_' + setup + '.feather'
			try: df = pd.read_feather(path)
			except FileNotFoundError: df = pd.DataFrame()
			df = pd.concat([df,add]).reset_index(drop = True)
			df.to_feather(path)

	def click(self):
		df = self.chart_info[self.i][0]
		ticker = self.chart_info[self.i][1]
		chart_size = self.x_size - 20
		if self.event == '-chart-':
			x = self.values['-chart-'][0]
			self.y = self.values['-chart-'][1]
			chart_click = x - 10
			percent = chart_click/chart_size
			self.selected_trainer_index = math.floor(len(df) * percent)
			if self.selected_trainer_index <= -1:
				self.selected_trainer_index = 0
			if self.selected_trainer_index >= len(df):
				self.selected_trainer_index = len(df) - 1
			try:
				self.date = df.index[self.selected_trainer_index]
			except:
				return
		else:
			if self.event == 'right_button' and self.selected_trainer_index < len(df) - 1:
				self.selected_trainer_index += 1
			elif self.event == 'left_button' and self.selected_trainer_index > 0:
				self.selected_trainer_index -= 1
			self.y = self.chart_height - 80
		
		round_x = int((self.selected_trainer_index + 1)/(len(df)) * (self.x_size - (self.chart_edge_offset * 2))) + self.chart_edge_offset - int((chart_size/len(df))/2)
		self.window['-chart-'].MoveFigure(self.select_line,round_x - self.select_line_x,0)
		self.select_line_x = round_x

	def log(self):
		if self.event == 'center_button':
			for i in range(len(self.current_setups)-1,-1,-1):
				if self.current_setups[i][0] == self.selected_trainer_index:
					for k in range(2):
						self.window['-chart-'].MoveFigure(self.current_setups[i][2][k],5000,0)

					del self.current_setups[i]
			self.window.refresh()
		else:
			try:
				i = int(self.event)
				setup = self.current_setup_list[i-1]
			except TimeoutError:
				return
			x = self.select_line_x
			y = self.y - 50
			if y < 10:
				y = 110
			text = self.window['-chart-'].draw_text( setup, (x,y) , color='black', font=None, angle=0, text_location='center_button')
			line = self.window['-chart-'].draw_line((self.select_line_x,0), (self.select_line_x,self.chart_height), color='black', width=1)
			self.current_setups.append([self.selected_trainer_index,setup,[line,text]])
			self.y -= 35

	def update(self):
		if self.init:
			self.chart_info = []

			self.i = 0
			if os.path.exists('C:/Stocks/local/trainer/charts'):
				shutil.rmtree('C:/Stocks/local/trainer/charts')
			os.mkdir('C:/Stocks/local/trainer/charts')
			self.chart_info = []
			self.chart_height = data.get_config('Trainer ch')
			self.chart_width = data.get_config('Trainer cw')
			data.combine_training_data()
			self.full_setup_list = data.get_setups_list()
			self.current_setup_list = [s for s in self.full_setup_list if self.current_tf in s]
			self.tf_list = [*set([s.split('_')[0] for s in self.full_setup_list])]
			sg.theme('DarkGrey')
			if self.current_menu == 'Trainer':
				def setup_len(s):
					df = pd.read_feather('C:/Stocks/local/data/'+s+'.feather')
					return len(df[df['value'] == 1])
				self.setup_count_stats = [setup_len(s) for s in self.full_setup_list]
				graph = sg.Graph( canvas_size=(self.chart_width, self.chart_height), graph_bottom_left=(0, 0), graph_top_right=(self.chart_width, 
				self.chart_height), key='-chart-', change_submits=True, background_color='grey', drag_submits=False)
				layout = [[graph],
				[sg.Text(key = '-setup_count_stats-')],
				[sg.Button('Use No'), sg.Button('Skip No')] + [sg.Button(tf) for tf in self.tf_list]]
			elif self.current_menu == 'Validator':
				layout = [[sg.Image(key = '-chart-')],
				[sg.Button(s) for s in self.full_setup_list],
				[ sg.Text(self.current_setup, key = '-curent_setup-'), sg.Text(key = '-counter-')]]
			elif self.current_menu == 'Tuner':
				layout = [[sg.Image(key = '-chart-')],
				[sg.Button(s) for s in self.full_setup_list],
				[sg.Text(self.current_setup, key = '-current_seutp-')],]
			elif self.current_menu == 'Tester' or self.menu == 'Manual':
				layout = [[sg.Text('Ticker'),sg.InputText(key = '-input_ticker-')],
				[sg.Text('Date'),sg.InputText(key = '-input_date-')],
				[sg.Text('Setup'),sg.InputText(key = '-input_setup-')],
				[sg.Button('Enter')]]
				if self.current_menu == 'Tester': layout.append(sg.Text(key = '-score-'))
			layout.append([sg.Button(m) for m in self.menu_list])
			self.window = sg.Window(self.current_menu, layout,margins = (10,10),scaling=data.get_config('Trainer ui'),finalize = True)
			self.init = False
			for k, v in [['<q>', '1'],['<w>', '2'],['<e>', '3'],['<a>', '4'],['<s>', '5'],['<d>', '6'],
				['<z>', '7'],['<x>', '8'],['<c>', '9'],['<p>', 'right_button'],['<i>', 'left_button'],['<o>', 'center_button']]:
				self.window.bind(k,v)
			self.preload(self)

			

		self.window.maximize()
		if self.current_menu == 'Trainer' or self.menu == 'Validatorf' or self.menu == 'Tuner':
			self.window['-chart-'].erase()
			while True:
				try:
					image1 = Image.open(r'C:/Stocks/local/trainer/charts/' + str(self.i) + '.png')
					bio1 = io.BytesIO()
					image1.save(bio1, format='PNG')
					self.window['-chart-'].draw_image(data=bio1.getvalue(), location=(0, self.chart_height))

					break
				except: pass
		if self.current_menu == 'Trainer':
			self.x_size = image1.size[0]
			self.select_line_x = -100
			self.select_line = self.window['-chart-'].draw_line((self.select_line_x,0), (self.select_line_x,self.chart_height), color='green', width=1)
			df = self.chart_info[self.i][0]
			chart_size = self.x_size - 20
			round_x = int((self.trainer_cutoff)/(len(df)) * (chart_size)) + 10 - int((chart_size/len(df))/2)
			self.window['-chart-'].draw_line((round_x,0), (round_x,self.chart_height), color='red', width=2)
			self.current_setups = []
	def preload(self):

		if self.i == 0:
			index_list = list(range(10))
		else:
			index_list = [self.i + 9]
		for i in index_list:
			if self.current_menu == 'Trainer':
				while True:
					try:
						ticker = self.full_ticker_list[random.randint(0,len(self.full_ticker_list)-1)]
						df = data.get(ticker,tf = self.current_tf)
						date_list = df.index.to_list()
						datetime = date_list[random.randint(0,len(date_list) - 1)]
						index = data.findex(df,datetime)
						left = index - 300
						if left < 0:left = 0
						df = df[left:index + 1]
						dolVol, adr, pmvol = screener.get_requirements(df,len(df) - 1,2,ticker)
						if self.current_tf == 'h':
							if dolVol > 2000000:
								break
						else:
							if dolVol > 2000000:
								break
					except TimeoutError as e:
						print(e)
				title = ''
			elif self.current_menu == 'Validator':
				ticker = self.setups_df.iat[self.i,0]
				datetime = self.setups_df.iat[self.i,1]
				df = data.get(ticker,self.current_setup.split('_')[0],datetime,250)
				title = self.current_setup

			elif self.current_menu == 'Tuner' and len(self.chart_info) < i + 10:
				while True:
					ticker = self.full_ticker_list[random.randint(0,len(self.full_ticker_list)-1)]
					df = data.get(ticker,tf = self.current_tf,bars = 10000)
					dfs = []
					for ii in range(len(df)):
						dfs.append(df[:ii])
					title = str(score)
					setups = data.score(dfs)
			self.chart_info.append([df,ticker,title,i])
		arglist = [self.chart_info[i] for i in index_list]
		self.pool.map_async(Trainer.plot,arglist)

	def plot(bar):
		df = bar[0]
		title = bar[2]
		i = bar[3]
		p = pathlib.Path('C:/Stocks/local/trainer/charts') / (str(i) + '.png')
		try:
			mc = mpf.make_marketcolors(up='g',down='r')
			s  = mpf.make_mpf_style(marketcolors=mc)
			fig, axlist = mpf.plot(df, type='candle', volume=True  ,                          
			style=s, warn_too_much_data=100000,returnfig = True,figratio = (data.get_config('Trainer fr'),1),
			figscale=data.get_config('Trainer fs'), panel_ratios = (5,1), title = title, tight_layout = True,axisoff=True)
			ax = axlist[0]
			ax.set_yscale('log')
			ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
			plt.savefig(p, bbox_inches='tight',dpi = data.get_config('Trainer dpi'))
		except Exception as e:
			print(e)
			shutil.copy(r'C:\Stocks\sync\files\blank.png',p)



if __name__ == '__main__':
	Trainer.run(Trainer)