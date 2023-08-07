

from discordwebhook import Discord
import pathlib
import time 
import selenium
import selenium.webdriver as webdriver
from selenium.webdriver.firefox.options import Options 
from selenium.webdriver.common.by import By 
import pandas as pd
import datetime
from Study import Study as study
import statistics
from Data import Data as data
from tqdm import tqdm
import mplfinance as mpf
import os
from tensorflow.keras.models import load_model


class Screener:

	def run(date = None,days = 1, ticker = None, tf = 'd',browser = None, fpath = None):
		path = 0
		data.consolidate_setups()
		if ticker == None:
			if date == None: 
				path = 0
				ticker_list = Screener.get('full').index.tolist()
			else:
				if 'd' in tf or 'w' in tf: 
					ticker_list = Screener.get('current').index.tolist()
					path = 1
				else: 
					ticker_list = Screener.get('intraday').index.tolist()
					path = 2
		else:
			path = 1
			ticker_list = [ticker]
		if fpath != None: path = fpath
		num_packages = int(data.get_config('Data cpu_cores'))
		package = [[[],date,tf,path] for _ in range(num_packages)]
		ii = 0
		for ticker in ticker_list:
			package[ii][0].append(ticker)
			ii += 1
			if ii == num_packages:
				ii = 0
		data.pool(Screener.screen, package)
		data.consolidate_setups()


	def screen(container):
		tickers = container[0]
		
		date = container[1]
		tf = container[2]
		path = container[3]
		setup_list = data.get_config('Screener active_setup_list').split(',')
		setup_list = [s for s in setup_list if tf in s]
		threshold = .25
		dfs = []
		for ticker in tickers:
			dfs.append(data.get(ticker,tf,date))
		for setup_type in setup_list:
			setups = data.score(dfs,setup_type,threshold)
			for ticker, z, df in setups:
				Screener.log(ticker,z,df,tf,path,setup_type)
  
	

	def log(ticker,z, df, tf,  path, setup_type):
		z = round(z * 100)
		if path == 3:
			print(f'{ticker} {df.index[-1]} {z} {setup_type} ')
		elif path == 2:
			mc = mpf.make_marketcolors(up='g',down='r')
			s  = mpf.make_mpf_style(marketcolors=mc)
			ourpath = pathlib.Path("C:/Screener/tmp")/ 'test.png'
			df = df[-100:]
			#df.set_index('datetime', inplace = True)
			mpf.plot(df, type='candle', mav=(10, 20), volume=True, title=f'{ticker}, {setup_type}, {z}, {tf}', style=s, savefig=ourpath)
			discordintraday = Discord(url="https://discord.com/api/webhooks/1071667193709858847/qwHcqShmotkEPkml8BSMTTnSp38xL1-bw9ESFRhBe5jPB9o5wcE9oikfAbt-EKEt7d3c")

			discordintraday.post(file={"test": open('tmp/test.png', "rb")})
		elif path == 1:
			d = "C:/Stocks/local/screener/subsetups/current_" + str(os.getpid()) + ".feather"
			try: setups = pd.read_feather(d)
			except: setups = pd.DataFrame()
			add =pd.DataFrame({'ticker': [ticker],
					'datetime':[df.index[-1]],
					'setup': [setup_type],
					'z':[z]})
			setups = pd.concat([setups,add]).reset_index(drop = True)
			setups.to_feather(d)
		elif path == 0:
			d = "C:/Stocks/local/screener/subsetups/historical_" + str(os.getpid()) + ".feather"
			try: setups = pd.read_feather(d)
			except: setups = pd.DataFrame()
			add = pd.DataFrame({'ticker':[ticker],
						'datetime': [df.index[-1]],
						'setup': [setup_type],
						'z': [z],
						'sub_setup':[setup_type],
						'pre_annotation': [""],
						'post_annotation': [""]
						})
			setups = pd.concat([setups,add]) .reset_index(drop = True)
			setups.to_feather(d)



	def get(type = 'full',  refresh = False, browser = None):

		def start_firefox():
			
			options = Options()
			options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"
			options.headless = True
			user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'
			FireFoxDriverPath = os.path.join(os.getcwd(), 'Drivers', 'geckodriver.exe')
			FireFoxProfile = webdriver.FirefoxProfile()
			FireFoxProfile.set_preference("General.useragent.override", user_agent)
			browser = webdriver.Firefox(options=options, executable_path=FireFoxDriverPath)
			browser.implicitly_wait(7)
			browser.set_window_size(2560, 1440)
			url = "https://www.tradingview.com/screener/"
			browser.get(url)
			time.sleep(1.5)
			browser.find_element(By.XPATH, '//button[@aria-label="Open user menu"]').click()
			time.sleep(1)
			browser.find_element(By.XPATH, '//button[@data-name="header-user-menu-sign-in"]').click()
			time.sleep(1)
			browser.find_element(By.XPATH, '//button[@class="emailButton-nKAw8Hvt light-button-bYDQcOkp with-start-icon-bYDQcOkp variant-secondary-bYDQcOkp color-gray-bYDQcOkp size-medium-bYDQcOkp typography-regular16px-bYDQcOkp"]').click()
			browser.find_element(By.XPATH, '//input[@name="id_username"]').send_keys("cs.benliu@gmail.com")
			time.sleep(0.5)
			password = browser.find_element(By.XPATH, '//input[@name="id_password"]').send_keys("tltShort!1")
			time.sleep(0.5)
			login_button = browser.find_element(By.XPATH, '//button[@class="submitButton-LQwxK8Bm button-D4RPB3ZC size-large-D4RPB3ZC color-brand-D4RPB3ZC variant-primary-D4RPB3ZC stretch-D4RPB3ZC"]').click()
			time.sleep(3)
			browser.refresh();
			time.sleep(5)
			browser.find_element(By.XPATH, '//div[@data-name="screener-field-sets"]').click()
			time.sleep(0.1)
			browser.find_element(By.XPATH, '//div[@title="Python Screener"]').click()
			filter_tab = browser.find_element(By.XPATH, '//div[@class="tv-screener-sticky-header-wrapper__fields-button-wrap"]')
			try: filter_tab.click()
			except: pass
			time.sleep(0.5)
			browser.find_element(By.XPATH, '//div[@class="tv-screener__standalone-title-wrap"]').click()
			time.sleep(0.5) 
			browser.find_element(By.XPATH, '//div[@data-name="screener-filter-sets"]').click()
			time.sleep(0.25)
			browser.find_element(By.XPATH, '//span[@class="js-filter-set-name"]').click()
			time.sleep(0.25)
			browser.find_element(By.XPATH, '//div[@data-field="relative_volume_intraday.5"]').click()
			return browser

		def get_full(refresh):
			df1 = pd.read_feather("C:/Stocks/sync/full_scan.feather")
			if not data.indentify() == 'desktop' or not refresh: return df1.set_index('Ticker')
			df2 = pd.read_feather("C:/Screener/sync/current_scan.feather")
			df3 = pd.concat([df1,df2]).drop_duplicates(subset = ['Ticker'])		
			removelist = []
			full = df3['Ticker'].to_list()
			current = df1['Ticker'].to_list()
			for ticker in full:
				if ticker not in current:
					ticker = str(ticker)
					if not os.path.exists(data.data(ticker,'1min')):
						removelist.append(ticker)
			df3 = df3.set_index('Ticker')
			for ticker in removelist:
				ticker = str(ticker)
				df3 = df3.drop(index = ticker)
			try: df3 = df3.reset_index()
			except: pass
			df3.to_feather("C:/Screener/sync/full_ticker_list.feather")
			return df3.set_index('Ticker')

		def get_current(refresh,browser = None):
			if not refresh:
				try: return pd.read_feather("C:/Stocks/sync/files/current_scan.feather").set_index('Ticker')
				except FileNotFoundError: pass
			today = str(datetime.date.today())
			try:
				if(browser == None):
					browser = start_firefox()
				time.sleep(0.5) 
				browser.find_element(By.XPATH, '//div[@data-name="screener-filter-sets"]').click()
				time.sleep(0.25)
				browser.find_element(By.XPATH, '//span[@class="js-filter-set-name"]').click()
				time.sleep(0.25)
				browser.find_element(By.XPATH, '//div[@data-field="relative_volume_intraday.5"]').click()
				browser.find_element(By.XPATH, '//div[@data-name="screener-export-data"]').click()
			except TimeoutError as e:
				print(e)
				print('manual csv fetch required')
			found = False
			while True:
				path = r"C:\Downloads"
				dir_list = os.listdir(path)
				for direct in dir_list:
					if today in direct:
						downloaded_file = path + "\\" + direct
						found = True
						break
				if found:
					break
			screener_data = pd.read_csv(downloaded_file)
			os.remove(downloaded_file)
			for i in range(len(screener_data)):
				if str(screener_data.iloc[i]['Exchange']) == "NYSE ARCA": screener_data.at[i, 'Exchange'] = "AMEX"
				if screener_data.iloc[i]['Pre-market Change'] is None: screener_data.at[i, 'Pre-market Change'] = 0
			screener_data = screener_data[~screener_data['Ticker'].str.contains('/')]
			screener_data = screener_data[~screener_data['Ticker'].str.contains('.')]
			screener_data.to_feather(r"C:\Screener\sync\current_scan.feather")
			return screener_data.set_index('Ticker'), browser

		def get_intraday(browser = None):
			while True:
				try:
					df, browser = get_current(True,browser)
					break
				except:
					try: browser.find_element(By.XPATH, '//button[@class="close-button-FuMQAaGA closeButton-zCsHEeYj defaultClose-zCsHEeYj"]').click()
					except AttributeError: print('tried closing popup')
					except selenium.common.exceptions.NoSuchElementException: print('tried closing popup')
			length = 60
			df = df.sort_values('Relative Volume at Time', ascending=False)
			left = 0
			right =  length
			df = df[left:right].reset_index(drop = True).set_index('Ticker')
			
			return df, browser

		if type == 'full':
			return get_full(refresh)
		elif type == 'current':
			return get_current(refresh,browser)
		elif type == 'intraday':
			return get_intraday(browser)
			
	
		

if __name__ == '__main__':
	
	Screener.run(datetime.datetime.now())
	'''if   ((datetime.datetime.now().hour) < 5 or (datetime.datetime.now().hour == 5 and datetime.datetime.now().minute < 40)) and not data.identify == 'laptop':
		Screener.run(datetime.datetime.now())
		study.current(study,True)
		browser = Screener.get.startFirefoxSession()
		while datetime.datetime.now().hour < 13:
			Screener.run(tf = '1min', date = '0',browser = browser)
	else:
		Screener.run(ticker = 'ENPH',fpath = 3)



		i = 0
		path = "C:/Screener/tmp/subtickerlists/"
		while True:
			print('enter cycle length (x50)')
			cycles = int(input())
			for _ in range(cycles):
				dir_list = os.listdir(path)
				direct = path + dir_list[0]
				tickers = pd.read_feather(direct)['Ticker'].to_list()
				Screener.queue(ticker = tickers,fpath = 0)
				os.remove(direct)'''