import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from pyarrow import feather
import pyarrow
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import websocket
import datetime
from tvDatafeed import TvDatafeed, Interval
import os
import datetime
import numpy
from multiprocessing import Pool, current_process
import warnings
import yfinance as yf
import shutil
import statistics
warnings.filterwarnings("ignore")

class Data:
	
	def add_setup(ticker,date,setup,val,req,ident = None):
		date = Data.format_date(date)
		add = pd.DataFrame({ 'ticker':[ticker], 'datetime':[date], 'value':[val], 'required':[req] })
		if ident == None: ident = Data.get_config('Data identity') + '_'
		path = 'C:/Stocks/sync/database/' + ident + setup + '.feather'
		try: df = pd.read_feather(path)
		except FileNotFoundError: df = pd.DataFrame()
		df = pd.concat([df,add]).drop_duplicates(subset = ['ticker','datetime'],keep = 'last').reset_index(drop = True)
		df.to_feather(path)

	def is_market_open():
		dayOfWeek = datetime.datetime.now().weekday()
		if(dayOfWeek == 5 or dayOfWeek == 6): return 0
		hour = datetime.datetime.now().hour
		minute = datetime.datetime.now().minute
		if(hour > 5 and hour < 12): return 1
		elif(hour == 5): 
			if(minute >= 30): return 1
		elif(hour == 12):
			if(minute <= 15): return 1
		return 0

	


	def pool(deff,arg):
		pool = Pool(processes = int(Data.get_config('Data cpu_cores')))
		data = list(tqdm(pool.imap_unordered(deff, arg), total=len(arg))) #might need to be imap
		return data

	def format_date(dt,tf  = '1min'):
		if type(dt) == str:
			try: dt = datetime.datetime.strptime(dt, '%Y-%m-%d')
			except: dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
		if type(dt) == datetime.date:
			time = datetime.time(9,30,0)
			dt = datetime.datetime.combine(dt,time)
		if dt.time() == datetime.time(0):
			time = datetime.time(9,30,0)
			dt = datetime.datetime.combine(dt.date(),time)

		df = pd.DataFrame({'datetime':[dt],'god':[3]}).set_index('datetime').resample(tf).apply({'god' : 'first'})
		if 'd' in tf or 'w' in tf: df.index = df.index.normalize() + pd.Timedelta(minutes = 570)
		dt = df.index[-1]

		return(dt)

	def findex(df,dt):
		dt = Data.format_date(dt)
		
		i = int(len(df)/2)
		k = i
		while True:
			k = int(k/2)
			date = df.index[i].to_pydatetime()
			if date > dt: i -= k
			elif date < dt: i += k
			if k == 0: break
		while True:
			if df.index[i].to_pydatetime() < dt: i += 1
			else: break
		while True:
			if i < 0:
				raise IndexError
			if df.index[i].to_pydatetime() > dt: i -= 1
			else: break
		return i

	def data_path(ticker,tf):
		drive = Data.get_config('Data data_drive_letter')
		if 'd' in tf or 'w' in tf: path = 'd/' 
		else: path = '1min/'
		return drive + ':/Stocks/local/data/' + path + ticker + '.feather'
	
	def get(ticker = 'NFLX',tf = 'd',dt = None, bars = 0, offset = 0):

		def adjust_date(dt,tf):
			if 'd' in tf or 'w' in tf: tf  = 'd'
			else: tf = '1min'
			if dt.hour <= 5 or (dt.hour == 5 and dt.minute < 30):
				time = datetime.time(15,59,0)
				dt = datetime.datetime.combine(dt.date() - datetime.timedelta(days = 1),time)
			if dt.hour > 12:
				time = datetime.time(15,59,0)
				dt = datetime.datetime.combine(dt.date(),time)
			if dt.second != 0:
				time = datetime.time(dt.hour,dt.minute,0)
				dt = datetime.datetime.combine(dt.date(),time)
			while dt.weekday() > 4:
				dt -= datetime.timedelta(days = 1)
			dt = pd.DataFrame({'datetime':[dt],'god':[3]}).set_index('datetime').resample(tf).apply({'god' : 'first'}).index[-1]
			if 'h' in tf: dt += datetime.timedelta(minutes = 30)
			return dt
		def append_tv(ticker,tf,df,pm):
			try: exchange = pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('Ticker').loc[ticker]['Exchange']
			except KeyError: raise TimeoutError()
			if not ('d' in tf or 'w' in tf): interval = Interval.in_1_minute
			else: interval = Interval.in_daily
			tv = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1")
			add = tv.get_hist(ticker, exchange, interval=interval, n_bars=100000, extended_session = pm)
			add.drop('symbol', axis = 1, inplace = True)
			add.index = add.index + pd.Timedelta(hours=4)
			if add.index[0] > df.index[-1]: return add
			add_index = Data.findex(add,df.index[-1]) + 1
			add = add[add_index:]
			return pd.concat([df,add])
		try: df = feather.read_feather(Data.data_path(ticker,tf))
		except (FileNotFoundError, pyarrow.lib.ArrowInvalid): df = pd.DataFrame()
		if dt != None:
			dt = Data.format_date(dt)
			adj_dt = adjust_date(dt,tf) #round date to nearest non premarket datetime
			try:  
				index = Data.findex(df,adj_dt)
			except IndexError: 
				try:
					df = append_tv(ticker,tf,df,False)
					if dt < df.index[0]: return pd.DataFrame()

					index = Data.findex(df,adj_dt)
				except websocket._exceptions.WebSocketAddressException : #no internet
					 ##no internet
					if df.empty: return df
					index = len(df) - 1
				except TimeoutError:
					#(f'ticker {ticker} has no data at {dt}')
					if df.empty: return pd.DataFrame()
					index = len(df) - 1
			if offset == 0: df = df[:index + 1]
			if dt.hour < 5 or (dt.hour == 5 and dt.minute < 30):  ####if date requested is premarket
				if 'd' in tf or 'w' in tf:
					try: df = append_tv(ticker,tf,df,True)
					except websocket._exceptions.WebSocketAddressException: pass ###no internet
				else:
					pm_bar = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('Ticker').loc[ticker]
					pm_change = pm_bar['Pre-market Change']
					pm_vol = pm_bar['Pre-market Volume']
					if numpy.isnan(pmchange): pmchange = 0
					if numpy.isnan(pm_vol): pm_vol = 0
					pm_price = pm_change +  df.iat[-1,3]
					date = pd.Timestamp(datetime.datetime.today())
					row  =pd.DataFrame({'datetime': [date], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_vol]}).set_index("datetime",drop = True)
					df = pd.concat([df, row])

		if df.empty: return df
		if 'w' in tf: 
			last_bar = df.tail(1)
			df.index = df.index - pd.Timedelta(days = 7)
			df = df[:-1]
		if 'h' in tf: df.index = df.index + pd.Timedelta(minutes = -30)
		if tf != '1min' and tf != 'd':
			df = df.resample(tf).apply({'open'  : 'first', 'high'  : 'max', 'low':'min', 'close' : 'last', 'volume': 'sum' })
		if 'h' in tf: df.index = df.index + pd.Timedelta(minutes = 30)
		elif 'w' in tf: 
			df = pd.concat([df,last_bar])
		if 'd' in tf or 'w' in tf:
			df.index = df.index.normalize() + pd.Timedelta(minutes = 570)
		if offset != 0: df = df[:Data.findex(df,dt)+offset]
		if 'd' not in tf and 'w' not in tf: df = df.between_time('09:30' , '15:59')
		df = df.dropna()
		df = df[-bars:]
		df['ticker'] = ticker
		return df

	def update(bar):
		ticker = bar[0]
		current_day = bar[1]
		tf = bar[2]
		if ticker == None or "/" in ticker  or '.' in ticker: return
		exists = True
		try:
			df = feather.read_feather(Data.data_path(ticker,tf))
			last_day = df.index[-1] 
			if last_day == current_day:
				return
		except:
			exists = False
		if tf == 'd':
			ytf = '1d'
			period = '25y'
		else:
			ytf = '1m'
			period = '5d'
		ydf = yf.download(tickers =  ticker,  period = period,  group_by='ticker',      
			interval = ytf, ignore_tz = True, progress=False,
			show_errors = False, threads = False, prepost = False) 

		ydf.drop(axis=1, labels="Adj Close",inplace = True)
		ydf.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace = True)
		ydf.dropna(inplace = True)
		if Data.is_market_open() == 1:
			ydf.drop(ydf.tail(1).index,inplace=True)
		if not exists:
			df = ydf
		else:
			try: index = Data.findex(ydf, last_day) 
			except: return
			ydf = ydf[index + 1:]
			df = pd.concat([df, ydf])
		df.index.rename('datetime', inplace = True)
		if tf == '1min':
			df = df.between_time('09:30' , '15:59')
		if not df.empty: feather.write_feather(df,Data.data_path(ticker,tf))

	def setup_directories():
		dirs = ['C:/Stocks/local','C:/Stocks/local/data','C:/Stocks/local/account','C:/Stocks/local/screener',
				'C:/Stocks/local/study','C:/Stocks/local/trainer','C:/Stocks/local/data/1min','C:/Stocks/local/data/d']
		if not os.path.exists("C:/Stocks/local"):
			for d in dirs:
				os.mkdir(d)
		if not os.path.exists("C:/Stocks/config.txt"):
			shutil.copyfile('C:/Stocks/sync/files/default_config.txt','C:/Stocks/config.txt')
			
	def run():
		

		tv = TvDatafeed()
		current_day = tv.get_hist('QQQ', 'NASDAQ', n_bars=2).index[Data.is_market_open()]
		current_minute = tv.get_hist('QQQ', 'NASDAQ', n_bars=2, interval=Interval.in_1_minute, extended_session = False).index[Data.is_market_open()]
		from Screener import Screener as screener
		#scan = pd.DataFrame({"Ticker": ["AAPL"]}).set_index("Ticker") 
		scan = screener.get('current')
		batches = []
		for i in range(len(scan)):
		   ticker = scan.index[i]
		   batches.append([ticker, current_day, 'd'])
		   batches.append([ticker, current_minute, '1min'])
		Data.pool(Data.update, batches)
		Data.refill_backtest()
		Data.combine_training_data()
		if Data.get_config("Data identity") == 'desktop':
			setup_list = Data.get_setups_list()
			weekday = datetime.datetime.now().weekday()
			if weekday%3 == 0:
				for s in setup_list:
					Data.train(s,.05,200,False)
			if  weekday == 4:
				Data.backup()

		

	def refill_backtest():
		from Screener import Screener as screener
		historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
		if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"):
			shutil.copy(r"C:\Stocks\sync\files\full_scan.feather", r"C:\Stocks\local\study\full_list_minus_annotated.feather")
		while(len(historical_setups[historical_setups["post_annotation"] == ""]) < 1500):
			full_list_minus_annotation = pd.read_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
			full_list_minus_annotation = full_list_minus_annotation.sample(frac=1)
			for t in range(8):
				screener.run(ticker=full_list_minus_annotation.iloc[t]["Ticker"], fpath=0)
			full_list_minus_annotation = full_list_minus_annotation[8:].reset_index(drop=True)
			full_list_minus_annotation.to_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
			historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")




	def backup():

		date = datetime.date.today()
		src = r'C:/Scan'
		dst = r'D:/Backups/' + str(date)
		shutil.copytree(src, dst)
		path = "D:/Backups/"
		dir_list = os.listdir(path)
		for b in dir_list:
			dt = datetime.datetime.strptime(b, '%Y-%m-%d')
			if (datetime.datetime.now() - dt).days > 30:
				shutil.rmtree((path + b))
   
	def consolidate_setups():
		path = "C:/Stocks/local/screener/subsetups/"
		if not os.path.exists(path):
			os.mkdir(path)

		dir_list = os.listdir(path)
		try:
			historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
		except:
			historical_setups = pd.DataFrame()
		todays_setups = pd.DataFrame()
		if len(dir_list) > 0:
			for f in dir_list:
				if "current" in f:
					df = pd.read_feather(path + f)
					todays_setups = pd.concat([todays_setups,df])
				else:
					df = pd.read_feather(path + f)
					historical_setups = pd.concat([historical_setups,df])
			if not historical_setups.empty:
				historical_setups.reset_index(inplace = True,drop = True)
				historical_setups.to_feather(r"C:\Stocks\local\study\historical_setups.feather")
			if not todays_setups.empty:
				todays_setups.reset_index(inplace = True,drop = True)
				todays_setups.to_feather(r"C:\Stocks\local\study\current_setups.feather")
		if os.path.exists(path):
			shutil.rmtree(path)
		os.mkdir(path)

	def combine_training_data(): 
		setups = Data.get_setups_list()
		for setup in setups:
			try:
				df1 = pd.read_feather(f"C:/Stocks/sync/database/ben_{setup}.feather")
				df1['sindex'] = df1.index
				df1['source'] = 'ben_'
			except:
				df1 = pd.DataFrame()
			try:
				df2 = pd.read_feather(f"C:/Stocks/sync/database/aj_{setup}.feather")
				df2['sindex'] = df2.index
				df2['source'] = 'aj_'
			except:
				df2 = pd.DataFrame()
			try:
				df3 = pd.read_feather(f"C:/Stocks/sync/database/laptop_{setup}.feather")
				df3['sindex'] = df3.index
				df3['source'] = 'laptop_'
			except:
				df3 = pd.DataFrame()
			df = pd.concat([df1, df2, df3]).reset_index(drop = True)
			df.to_feather(f"C:/Stocks/local/data/{setup}.feather")

	def get_setups_list():
		setups = []
		path = "C:/Stocks/sync/database/"
		dir_list = os.listdir(path)
		for p in dir_list:
			s = p.split('_')
			s = s[1] + '_' + s[2].split('.')[0]
			use = True
			for h in setups:
				if s == h:
					use = False
					break
			if use:
				setups.append(s)
		return setups

	def score(dfs,st,use_whole_df = False,threshold = 0,model = None):
		if not isinstance(dfs, list):dfs = [dfs]
		if model == None: model = load_model('C:/Stocks/sync/models/model_' + st)
		setups = []
		x,_,info= Data.format(dfs,st,use_whole_df)
		sys.stdout = open(os.devnull, 'w')
		scores = model.predict(x)
		sys.stdout = sys.__stdout__
		scores = scores[:,1]
		for i in range(len(scores)):
			score = scores[i]
			if score > threshold:
				bar = info[i]
				ticker = bar[0]
				dt = bar[1]
				key = bar[2]
				df = dfs[key]
				df = df[:Data.findex(df,dt) + 1]
				if Data.get_requirements(ticker, df,st):
					setups.append([ticker,dt,score,df])
		return setups
		
	def worker(bar):
		def time_series(df: pd.DataFrame,col: str,name: str, sample_size) -> pd.DataFrame:
			return df.assign(**{
				f'{name}_t-{lag}': col.shift(lag)
				for lag in range(0, sample_size)})
		def get_classification(df: pd.DataFrame,value,ticker,dt,key) -> pd.DataFrame:
			df['classification'] = value
			df['ticker'] = ticker
			df['dt'] = dt
			df['key'] = key

			return df
		def get_lagged_returns(df: pd.DataFrame, sample_size) -> pd.DataFrame:
			FEAT_COLS = ['open', 'low', 'high', 'close']
			for col in FEAT_COLS:
				return_col = df[col]/df['close'].shift(1)  - 1
				df = time_series(df, return_col, f'feat_{col}_ret', sample_size)
			return df


		df = bar[0]
		ticker = df['ticker'][0]
		try: value = df['value'][0]
		except: value = 0
		use_whole_df = bar[2]
		st = bar[1]
		key = bar[3]
		sample_size = Data.setup_size(st)
		if df.empty: return
		if use_whole_df:
			add = pd.DataFrame(df.iat[-1,3], index=np.arange(sample_size ), columns=df.columns)
			df = pd.concat([add,df])
		else:
			df = df[-sample_size - 1:]
			if len(df) < sample_size + 1:
				add = pd.DataFrame(df.iat[-1,3], index=np.arange(sample_size - len(df) + 1), columns=df.columns)
				df = pd.concat([add,df])
		dt = df.index

		df = get_lagged_returns(df, sample_size)
		for col in ['high','low','close']: df[f'feat_{col}_ret_t-{sample_size - 1}'] = df[f'feat_open_ret_t-{sample_size-1}'] #make last high low close the same as open so no hinsdight
		df = get_classification(df,value,ticker,dt,key)
		df = df.replace([np.inf, -np.inf], np.nan).dropna()[[col for col in df.columns if 'feat_' in col] + ['classification'] + ['ticker'] + ['dt'] + ['key']]
		return  df

	def format(dfs, st,use_whole_df = False):
		def reshape_x(x: np.array,FEAT_LENGTH) -> np.array:
			num_feats = x.shape[1]//FEAT_LENGTH
			x_reshaped = np.zeros((x.shape[0], FEAT_LENGTH, num_feats))
			for n in range(0, num_feats):
				x_reshaped[:, :, n] = x[:, n*FEAT_LENGTH:(n+1)*FEAT_LENGTH]
			return x_reshaped
		arglist = []
		for i in range(len(dfs)):
			df = dfs[i]
			arglist.append([df,st,use_whole_df,i])
		if current_process().name == 'MainProcess': dfs = Data.pool(Data.worker,arglist)  ##if not a worker process already
		else: dfs = [Data.worker(arglist[i]) for i in range(len(arglist))]
		values = pd.concat(dfs).values
		y = values[:,-4]
		info = values[:,-3:]
		x_values = values[:,:-4]
		x = reshape_x(x_values,Data.setup_size(st))
		return x , y , info
	
	def setup_size(setup_type):
		if 'F' in setup_type:
			return 80
		return 40

	def sample(st,use):
		buffer = 2
		allsetups = pd.read_feather('C:/Stocks/local/data/' +st + '.feather').sort_values(by='date', ascending = False).reset_index(drop = True)
		yes = []
		no = []
		req_no = []
		g = allsetups.groupby(pd.Grouper(key='ticker'))
		dfs = [group for _,group in g]
		for df in dfs:
			df = df.reset_index(drop = True)
			rem = 0
			for i in range(len(df)):
				bar = df.iloc[i]
				setup = bar[2]
				if setup == 1:
					if df.iat[i-1,2] == 0:
						for j in range(buffer):
							try:
								req_no.append(df.iloc[i - j - 1])
							except:
								pass
					yes.append(bar)
					rem = buffer
				else:
					if rem > 0:
						req_no.append(bar)
						rem -= 1
					else:
						#needs to be fixes because if setup occurs it will ad to no and to no_req because it adds to no req on a 1
						no.append(bar)
		yes = pd.DataFrame(yes)
		no = pd.DataFrame(no)
		req_no = pd.DataFrame(req_no)
		required_no = allsetups[allsetups['req'] == 1]
		required_no = required_no[required_no['setup'] == 0]
		req_no = pd.concat([req_no,required_no])
		length = ((len(yes) / use) - len(yes)) - len(req_no)
		use = length / len(no)
		if use > 1:
			use = 1
		if use < 0:
			use = 0
		no = no.sample(frac = use)
		allsetups = pd.concat([yes,no,req_no]).sample(frac = 1).reset_index(drop = True)
		setups = allsetups
		dfs = []
		for i in range(len(setups)):
			bar = setups.iloc[i]
			ticker = bar[0]
			dt = bar[1]
			v = bar[2]
			df = Data.get(ticker,dt,st.split('_')[0])
			df['value'] = v
			dfs.append(df)
		return dfs
   
	def train(st,percent_yes,epochs):
		dfs = Data.sample(st, percent_yes)
		x, y, t = Data.format(dfs,st,True)
		model = Sequential([ Bidirectional( LSTM( 64,  input_shape = (x.shape[1], x.shape[2]), return_sequences = True,),),
			Dropout(0.2), Bidirectional(LSTM(32)), Dense(3, activation = 'softmax'),])
		model.compile(loss = 'sparse_categorical_crossentropy',optimizer = Adam(learning_rate = 1e-3),metrics = ['accuracy'])
		model.fit(x,y,epochs = epochs,batch_size = 64,validation_split = .2,)
		model.save('C:/Stocks/sync/models/model_' + st)

	def get_config(name):
		s  = open("C:/Stocks/config.txt", "r").read()
		trait = name.split(' ')[1]
		trait.replace(' ','')
		bars = s.split('-')
		for bar in bars:
			if name.split(' ')[0] in bar: break
		lines = bar.splitlines()
		for line in lines:
			if trait in line: break
		value = line.split('=')[1].replace(' ','')
		try: value = float(value)
		except: pass
		return value
	def get_requirements(ticker, df, st = ''):

		def setup_requirements(st):


			if 'd_' in st:
				reqDolVol = 8000000
				reqAdr = 3
				reqpmDolVol = 1000000
			if 'h_' in st:
				reqAdr = 1.4
				reqDolVol = 1500000
				reqpmDolVol = 1000000
			if '1min_' in st:
				reqAdr = 0
				reqDolVol = 10000
				reqpmDolVol = 1000000
			if 'w_' in st:
				reqDolVol = 16000000
				reqAdr = 5
				reqpmDolVol = 1000000
			return reqDolVol, reqAdr, reqpmDolVol


		def pm_dol_vol(df):
			time = datetime.time(0,0,0)
			today = datetime.date.today()
			today = datetime.datetime.combine(today,time)
			if df.index[-1] < today or Data.is_market_open == 1: return 0
			return df.iat[-1,4] * df.iat[-1,0]

		length = len(df)
		if length < 5:
			return False
		dol_vol_l = 15
		adr_l = 15
		if dol_vol_l > length - 1:
			dol_vol_l = length - 1
		if adr_l > length - 1:
			adr_l = length - 1
		dolVol = []
		for i in range(dol_vol_l):
			dolVol.append(df.iat[-2-i,3]*df.iat[-2-i,4])
		dolVol = statistics.mean(dolVol)              
		adr= []
		for j in range(adr_l): 
			high = df.iat[-j-2,1]
			low = df.iat[-j-2,2]
			val = (high/low - 1) * 100
			adr.append(val)
		adr = statistics.mean(adr)  
		reqDolVol, reqAdr, reqpmDolVol = setup_requirements(st)
		if((adr > reqAdr) and ((dolVol > reqDolVol) or (pm_dol_vol(df) > reqpmDolVol))):return True
		return False

if __name__ == '__main__':
	Data.run()
	#Data.consolidate_setups()







