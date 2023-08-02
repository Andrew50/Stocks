

#remove all premarket from saved df


#screen for histrocial setups every night if unmarked less than 500 or something


#default scale file

#allow no interent functionality

#pool with async funcitonality
#pool using map and working tqdm


import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from pyarrow import feather
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import datetime
from tvDatafeed import TvDatafeed, Interval
import os
import datetime
import numpy
from multiprocessing import Pool
import warnings
import yfinance as yf
import shutil

warnings.filterwarnings("ignore")

class Data:
	
	def add_setup(ticker,date,setup,val):
		add = pd.DataFrame({ 'ticker':[date], 'datetime':[ticker], 'value':[val], 'required':[1] })
		ident = Data.identify()
		path = 'C:/Stocks/sync/database/' + ident + '_' + setup + '.feather'
		try: df = pd.read_feather(path)
		except FileNotFoundError: df = pd.DataFrame()
		df = pd.concat([df,add])
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

	def identify():
		if os.path.exists("C:/Stocks/local/data/laptop.txt"): return 'laptop'
		if os.path.exists("C:/Stocks/local/data/desktop.txt"): return 'desktop'
		if os.path.exists("C:/Stocks/local/data/ben.txt"): return 'ben'
		if os.path.exists("C:/Stocks/local/data/tae.txt"): return 'tae'
		raise Exception('No idetifcation file')

	def get_nodes():
		if Data.identify == 'laptop':
			return 8
		return 5

	def pool(deff,arg):
		pool = Pool(processes = Data.get_nodes())
		data = list(tqdm(pool.imap(deff, arg), total=len(arg))) #might need to be imap
		return data

	def format_date(dt):
		if type(dt) == str:
			try: dt = datetime.datetime.strptime(dt, '%Y-%m-%d')
			except: dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
		if type(dt) == datetime.date:
			time = datetime.time(9,30,0)
			dt = datetime.datetime.combine(dt,time)
		if dt.time() == datetime.time(0):
			time = datetime.time(9,30,0)
			dt = datetime.datetime.combine(dt.date(),time)
		return(dt)

	def findex(df,dt):
		dt = Data.format_date(dt)
		if df.index[0] > dt:
			raise IndexError
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
			if df.index[i].to_pydatetime() > dt: i -= 1
			else: break
		return i

	def data_path(ticker,tf):
		if Data.identify == 'ben': drive = 'F:/Stocks/local/data/'
		else: drive = 'C:/Stocks/local/data/'
		if 'd' in tf or 'w' in tf: path = 'd/' 
		else: path = '1min/'
		return drive + path + ticker + '.feather'
	
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
			exchange = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('Ticker').loc[ticker]['Exchange']
			if not ('d' in tf or 'w' in tf): interval = Interval.in_1_minute
			else: interval = Interval.in_daily
			tv = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1")
			add = tv.get_hist(ticker, exchange, interval=interval, n_bars=100000, extended_session = pm)
			add.drop('symbol', axis = 1, inplace = True)
			add.index = add.index + pd.Timedelta(hours=4)
			add_index = Data.findex(add,df.index[-1]) + 1
			add = add[add_index:]
			return pd.concat([df,add])
		df = feather.read_feather(Data.data_path(ticker,tf))
		if dt != None:
			dt = Data.format_date(dt)
			adj_dt = adjust_date(dt,tf) #round date to nearest non premarket datetime
			try:  
				index = Data.findex(df,adj_dt)
			except IndexError: 
				df = append_tv(ticker,tf,df,False)
				index = Data.findex(df,adj_dt)
			if offset == 0: df = df[:index + 1]
			if dt.hour < 5 or (dt.hour == 5 and dt.minute < 30):  ####if date requested is premarket
				if 'd' in tf or 'w' in tf:
					df = append_tv(ticker,tf,df,True)
				else:
					pmchange = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('Ticker').loc[ticker]['Pre-market Change']
					if numpy.isnan(pmchange): pmchange = 0
					pm = df.iat[-1,3] + pmchange
					date = pd.Timestamp(datetime.datetime.today())
					row  =pd.DataFrame({'datetime': [date], 'open': [pm],'high': [pm], 'low': [pm], 'close': [pm], 'volume': [0]}).set_index("datetime")
					df = pd.concat([df, row])
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
			df.index = df.index + pd.Timedelta(minutes = 570)
		if offset != 0: df = df[:Data.findex(df,dt)+offset]
		df = df[-bars:]
		return df

	def update(bar):
		ticker = bar[0]
		current_day = bar[1]
		tf = bar[2]
		if ticker == None or "/" in ticker  or '.' in ticker: return
		exists = True
		try:
			df = Data.get(ticker,tf)
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
		if Data.isMarketOpen() == 1:
			ydf.drop(ydf.tail(1).index,inplace=True)
		if not exists:
			df = ydf
		else:
			try: index = Data.findex(ydf, last_day) 
			except: return
			ydf = ydf[index + 1:]
			df = pd.concat([df, ydf])
		df.index.rename('datetime', inplace = True)
		feather.write_feather(df,Data.data_path(ticker,tf))
	
	def run():
		dirs = ['C:/Stocks/local','C:/Stocks/local/data','C:/Stocks/local/account','C:/Stocks/local/screener',
				'C:/Stocks/local/study','C:/Stocks/local/trainer','C:/Stocks/local/data/1min','C:/Stocks/local/data/d']
		if not os.path.exists("C:/Stocks/local"):
			for d in dirs:
				os.mkdir(d)
		tv = TvDatafeed()
		current_day = tv.get_hist('QQQ', 'NASDAQ', n_bars=2).index[Data.isMarketOpen()]
		current_minute = tv.get_hist('QQQ', 'NASDAQ', n_bars=2, interval=Interval.in_1_minute, extended_session = False).index[Data.isMarketOpen()]
		from Screener import Screener as screener
		scan = screener.get('current')
		batches = []
		for i in range(len(scan)):
		   ticker = scan.index[i]
		   batches.append([ticker, current_day, 'd'])
		   batches.append([ticker, current_minute, '1min'])
		Data.pool(Data.update, batches)
		setup_list = Data.get_setups_list()
		Data.combine_training_data()
		epochs = 200
		prcnt_setup = .05
		if Data.indentify == 'desktop':
			for s in setup_list:
				
				Data.run(s,prcnt_setup,epochs,False)
			if datetime.datetime.now().weekday() == 4:
				Data.backup()

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

	def score(dfs,ticker_list,setup_type,model,threshold):
		x,y= Data.format(dfs,setup_type,False)
		sys.stdout = open(os.devnull, 'w')
		scores = model.predict(x)
		sys.stdout = sys.__stdout__
		scores = scores[:,1]
		setups = []
		for i in range(len(scores)):
			score = scores[i]
			if score >= threshold:
				df = dfs[i]
				ticker = ticker_list[i]
				setups.append([ticker,score, df])
		return setups
		
	def worker(bar):
		def time_series(df: pd.DataFrame,
					col: str,
					name: str, sample_size) -> pd.DataFrame:
			return df.assign(**{
				f'{name}_t-{lag}': col.shift(lag)
				for lag in range(0, sample_size)
			})
		def get_classification(df: pd.DataFrame,value) -> pd.DataFrame:
			df['classification'] = value
			return df
		def get_lagged_returns(df: pd.DataFrame, sample_size) -> pd.DataFrame:
			FEAT_COLS = ['open', 'low', 'high', 'close']
			#FEAT_COLS = ['open', 'low', 'high', 'close','volume']
			for col in FEAT_COLS:
				#return_col = df[col]/df[col].shift(1)-1
				return_col = df[col]/df['close'].shift(1)  - 1
				#return_col = df[col].div(close) - 1
				df = time_series(df, return_col, f'feat_{col}_ret', sample_size)
			return df
		setup_type = bar[1]
		sample_size = Data.setup_size(setup_type)
		try:
			ticker = bar[0][0]
			date = bar[0][1]
			value = bar[0][2]
			tf = setup_type.split('_')[0]
			df = Data.get(ticker,tf,date,sample_size + 1)
		except :
			df = bar[0]
			df = df[-(sample_size + 1):]
			value = 0
		if df.empty:
			return
		open_price = df.iat[-1,0]
		for i in range(1,4):
			df.iat[-1,i] = open_price
		if len(df) < sample_size + 1:
			add = pd.DataFrame(df.iat[-1,3], index=np.arange(sample_size - len(df) + 1), columns=df.columns)
			df = pd.concat([add,df])
		df = get_lagged_returns(df, sample_size)
			
		df = get_classification(df,value)
		df = df.replace([np.inf, -np.inf], np.nan).dropna()[[col for col in df.columns if 'feat_' in col] + ['classification']]
		return  df

	def format(setups, setup_type,pool = True):
		def reshape_x(x: np.array,FEAT_LENGTH) -> np.array:
			num_feats = x.shape[1]//FEAT_LENGTH
			x_reshaped = np.zeros((x.shape[0], FEAT_LENGTH, num_feats))
			for n in range(0, num_feats):
				x_reshaped[:, :, n] = x[:, n*FEAT_LENGTH:(n+1)*FEAT_LENGTH]
			return x_reshaped
		arglist = []
		if isinstance(setups, pd.DataFrame):
			for i in range(len(setups)):
				ticker = setups.iat[i,0]
				date = setups.iat[i,1]
				value = setups.iat[i,2]
				arglist.append([[ticker,date,value],setup_type])
		else:
			for df in setups:
				arglist.append([df,setup_type])
		if pool: dfs = Data.pool(Data.worker,arglist)
		else: dfs = [Data.worker(arglist[i]) for i in range(len(arglist))]
		values = pd.concat(dfs).values
		y = values[:,-1]
		x = reshape_x(values[:,:-1],Data.setup_size(setup_type))
		return x , y
	
	def setup_size(setup_type):
		if 'F' in setup_type:
			return 80
		return 40

	def sample(setuptype,use):
		buffer = 2
		allsetups = pd.read_feather('C:/Stocks/local/data/' + setuptype + '.feather').sort_values(by='date', ascending = False).reset_index(drop = True)
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
		return setups
   
	def train(setup_type,percent_yes,epochs):
		setups = Data.sample(setup_type, percent_yes)
		x, y = Data.format(setups,setup_type)
		model = Sequential([ Bidirectional( LSTM( 64,  input_shape = (x.shape[1], x.shape[2]), return_sequences = True,),),
			Dropout(0.2), Bidirectional(LSTM(32)), Dense(3, activation = 'softmax'),])
		model.compile(loss = 'sparse_categorical_crossentropy',optimizer = Adam(learning_rate = 1e-3),metrics = ['accuracy'])
		model.fit(x,y,epochs = epochs,batch_size = 64,validation_split = .2,)
		model.save('C:/Stocks/sync/models/model_' + setup_type)

	def get_scale(name):
		s  = open("C:/Stocks/scale.txt", "r").read()
		bars = s.split('-')
		for bar in bars:
			if name.split(' ')[0] in bar: break
		lines = bar.splitlines()
		for line in lines:
			if name.split(' ')[1] in line: break
		return float(line.split('=')[1].replace(' ',''))
if __name__ == '__main__':
	#Data.update()
	Data.consolidate_setups()







