

#remove all premarket from saved df


import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from pyarrow import feather
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import datetime
from tvDatafeed import TvDatafeed, Interval
from Screener import Screener as screener
import os 
import datetime
import numpy
from multiprocessing import Pool
import warnings
import yfinance as yf
import shutil

warnings.filterwarnings("ignore")

class Data:

	def isMarketOpen():
		dayOfWeek = datetime.datetime.now().weekday()
		if(dayOfWeek == 5 or dayOfWeek == 6):
			return 0
		hour = datetime.datetime.now().hour
		minute = datetime.datetime.now().minute
		if(hour > 5 and hour < 12):
			return 1
		elif(hour == 5):
			if(minute >= 30):
				return 1
		elif(hour == 12):
			if(minute <= 15): 
				return 1
		return 0

	def identify():
		if os.path.exists("C:/Screener/laptop.txt"): return 'laptop'
		if os.path.exists("C:/Screener/desktop.txt"): return 'desktop'
		if os.path.exists("C:/Screener/ben.txt"): return 'ben'
		if os.path.exists("C:/Screener/tae.txt"): return 'tae'
		raise Exception('No idetifcation file')

	def pool(deff,arg):
			if Data.identify() == 'laptop': nodes = 8
			else: nodes = 5
			pool = Pool(processes = nodes)
			data = list(tqdm(pool.map(deff, arg), total=len(arg))) #might need to be imap
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

	def isToday(dt):
		if dt == 'now':
			return True
		if dt == None:
			return False
		if dt == 'Today' or dt == '0' or dt == 0:
			return True
		time = datetime.time(0,0,0)
		today = datetime.date.today()
		today = datetime.datetime.combine(today,time)
		dt = Data.convert_date(dt)
		if dt >= today:
			return True
		return False

	def findex(df,dt):
		if dt == '0' or dt == 0:
			return len(df) - 1
		dt = Data.convert_date(dt)
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
		if Data.identify == 'ben': drive = 'F:/Stocks/local/'
		else: drive = 'C:/Stocks/local/'
		if 'd' in tf or 'w' in tf: path = 'd/' 
		else: path = '1min/'
		return drive + path + ticker + '.feather'

	def get(ticker = 'QQQ',tf = 'd',dt = None, bars = 0):
		df = pd.read_feather(Data.data_path)
		fetch = False
		pm = False
		if date != None:
			try:
				index = Data.findex(df,dt)
			except:
				fetch = True
			if  datetime.datetime.now().hour < 5 or (datetime.datetime.now().hour < 6 and datetime.datetime.now().minute < 30): 
				pm = True
				if 'min' in tf:
					fetch = True
			if fetch:
				scan = screener.get('full')
				exchange = scan.loc[ticker]['Exchange']
				if not ('d' in tf or 'w' in tf): interval = Interval.in_1_minute
				else: interval = Interval.in_1_day
				tv = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1")
				add = tv.get_hist(ticker, exchange, interval=interval, n_bars=10000, extended_session = pm)
				add.drop('symbol', axis = 1, inplace = True)
				add.index = add.index + pd.Timedelta(hours=4)
				add_index = Data.findex(add,df.index[-1]) + 1
				add = add[add_index:]
				df = pd.concat([df,add]).reset_index(drop = True)
			if bars > 0:
				df = df[:Data.findex(df,dt) + 1]
		if 'h' in tf:
			df.index = df.index + pd.Timedelta(minutes = -30)
		if tf != '1min' and tf != 'd':
			logic = {'open'  : 'first', 'high'  : 'max', 'low':'min', 'close' : 'last', 'volume': 'sum' }
			df = df.resample(tf).apply(logic)
		df = df[-bars:]
		if 'h' in tf:
			df.index = df.index + pd.Timedelta(minutes = 30)

		if not fetch and pm:
			screenbar = screener.get('0','d').loc[ticker]
			pmchange =  screenbar['Pre-market Change']
			if numpy.isnan(pmchange):
				pmchange = 0
			pm = df.iat[-1,3] + pmchange
			date = pd.Timestamp(datetime.datetime.today())
			row  =pd.DataFrame({'datetime': [date],
					'open': [pm],
					'high': [pm],
					'low': [pm],
					'close': [pm],
					'volume': [0]}).set_index("datetime")
			df = pd.concat([df, row])
		df = df[-bars:]
			
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
		if tf == 'daily':
			ytf = '1d'
			period = '25y'
		else:
			ytf = '1m'
			period = '5d'
		ydf = yf.download(tickers =  ticker,  period = period,  group_by='ticker',      
			interval = ytf, ignore_tz = True, progress=False,
			show_errors = False, threads = False, prepost = False) 
		ydf = ydf.drop(axis=1, labels="Adj Close").rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace = True)
		if Data.isMarketOpen() == 1:
			ydf.drop(ydf.tail(1).index,inplace=True)
		ydf.dropna(inplace = True)
		if not exists:
			df = ydf
		else:
			try: index = Data.findex(ydf, last_day) 
			except: return
			ydf = ydf[index + 1:]
			df = pd.concat([df, ydf])
		df.index.rename('datetime', inplace = True)
		feather.write_feather(df,Data.get_path(ticker,tf))
	
	def runUpdate():

		dirs = ['C:/Stocks/local','C:/Stocks/local/data','C:/Stocks/local/account','C:/Stocks/local/screener',
				'C:/Stocks/local/study','C:/Stocks/local/trainer','C:/Stocks/local/1min','C:/Stocks/local/d']


		if not os.path.exists("C:/Stocks/local"):
			for d in dirs:
				
				os.mkdir(d)

		tv = TvDatafeed()
		current_day = tv.get_hist('QQQ', 'NASDAQ', n_bars=2).index[Data.isMarketOpen()]
		current_minute = tv.get_hist('QQQ', 'NASDAQ', n_bars=2, interval=Interval.in_1_minute, extended_session = False).index[Data.isMarketOpen()]
		scan = screener.get()
		batches = []
		for i in range(len(scan)):
		   ticker = scan.index[i]
		   batches.append([ticker, current_day, 'd'])
		   batches.append([ticker, current_minute, '1min'])
		Data.pool(Data.update, batches)
		setup_list = Data.get_setups_list()
		epochs = 200
		new = True
		prcnt_setup = .05
		if Data.indentify == 'desktop':
			for s in setup_list:
				Data.combine(new,s)
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
		path = "C:/Screener/tmp/subsetups/"
		dir_list = os.listdir(path)
		try:
			setups = pd.read_feather(r"C:\Screener\sync\setups.feather")
		except:
			setups = pd.DataFrame()
		todays_setups = pd.DataFrame()
		if len(dir_list) > 0:
			for f in dir_list:
				if "today" in f:
					df = pd.read_feather(path + f)
					todays_setups = pd.concat([todays_setups,df])
				else:
					df = pd.read_feather(path + f)
					setups = pd.concat([setups,df])
			if not setups.empty:
				setups.reset_index(inplace = True,drop = True)
				setups.to_feather(r"C:\Screener\sync\setups.feather")
			if not todays_setups.empty:
				todays_setups.reset_index(inplace = True,drop = True)
				todays_setups.to_feather(r"C:\Screener\sync\todays_setups.feather")
		if os.path.exists("C:/Screener/tmp/subsetups"):
			shutil.rmtree("C:/Screener/tmp/subsetups")
		os.mkdir("C:/Screener/tmp/subsetups")

	def combine_training_data(): 
		setups = Data.get_setups_list()
		for setup in setups:
			try:
				df1 = pd.read_feather(f"C:/Screener/sync/database/ben_{setup}.feather")
				df1['sindex'] = df1.index
				df1['source'] = 'ben_'
			except:
				df1 = pd.DataFrame()
			try:
				df2 = pd.read_feather(f"C:/Screener/sync/database/aj_{setup}.feather")
				df2['sindex'] = df2.index
				df2['source'] = 'aj_'
			except:
				df2 = pd.DataFrame()
			try:
				df3 = pd.read_feather(f"C:/Screener/sync/database/laptop_{setup}.feather")
				df3['sindex'] = df3.index
				df3['source'] = 'laptop_'
			except:
				df3 = pd.DataFrame()
			df = pd.concat([df1, df2, df3]).reset_index(drop = True)
			df.to_feather(f"C:/Screener/setups/database/{setup}.feather")



			###################################


	def get_setups_list():
		setups = []
		path = "C:/Screener/sync/database/"
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


	def score(dfs,setup_type,model,threshold):
		sample = Data.fromat(dfs,setup_type)
		scores = model.predict(sample)


		####################################################

	def format(setups, setup_type):
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

			#close = df.iat[-2,3]
				FEAT_COLS = ['open', 'low', 'high', 'close']
				#FEAT_COLS = ['open', 'low', 'high', 'close','volume']
				for col in FEAT_COLS:
					#return_col = df[col]/df[col].shift(1)-1
					return_col = df[col]/df['close'].shift(1)  -1
					#return_col = df[col].div(close) - 1
					df = time_series(df, return_col, f'feat_{col}_ret', sample_size)
				return df

			setup_type = bar[1]
			sample_size = Data.setup_size(setup_type)
			try: #no df
				ticker = bar[0][0]
				date = bar[0][1]
				value = bar[0][2]
				tf = setup_type.split('_')[0]
				df = Data.get(ticker,tf,date,sample_size)
			except :
				df = bar[0]
				value = pd.NA
		
			open_price = df.iat[-1,0]
			for i in range(1,4):
				df.iat[-1,i] = open_price
			if len(df) < sample_size:
				add = pd.DataFrame(df.iat[-1,3], index=np.arange(sample_size - len(df) + 1), columns=df.columns)
				df = pd.concat([add,df])
			df = get_lagged_returns(df, sample_size)
			df = get_classification(df,value)
			df = df.replace([np.inf, -np.inf], np.nan).dropna()[[col for col in df.columns if 'feat_' in col] + ['classification']]
			return  df
		arglist = []
		if isinstance(setups, pd.DataFrame):
			for i in range(len(setups)):
				ticker = setups.iat[i,0]
				date = setups.iat[i,1]
				value = setups.iat[i,2]
				arglist.append([[ticker,date,value],setup_type])
		else:
			for i in range(len(setups)):
				df = setups
				arglist.append([df,setup_type])
		dfs = Data.pool(worker,arglist)
		values = np.random.shuffle(pd.concat(dfs).values)
		y = values[0:, -1]
		x_values = values[0:, :-1]
		sample_size = Data.setup_size(setup_type)
		num_feats = x_values.shape[1]//sample_size
		x = np.zeros((x_values.shape[0], sample_size, num_feats))
		for n in range(0, num_feats):
			x[:, :, n] = x_values[:, n*num_feats:(n+1)*sample_size]
		return x , y

	def setup_size(setup_type):
		if 'F' in setup_type:
			return 80
		return 40

	def sample(setuptype,use):
		buffer = 2
		allsetups = pd.read_feather('C:/Screener/setups/database/' + setuptype + '.feather').sort_values(by='date', ascending = False).reset_index(drop = True)
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
		print(f'{len(yes)} setups')
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
		x, y = Data.fromat(setups,setup_type)
		model = Sequential([ Bidirectional( LSTM( 64,  input_shape = (x.shape[1], x.shape[2]), return_sequences = True,),),
			Dropout(0.2), Bidirectional(LSTM(32)), Dense(3, activation = 'softmax'),])
		model.compile(loss = 'sparse_categorical_crossentropy',optimizer = Adam(learning_rate = 1e-3),metrics = ['accuracy'])
		model.fit(x,y,epochs = epochs,batch_size = 64,validation_split = .2,)
		model.save('C:/Screener/sync/models/model_' + setuptype)

if __name__ == '__main__':
	Data.runUpdate()
	







