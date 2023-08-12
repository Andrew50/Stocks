import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from pyarrow import feather
from tvDatafeed import TvDatafeed, Interval
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from multiprocessing import Pool, current_process
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
import websocket, datetime, os, pyarrow, shutil,statistics, warnings, math
warnings.filterwarnings("ignore")

class Data:

	def is_pre_market(dt):
		if dt == None: return False
		if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30): return True
		return False

	def get(ticker = 'NFLX',tf = 'd',dt = None, bars = 0, offset = 0):
		try:
			if len(tf) == 1: tf = '1' + tf
			dt = Data.format_date(dt)
			if 'd' in tf or 'w' in tf: base_tf = '1d'
			else: base_tf = '1min'
			try: df = feather.read_feather(Data.data_path(ticker,tf))
			except FileNotFoundError: df = pd.DataFrame()
			print(df)
			if df.empty or (dt != None and (dt < df.index[0] or dt > df.index[-1])): 
				if df.empty or not (base_tf == '1d' and Data.is_pre_market(dt)):
					try: add = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1").get_hist(ticker,pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('Ticker').loc[ticker]['Exchange'], interval=base_tf, n_bars=100000, extended_session = Data.is_pre_market(dt))
					except (websocket._exceptions.WebSocketAddressException, KeyError): raise TimeoutError
					if pd.isna(add) or add.empty: raise TimeoutError
					add.drop('symbol', axis = 1, inplace = True)
					add.index = add.index + pd.Timedelta(hours=4)
					if df.empty or add.index[0] > df.index[-1]: df = add
					else: df = pd.concat([df,add[Data.findex(add,df.index[-1]) + 1:]])
			if dt != None and not Data.is_pre_market(dt):
				try: df = df[:Data.findex(df,dt) + 1 + int(offset*(pd.Timedelta(tf) / pd.Timedelta(base_tf)))]
				except IndexError: raise TimeoutError
			if 'w' in tf and not Data.is_pre_market(dt):
				last_bar = df.tail(1)
				df = df[:-1]
			df = df.resample(tf,closed = 'left',label = 'left',origin = pd.Timestamp('2008-01-07 09:30:00')).apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
			if 'w' in tf and not Data.is_pre_market(dt): df = pd.concat([df,last_bar])
			if base_tf == '1d' and Data.is_pre_market(dt): 
				pm_bar = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('ticker').loc[ticker]
				pm_price = pm_bar['pm change'] +  df.iat[-1,3]
				df = pd.concat([df,pd.DataFrame({'datetime': [dt], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime",drop = True)])
					
			df['ticker'] = ticker
			return df.dropna()[-bars:]
		except IndexError: return pd.DataFrame()

	def findex(df,dt):
		dt = Data.format_date(dt)
		i = int(len(df)/2)
		k = int(i/2)
		while k != 0:
			date = df.index[i].to_pydatetime()
			if date > dt: i -= k
			elif date < dt: i += k
			k = int(k/2)
		while df.index[i].to_pydatetime() < dt: i += 1
		while df.index[i].to_pydatetime() > dt: i -= 1
		return i



	def load_model(st):
		start = datetime.datetime.now()
		model = load_model('C:/Stocks/sync/models/model_' + st)
		print(f'{st} model loaded in {datetime.datetime.now() - start}')
		return model

	def score(x,info,dfs,st,model,skip_req = False):
		threshold = Data.get_config('Screener threshold')
		setups = []
		scores = model.predict(x)[:,1]
		for i in range(len(scores)):
			score = round(scores[i]*100)
			if score >= threshold:
				bar = info[i]
				ticker = bar[0]
				dt = bar[1]
				key = bar[2]
				df = dfs[key]
				try: df = df[:Data.findex(df,dt) + 1]
				except: pass
				else:
					if skip_req or Data.get_requirements(ticker, df,st): 
						setups.append([ticker,dt,score,df])
		return setups


		
	def worker(bar):

		def time_series(df: pd.DataFrame,col: str,name: str, sample_size) -> pd.DataFrame:
			return df.assign(**{f'{name}_t-{lag}': col.shift(lag) for lag in range(0, sample_size)})

		def get_classification(df: pd.DataFrame,value,ticker,dt,key) -> pd.DataFrame:
			df['classification'] = value
			df['ticker'] = ticker
			df['dt'] = dt
			df['key'] = key
			return df

		def get_lagged_returns(df: pd.DataFrame, sample_size) -> pd.DataFrame:
			for col in ['open', 'low', 'high', 'close']:
				return_col = df[col]/df['close'].shift(1)  - 1
				df = time_series(df, return_col, f'feat_{col}_ret', sample_size)
			return df

		df = bar[0]
		ticker = df['ticker'][0]
		try: value = df['value'][0]
		except: value = 0
		use_whole_df = bar[1]
		key = bar[2]
		sample_size = 100
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
		for col in ['high','low','close']: df[f'feat_{col}_ret_t-{sample_size - 1}'] = df[f'feat_open_ret_t-{sample_size-1}']
		df = get_classification(df,value,ticker,dt,key)
		df = df.replace([np.inf, -np.inf], np.nan).dropna()[[col for col in df.columns if 'feat_' in col] + ['classification'] + ['ticker'] + ['dt'] + ['key']]
		return  df



	def format(dfs, use_whole_df = False):

		def reshape_x(x: np.array,FEAT_LENGTH) -> np.array:
			num_feats = x.shape[1]//FEAT_LENGTH
			x_reshaped = np.zeros((x.shape[0], FEAT_LENGTH, num_feats))
			for n in range(0, num_feats): x_reshaped[:, :, n] = x[:, n*FEAT_LENGTH:(n+1)*FEAT_LENGTH]
			return x_reshaped

		arglist = [[dfs[i],use_whole_df,i] for i in range(len(dfs))]
		if current_process().name == 'MainProcess': dfs = Data.pool(Data.worker,arglist)
		else:
			dfs = []
			pbar = tqdm(total = len(arglist))
			for arg in arglist:
				dfs.append(Data.worker(arg))
				pbar.update(1)
			pbar.close()
		values = pd.concat(dfs).values
		y = values[:,-4]
		info = values[:,-3:]
		x_values = values[:,:-4]
		x = reshape_x(x_values,100)
		return np.asarray(x).astype(np.float32),np.asarray(y).astype(np.float32) , info

	def sample(st,use):
		Data.consolidate_database()
		allsetups = pd.read_feather('C:/Stocks/local/data/' + st + '.feather').sort_values(by='dt', ascending = False).reset_index(drop = True)
		yes = []
		no = []
		groups = allsetups.groupby(pd.Grouper(key='ticker'))
		dfs = [group for _,group in groups]
		for df in dfs:
			df = df.reset_index(drop = True)
			for i in range(len(df)):
				bar = df.iloc[i]
				if bar[2] == 1:
					for ii in [i + ii for ii in [-2,-1,1,2]]:
						if abs(ii) < len(df): 
							bar = df.iloc[ii]
							if bar[2] == 0: no.append(bar)
					yes.append(bar)
		yes = pd.DataFrame(yes)
		no = pd.DataFrame(no)
		while True:
			no = no.drop_duplicates(subset = ['ticker','dt'])
			required =  int(len(yes) - (len(no) * use))
			if required < 0: break
			sample = allsetups[allsetups['value'] == 0].sample(frac = 1)[:required + 1]
			no = pd.concat([no,sample])
		df = pd.concat([yes,no]).sample(frac = 1).reset_index(drop = True)
		df['tf'] = st.split('_')[0]
		num_dfs = int(Data.get_config('Data cpu_cores'))
		s = math.ceil(len(df) / num_dfs)
		dfs = []

		for i in range(num_dfs):
			dfs.append(df[int(s*i):int(s*(i+1))])
		values = Data.pool(Data.create_arrays,dfs)
		x = np.concatenate([bar[0] for bar in values])
		y = np.concatenate([bar[1] for bar in values])

		return x, y



	def create_arrays(df):
		pbar = tqdm(total = len(df))
		dfs = []
		for i in range(len(df)):
			bar = df.iloc[i]
			ticker = bar['ticker']
			tf = bar['tf']
			dt = bar['dt']
			data = Data.get(ticker,tf,dt)
			if not data.empty: dfs.append(data)
			pbar.update(1)
		pbar.close()
		x, y, info = Data.format(dfs,(dt == None))
		return x, y, info, dfs


	def train(st,percent_yes,epochs):
		x, y  = Data.sample(st, percent_yes)
		model = Sequential([ Bidirectional( LSTM( 64,  input_shape = (x.shape[1], x.shape[2]), return_sequences = True,),),Dropout(0.2), Bidirectional(LSTM(32)), Dense(3, activation = 'softmax'),])
		model.compile(loss = 'sparse_categorical_crossentropy',optimizer = Adam(learning_rate = 1e-3),metrics = ['accuracy'])
		model.fit(x,y,epochs = epochs,batch_size = 64,validation_split = .2,)
		#model.save('C:/Stocks/sync/models/model_' + st)

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

			if st == 'd_EP':
				reqDolVol = 2000000
				reqAdr = 2.5
				reqpmDolVol = 1000000
			elif st == 'd_NEP':
				reqDolVol = 4000000
				reqAdr = 2.5
				reqpmDolVol = 1000000
			elif st == 'd_P':
				reqDolVol = 5000000
				reqAdr = 2.7
				reqpmDolVol = 800000
			elif st == 'd_NP':
				reqDolVol = 5000000
				reqAdr = 2.7
				reqpmDolVol = 800000
			elif st == 'd_F':
				reqDolVol = 8000000
				reqAdr = 2.5
				reqpmDolVol = 500000
			elif st == 'd_NF':
				reqDolVol = 8000000
				reqAdr = 2.5
				reqpmDolVol = 500000
			elif st == 'd_MR':
				reqDolVol = 10000000
				reqAdr = 3
				reqpmDolVol = 1000000
			elif 'd_' in st:
				reqDolVol = 8000000
				reqAdr = 3
				reqpmDolVol = 1000000
			elif 'h_' in st:
				reqAdr = 1.4
				reqDolVol = 1500000
				reqpmDolVol = 1000000
			elif '1min_' in st:
				reqAdr = 0
				reqDolVol = 10000
				reqpmDolVol = 1000000
			elif 'w_' in st:
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
		if length < 5: return False
		ma_length = 15
		if ma_length > length - 1: ma_length = length - 1
		adr = []
		dol_vol = []
		for i in range(ma_length):
			adr.append((df.iat[-i-2,1]/df.iat[-i-2,2] - 1) * 100)
			dol_vol.append(df.iat[-2-i,3]*df.iat[-2-i,4])
		dol_vol = statistics.mean(dol_vol)              
		adr = statistics.mean(adr)  
		dol_vol_req, adr_req, pm_dol_vol_req = setup_requirements(st)
		if adr > adr_req and (dol_vol > dol_vol_req or pm_dol_vol(df) > pm_dol_vol_req): return True
		return False

	def run():
		Data.setup_directories()
		tv = TvDatafeed()
		current_day = tv.get_hist('QQQ', 'NASDAQ', n_bars=2).index[Data.is_market_open()]
		current_minute = tv.get_hist('QQQ', 'NASDAQ', n_bars=2, interval=Interval.in_1_minute, extended_session = False).index[Data.is_market_open()]
		from Screener import Screener as screener
		scan = screener.get('full',True)
		batches = []
		for i in range(len(scan)):
		   ticker = scan[i]
		   batches.append([ticker, current_day, 'd'])
		   batches.append([ticker, current_minute, '1min'])
		Data.pool(Data.update, batches)
		Data.refill_backtest()
		if Data.get_config("Data identity") == 'desktop':
			setup_list = Data.get_setups_list()
			weekday = datetime.datetime.now().weekday()
			if weekday%3 == 0:
				for s in setup_list: Data.train(s,.05,200)
			if  weekday == 4: Data.backup()

	def setup_directories():
		dirs = ['C:/Stocks/local','C:/Stocks/local/data','C:/Stocks/local/account','C:/Stocks/local/screener',
				'C:/Stocks/local/study','C:/Stocks/local/trainer','C:/Stocks/local/data/1min','C:/Stocks/local/data/d']
		if not os.path.exists("C:/Stocks/local"):
			for d in dirs: os.mkdir(d)
		if not os.path.exists("C:/Stocks/config.txt"): shutil.copyfile('C:/Stocks/sync/files/default_config.txt','C:/Stocks/config.txt')

	def update(bar):
		ticker = bar[0]
		current_day = bar[1]
		tf = bar[2]
		exists = True
		try:
			df = feather.read_feather(Data.data_path(ticker,tf))
			last_day = df.index[-1] 
			if last_day == current_day: return
		except: exists = False
		if tf == 'd':
			ytf = '1d'
			period = '25y'
		else:
			ytf = '1m'
			period = '5d'
		ydf = yf.download(tickers = ticker,  period = period,  group_by='ticker',      
			interval = ytf, ignore_tz = True, progress=False,
			show_errors = False, threads = False, prepost = False) 
		ydf.drop(axis=1, labels="Adj Close",inplace = True)
		ydf.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace = True)
		ydf.dropna(inplace = True)
		if Data.is_market_open() == 1: ydf.drop(ydf.tail(1).index,inplace=True)
		if not exists:df = ydf
		else:
			try: index = Data.findex(ydf, last_day) 
			except: return
			ydf = ydf[index + 1:]
			df = pd.concat([df, ydf])
		df.index.rename('datetime', inplace = True)
		if tf == '1min': df = df.between_time('09:30' , '15:59')
		if not df.empty: feather.write_feather(df,Data.data_path(ticker,tf))

	def refill_backtest():
		from Screener import Screener as screener
		historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
		if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"): shutil.copy(r"C:\Stocks\sync\files\full_scan.feather", r"C:\Stocks\local\study\full_list_minus_annotated.feather")
		while(len(historical_setups[historical_setups["post_annotation"] == ""]) < 1500):
			full_list_minus_annotation = pd.read_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather").sample(frac=1)
			screener.run(ticker = full_list_minus_annotation[:20]['ticker'].tolist(), fpath=0)
			full_list_minus_annotation = full_list_minus_annotation[20:].reset_index(drop=True)
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
			if (datetime.datetime.now() - dt).days > 30: shutil.rmtree((path + b))

	def consolidate_database(): 
		setups = Data.get_setups_list()
		for setup in setups:
			df = pd.DataFrame()
			for ident in ['ben_','desktop_','laptop_']:
				try: 
					df1 = pd.read_feather(f"C:/Stocks/sync/database/{ident}{setup}.feather")
					df1['sindex'] = df1.index
					df1['source'] = ident
					df = pd.concat([df,df1]).reset_index(drop = True)
				except FileNotFoundError: pass
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
			if use: setups.append(s)
		return setups
	
	def add_setup(ticker,date,setup,val,req,ident = None):
		date = Data.format_date(date)
		add = pd.DataFrame({ 'ticker':[ticker], 'dt':[date], 'value':[val], 'required':[req] })
		if ident == None: ident = Data.get_config('Data identity') + '_'
		path = 'C:/Stocks/sync/database/' + ident + setup + '.feather'
		try: df = pd.read_feather(path)
		except FileNotFoundError: df = pd.DataFrame()
		df = pd.concat([df,add]).drop_duplicates(subset = ['ticker','dt'],keep = 'last').reset_index(drop = True)
		print(df)
		df.to_feather(path)

	def is_market_open():
		dayOfWeek = datetime.datetime.now().weekday()
		if(dayOfWeek == 5 or dayOfWeek == 6): return 0
		hour = datetime.datetime.now().hour
		minute = datetime.datetime.now().minute
		if hour > 5 and hour < 12: return 1
		elif hour == 5 and minute >= 30: return 1
		return 0

	def pool(deff,arg):
		pool = Pool(processes = int(Data.get_config('Data cpu_cores')))
		data = list(tqdm(pool.imap_unordered(deff, arg), total=len(arg)))
		return data

	def format_date(dt):
		if dt == 'current': return datetime.datetime.now()
		if dt == None: return None
		if isinstance(dt,str):
			try: dt = datetime.datetime.strptime(dt, '%Y-%m-%d')
			except: dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
		if isinstance(dt,datetime.datetime):
			time = datetime.time(dt.hour,dt.minute,0)
			dt = datetime.datetime.combine(dt.date(),time)

		if dt.hour == 0 and dt.minute == 0:
			time = datetime.time(9,30,0)
			dt = datetime.datetime.combine(dt.date(),time)
		return dt

	def data_path(ticker,tf):
		drive = Data.get_config('Data data_drive_letter')
		if 'd' in tf or 'w' in tf: path = 'd/' 
		else: path = '1min/'
		return drive + ':/Stocks/local/data/' + path + ticker + '.feather'
	


if __name__ == '__main__':
	Data.refill_backtest()
	#Data.run()