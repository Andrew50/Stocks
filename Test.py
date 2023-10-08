


from typing import Any
import numpy as np
from numpy.core import overrides
import pandas as pd
from pandas.core.series import Series
from scipy.stats import alpha
import yfinance as yf
from tqdm import tqdm
from pyarrow import feather
from tvDatafeed import TvDatafeed
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.models import Sequential, load_model
from multiprocessing import Pool, current_process
#from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
import websocket, datetime, os, pyarrow, shutil,statistics, warnings, math, time, pytz, tensorflow, random
warnings.filterwarnings("ignore")
import numpy
import torch

# import numpy as np
# from sklearn import preprocessing
# import pyts

#from pyts.approximation import SymbolicAggregateApproximation
# from pyts.metrics import dtw
from sklearn.preprocessing import normalize








class Main:
	
	def sample(st,use):
		Main.consolidate_database()
		allsetups = pd.read_feather('C:/Stocks/local/data/' + st + '.feather').sort_values(by='dt', ascending = False).reset_index(drop = True)
		yes = []
		no = []
		groups = allsetups.groupby(pd.Grouper(key='ticker'))
		dfs = [group for _,group in groups]
		for df in dfs:
			df = df.reset_index(drop = True)
			for i in range(len(df)):
				bar = df.iloc[i]
				if bar['value'] == 1:
					for ii in [i + ii for ii in [-2,-1,1,2]]:
						if abs(ii) < len(df): 
							bar2 = df.iloc[ii]
							if bar2['value'] == 0: no.append(bar2)
					yes.append(bar)
		yes = pd.DataFrame(yes)
		no = pd.DataFrame(no)
		required =  int(len(yes) - ((len(no)+len(yes)) * use))
		if required < 0:
			no = no[:required]
		while True:
			no = no.drop_duplicates(subset = ['ticker','dt'])
			required =  int(len(yes) - ((len(no)+len(yes)) * use))
			sample = allsetups[allsetups['value'] == 0].sample(frac = 1)
			if required < 0 or len(sample) == len(no): break
			sample = sample[:required + 1]
			no = pd.concat([no,sample])
		df = pd.concat([yes,no]).sample(frac = 1).reset_index(drop = True)
		df['tf'] = st.split('_')[0]
		return df

	# def worker(bar):
	# 	ticker, dt, value, tf = bar
	# 	ss = 50
	# 	try:
	# 		df = Main.get(ticker,tf,dt,ss)
	# 		df = df.drop(columns = ['ticker','volume'])
	# 		if len(df) < ss:
	# 			add = pd.DataFrame(df.iat[-1,3], index=np.arange(ss - len(df)), columns=df.columns)
	# 			df = pd.concat([add,df])
	# 		df = df.values.tolist()
	# 		df = np.array(df)
	# 		o = df[-1,0]
	# 		for ii in range(1,3): df[-1,ii] = o
	# 		df = df/np.array(o)
	# 		df = np.log(df)
	# 		np_array = df
	# 	except:
	# 		df = pd.DataFrame()
	# 		np_array = np.zeros((ss,4))
		
	# 	return [ticker,dt,tf,df,'',0,np_array,value]


	def worker(bar):
		ticker, dt, value, tf = bar
		#df = Get(ticker,tf,dt,value = value)
		

	def train(st, percent_yes, epochs):
		df = pd.read_feather('C:/Stocks/local/data/' + st + '.feather')
		ones = len(df[df['value'] ==1])
		if ones < 150: 
			print(f'{st} cannot be trained with only {ones} positives')
			return
		df  = Main.sample(st, percent_yes)
		df = df[['ticker','dt','value']]
		df['tf'] = 'd'
		df = df.values.tolist()
		info = Main.pool(Main.worker,df)
		x = np.array([x[6] for x in info])
		y = np.array([x[7] for x in info])
		model = Sequential([Bidirectional(LSTM(64, input_shape = (x.shape[1], x.shape[2]), return_sequences = True,),),Dropout(0.2), Bidirectional(LSTM(32)), Dense(3, activation = 'softmax'),])
		model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 1e-3), metrics = ['accuracy'])
		model.fit(x, y, epochs = epochs, batch_size = 64, validation_split = .2,)
		model.save('C:/Stocks/sync/models/model_' + st)
		tensorflow.keras.backend.clear_session()	
		
	def worker2(bar):
		info, st = bar
		model = Main.load_model(st)
		for i in range(len(info)):
			np_array = info[i][6]
			if np_array == None: continue
			score = model.score(np_array)[0][1]
			info[i][4] = st
			info[i][5] = score
		return info

	def score(df:list,st):
		if type(st) != list: st = [st]
		pool = Pool(processes = int(Main.get_config('Data cpu_cores')))
		info = Main.pool(Main.worker,df)
	
		info = [item for sublist in pool.map(Main.worker2,[[info,s] for s in st]) for item in sublist]

	def get_requirements(ticker, df, st):

		def pm_dol_vol(df):
			time = datetime.time(0,0,0)
			today = datetime.date.today()
			today = datetime.datetime.combine(today,time)
			if df.index[-1] < today or Main.is_market_open == 1: return 0
			return df.iat[-1,4] * df.iat[-1,0]

		length = len(df)
		if length < 5: return False
		ma_length = 15
		if ma_length > length - 1: ma_length = length - 1
		adr_list = []
		dol_vol_list = []
		for i in range(-1,-ma_length-1,-1):
			adr_list.append((df.iat[i,1]/df.iat[i,2] - 1) * 100)
			dol_vol_list.append(df.iat[i,3]*df.iat[i,4])
		reqs = [float(r) for r in Main.get_config(f'Screener {st}').split(',')]
		dol_vol_req = reqs[0] * 1000000
		adr_req = reqs[1]
		pm_dol_vol_req = reqs[2] * 1000000
		if statistics.mean(adr_list) > adr_req and (statistics.mean(dol_vol_list) > dol_vol_req or pm_dol_vol(df) > pm_dol_vol_req): return True
		return False

	def run():
		Main.check_directories()
		#current_day = Main.format_date(yf.download(tickers = 'QQQ', period = '25y', group_by='ticker', interval = '1d', ignore_tz = True, progress = False, show_errors = False, threads = False, prepost = False).index[-1-Main.is_market_open()])
		#current_minute = Main.format_date(yf.download(tickers = 'QQQ', period = '5d', group_by='ticker', interval = '1m', ignore_tz = True, progress = False, show_errors = False, threads = False, prepost = False).index[-1-Main.is_market_open()])
		from Screener import Screener as screener
		scan = screener.get('full',True)
		batches = []
		#for i in range(len(scan)):
		#   ticker = scan[i]
		#   batches.append([ticker, current_day, 'd'])
		#   batches.append([ticker, current_minute, '1min'])
		for i in range(len(scan)):
			for tf in ['d','1min']: batches.append([scan[i],tf])
		Main.pool(Main.update, batches)
		if Main.get_config("Data identity") == 'laptop':
			weekday = datetime.datetime.now().weekday()
			if weekday == 4: Main.backup()
			elif weekday == 5: Main.retrain_models()
		Main.refill_backtest()

	def retrain_models():
		Main.consolidate_database()
		setup_list = Main.get_setups_list()
		for s in setup_list: Main.train(s,.05,300)#######

	def update(bar):
		ticker = bar[0]
		#current_day = bar[1]
		#tf = bar[2]
		tf = bar[1]
		exists = True
		try:
			df = feather.read_feather(Main.data_path(ticker,tf)).set_index('datetime',drop = True)######
			last_day = df.index[-1] 
			#if last_day == current_day and False: return
		except: exists = False
		if tf == 'd':
			ytf = '1d'
			period = '25y'
		else:
			ytf = '1m'
			period = '5d'
		ydf = yf.download(tickers = ticker, period = period, group_by='ticker', interval = ytf, ignore_tz = True, progress=False, show_errors = False, threads = False, prepost = True) 
		#ydf = yf.download(tickers = ticker, period = period, group_by='ticker', interval = ytf, ignore_tz = True, progress=False, show_errors = False, threads = False, prepost = False) 
		ydf.drop(axis=1, labels="Adj Close",inplace = True)
		ydf.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace = True)
		ydf.dropna(inplace = True)
		if Main.is_market_open() == 1: ydf.drop(ydf.tail(1).index,inplace=True)
		if not exists: df = ydf
		else:
			try: index = Main.findex(ydf, last_day) 
			except: return
			ydf = ydf[index + 1:]
			df = pd.concat([df, ydf])
		df.index.rename('datetime', inplace = True)
		if not df.empty: 
			if tf == '1min': pass
				#df = df.between_time('09:30', '15:59')
			elif tf == 'd': df.index = df.index.normalize() + pd.Timedelta(minutes = 570)
			df = df.reset_index()
			feather.write_feather(df,Main.data_path(ticker,tf))

	def get_config(name):
		s  = open("C:/Stocks/config.txt", "r").read()
		trait = name.split(' ')[1]
		script = name.split(' ')[0]
		trait.replace(' ','')
		bars = s.split('-')
		found = False
		for bar in bars:
			if script in bar: 
				found = True
				break
		if not found: raise Exception(str(f'{script} not found in config'))
		lines = bar.splitlines()
		found = False
		for line in lines:
			if trait in line.split('=')[0]: 
				found = True
				break
		if not found: raise Exception(str(f'{trait} not found in config'))
		value = line.split('=')[1].replace(' ','')
		try: value = float(value)
		except: pass
		return value


	def load_model(st):
		start = datetime.datetime.now()
		model = load_model('C:/Stocks/sync/models/model_' + st)
		print(f'{st} model loaded in {datetime.datetime.now() - start}')
		return model

	def check_directories():
		dirs = ['C:/Stocks/local','C:/Stocks/local/data','C:/Stocks/local/account','C:/Stocks/local/study','C:/Stocks/local/trainer','C:/Stocks/local/data/1min','C:/Stocks/local/data/d']
		if not os.path.exists(dirs[0]): 
			for d in dirs: os.mkdir(d)
		if not os.path.exists("C:/Stocks/config.txt"): shutil.copyfile('C:/Stocks/sync/files/default_config.txt','C:/Stocks/config.txt')

	def refill_backtest():
		from Screener import Screener as screener
		try: historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
		except: historical_setups = pd.DataFrame()
		if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"): shutil.copy(r"C:\Stocks\sync\files\full_scan.feather", r"C:\Stocks\local\study\full_list_minus_annotated.feather")
		while historical_setups.empty or (len(historical_setups[historical_setups["pre_annotation"] == ""]) < 2500):
			full_list_minus_annotation = pd.read_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather").sample(frac=1)
			screener.run(ticker = full_list_minus_annotation[:20]['ticker'].tolist(), fpath = 0)
			full_list_minus_annotation = full_list_minus_annotation[20:].reset_index(drop=True)
			full_list_minus_annotation.to_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
			historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")

	def backup():
		date = datetime.date.today()
		src = r'C:/Stocks'
		dst = r'C:/Backups/' + str(date)
		shutil.copytree(src, dst)
		path = "C:/Backups/"
		dir_list = os.listdir(path)
		for b in dir_list:
			dt = datetime.datetime.strptime(b, '%Y-%m-%d')
			if (datetime.datetime.now() - dt).days > 30: shutil.rmtree((path + b))

	def add_setup(ticker,date,setup,val,req,ident = None):
		date = Main.format_date(date)
		add = pd.DataFrame({ 'ticker':[ticker], 'dt':[date], 'value':[val], 'required':[req] })
		if ident == None: ident = Main.get_config('Data identity') + '_'
		path = 'C:/Stocks/sync/database/' + ident + setup + '.feather'
		try: df = pd.read_feather(path)
		except FileNotFoundError: df = pd.DataFrame()
		df = pd.concat([df,add]).drop_duplicates(subset = ['ticker','dt'],keep = 'last').reset_index(drop = True)
		df.to_feather(path)

	def consolidate_database(): 
		setups = Main.get_setups_list()
		for setup in setups:
			df = pd.DataFrame()
			#for ident in ['ben_','desktop_','laptop_', 'ben_laptop_']:
			for ident in ['desktop_','laptop_']:
				try: 
					df1 = pd.read_feather(f"C:/Stocks/sync/database/{ident}{setup}.feather").dropna()
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
	
	def format_date(dt):
		if dt == 'current': return datetime.datetime.now(pytz.timezone('EST'))
		if dt == None: return None
		if isinstance(dt,str):
			try: dt = datetime.datetime.strptime(dt, '%Y-%m-%d')
			except: dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
		time = datetime.time(dt.hour,dt.minute,0)
		dt = datetime.datetime.combine(dt.date(),time)
		if dt.hour == 0 and dt.minute == 0:
			time = datetime.time(9,30,0)
			dt = datetime.datetime.combine(dt.date(),time)
		return dt

	def is_market_open():
		dayOfWeek = datetime.datetime.now().weekday()
		if(dayOfWeek == 5 or dayOfWeek == 6): return 0
		dt = datetime.datetime.now(pytz.timezone('US/Eastern'))
		hour = dt.hour
		minute = dt.minute
		if hour >= 10 and hour <= 16: return 1
		elif hour == 9 and minute >= 30: return 1
		return 0

	def pool(deff,arg):
		pool = Pool(processes = int(Main.get_config('Data cpu_cores')))
		data = list(tqdm(pool.imap_unordered(deff, arg), total=len(arg)))
		return data

	def is_pre_market(dt):
		if dt == None: return False
		if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30): return True
		return False

	def data_path(ticker,tf):
		if 'd' in tf or 'w' in tf: path = 'd/' 
		else: path = '1min/'
		return Main.get_config('Data data_drive_letter') + ':/Stocks/local/data/' + path + ticker + '.feather'
	
		
class Data:
	
	def __init__(self,ticker = 'QQQ',tf = 'd',dt = None,bars = 0,offset = 0,value = None):
		self.ticker = ticker
		self.tf = tf
		self.dt = dt
		self.value = value
		self.bars = bars
		self.offset = offset
		self.scores = []
		try:
			if len(tf) == 1: tf = '1' + tf
			dt = Main.format_date(dt)
			if 'd' in tf or 'w' in tf: base_tf = '1d'
			else: base_tf = '1min'
			try: df = feather.read_feather(Main.data_path(ticker,tf)).set_index('datetime',drop = True)
			except FileNotFoundError: df = pd.DataFrame()
			if (df.empty or (dt != None and (dt < df.index[0] or dt > df.index[-1]))) and not (base_tf == '1d' and Main.is_pre_market(dt)): 
				try: 
					add = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1").get_hist(ticker,pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange'], interval=base_tf, n_bars=100000, extended_session = Main.is_pre_market(dt))
					add.iloc[0]
				except: pass
				else:
					add.drop('symbol', axis = 1, inplace = True)
					add.index = add.index + pd.Timedelta(hours=(13-(time.timezone/3600)))
					if df.empty or add.index[0] > df.index[-1]: df = add
					else: df = pd.concat([df,add[Main.findex(add,df.index[-1]) + 1:]])
			if df.empty: raise TimeoutError
			if dt != None and not Main.is_pre_market(dt):
				try: df = df[:Data.findex(df,dt) + 1 + int(offset*(pd.Timedelta(tf) / pd.Timedelta(base_tf)))]
				except IndexError: raise TimeoutError
			if 'min' not in tf and base_tf == '1min': df = df.between_time('09:30', '15:59')##########
			if 'w' in tf and not Main.is_pre_market(dt):
				last_bar = df.tail(1)
				df = df[:-1]
			df = df.resample(tf,closed = 'left',label = 'left',origin = pd.Timestamp('2008-01-07 09:30:00')).apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
			if 'w' in tf and not Main.is_pre_market(dt): df = pd.concat([df,last_bar])
			if base_tf == '1d' and Main.is_pre_market(dt): 
				pm_bar = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('ticker').loc[ticker]
				pm_price = pm_bar['pm change'] + df.iat[-1,3]
				df = pd.concat([df,pd.DataFrame({'datetime': [dt], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime",drop = True)])
			df = df.dropna()[-bars:]
		except TimeoutError:
			df = pd.DataFrame()
			
		self.df = df
		self.len = len(df)
		
	def np(self,bars,only_close):
		returns = []
		try:
			df = self.df
			if only_close:
				df = df.iloc[:,3]
				partitions = bars//2
			else:
				partitions = 1
			x = df.to_numpy()
			d = np.zeros((x.shape[0]-1))
			for i in range(len(d)): #add ohlc
				d[i] = x[i+1]/x[i] - 1
				
			if partitions != 0:
				# if d.shape[0] == bars-1:
				# 	x = d.reshape(-1, 1)
				# 	#x = normalize(x)
				# 	x = np.flip(x,0)
				# 	if only_close: x = np.column_stack((x, numpy.arange(  x.shape[0])))
				# 	returns = x
				# else:
				for i in list(range(bars,d.shape[0]+1,partitions)) + [bars]:
					try:
						x = d[i-bars:i+1]		
						x = x.reshape(-1, 1)
						#x = normalize(x)
						x = np.flip(x,0)
						#if only_close: x = np.column_stack((x, numpy.arange(  x.shape[0])))
						#x = np.array(x)

						x = torch.tensor(list(x), requires_grad=True).cuda()
						#sequence2 = torch.tensor([1.0, 2.0, 2.5, 3.5], requires_grad=True).cuda()




						x = x.cpu()
						returns.append(x.detach())
					except:
						pass
				
		except: 
			return returns
		
		#self.np = returns.detach().cpu().numpy()
		self.np = returns

	def findex(self,dt):
		dt = Main.format_date(dt)
		if isinstance(self,pd.DataFrame):
			df = self
		else:
			df = self.df
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
		
		
	def get_scores(self,threshold = 0):
		table = []
		for i in range(len(self.scores)):
			score = self.scores[i]
			if score > threshold:
				table.append([self.ticker,i,score])
				

		return table


	
if __name__=='__main__':
	
	
	df = Data('COIN') #create dataframe
	
	df.df = df.df[:100] #do something with a pandas method
	
 #reassign the df to a new DF object
	
	#print(df.findex('2023-04-10'))
	
	
	

# # class CustomDataFrame(pd.DataFrame):
# # 	def __init__(self, data, name):
# # 		super().__init__(data)
# # 		self.name = name























































		# if False:
		# 	plt.figure(figsize=(6, 4))
		# 	plt.subplot(121)
		# 	plt.title("Distance matrix")
		# 	plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
		# 	plt.subplot(122)
		# 	plt.title("Cost matrix")
		# 	plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
		# 	x_path, y_path = zip(*path)
		# 	plt.plot(y_path, x_path)
		# 	plt.show()    
		# 	plt.figure()
		# 	for x_i, y_j in path:
		# 		plt.plot([x_i, y_j], [x[x_i] + .5, y[y_j] - .5], c="C7")
		# 	plt.plot(np.arange(x.shape[0]), x + .5, "-o", c="C3")
		# 	plt.plot(np.arange(y.shape[0]), y - .5, "-o", c="C0")
		# 	plt.axis("off")
		# 	plt.show()
	



# def dp(dist_mat):

# 		N, M = dist_mat.shape
	
# 		# Initialize the cost matrix
# 		cost_mat = np.zeros((N + 1, M + 1))
# 		for i in range(1, N + 1):
# 			cost_mat[i, 0] = np.inf
# 		for i in range(1, M + 1):
# 			cost_mat[0, i] = np.inf

# 		# Fill the cost matrix while keeping traceback information
# 		traceback_mat = np.zeros((N, M))
# 		for i in range(N):
# 			for j in range(M):
# 				penalty = [
# 					cost_mat[i, j],      # match (0)
# 					cost_mat[i, j + 1],  # insertion (1)
# 					cost_mat[i + 1, j]]  # deletion (2)
# 				i_penalty = np.argmin(penalty)
# 				cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
# 				traceback_mat[i, j] = i_penalty

# 		# Traceback from bottom right
# 		i = N - 1
# 		j = M - 1
# 		path = [(i, j)]
# 		while i > 0 or j > 0:
# 			tb_type = traceback_mat[i, j]
# 			if tb_type == 0:
# 				# Match
# 				i = i - 1
# 				j = j - 1
# 			elif tb_type == 1:
# 				# Insertion
# 				i = i - 1
# 			elif tb_type == 2:
# 				# Deletion
# 				j = j - 1
# 			path.append((i, j))

# 		# Strip infinity edges from cost_mat before returning
# 		cost_mat = cost_mat[1:, 1:]
# 		return (path[::-1], cost_mat)








# 	import numpy as np
# 	from scipy.spatial.distance import euclidean

# 	from fastdtw import fastdtw

# 	x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
# 	y = np.array([[2,2], [3,3], [4,4]])
# 	distance, path = fastdtw(x, y, dist=euclidean)
# 	#Main.train('d_EP',.05,200)
# 	#train('d_EP',.1,200)
# 	# df = pd.read_feather('C:/Stocks/local/data/d_EP.feather')
# 	# df = df[df['value'] == 1]
# 	# df = df[['ticker','dt']]
# 	# df['tf'] = 'd'
# 	# df = df.values.tolist()





















# def god(p1,p2):
#     return [x for x in p1 if x not in p2] + [x for x in p2 if x not in p1]












# '''
# df = pd.read_csv("C:/Users/csben/Downloads/america_2023-08-18.csv")
# lis = ""

# text_file = open(r"C:/Stocks/full_ticker_list_list.txt", 'w')
# for i in range(len(df)):
# 	if('/' not in str(df.iloc[i]['Ticker'])):
# 		text_file.write(str(df.iloc[i]['Ticker']) + "\n")  
# text_file.close()'''
# def dp(dist_mat):
# 	"""
# 	Find minimum-cost path through matrix `dist_mat` using dynamic programming.

# 	The cost of a path is defined as the sum of the matrix entries on that
# 	path. See the following for details of the algorithm:

# 	- http://en.wikipedia.org/wiki/Dynamic_time_warping
# 	- https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

# 	The notation in the first reference was followed, while Dan Ellis's code
# 	(second reference) was used to check for correctness. Returns a list of
# 	path indices and the cost matrix.
# 	"""

# 	N, M = dist_mat.shape
	
# 	# Initialize the cost matrix
# 	cost_mat = np.zeros((N + 1, M + 1))
# 	for i in range(1, N + 1):
# 		cost_mat[i, 0] = np.inf
# 	for i in range(1, M + 1):
# 		cost_mat[0, i] = np.inf

# 	# Fill the cost matrix while keeping traceback information
# 	traceback_mat = np.zeros((N, M))
# 	for i in range(N):
# 		for j in range(M):
# 			penalty = [
# 				cost_mat[i, j],      # match (0)
# 				cost_mat[i, j + 1],  # insertion (1)
# 				cost_mat[i + 1, j]]  # deletion (2)
# 			i_penalty = np.argmin(penalty)
# 			cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
# 			traceback_mat[i, j] = i_penalty

# 	# Traceback from bottom right
# 	i = N - 1
# 	j = M - 1
# 	path = [(i, j)]
# 	while i > 0 or j > 0:
# 		tb_type = traceback_mat[i, j]
# 		if tb_type == 0:
# 			# Match
# 			i = i - 1
# 			j = j - 1
# 		elif tb_type == 1:
# 			# Insertion
# 			i = i - 1
# 		elif tb_type == 2:
# 			# Deletion
# 			j = j - 1
# 		path.append((i, j))

# 	def set_birthday(self,birthday):
# 		self.birthday = birthday

# 	def read_birthday(self):



# 	def rolling_change(df):
# 		d = np.zeros((df.shape[0]-1))
# 		for i in range(len(d)):
# 			d[i] = df[i+1]/df[i] - 1
# 		return d
# 	ticker1 = "QQQ"
# 	#ticker2list = ['ENPH','NUE','IOT','KWEB','AAPL','U',"FSLR", "COIN", "AMR", "MRNA", "COST", "T", "W", "ARKK", "X", "CLF", "UUUU"]
# 	ticker2list = ['']
# 	scores = []
# 	start = datetime.datetime.now()
# 	for ticker2 in ticker2list:
# 		try:     
# 			dt1 = None
# 			dt2 = None
# 			df = Main.get(ticker1, 'd', bars=50, dt=dt1)
# 			df = df.iloc[:, 3]
# 			x = df.to_numpy() 
# 			#x = x / np.array(x[0])
# 			#x = df.to_numpy()
# 			df = Main.get(ticker2, 'd', bars=50, dt=dt2)
# 			df = df.iloc[:, 3]
# 			y = df.to_numpy()   
		
# 			x = rolling_change(x)
# 			y = rolling_change(y)
# 			#y = y / np.array(y[0])
# 				# Distance matrix
# 			N = x.shape[0]
# 			M = y.shape[0]
# 			dist_mat = np.zeros((N, M))
# 			for i in range(N):
# 				for j in range(M):
# 					dist_mat[i, j] = abs(x[i] - y[j])
# 							# DTW
# 			path, cost_mat = dp(dist_mat)
				
# 			score = cost_mat[N - 1, M - 1]/(N + M)
# 			scores.append([ticker2,score])

# 	def ident(self):
# 			if False:
# 				plt.figure(figsize=(6, 4))
# 				plt.subplot(121)
# 				plt.title("Distance matrix")
# 				plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
# 				plt.subplot(122)
# 				plt.title("Cost matrix")
# 				plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
# 				x_path, y_path = zip(*path)
# 				plt.plot(y_path, x_path)
# 				plt.show()    
# 				plt.figure()
# 				for x_i, y_j in path:
# 					plt.plot([x_i, y_j], [x[x_i] + .5, y[y_j] - .5], c="C7")
# 				plt.plot(np.arange(x.shape[0]), x + .5, "-o", c="C3")
# 				plt.plot(np.arange(y.shape[0]), y - .5, "-o", c="C0")
# 				plt.axis("off")
# 				plt.show()
# 		except:
# 			pass
			
# 	scores.sort(key=lambda x: x[1])
#student = Student('dd')
#student.set_birthday('september')
#student.read_birthday()

#st = idk('d')
#st.gosh()
#st.ident()

#Main.refill_backtest()


#path = 'C:/Stocks/local/study/historical_setups.feather'

#df = pd.read_feather(path)
#df = df[df['pre_annotation'] != ''].reset_index(drop = True)
#df.to_feather(path)
#path = "C:/Stocks/local/data/d_2/"
#path2 = "C:/Stocks/local/data/god/"
#dir_list = os.listdir(path)
#pbar = tqdm(total = len(dir_list))
#for p in dir_list:
#	df = pd.read_csv(path+p)
#	#df = df.set_index('datetime',drop = True)
#	df['datetime'] = pd.to_datetime(df['datetime'])
#	df.to_feather(path2+(p.split('.')[0]) + '.feather')
#	pbar.update(1)
#pbar.close()
	

#df = pd.read_feather("C:/Stocks/local/study/historical_setups.feather")
#rint(df)
			
				

#path = "C:/Stocks/local/data/d/"
#dir_list = os.listdir(path)
#pbar = tqdm(total = len(dir_list))
#for p in dir_list:
#	df = pd.read_feather(path+p)
#	df = df.reset_index()
#	df.to_feather(path+p)
#	pbar.update(1)
#pbar.close()


#df = (yf.download(tickers = 'QQQ', period = '5d', group_by='ticker', interval = '1m', ignore_tz = True, progress = False, show_errors = False, threads = False, prepost = True))
#path = "C:/Stocks/sync/database/"
#dir_list = os.listdir(path)
#for p in dir_list:
#	d = path + p
	
#	df = pd.read_feather(d)
#	df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)	
#	df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)	
#	for i in range(len(df)):
#		if Main.is_pre_market( df.at[i,'dt']):
#			df.at[i,'dt'] =  df.at[i,'dt'].replace(hour=9, minute=30)
#	df.to_feather(d)










#p = 'C:/Stocks/sync/database/laptop_d_EP.feather'
#df = pd.read_feather(p)
#df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)
#df.to_feather(p)



#ticker = 'AEHR'
#base_tf = '1min'
#dt = None
#exchange = pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange']
#add = TvDatafeed(username="billingsandrewjohn@gmail.com",password="Steprapt04").get_hist(ticker, exchange, interval=base_tf, n_bars=100000, extended_session = True)













#df_traits = pd.read_feather('C:/Stocks/local/account/traits.feather')

#df = pd.read_feather('C:/Stocks/sync/files/full_scan.feather')
#df = df[['Ticker','Pre-market Change','Pre-market Volume','Relative Volume at Time','Exchange']]
#df = df.rename(columns={'Ticker':'ticker','Exchange':'exchange','Pre-market Change':'pm change','Pre-market Volume':'pm volume','Relative Volume at Time':'rvol'})

#df.to_feather('C:/Stocks/sync/files/full_scan.feather')



#traits = df_traits[df_traits['datetime'] >= datetime.datetime.now() - datetime.timedelta(days = 180)].reset_index(drop = True)

#low = int(max(traits['min %']))
#data_table = []
#for thresh in range(low,0,-1):
#	df = traits.copy()
#	for i in range(len(df)):

#		if df.at[i,'min %'] >= thresh:
#			df.at[i,'pnl $'] = df.at[i,'size $'] * - thresh / 100

#	data_table.append([sum(df['pnl $'].tolist()),thresh])
#top =int( max(traits['high %']))
#data_table = []
#for thresh in range(top, 0, -1):
#    df = traits.copy()
#    for i in range(len(df)):

#        if df.at[i,'high %'] >= thresh:
#            df.at[i,'pnl $'] *= (thresh / df.at[i,'pnl %'])
#        elif df.at[i,'pnl $'] > 0:
#            df.at[i,'pnl $'] = 0

#    data_table.append([sum(df['pnl $'].tolist()),thresh])



#df = pd.read_feather('C:/Stocks/local/study/historical_setups.feather')
#coin = pd.read_feather("F:/Stocks/local/data/d/COIN.feather")
##df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
##df.to_feather('C:/Stocks/sync/database/aj_d_EP.feather')
##path = "C:/Stocks/sync/database/"
##dir_list = os.listdir(path)
##for p in dir_list:
##    d = path + p
##    df = pd.read_feather(d)
##    df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
##    df.to_feather(d)

#	historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
#	if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"):
#		shutil.copy(r"C:\Stocks\sync\files\full_scan.feather", r"C:\Stocks\local\study\full_list_minus_annotated.feather")
#	while(len(historical_setups[historical_setups["post_annotation"] == ""]) < 1500):
#		full_list_minus_annotation = pd.read_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
#		full_list_minus_annotation = full_list_minus_annotation.sample(frac=1)
#		for t in range(8):
#			screener.run(ticker=full_list_minus_annotation.iloc[t]["Ticker"], fpath=0)
#		full_list_minus_annotation = full_list_minus_annotation[8:].reset_index(drop=True)
#		full_list_minus_annotation.to_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")



#def create(bar):
#    i = bar[0]
#    df = bar[1]
#    if (os.path.exists(r"C:\Stocks\local\account\charts" + f"\{0}_{i}" + "1min.png") == False):
#        trait_bar = df.iloc[i]
#        ticker = trait_bar['ticker']
#        dt = trait_bar['datetime']
#        tflist = ['1min','h','d']
#        mc = mpf.make_marketcolors(up='g',down='r')
#        s  = mpf.make_mpf_style(marketcolors=mc)

#        for ii  in range(len(tflist)):
#            tf = tflist[ii]
#            string1 = str(ii) + '_' + str(i)  + ".png"
#            p1 = pathlib.Path("C:/Stocks/local/account/charts") / string1
#            datelist = []
#            colorlist = []
#            trades = []
#            for k in range(len(df.iat[i,2])):
#                date = datetime.datetime.strptime(df.iat[i,2][k][1], '%Y-%m-%d %H:%M:%S')
#                if tf == 'd':
#                    date = date.date()
#                val = float(df.iat[i,2][k][2])
#                if val > 0:
#                    colorlist.append('g')
#                    add = pd.DataFrame({
#                            'Datetime':[df.iat[i,2][k][1]], 
#                            'Symbol':[df.iat[i,2][k][0]],
#                            'Action':"Buy",
#                            'Price':[float(df.iat[i,2][k][3])]
#                            })
#                    trades.append(add)
#                else:
#                    colorlist.append('r')
#                datelist.append(date)
#            god = bar[1].iloc[i]['arrow_list']
#            god = [list(x) for x in god]
#            dfall= pd.DataFrame(god, columns=['Datetime', 'Price', 'Color', 'Marker'])
#            dfall['Datetime'] = pd.to_datetime(dfall['Datetime'])
#            dfall = dfall.sort_values('Datetime')
#            colors = []
#            dfsByColor = []
#            for zz in range(len(dfall)):
#                if(dfall.iloc[zz]['Color'] not in colors):
#                    colors.append(dfall.iloc[zz]['Color'])
#            for yy in range(len(colors)):
#                colordf = dfall.loc[dfall['Color'] == colors[yy]] 
#                dfsByColor.append(colordf)
#            startdate = dfall.iloc[0]['Datetime']
#            enddate = dfall.iloc[-1]['Datetime']
#            df1 = Main.get(ticker,tf,dt,100,50)
#            if df1.empty: 
#                shutil.copy(r"C:\Stocks\sync\files\blank.png",p1)
#                continue
				
#            minmax = 300
			
#            times = df1.index.to_list()
#            timesdf = []
#            for _ in range(len(df1)):
#                nextTime = pd.DataFrame({ 
#                    'Datetime':[df1.index[_]]
#                    })
#                timesdf.append(nextTime)
#            mainindidf = pd.concat(timesdf).set_index('Datetime', drop=True)
#            apds = [mpf.make_addplot(mainindidf)]
#            for datafram in dfsByColor:
#                datafram['Datetime'] = pd.to_datetime(datafram['Datetime'])
#                tradelist = []
#                for t in range(len(datafram)): 
#                    tradeTime = datafram.iloc[t]['Datetime']
#                    for q in range(len(times)):
#                        if(q+1 != len(times)):
#                            if(times[q+1] >= tradeTime):
#                                test = pd.DataFrame({
#                                    'Datetime':[times[q]],
#                                    'Marker':[datafram.iloc[t]['Marker']],
#                                    'Price':[float(datafram.iloc[t]['Price'])]
#                                    })
#                                tradelist.append(test)
#                                break
#                        else:
#                            test = pd.DataFrame({
#                                    'Datetime':[times[q]],
#                                    'Marker':[datafram.iloc[t]['Marker']],
#                                    'Price':[float(datafram.iloc[t]['Price'])]
#                                    })
#                            tradelist.append(test)
#                            break
#                df2 = pd.concat(tradelist).reset_index(drop = True)
#                df2['Datetime'] = pd.to_datetime(df2['Datetime'])
#                df2 = df2.sort_values(by=['Datetime'])
#                df2['TradeDate_count'] = df2.groupby("Datetime").cumcount() + 1
#                newdf = (df2.pivot(index='Datetime', columns='TradeDate_count', values="Price")
#                    .rename(columns="price{}".format)
#                    .rename_axis(columns=None))
#                series = mainindidf.merge(newdf, how='left', left_index=True, right_index=True)[newdf.columns]
#                if series.isnull().values.all(axis=0)[0]:
#                    pass
#                else: 
#                    apds.append(mpf.make_addplot(series,type='scatter',markersize=300,alpha = .4,marker=datafram.iloc[0]['Marker'],edgecolors='black', color=datafram.iloc[0]['Color']))
#            if tf != '1min': mav = (10,20,50)
#            else: mav = ()
#            _, axlist = mpf.plot(df1, type='candle', volume=True  , 
#                                    title=str(f'{ticker} , {tf}'), 
#                                    style=s, warn_too_much_data=100000,returnfig = True,figratio = (Main.get_config('Plot chart_aspect_ratio'),1),
#                                    figscale=Main.get_config('Plot chart_size'), panel_ratios = (5,1), mav=mav, 
#                                    tight_layout = True,
#                                    addplot=apds)
#            ax = axlist[0]
#            ax.set_yscale('log')
#            ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
#            plt.savefig(p1, bbox_inches='tight',dpi = Main.get_config('Plot chart_dpi')) 




			   #    c = self.event[2][1]



		#    size = (Main.get_config('Traits fw')*Main.get_config('Traits fs'),Main.get_scal('Traits fh')*Main.get_config('Traits fs'))
		#    if c == 0:
		#        return
		#    plt.clf()
		#    y = [p[5] for p in self.monthly[1:] if not np.isnan(p[c])]
		#    x = [p[c] for p in self.monthly[1:] if not np.isnan(p[c])]
		#    plt.scatter(x,y)
		#    z = np.polyfit(x, y, 1)
		#    p = np.poly1d(z)
		#    plt.plot(x,p(x),"r--")
		#    plt.gcf().set_size_inches(size)
		#    string1 = "traits.png"
		#    p1 = pathlib.Path("C:/Screener/tmp/pnl") / string1
		#    plt.savefig(p1,bbox_inches='tight')
		#    bio1 = io.BytesIO()
		#    image1 = Image.open(r"C:\Screener\tmp\pnl\traits.png")
		#    image1.save(bio1, format="PNG")
		#    self.window["-CHART-"].update(data=bio1.getvalue())
		#elif self.event == '-table_traits-':
		#    i = self.values['-table_traits-'][0]
		#    inp = self.traits_table[i][0]
		#elif self.event == '-table_gainers-' or self.event == '-table_losers-':
		#    if self.event == '-table_gainers-':
		#        df = self.gainers
		#        i = self.values['-table_gainers-'][0]
		#    else:
		#        df = self.losers
		#        i = self.values['-table_losers-'][0]
		#    bar = [i,df,1]
		#    if os.path.exists("C:/Screener/tmp/pnl/charts"):
		#        shutil.rmtree("C:/Screener/tmp/pnl/charts")
		#    os.mkdir("C:/Screener/tmp/pnl/charts")
		#    Plot.create(bar)
		#    bio1 = io.BytesIO()
		#    image1 = Image.open(f'C:/Screener/tmp/pnl/charts/{i}d.png')
		#    image1.save(bio1, format="PNG")
		#    self.window["-CHART-"].update(data=bio1.getvalue())
		#elif self.event == 'Traits':
		#    inp = 'account'
		#    gainers2 = self.df_traits.sort_values(by = ['pnl %'])[:10].reset_index(drop = True)
		#    gainers = pd.DataFrame()
		#    gainers['#'] = gainers2.index + 1
		#    gainers['Ticker'] = gainers2['ticker']
		#    gainers['$'] = gainers2['pnl %'].round(2)
		#    losers2 = self.df_traits.sort_values(by = ['pnl %'] , ascending = False)[:10].reset_index(drop = True)
		#    losers = pd.DataFrame()
		#    losers['#'] = losers2.index + 1
		#    losers['Ticker'] = losers2['ticker']
		#    losers['$'] = losers2['pnl %'].round(2)
		#    self.losers = losers2
		#    self.gainers = gainers2
		#    self.monthly = Traits.build_rolling_traits(self)
		#    traits = Traits.build_traits(self)
		#    self.window["-table_gainers-"].update(gainers.values.tolist())
		#    self.window["-table_losers-"].update(losers.values.tolist())
		#    self.window["-table_traits-"].update(traits)
		#    self.window["-table_monthly-"].update(self.monthly)
		#if inp != False:
		#    bins = 50
		#    if os.path.exists("C:/Screener/laptop.txt"): #if laptop
		#        size = (49,25)
		#    else:
		#        size = (25,10)
		#    if inp == "":
		#        inp = 'p10'
		#    try:
		#        plt.clf()
		#        if ':'  in inp:
		#            inp = inp.split(':')
		#            inp1 = inp[0]
		#            inp2 = inp[1]
		#            x = self.df_traits[inp1].to_list()
		#            y = self.df_traits[inp2].to_list()
		#            plt.scatter(x,y)
		#            z = np.polyfit(x, y, 1)
		#            p = np.poly1d(z)
		#            plt.plot(x,p(x),"r--")
		#        else:
		#            fifty = self.df_traits[inp].dropna().to_list()
		#            plt.hist(fifty, bins, alpha=1, ec='black',label='Percent') 
		#        plt.gcf().set_size_inches(size)
		#        string1 = "traits.png"
		#        p1 = pathlib.Path("C:/Screener/tmp/pnl") / string1
		#        plt.savefig(p1,bbox_inches='tight')
				
		#        bio1 = io.BytesIO()
		#        image1 = Image.open(r"C:\Screener\tmp\pnl\traits.png")
		#        image1.save(bio1, format="PNG")
		#        self.window["-CHART-"].update(data=bio1.getvalue())
		#    except:
		#        pass
	  
