
#import yfinance as yf
#from tqdm import tqdm

#import time
#from tvDatafeed import TvDatafeed
#import pytz
bars = 10
partitions = bars//2
len = 10
print(list(range(bars,len+1,partitions)))

	


	
# # # # # # class DF_Class(pd.DataFrame):
	
	
# # # # # # 	def __init__(self,ticker,tf,dt,bars,offset,value,df,np_df):
# # # # # # 		self.ticker = ticker
# # # # # # 		self.tf = tf
# # # # # # 		self.dt = dt
# # # # # # 		self.value = value
# # # # # # 		self.bars = bars
# # # # # # 		self.offset = offset
		
# # # # # # 		self.np = np_df
# # # # # # 		if df.empty:
# # # # # # 			try:
# # # # # # 				if len(tf) == 1: tf = '1' + tf
# # # # # # 				dt = Data.format_date(dt)
# # # # # # 				if 'd' in tf or 'w' in tf: base_tf = '1d'
# # # # # # 				else: base_tf = '1min'
# # # # # # 				#try: df = feather.read_feather(Data.data_path(ticker,tf))
# # # # # # 				try: df = feather.read_feather(Data.data_path(ticker,tf)).set_index('datetime',drop = True)
# # # # # # 				except FileNotFoundError: df = pd.DataFrame()
# # # # # # 				if (df.empty or (dt != None and (dt < df.index[0] or dt > df.index[-1]))) and not (base_tf == '1d' and Data.is_pre_market(dt)): 
# # # # # # 					try: 
# # # # # # 						add = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1").get_hist(ticker,pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange'], interval=base_tf, n_bars=100000, extended_session = Data.is_pre_market(dt))
# # # # # # 						add.iloc[0]
# # # # # # 					except: pass
# # # # # # 					else:
# # # # # # 						add.drop('symbol', axis = 1, inplace = True)
# # # # # # 						add.index = add.index + pd.Timedelta(hours=(13-(time.timezone/3600)))
# # # # # # 						if df.empty or add.index[0] > df.index[-1]: df = add
# # # # # # 						else: df = pd.concat([df,add[Data.findex(add,df.index[-1]) + 1:]])
# # # # # # 				if df.empty: raise TimeoutError
# # # # # # 				if dt != None and not Data.is_pre_market(dt):
# # # # # # 					try: df = df[:DF_Class.findex(df,dt) + 1 + int(offset*(pd.Timedelta(tf) / pd.Timedelta(base_tf)))]
# # # # # # 					except IndexError: raise TimeoutError
# # # # # # 				if 'min' not in tf and base_tf == '1min': df = df.between_time('09:30', '15:59')##########
# # # # # # 				if 'w' in tf and not Data.is_pre_market(dt):
# # # # # # 					last_bar = df.tail(1)
# # # # # # 					df = df[:-1]
# # # # # # 				df = df.resample(tf,closed = 'left',label = 'left',origin = pd.Timestamp('2008-01-07 09:30:00')).apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
# # # # # # 				if 'w' in tf and not Data.is_pre_market(dt): df = pd.concat([df,last_bar])
# # # # # # 				if base_tf == '1d' and Data.is_pre_market(dt): 
# # # # # # 					pm_bar = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('ticker').loc[ticker]
# # # # # # 					pm_price = pm_bar['pm change'] + df.iat[-1,3]
# # # # # # 					df = pd.concat([df,pd.DataFrame({'datetime': [dt], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime",drop = True)])
# # # # # # 				df = df.dropna()[-bars:]
# # # # # # 			except TimeoutError:
# # # # # # 				pass
			
# # # # # # 		super().__init__(df)
		
# # # # # # 	def __getattribute__(self, name):
# # # # # # 		return super().__getattribute__(name)
	
# # # # # # 	def __getattr__(self, name):
# # # # # # 		#print(name)
# # # # # # 		raise AttributeError
	
# # # # # # 	def scores_table(self,threshold = 0):
# # # # # # 		table = []
# # # # # # 		for i in range(len(self.scores)):
# # # # # # 			score = self.scores[i]
# # # # # # 			if score > threshold:
# # # # # # 				table.append([self.ticker,self.dt,score])
				

		
# # # # # # 		return table
	
# # # # # # 	def preload_np(self,bars,use_1 = False):
# # # # # # 		df = self
# # # # # # 		#print(df)
# # # # # # 		if use_1:
# # # # # # 			df = df.iloc[:,3]
# # # # # # 		x = df.to_numpy()
# # # # # # 		d = np.zeros((x.shape[0]-1))
# # # # # # 		for i in range(len(d)): #add ohlc
# # # # # # 			d[i] = x[i+1]/x[i] - 1
# # # # # # 		partitions = bars//2
# # # # # # 		if partitions == 0:
# # # # # # 			self.np = []
# # # # # # 		else:
# # # # # # 			returns = []
# # # # # # 			for i in range(bars,d.shape[0],partitions):
# # # # # # 				try:
# # # # # # 					d = d[i-bars:i]		
# # # # # # 					d = preprocessing.normalize()
# # # # # # 					transformer = SymbolicAggregateApproximation()
# # # # # # 					d = transformer.transform(d)
# # # # # # 					returns.append(d)
# # # # # # 				except:
# # # # # # 					pass
		
# # # # # # 			self.np = returns
			
# # # # # # 	def __str__(self):
# # # # # # 		return f'{super().copy()} {self.ticker} {self.tf} {self.value}'
		
# # # # # # 	def findex(self,dt):
# # # # # # 		dt = Data.format_date(dt)
# # # # # # 		df = self
# # # # # # 		i = int(len(df)/2)
# # # # # # 		k = int(i/2)
# # # # # # 		while k != 0:
# # # # # # 			date = df.index[i].to_pydatetime()
# # # # # # 			if date > dt: i -= k
# # # # # # 			elif date < dt: i += k
# # # # # # 			k = int(k/2)
# # # # # # 		while df.index[i].to_pydatetime() < dt: i += 1
# # # # # # 		while df.index[i].to_pydatetime() > dt: i -= 1
# # # # # # 		return i
	




# class God():
    
# 	def __init__ (self):
# 		self.df = pd.read_feather('C:/Stocks/match_training_data.feather')

# df  = pd.read_feather('C:/Stocks/match_training_data.feather')
# df.ticker = 10
# df = df.copy()
# print(df.ticker)
# # class Train:


# # 	def add_setup(ticker1,dt1,ticker2, dt2, val):
# # 		try: df = pd.read_feather('C:/Stocks/match_training_data.feather')
# # 		except: df = pd.DataFrame()

# # 		add = pd.DataFrame({'ticker1':[ticker1], 'dt1':[dt1], 'ticker2':[ticker2], 'dt2':[dt2], 'val':[val]})

# # 		df = pd.concat([df,add]).reset_index(drop = True)
# # 		df.to_feather('C:/Stocks/match_training_data.feather')

# # 		print(df)


# # 	def run(self):
# # 		with Pool(int(data.get_config('Data cpu_cores'))) as self.pool:
# # 			self.chart_height = data.get_config('Trainer image_box_height')
# # 			self.chart_width = data.get_config('Trainer image_box_width')/2
# # 			sg.theme('DarkGrey')
# # 			self.data = []
# # 			self.i = 0
# # 			self.full_ticker_list = screener.get('full')
# # 			self.init = True
# # 			self.preload_amount = 10
# # 			self.preload(self)
# # 			self.update(self)
# # 			while True:
# # 				self.event, self.values = self.window.read()

# # 				bar = self.data[self.i]
# # 				Train.add_setup(bar[0].ticker,bar[0].dt,bar[1].ticker,bar[1].dt,self.event)
# # 				self.i += 1
# # 				self.preload(self)
# # 				self.update(self)

				

				

			


# # 	def preload(self):

# # 		while len(self.data) < self.i + self.preload_amount: self.data.append([Sample(self.pool),Sample(self.pool)])
# # 		print(self.data)
# # 	def update(self):
# # 		graph1 = sg.Graph(canvas_size = (self.chart_width, self.chart_height), graph_bottom_left = (0, 0), graph_top_right = (self.chart_width, 
# # 				self.chart_height), key = '-chart1-', background_color = 'grey')
# # 		graph2 = sg.Graph(canvas_size = (self.chart_width, self.chart_height), graph_bottom_left = (0, 0), graph_top_right = (self.chart_width, 
# # 				self.chart_height), key = '-chart2-', background_color = 'grey')
# # 		if self.init:
# # 			self.init = False
# # 			layout = [[graph1,graph2],[sg.Button('Yes'),sg.Button('No')]]
# # 			self.window = sg.Window("Match Trainer",layout,margins = (10,10), scaling = data.get_config('Trainer ui_scale'), finalize = True)
			
# # 		self.window['-chart1-'].draw_image(data=self.data[self.i][0].image.get().getvalue(), location=(0, self.chart_height))
# # 		self.window['-chart2-'].draw_image(data=self.data[self.i][1].image.get().getvalue(), location=(0, self.chart_height))

# # 		self.window.maximize()

		

# # class Sample:

# # 	def plot(df):


# # 		_, axlist = mpf.plot(df, type = 'candle', volume=True, style = mpf.make_mpf_style(marketcolors = mpf.make_marketcolors(up = 'g', down = 'r')), warn_too_much_data=100000, returnfig = True, figratio = (data.get_config('Trainer chart_aspect_ratio')/2,1),
# # 		figscale = data.get_config('Trainer chart_size'), tight_layout = True,axisoff=True)
# # 		ax = axlist[0]
# # 		ax.set_yscale('log')
# # 		ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
# # 		buf = io.BytesIO()
# # 		plt.savefig(buf, bbox_inches='tight',dpi = data.get_config('Trainer chart_dpi'))
# # 		buf.seek(0)
# # 		return buf

# # 	def __init__(self,pool):
# # 		while True:
# # 			try:
# # 				sample_size = 50
# # 				full_ticker_list = screener.get('full')
# # 				self.ticker = full_ticker_list[random.randint(20,len(full_ticker_list) - 1)]
# # 				df = data.get(self.ticker)
# # 				index = random.randint(0,len(df)-1)
		

# # 				self.df = df[index-sample_size:index]
# # 				self.dt = df.index[-1]
# # 				#print(self.df)
# # 				self.image = pool.apply_async(Sample.plot,[self.df])
# # 			except: pass
# # 			else: break


# # if __name__ == '__main__':
# # 	Train.run(Train)

# # #class Match:



# # #	def train():

# # #		while True






	
	
# #	def match(ticker, tf = 'd', dt = None):
# #		start_time = time.time()
# #		for i in range(1):
# #			df = data.get(ticker, tf, dt, bars=40)
# #			print(df)
# #		print("--- %s seconds ---" % (time.time() - start_time))
		
# #if __name__ == "__main__":

# #	Match.match('IOT', 'd', '2023-06-02')



# #---------------------------------------------------------------------------------------------

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

# 	# Strip infinity edges from cost_mat before returning
# 	cost_mat = cost_mat[1:, 1:]
# 	return (path[::-1], cost_mat)



# def god(bar):
# 	ticker2, x = bar
# 	try:
# 		df = data.get(ticker2, 'd', bars=10, dt=None)

# 		df = df.iloc[:, 3]
# 		y = df.to_numpy()   	
# 		y = rolling_change(y)
		
# 		N = x.shape[0]
# 		M = y.shape[0]
# 		dist_mat = np.zeros((N, M))
# 		for i in range(N):
# 			for j in range(M):
# 				dist_mat[i, j] = abs(x[i] - y[j])
# 						# DTW
# 		path, cost_mat = dp(dist_mat)
				
# 		score = cost_mat[N - 1, M - 1]/(N + M)
# 		return [ticker2,score]


# 		#print("Alignment cost: {:.4f}".format(cost_mat[N - 1, M - 1]))
# 		#print("Normalized alignment cost: {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))
# 		if False:
# 			plt.figure(figsize=(6, 4))
# 			plt.subplot(121)
# 			plt.title("Distance matrix")
# 			plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
# 			plt.subplot(122)
# 			plt.title("Cost matrix")
# 			plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
# 			x_path, y_path = zip(*path)
# 			plt.plot(y_path, x_path)
# 			plt.show()    
# 			plt.figure()
# 			for x_i, y_j in path:
# 				plt.plot([x_i, y_j], [x[x_i] + .5, y[y_j] - .5], c="C7")
# 			plt.plot(np.arange(x.shape[0]), x + .5, "-o", c="C3")
# 			plt.plot(np.arange(y.shape[0]), y - .5, "-o", c="C0")
# 			plt.axis("off")
# 			plt.show()
		

# 	except: return ['god',100000]

# def rolling_change(df):
# 	d = np.zeros((df.shape[0]-1))
# 	for i in range(len(d)):
# 		d[i] = df[i+1]/df[i] - 1
# 	return d

# if __name__ == '__main__':
# 	lis  = screener.get('full')
	

# 	ticker1 = 'JBL'
# 	dt1 = None
# 	dt2 = None
# 	df = data.get(ticker1, 'd', bars=10, dt=dt1)
# 	df = df.iloc[:, 3]
# 	x = df.to_numpy() 
# 	x = rolling_change(x)
	
# 	lis = [[tick, x] for tick in lis]

# 	scores = data.pool(god,lis)
# 	scores.sort(key=lambda x: x[1])
# 	[print(x) for x,s in scores[:50]]
# 	#print(scores)
# #student = Student('dd')
# #student.set_birthday('september')
# #student.read_birthday()

# #st = idk('d')
# #st.gosh()
# #st.ident()

# #data.refill_backtest()


# #print(pd.read_feather('C:/Stocks/local/data/1min/ROKU.feather'))
# #path = 'C:/Stocks/local/study/historical_setups.feather'

# #df = pd.read_feather(path)
# #df = df[df['pre_annotation'] != ''].reset_index(drop = True)
# #df.to_feather(path)
# #path = "C:/Stocks/local/data/d_2/"
# #path2 = "C:/Stocks/local/data/god/"
# #dir_list = os.listdir(path)
# #pbar = tqdm(total = len(dir_list))
# #for p in dir_list:
# #	df = pd.read_csv(path+p)
# #	#df = df.set_index('datetime',drop = True)
# #	df['datetime'] = pd.to_datetime(df['datetime'])
# #	df.to_feather(path2+(p.split('.')[0]) + '.feather')
# #	pbar.update(1)
# #pbar.close()
	

# #df = pd.read_feather("C:/Stocks/local/study/historical_setups.feather")
# #rint(df)
			
				

# #path = "C:/Stocks/local/data/d/"
# #dir_list = os.listdir(path)
# #pbar = tqdm(total = len(dir_list))
# #for p in dir_list:
# #	df = pd.read_feather(path+p)
# #	df = df.reset_index()
# #	df.to_feather(path+p)
# #	pbar.update(1)
# #pbar.close()


# #df = (yf.download(tickers = 'QQQ', period = '5d', group_by='ticker', interval = '1m', ignore_tz = True, progress = False, show_errors = False, threads = False, prepost = True))
# #print(df)
# #print(data.get_requirements('',df,'d_EP'))
# #path = "C:/Stocks/sync/database/"
# #dir_list = os.listdir(path)
# #for p in dir_list:
# #	d = path + p
	
# #	df = pd.read_feather(d)
# #	df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)	
# #	df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)	
# #	for i in range(len(df)):
# #		if data.is_pre_market( df.at[i,'dt']):
# #			df.at[i,'dt'] =  df.at[i,'dt'].replace(hour=9, minute=30)
# #	df.to_feather(d)










# #p = 'C:/Stocks/sync/database/laptop_d_EP.feather'
# #df = pd.read_feather(p)
# #df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)
# #df.to_feather(p)
# #print(df)



# #ticker = 'AEHR'
# #base_tf = '1min'
# #dt = None
# #exchange = pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange']
# #add = TvDatafeed(username="billingsandrewjohn@gmail.com",password="Steprapt04").get_hist(ticker, exchange, interval=base_tf, n_bars=100000, extended_session = True)
# #print(add)













# #print((data.get(tf = 'd',dt = datetime.datetime(2023,8,10,9,15),bars = 100)).to_string())
# #df_traits = pd.read_feather('C:/Stocks/local/account/traits.feather')
# #print(pd.read_feather('C:/Stocks/local/study/current_setups.feather'))
# #print(data.get(tf = '2min', dt = datetime.datetime(2022,8,10,9,45)))

# #df = pd.read_feather('C:/Stocks/sync/files/full_scan.feather')
# #print(df.columns)
# #df = df[['Ticker','Pre-market Change','Pre-market Volume','Relative Volume at Time','Exchange']]
# #df = df.rename(columns={'Ticker':'ticker','Exchange':'exchange','Pre-market Change':'pm change','Pre-market Volume':'pm volume','Relative Volume at Time':'rvol'})

# #print(df)
# #df.to_feather('C:/Stocks/sync/files/full_scan.feather')



# #traits = df_traits[df_traits['datetime'] >= datetime.datetime.now() - datetime.timedelta(days = 180)].reset_index(drop = True)

# #low = int(max(traits['min %']))
# #data_table = []
# #for thresh in range(low,0,-1):
# #	df = traits.copy()
# #	for i in range(len(df)):

# #		if df.at[i,'min %'] >= thresh:
# #			df.at[i,'pnl $'] = df.at[i,'size $'] * - thresh / 100

# #	data_table.append([sum(df['pnl $'].tolist()),thresh])
# #print(data_table)
# #print(traits)
# #top =int( max(traits['high %']))
# #data_table = []
# #for thresh in range(top, 0, -1):
# #    df = traits.copy()
# #    for i in range(len(df)):

# #        if df.at[i,'high %'] >= thresh:
# #            df.at[i,'pnl $'] *= (thresh / df.at[i,'pnl %'])
# #        elif df.at[i,'pnl $'] > 0:
# #            df.at[i,'pnl $'] = 0

# #    data_table.append([sum(df['pnl $'].tolist()),thresh])

# #print(data_table)


# ##print(data.get_scale('Account fw'))
# #df = pd.read_feather('C:/Stocks/local/study/historical_setups.feather')
# #coin = pd.read_feather("F:/Stocks/local/data/d/COIN.feather")
# ##print(df)
# ##print(coin.to_string())
# ##df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
# ##df.to_feather('C:/Stocks/sync/database/aj_d_EP.feather')
# ##path = "C:/Stocks/sync/database/"
# ##dir_list = os.listdir(path)
# ##for p in dir_list:
# ##    d = path + p
# ##    df = pd.read_feather(d)
# ##    df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
# ##    df.to_feather(d)

# #if __name__ == "__main__":
# #	historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
# #	if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"):
# #		shutil.copy(r"C:\Stocks\sync\files\full_scan.feather", r"C:\Stocks\local\study\full_list_minus_annotated.feather")
# #	while(len(historical_setups[historical_setups["post_annotation"] == ""]) < 1500):
# #		full_list_minus_annotation = pd.read_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
# #		print(len(historical_setups[historical_setups["post_annotation"] == ""]))
# #		full_list_minus_annotation = full_list_minus_annotation.sample(frac=1)
# #		for t in range(8):
# #			screener.run(ticker=full_list_minus_annotation.iloc[t]["Ticker"], fpath=0)
# #		full_list_minus_annotation = full_list_minus_annotation[8:].reset_index(drop=True)
# #		full_list_minus_annotation.to_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")



# #def create(bar):
# #    i = bar[0]
# #    df = bar[1]
# #    if (os.path.exists(r"C:\Stocks\local\account\charts" + f"\{0}_{i}" + "1min.png") == False):
# #        trait_bar = df.iloc[i]
# #        ticker = trait_bar['ticker']
# #        dt = trait_bar['datetime']
# #        tflist = ['1min','h','d']
# #        mc = mpf.make_marketcolors(up='g',down='r')
# #        s  = mpf.make_mpf_style(marketcolors=mc)

# #        for ii  in range(len(tflist)):
# #            tf = tflist[ii]
# #            string1 = str(ii) + '_' + str(i)  + ".png"
# #            p1 = pathlib.Path("C:/Stocks/local/account/charts") / string1
# #            datelist = []
# #            colorlist = []
# #            trades = []
# #            for k in range(len(df.iat[i,2])):
# #                date = datetime.datetime.strptime(df.iat[i,2][k][1], '%Y-%m-%d %H:%M:%S')
# #                if tf == 'd':
# #                    date = date.date()
# #                val = float(df.iat[i,2][k][2])
# #                if val > 0:
# #                    colorlist.append('g')
# #                    add = pd.DataFrame({
# #                            'Datetime':[df.iat[i,2][k][1]], 
# #                            'Symbol':[df.iat[i,2][k][0]],
# #                            'Action':"Buy",
# #                            'Price':[float(df.iat[i,2][k][3])]
# #                            })
# #                    trades.append(add)
# #                else:
# #                    colorlist.append('r')
# #                datelist.append(date)
# #            god = bar[1].iloc[i]['arrow_list']
# #            god = [list(x) for x in god]
# #            dfall= pd.DataFrame(god, columns=['Datetime', 'Price', 'Color', 'Marker'])
# #            dfall['Datetime'] = pd.to_datetime(dfall['Datetime'])
# #            dfall = dfall.sort_values('Datetime')
# #            colors = []
# #            dfsByColor = []
# #            for zz in range(len(dfall)):
# #                if(dfall.iloc[zz]['Color'] not in colors):
# #                    colors.append(dfall.iloc[zz]['Color'])
# #            for yy in range(len(colors)):
# #                colordf = dfall.loc[dfall['Color'] == colors[yy]] 
# #                dfsByColor.append(colordf)
# #            startdate = dfall.iloc[0]['Datetime']
# #            enddate = dfall.iloc[-1]['Datetime']
# #            df1 = data.get(ticker,tf,dt,100,50)
# #            if df1.empty: 
# #                shutil.copy(r"C:\Stocks\sync\files\blank.png",p1)
# #                continue
				
# #            minmax = 300
			
# #            times = df1.index.to_list()
# #            timesdf = []
# #            for _ in range(len(df1)):
# #                nextTime = pd.DataFrame({ 
# #                    'Datetime':[df1.index[_]]
# #                    })
# #                timesdf.append(nextTime)
# #            mainindidf = pd.concat(timesdf).set_index('Datetime', drop=True)
# #            apds = [mpf.make_addplot(mainindidf)]
# #            for datafram in dfsByColor:
# #                datafram['Datetime'] = pd.to_datetime(datafram['Datetime'])
# #                tradelist = []
# #                for t in range(len(datafram)): 
# #                    tradeTime = datafram.iloc[t]['Datetime']
# #                    for q in range(len(times)):
# #                        if(q+1 != len(times)):
# #                            if(times[q+1] >= tradeTime):
# #                                test = pd.DataFrame({
# #                                    'Datetime':[times[q]],
# #                                    'Marker':[datafram.iloc[t]['Marker']],
# #                                    'Price':[float(datafram.iloc[t]['Price'])]
# #                                    })
# #                                tradelist.append(test)
# #                                break
# #                        else:
# #                            test = pd.DataFrame({
# #                                    'Datetime':[times[q]],
# #                                    'Marker':[datafram.iloc[t]['Marker']],
# #                                    'Price':[float(datafram.iloc[t]['Price'])]
# #                                    })
# #                            tradelist.append(test)
# #                            break
# #                df2 = pd.concat(tradelist).reset_index(drop = True)
# #                df2['Datetime'] = pd.to_datetime(df2['Datetime'])
# #                df2 = df2.sort_values(by=['Datetime'])
# #                df2['TradeDate_count'] = df2.groupby("Datetime").cumcount() + 1
# #                newdf = (df2.pivot(index='Datetime', columns='TradeDate_count', values="Price")
# #                    .rename(columns="price{}".format)
# #                    .rename_axis(columns=None))
# #                series = mainindidf.merge(newdf, how='left', left_index=True, right_index=True)[newdf.columns]
# #                if series.isnull().values.all(axis=0)[0]:
# #                    pass
# #                else: 
# #                    apds.append(mpf.make_addplot(series,type='scatter',markersize=300,alpha = .4,marker=datafram.iloc[0]['Marker'],edgecolors='black', color=datafram.iloc[0]['Color']))
# #            if tf != '1min': mav = (10,20,50)
# #            else: mav = ()
# #            _, axlist = mpf.plot(df1, type='candle', volume=True  , 
# #                                    title=str(f'{ticker} , {tf}'), 
# #                                    style=s, warn_too_much_data=100000,returnfig = True,figratio = (data.get_config('Plot chart_aspect_ratio'),1),
# #                                    figscale=data.get_config('Plot chart_size'), panel_ratios = (5,1), mav=mav, 
# #                                    tight_layout = True,
# #                                    addplot=apds)
# #            ax = axlist[0]
# #            ax.set_yscale('log')
# #            ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
# #            plt.savefig(p1, bbox_inches='tight',dpi = data.get_config('Plot chart_dpi')) 




# 			   #    c = self.event[2][1]



# 		#    size = (data.get_config('Traits fw')*data.get_config('Traits fs'),data.get_scal('Traits fh')*data.get_config('Traits fs'))
# 		#    if c == 0:
# 		#        return
# 		#    plt.clf()
# 		#    y = [p[5] for p in self.monthly[1:] if not np.isnan(p[c])]
# 		#    x = [p[c] for p in self.monthly[1:] if not np.isnan(p[c])]
# 		#    plt.scatter(x,y)
# 		#    z = np.polyfit(x, y, 1)
# 		#    p = np.poly1d(z)
# 		#    plt.plot(x,p(x),"r--")
# 		#    plt.gcf().set_size_inches(size)
# 		#    string1 = "traits.png"
# 		#    p1 = pathlib.Path("C:/Screener/tmp/pnl") / string1
# 		#    plt.savefig(p1,bbox_inches='tight')
# 		#    bio1 = io.BytesIO()
# 		#    image1 = Image.open(r"C:\Screener\tmp\pnl\traits.png")
# 		#    image1.save(bio1, format="PNG")
# 		#    self.window["-CHART-"].update(data=bio1.getvalue())
# 		#elif self.event == '-table_traits-':
# 		#    i = self.values['-table_traits-'][0]
# 		#    inp = self.traits_table[i][0]
# 		#elif self.event == '-table_gainers-' or self.event == '-table_losers-':
# 		#    if self.event == '-table_gainers-':
# 		#        df = self.gainers
# 		#        i = self.values['-table_gainers-'][0]
# 		#    else:
# 		#        df = self.losers
# 		#        i = self.values['-table_losers-'][0]
# 		#    bar = [i,df,1]
# 		#    if os.path.exists("C:/Screener/tmp/pnl/charts"):
# 		#        shutil.rmtree("C:/Screener/tmp/pnl/charts")
# 		#    os.mkdir("C:/Screener/tmp/pnl/charts")
# 		#    Plot.create(bar)
# 		#    bio1 = io.BytesIO()
# 		#    image1 = Image.open(f'C:/Screener/tmp/pnl/charts/{i}d.png')
# 		#    image1.save(bio1, format="PNG")
# 		#    self.window["-CHART-"].update(data=bio1.getvalue())
# 		#elif self.event == 'Traits':
# 		#    inp = 'account'
# 		#    gainers2 = self.df_traits.sort_values(by = ['pnl %'])[:10].reset_index(drop = True)
# 		#    gainers = pd.DataFrame()
# 		#    gainers['#'] = gainers2.index + 1
# 		#    gainers['Ticker'] = gainers2['ticker']
# 		#    gainers['$'] = gainers2['pnl %'].round(2)
# 		#    losers2 = self.df_traits.sort_values(by = ['pnl %'] , ascending = False)[:10].reset_index(drop = True)
# 		#    losers = pd.DataFrame()
# 		#    losers['#'] = losers2.index + 1
# 		#    losers['Ticker'] = losers2['ticker']
# 		#    losers['$'] = losers2['pnl %'].round(2)
# 		#    self.losers = losers2
# 		#    self.gainers = gainers2
# 		#    self.monthly = Traits.build_rolling_traits(self)
# 		#    traits = Traits.build_traits(self)
# 		#    self.window["-table_gainers-"].update(gainers.values.tolist())
# 		#    self.window["-table_losers-"].update(losers.values.tolist())
# 		#    self.window["-table_traits-"].update(traits)
# 		#    self.window["-table_monthly-"].update(self.monthly)
# 		#if inp != False:
# 		#    bins = 50
# 		#    if os.path.exists("C:/Screener/laptop.txt"): #if laptop
# 		#        size = (49,25)
# 		#    else:
# 		#        size = (25,10)
# 		#    if inp == "":
# 		#        inp = 'p10'
# 		#    try:
# 		#        plt.clf()
# 		#        if ':'  in inp:
# 		#            inp = inp.split(':')
# 		#            inp1 = inp[0]
# 		#            inp2 = inp[1]
# 		#            x = self.df_traits[inp1].to_list()
# 		#            y = self.df_traits[inp2].to_list()
# 		#            plt.scatter(x,y)
# 		#            z = np.polyfit(x, y, 1)
# 		#            p = np.poly1d(z)
# 		#            plt.plot(x,p(x),"r--")
# 		#        else:
# 		#            fifty = self.df_traits[inp].dropna().to_list()
# 		#            plt.hist(fifty, bins, alpha=1, ec='black',label='Percent') 
# 		#        plt.gcf().set_size_inches(size)
# 		#        string1 = "traits.png"
# 		#        p1 = pathlib.Path("C:/Screener/tmp/pnl") / string1
# 		#        plt.savefig(p1,bbox_inches='tight')
				
# 		#        bio1 = io.BytesIO()
# 		#        image1 = Image.open(r"C:\Screener\tmp\pnl\traits.png")
# 		#        image1.save(bio1, format="PNG")
# 		#        self.window["-CHART-"].update(data=bio1.getvalue())
# 		#    except:
# 		#        pass
	  
# '''
# df = pd.read_csv("C:/Users/csben/Downloads/america_2023-08-18.csv")
# lis = ""

# text_file = open(r"C:/Stocks/full_ticker_list_list.txt", 'w')
# for i in range(len(df)):
# 	if('/' not in str(df.iloc[i]['Ticker'])):
# 		text_file.write(str(df.iloc[i]['Ticker']) + "\n")  
# text_file.close()'''
