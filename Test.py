import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from Data import Data as data
from pyarrow import feather
from tvDatafeed import TvDatafeed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from multiprocessing import Pool, current_process
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
import websocket, datetime, os, pyarrow, shutil,statistics, warnings, math, time, pytz, tensorflow, random


def sample(st,use):

	data.consolidate_database()
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




def worker(bar):
	ticker, dt, value, tf = bar
	
	ss = 50
	try:
		df = data.get(ticker,tf,dt,ss)
		df = df.drop(columns = ['ticker','volume'])
		if len(df) < ss:
			add = pd.DataFrame(df.iat[-1,3], index=np.arange(ss - len(df)), columns=df.columns)
			df = pd.concat([add,df])
		df = df.values.tolist()
		df = np.array(df)
		o = df[-1,0]
		#v = df[-2,4]
		for ii in range(1,3): df[-1,ii] = o
		#df[-1,4] = 0

		df = df/np.array(o)
		#df[4] = df[4]/np.array(v)
		df = np.log(df)
		

		np_array = df
		


	except:
		df = pd.DataFrame()
		np_array = np.zeros((ss,4))
		print('god')
		
	return [ticker,dt,tf,df,'',0,np_array,value]



def train(st, percent_yes, epochs):
	df = pd.read_feather('C:/Stocks/local/data/' + st + '.feather')
	ones = len(df[df['value'] ==1])
	if ones < 150: 
		print(f'{st} cannot be trained with only {ones} positives')
		return
	df  = sample(st, percent_yes)
	df = df[['ticker','dt','value']]
	df['tf'] = 'd'
	df = df.values.tolist()
	info = data.pool(worker,df)
	x = np.array([x[6] for x in info])
	y = np.array([x[7] for x in info])
	print(x)
	model = Sequential([Bidirectional(LSTM(64, input_shape = (x.shape[1], x.shape[2]), return_sequences = True,),),Dropout(0.2), Bidirectional(LSTM(32)), Dense(3, activation = 'softmax'),])
	model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 1e-3), metrics = ['accuracy'])
	model.fit(x, y, epochs = epochs, batch_size = 64, validation_split = .2,)
	model.save('C:/Stocks/sync/models/model_' + st)
	tensorflow.keras.backend.clear_session()	
		

def worker2(bar):
	info, st = bar
	model = data.load_model(st)
	for i in range(len(info)):
		np_array = info[i][6]
		if np_array == None: continue
		score = model.score(np_array)[0][1]
		
		info[i][4] = st
		info[i][5] = score
		
	return info

def score(df:list,st):
	if type(st) != list: st = [st]
	pool = Pool(processes = int(data.get_config('Data cpu_cores')))
	info = data.pool(worker,df)
	
	info = [item for sublist in pool.map(worker2,[[info,s] for s in st]) for item in sublist]

	

class Get:
	
	def __init__(self,ticker,tf,dt = None,bars = None,offset = 0):
		self.ticker = ticker
		self.tf = tf
		self.date = dt
		
		self.df = None
		


		try:
			if len(tf) == 1: tf = '1' + tf
			dt = Data.format_date(dt)
			if 'd' in tf or 'w' in tf: base_tf = '1d'
			else: base_tf = '1min'
			#try: df = feather.read_feather(Data.data_path(ticker,tf))
			try: df = feather.read_feather(Data.data_path(ticker,tf)).set_index('datetime',drop = True)
			except FileNotFoundError: df = pd.DataFrame()
			if (df.empty or (dt != None and (dt < df.index[0] or dt > df.index[-1]))) and not (base_tf == '1d' and Data.is_pre_market(dt)): 
				try: 
					add = TvDatafeed(username="cs.benliu@gmail.com",password="tltShort!1").get_hist(ticker,pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange'], interval=base_tf, n_bars=100000, extended_session = Data.is_pre_market(dt))
					add.iloc[0]
				except: pass
				else:
					add.drop('symbol', axis = 1, inplace = True)
					add.index = add.index + pd.Timedelta(hours=(13-(time.timezone/3600)))
					if df.empty or add.index[0] > df.index[-1]: df = add
					else: df = pd.concat([df,add[Data.findex(add,df.index[-1]) + 1:]])
			if df.empty: raise TimeoutError
			if dt != None and not Data.is_pre_market(dt):
				try: df = df[:Data.findex(df,dt) + 1 + int(offset*(pd.Timedelta(tf) / pd.Timedelta(base_tf)))]
				except IndexError: raise TimeoutError
			if 'min' not in tf and base_tf == '1min': df = df.between_time('09:30', '15:59')##########
			if 'w' in tf and not Data.is_pre_market(dt):
				last_bar = df.tail(1)
				df = df[:-1]
			df = df.resample(tf,closed = 'left',label = 'left',origin = pd.Timestamp('2008-01-07 09:30:00')).apply({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
			if 'w' in tf and not Data.is_pre_market(dt): df = pd.concat([df,last_bar])
			if base_tf == '1d' and Data.is_pre_market(dt): 
				pm_bar = pd.read_feather('C:/Stocks/sync/files/current_scan.feather').set_index('ticker').loc[ticker]
				pm_price = pm_bar['pm change'] + df.iat[-1,3]
				df = pd.concat([df,pd.DataFrame({'datetime': [dt], 'open': [pm_price],'high': [pm_price], 'low': [pm_price], 'close': [pm_price], 'volume': [pm_bar['pm volume']]}).set_index("datetime",drop = True)])
			df['ticker'] = ticker
			return df.dropna()[-bars:]
		except TimeoutError: return pd.DataFrame()





		
	def get_numpy_array(self):
		
		ss = 50
		
		df = self.df.drop(columns = ['volume'])
		if len(df) < ss:
			add = pd.DataFrame(df.iat[-1,3], index=np.arange(ss - len(df)), columns=df.columns)
			df = pd.concat([add,df])
		df = df.values.tolist()
		df = np.array(df)
		o = df[-1,0]
		#v = df[-2,4]
		for ii in range(1,3): df[-1,ii] = o
		#df[-1,4] = 0

		df = df/np.array(o)
		#df[4] = df[4]/np.array(v)
		df = np.log(df)
		
		self.np_array = df
		




		
	def findex(self,dt):


		
		


if __name__ == '__main__':
	#data.train('d_EP',.05,200)
	train('d_EP',.1,200)
	# df = pd.read_feather('C:/Stocks/local/data/d_EP.feather')
	# df = df[df['value'] == 1]
	# print(df)
	# df = df[['ticker','dt']]
	# df['tf'] = 'd'
	# df = df.values.tolist()
	# print(score(df,'d_EP'))





















# def god(p1,p2):
#     return [x for x in p1 if x not in p2] + [x for x in p2 if x not in p1]












'''
df = pd.read_csv("C:/Users/csben/Downloads/america_2023-08-18.csv")
lis = ""

text_file = open(r"C:/Stocks/full_ticker_list_list.txt", 'w')
for i in range(len(df)):
	if('/' not in str(df.iloc[i]['Ticker'])):
		text_file.write(str(df.iloc[i]['Ticker']) + "\n")  
text_file.close()'''


#print(pd.read_feather("C:/Stocks/local/data/2h_mF.feather"))
#data.consolidate_database()
#setup_list = data.get_setups_list()
#print(data.get_setups_list())
#if __name__ == '__main__': data.train('d_F',.05,300)

#df = pd.read_feather('C:/Stocks/sync/database/laptop_2h_mF.feather')
#print(len(df[df['value'] == 1]))



#class Student:


#	def __init__(self,name='none'):
#		self.name = name

#	def ident(self):
#		print(self.name)

#	def set_birthday(self,birthday):
#		self.birthday = birthday

#	def read_birthday(self):
#		print(self.birthday)


#class idk(Student):


#	#def __init__(self,name='none'):
#		#self.name1 = name

#	def ident(self):
#		print('odddd')

#	def gosh(self):
#		print (self.name)

##student = Student('dd')
##student.set_birthday('september')
##student.read_birthday()

#st = idk('d')
#st.gosh()
#st.ident()

#data.refill_backtest()


#print(pd.read_feather('C:/Stocks/local/data/1min/ROKU.feather'))
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
#print(df)
#print(data.get_requirements('',df,'d_EP'))
#path = "C:/Stocks/sync/database/"
#dir_list = os.listdir(path)
#for p in dir_list:
#	d = path + p
	
#	df = pd.read_feather(d)
#	df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)	
#	df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)	
#	for i in range(len(df)):
#		if data.is_pre_market( df.at[i,'dt']):
#			df.at[i,'dt'] =  df.at[i,'dt'].replace(hour=9, minute=30)
#	df.to_feather(d)










#p = 'C:/Stocks/sync/database/laptop_d_EP.feather'
#df = pd.read_feather(p)
#df.rename(columns={'date':'dt','req':'required','setup':'value'}, inplace = True)
#df.to_feather(p)
#print(df)



#ticker = 'AEHR'
#base_tf = '1min'
#dt = None
#exchange = pd.read_feather('C:/Stocks/sync/files/full_scan.feather').set_index('ticker').loc[ticker]['exchange']
#add = TvDatafeed(username="billingsandrewjohn@gmail.com",password="Steprapt04").get_hist(ticker, exchange, interval=base_tf, n_bars=100000, extended_session = True)
#print(add)













#print((data.get(tf = 'd',dt = datetime.datetime(2023,8,10,9,15),bars = 100)).to_string())
#df_traits = pd.read_feather('C:/Stocks/local/account/traits.feather')
#print(pd.read_feather('C:/Stocks/local/study/current_setups.feather'))
#print(data.get(tf = '2min', dt = datetime.datetime(2022,8,10,9,45)))

#df = pd.read_feather('C:/Stocks/sync/files/full_scan.feather')
#print(df.columns)
#df = df[['Ticker','Pre-market Change','Pre-market Volume','Relative Volume at Time','Exchange']]
#df = df.rename(columns={'Ticker':'ticker','Exchange':'exchange','Pre-market Change':'pm change','Pre-market Volume':'pm volume','Relative Volume at Time':'rvol'})

#print(df)
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
#print(data_table)
#print(traits)
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

#print(data_table)


##print(data.get_scale('Account fw'))
#df = pd.read_feather('C:/Stocks/local/study/historical_setups.feather')
#coin = pd.read_feather("F:/Stocks/local/data/d/COIN.feather")
##print(df)
##print(coin.to_string())
##df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
##df.to_feather('C:/Stocks/sync/database/aj_d_EP.feather')
##path = "C:/Stocks/sync/database/"
##dir_list = os.listdir(path)
##for p in dir_list:
##    d = path + p
##    df = pd.read_feather(d)
##    df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
##    df.to_feather(d)

#if __name__ == "__main__":
#	historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
#	if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"):
#		shutil.copy(r"C:\Stocks\sync\files\full_scan.feather", r"C:\Stocks\local\study\full_list_minus_annotated.feather")
#	while(len(historical_setups[historical_setups["post_annotation"] == ""]) < 1500):
#		full_list_minus_annotation = pd.read_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
#		print(len(historical_setups[historical_setups["post_annotation"] == ""]))
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
#            df1 = data.get(ticker,tf,dt,100,50)
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
#                                    style=s, warn_too_much_data=100000,returnfig = True,figratio = (data.get_config('Plot chart_aspect_ratio'),1),
#                                    figscale=data.get_config('Plot chart_size'), panel_ratios = (5,1), mav=mav, 
#                                    tight_layout = True,
#                                    addplot=apds)
#            ax = axlist[0]
#            ax.set_yscale('log')
#            ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
#            plt.savefig(p1, bbox_inches='tight',dpi = data.get_config('Plot chart_dpi')) 




			   #    c = self.event[2][1]



		#    size = (data.get_config('Traits fw')*data.get_config('Traits fs'),data.get_scal('Traits fh')*data.get_config('Traits fs'))
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
	  
