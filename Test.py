from Data import Data as data
import pandas as pd
import datetime
import os
import yfinance as yf


#print(data.get_scale('Account fw'))
df = pd.read_feather('C:/Stocks/local/study/historical_setups.feather')
coin = pd.read_feather("F:/Stocks/local/data/d/COIN.feather")
#print(df)
print(coin.to_string())
ydf = yf.download(tickers = ["COIN"],  period = '25y',  group_by='ticker',      
			interval = '1d', ignore_tz = True, progress=False,
			show_errors = False, threads = False, prepost = False)
#df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
#df.to_feather('C:/Stocks/sync/database/aj_d_EP.feather')
#path = "C:/Stocks/sync/database/"
#dir_list = os.listdir(path)
#for p in dir_list:
#    d = path + p
#    df = pd.read_feather(d)
#    df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
#    df.to_feather(d)
