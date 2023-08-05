from Data import Data as data
import pandas as pd
import datetime
import os
import yfinance as yf
import shutil
from Screener import Screener as screener

#print(data.get_scale('Account fw'))
df = pd.read_feather('C:/Stocks/local/study/historical_setups.feather')
coin = pd.read_feather("F:/Stocks/local/data/d/COIN.feather")
#print(df)
#print(coin.to_string())
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

if __name__ == "__main__":
	historical_setups = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
	if not os.path.exists("C:\Stocks\local\study\full_list_minus_annotated.feather"):
		shutil.copy(r"C:\Stocks\sync\files\full_scan.feather", r"C:\Stocks\local\study\full_list_minus_annotated.feather")
	while(len(historical_setups[historical_setups["post_annotation"] == ""]) < 1500):
		full_list_minus_annotation = pd.read_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
		print(len(historical_setups[historical_setups["post_annotation"] == ""]))
		full_list_minus_annotation = full_list_minus_annotation.sample(frac=1)
		for t in range(8):
			screener.run(ticker=full_list_minus_annotation.iloc[t]["Ticker"], fpath=0)
		full_list_minus_annotation = full_list_minus_annotation[8:].reset_index(drop=True)
		full_list_minus_annotation.to_feather(r"C:\Stocks\local\study\full_list_minus_annotated.feather")
