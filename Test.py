from Data import Data as data
import pandas as pd
import datetime
import os



#print(data.get_scale('Account fw'))
print(data.get('NFLX','1min',datetime.datetime.now()))

#path = "C:/Stocks/sync/database/"
#dir_list = os.listdir(path)
#for p in dir_list:
#    d = path + p
#    df = pd.read_feather(d)
#    df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
#    df.to_feather(d)
