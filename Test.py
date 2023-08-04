from Data import Data as data
import pandas as pd
import datetime
import os



#print(data.get_scale('Account fw'))
df = pd.read_feather('C:/Stocks/sync/files/full_scan.feather')
print(df)
print(df[:-0])
print(df[:-1])

#df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
#df.to_feather('C:/Stocks/sync/database/aj_d_EP.feather')
#path = "C:/Stocks/sync/database/"
#dir_list = os.listdir(path)
#for p in dir_list:
#    d = path + p
#    df = pd.read_feather(d)
#    df.rename(columns={'date':'datetime','req':'required','setup':'value'}, inplace = True)
#    df.to_feather(d)
