import os

dirs = ['C:/Stocks/local','C:/Stocks/local/data','C:/Stocks/local/account','C:/Stocks/local/screener',
                'C:/Stocks/local/study','C:/Stocks/local/trainer','C:/Stocks/local/1min','C:/Stocks/local/d']


if not os.path.exists("C:/Stocks/local"):
    for d in dirs:
                
        os.mkdir(d)
