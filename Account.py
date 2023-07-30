




from Data import Data as data
import pandas as pd
import mplfinance as mpf
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
import PySimpleGUI as sg
import matplotlib.ticker as mticker
import datetime
from PIL import Image
import io
import pathlib
import shutil
import os
import statistics
from tqdm import tqdm
import os
import numpy as np
from imbox import Imbox


class 


class Run(self): 

class Account():

    def account(self):
        
    def log(self):


    def traits(self):


    def plot(self):














    def update(self):
        if self.menu == None:
            sg.theme('DarkGrey')
            try: self.df_log = pd.read_feather(r"C:\Stocks\local\account\log.feather").sort_values(by='datetime',ascending = False)
            except FileNotFoundError: self.df_log = pd.DataFrame()
            try: self.df_traits = pd.read_feather(r"C:\Stocks\local\account\traits.feather").sort_values(by='datetime',ascending = False)
            except FileNotFoundError: self.df_traits = pd.DataFrame()
            try: self.df_pnl = pd.read_feather(r"C:\Stocks\local\account\account.feather").set_index('datetime', drop = True)
            except FileNotFoundError: self.df_pnl = pd.DataFrame()
            self.menu = "Log"
        else:
            self.window.close()
        if os.path.exists("C:/Screener/laptop.txt"): #if laptop
            scalelog = 6
            scaleplot = 4.5
            scaleaccount = 5
            scaletraits = 4
        else:
            scalelog = 3.7
            scaleplot = 2.98
            scaleaccount = 3
            scaletraits = 3
        if self.menu == "Log":
            toprow = ['Ticker        ','Datetime        ','Shares ', 'Price   ','Setup    ']
            c1 = [  
            [(sg.Text("Ticker    ")),sg.InputText(key = 'input-ticker')],
            [(sg.Text("Datetime")),sg.InputText(key = 'input-datetime')],
            [(sg.Text("Shares   ")),sg.InputText(key = 'input-shares')],
            [(sg.Text("Price     ")),sg.InputText(key = 'input-price')],
            [(sg.Text("Setup    ")),sg.InputText(key = 'input-setup')],
            [sg.Text("",key = '-index-')],
            [sg.Button('Delete'),sg.Button('Clear'),sg.Button('Enter')],
            [sg.Button('Pull')],
            [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            c2 = [[sg.Table([],headings=toprow,key = '-table-',auto_size_columns=True,num_rows = 30,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            layout = [
            [sg.Column(c1),
             sg.VSeperator(),
             sg.Column(c2),]]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=scalelog,finalize = True)
            log.log(self)
        if self.menu == "Account":
            layout =[
            [sg.Image(key = '-CHART-')],
            [(sg.Text("Timeframe")),sg.InputText(key = 'input-timeframe')],
            #[(sg.Text("Datetime  ")),sg.InputText(key = 'input-datetime')],
            [(sg.Text("Bars  ")),sg.InputText(key = 'input-bars')],
            [sg.Button('Trade'),sg.Button('Real'),sg.Button('Recalc'),sg.Button('Load')],
            [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=scaleaccount,finalize = True)
            account.account(self)
        if self.menu == "Traits":
            toprow = ['    ','Ticker  ','$      ']
            c1 = [[sg.Table([],headings=toprow,key = '-table_gainers-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            c2 = [[sg.Table([],headings=toprow,key = '-table_losers-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            toprow = [' Trait ', 'Value ', 'Residual ']
            c3 = [[sg.Table([],headings=toprow,key = '-table_traits-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            toprow = ['Month    ', 'Avg Gain    ', 'Avg Loss    ', 'Win %    ', 'Trades     ', 'PNL     ']
            c4 = [[sg.Table([],headings=toprow,key = '-table_monthly-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,enable_click_events=True)]]
            c5 = [[sg.Button('Recalc')],
                [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            c6 = [[sg.Image(key = '-CHART-')]]
            layout = [
            [sg.Column(c1),
             sg.VSeperator(),
             sg.Column(c2),
             sg.VSeperator(),
             sg.Column(c3),
             sg.VSeperator(),
             sg.Column(c4),
             sg.VSeperator(),
             ],
             [sg.Column(c5),
             sg.VSeperator(),
             sg.Column(c6),
             ]]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=scaletraits,finalize = True)
            traits.traits(self)
        if self.menu == "Plot":
            toprow = ['Date             ','Shares   ','Price  ', 'Percent  ',' Timedelta   ','Size  ']
            toprow2 = ['Actual  ','Fsell  ','Fbuy   ', '10h     ','20h     ','50h     ','5d     ','10d     ']
            c2 = [  
             [sg.Image(key = '-IMAGE3-')],
             [sg.Image(key = '-IMAGE1-')]]
            c1 = [
             [sg.Image(key = '-IMAGE2-')],
             [(sg.Text((str(f"{self.i + 1} of {len(self.df_traits)}")), key = '-number-'))], 
             [sg.Table([],headings=toprow2,num_rows = 2, key = '-table2-',auto_size_columns=True,justification='left', 
                       expand_y = False)],
              [sg.Table([],headings=toprow,key = '-table-',auto_size_columns=True,justification='left',num_rows = 5, 
                       expand_y = False)],
            [(sg.Text("Ticker  ")),sg.InputText(key = 'input-ticker')],
            [(sg.Text("Date   ")),sg.InputText(key = 'input-datetime')],
            [(sg.Text("Setup  ")),sg.InputText(key = 'input-setup')],
            [(sg.Text("Sort    ")),sg.InputText(key = 'input-sort')],
            [sg.Button('Prev'),sg.Button('Next'),sg.Button('Load')],
            [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            layout = [
            [sg.Column(c1),
             sg.VSeperator(),
             sg.Column(c2)],]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=scaleplot,finalize = True)
            plot.plot(self)
        self.window.maximize()

    def loop(self):
        with Pool(6) as self.pool:
            if os.path.exists("C:/Screener/tmp/pnl/charts"):
                shutil.rmtree("C:/Screener/tmp/pnl/charts")
            os.mkdir("C:/Screener/tmp/pnl/charts")
            self.preloadamount = 7
            self.i = 0
            self.menu = None
            self.event = [None]
            self.index = None
            self.update(self)
            self.account_type = 'Real'
            lap = datetime.datetime.now()
            while True:
                self.event, self.values = self.window.read(timeout=15000)
                if self.event == "Traits" or self.event == "Plot" or self.event == "Account" or self.event == "Log":
                    self.index = None
                    self.menu = self.event
                    self.update(self)
                elif self.event != '__TIMEOUT__':
                    if self.menu == "Traits":

                        traits.traits(self)
                    elif self.menu == "Plot":
                        plot.plot(self)
                    elif self.menu == "Account":
                        account.account(self)
                    elif self.menu == "Log":
                        log.log(self)
                else:
                  
                    if self.menu == "Account": #and (data.isMarketOpen()):
                        print('refresh')
                        self.df_pnl = pd.read_feather(r"C:\Screener\sync\pnl.feather").set_index('datetime',drop = True)
                        account.plot_update(self)
                        pool = self.pool
                        tf = self.values['input-timeframe']
                        bars = self.values['input-bars']
                        if tf == '':
                            tf = 'd'
                        if bars == '':
                            bars = '375'
                        pool.apply_async(account.calcaccount,args = (self.df_pnl,self.df_log,'now',tf,bars,self.account_type, self.df_traits), callback = account.account_plot)
    def pull_mail():

        host = "imap.gmail.com"
        username = "billingsandrewjohn@gmail.com"
        password = 'kqnrpkqscmvkrrnm'
        download_folder = "C:/Screener/tmp/pnl"
        if not os.path.isdir(download_folder):
            os.makedirs(download_folder, exist_ok=True)
        mail = Imbox(host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)
        #messages = mail.messages() # defaults to inbox
        dt = datetime.date.today() - datetime.timedelta(days = 1)
        messages = mail.messages(sent_from='noreply@email.webull.com',date__gt=dt)
        for (uid, message) in messages:
            mail.mark_seen(uid) # optional, mark message as read
            for idx, attachment in enumerate(message.attachments):
                att_fn = attachment.get('filename')
                print(att_fn)
                if not 'IPO' in att_fn and not  'Options' in att_fn:
                    download_path = f"{download_folder}/{att_fn}"
                    with open(download_path, "wb") as fp:
                        fp.write(attachment.get('content').read())
        mail.logout()
        log = pd.read_csv('C:/Screener/tmp/pnl/Webull_Orders_Records.csv')
        log2 = pd.DataFrame()
        log2['ticker'] = log['Symbol']
        log2['datetime']  = pd.to_datetime(log['Filled Time'])
        log2['shares'] = log['Filled']
        for i in range(len(log)):
            if log.at[i,'Side'] != 'Buy':
                log2.at[i,'shares'] *= -1
        log2['price'] = log['Avg Price']
        log2['setup'] = ''
        log2 = log2.dropna()
        log2 = log2[(log2['datetime'] > '2021-12-01')]
        log2 = log2.sort_values(by='datetime', ascending = False).reset_index(drop = True)
        return log2

    def log(self):
        if not self.df_log.empty:
            self.df_log = self.df_log.sort_values(by='datetime', ascending = False)
        if self.event == 'Pull':
            new_log = Log.pull_mail()
            print(new_log)
            non_dep = self.df_log[self.df_log['ticker'] != 'Deposit']
            new = pd.concat([non_dep, new_log]).drop_duplicates(keep=False)
            print(new)
            new = new.sort_values(by='datetime', ascending = False)
            dt = new.iloc[-1]['datetime']
            print(new)
            df_log = pd.concat([self.df_log,new])
            df_log = df_log.sort_values(by='datetime', ascending = True).reset_index(drop = True)
            self.df_pnl = account.calcaccount(self.df_pnl,df_log,dt)
            self.df_traits = traits.update(new.values.tolist(), df_log,self.df_traits,self.df_pnl)
            self.df_log = df_log
            if os.path.exists("C:/Screener/tmp/pnl/charts"):
                shutil.rmtree("C:/Screener/tmp/pnl/charts")
            os.mkdir("C:/Screener/tmp/pnl/charts")
        if self.event == '-table-':
            try:
                index = self.values['-table-'][0]
                if type(index) == int:
                    self.index = index
                    bar = self.df_log.iloc[index]
                    self.window["input-ticker"].update(bar[0])
                    self.window["input-datetime"].update(bar[1])
                    self.window["input-shares"].update(bar[2])
                    self.window["input-price"].update(bar[3])
            except:
                pass
        if self.event == "Enter":
            ticker = str(self.values['input-ticker'])
            if ticker != "":
                try:
                    dt = self.values['input-datetime']
                    dt  = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                    shares = float(self.values['input-shares'])
                    price = float(self.values['input-price'])
                    setup = str(self.values['input-setup'])
                    add = pd.DataFrame({
                        'ticker': [ticker],
                        'datetime':[dt],
                        'shares': [shares],
                        'price': [price],
                        'setup': [setup]
                        })
                    df_log = self.df_log
                    if self.index == None:
                        df_log = pd.concat([df_log,add])
                        df_log.reset_index(inplace = True, drop = True)
                    else:
                        df_log.iat[self.index,0] = ticker
                        old_date = df_log.iat[self.index,1]
                        df_log.iat[self.index,1] = dt
                        df_log.iat[self.index,2] = shares
                        df_log.iat[self.index,3] = price
                        df_log.iat[self.index,4] = setup
                        if old_date < dt:
                            dt = old_date
                    df_log = df_log.sort_values(by='datetime', ascending = True).reset_index(drop = True)
                    self.df_pnl = account.calcaccount(self.df_pnl,df_log,dt)
                    self.df_traits = traits.update(add.values.tolist(), df_log,self.df_traits,self.df_pnl)
                    self.df_log = df_log
                    if os.path.exists("C:/Screener/tmp/pnl/charts"):
                        shutil.rmtree("C:/Screener/tmp/pnl/charts")
                    os.mkdir("C:/Screener/tmp/pnl/charts")
                except TimeoutError:
                    sg.Popup(str(e))
        if self.event == "Delete":
            if self.index != None:
                bar = self.df_log.iloc[self.index].to_list()
                df_log = self.df_log.drop(self.index).reset_index(drop = True)
                df_log = df_log.sort_values(by='datetime', ascending = True)
                self.df_pnl = account.calcaccount(self.df_pnl,df_log,bar[1])
                self.df_traits = traits.update([bar], df_log,self.df_traits,self.df_pnl)
                self.df_log = df_log
                self.index = None
                if os.path.exists("C:/Screener/tmp/pnl/charts"):
                    shutil.rmtree("C:/Screener/tmp/pnl/charts")
                os.mkdir("C:/Screener/tmp/pnl/charts")
        elif self.event == "Clear":
            if self.index == None:
                self.window["input-ticker"].update("")
                self.window["input-shares"].update("")
                self.window["input-price"].update("")
                self.window["input-setup"].update("")
                self.window["input-datetime"].update("")
            else:
                self.index = None
        try:
            self.window['-index-'].update(f'Index {self.index}')
        except:
            pass
        self.df_log = self.df_log.reset_index(drop = True)
        if not self.df_log.empty:
            self.df_log.to_feather(r"C:\Screener\sync\log.feather")
            table = self.df_log.sort_values(by='datetime', ascending = False).values.tolist()
            self.window["-table-"].update(table)

    def pull_ben()
        df = pd.read_csv("C:/Screener/11-26-21 orders.csv")
        df = df.drop([0, 1, 2])
        print(df.head())
        dfnew = []
        for i in range(len(df)):
            if "FILLED" in df.iloc[i][1]:
                if "-" not in df.iloc[i][0]:
                    dfnew.append(df.iloc[i])
        dfnew = pd.DataFrame(dfnew)
        print(dfnew.head())
        dfn = []
        dfn = pd.DataFrame(dfn)
        for i in range(len(dfnew)):
            ticker = dfnew.iloc[i][0]
            price = dfnew.iloc[i][1].split("$")[1]
            shareSplit = dfnew.iloc[i][3].split(" ")
            shares = None
            for j in range(len(shareSplit)):
                if(shareSplit[j].isnumeric() == True):
                    shares = shareSplit[j]
            shares = float(shares)
            for e in range(len(shareSplit)):
                if("Sell" in shareSplit[e]):
                    shares = -shares

            dateSplit = dfnew.iloc[i][4].split(" ")
            dateS = dateSplit[1].split("\n")
            print(dfnew.iloc[i][4])
            date = dateS[1] + " " + dateSplit[0]
            date = pd.to_datetime(date)
            le = len(dfn)
            dfn.at[le, 'ticker'] = str(ticker)
            dfn.at[le, 'datetime'] = date
            dfn.at[le, 'shares'] = float(shares)
            dfn.at[le, 'price'] = float(price)
            dfn.at[le, 'setup'] = ""
        dfn = pd.DataFrame(dfn)
        print(dfn)
        dfn.to_feather('C:/Screener/tmp/pnl/log.feather')

    def plot(self):
        if self.event == 'Load':
            scan = pd.read_feather(r"C:\Screener\sync\traits.feather")
            dt = self.values['input-datetime']
            ticker = self.values['input-ticker']
            setup = self.values['input-setup']
            sort = self.values['input-sort']
            if ticker  != "":
                scan = scan[scan['ticker'] == ticker]
            if setup  != "":
                scan = scan[scan['setup'] == setup]
            if len(scan) < 1:
                sg.Popup('No Setups Found')
                return
            if sort != "":
                try:
                    scan = scan.sort_values(by=[sort], ascending=False)
                except KeyError:
                    sg.Popup('Not a Trait')
                    return
            self.df_traits = scan
            if os.path.exists("C:/Screener/tmp/pnl/charts"):
                shutil.rmtree("C:/Screener/tmp/pnl/charts")
            os.mkdir("C:/Screener/tmp/pnl/charts")
            if dt != "":
                self.i = data.findex(self.df_traits.set_index('datetime',drop = True),dt,-1)
            else:
                self.i = 0
        if self.event == 'Next' :
            if self.i == len(self.df_traits) - 1:
                self.i = 0
            else:
                self.i += 1
        if self.event == 'Prev':
            if self.i == 0:
                self.i = len(self.df_traits) - 1
            else:
                self.i -= 1
        i = list(range(self.i,self.preloadamount+self.i))
        if self.i < 5:
            i += list(range(len(self.df_traits) - 1,len(self.df_traits) - self.preloadamount - 1,-1))
        else:
            i += list(range(self.i,self.i - self.preloadamount,-1))
        i = [x for x in i if x >= 0 and x < len(self.df_traits)]
        arglist = []
        for index in i:
            arglist.append([index,self.df_traits])
        pool = self.pool
        pool.map_async(Plot.create,arglist) 
        image1 = None
        image2 = None
        image3 = None
        start = datetime.datetime.now()
        while image1 == None or image2 == None or image3 == None:
            if (datetime.datetime.now() - start).seconds > 8:
                pool.map_async(Plot.create,[[self.i,self.df_traits]])
            try:
                image1 = Image.open(r"C:\Screener\tmp\pnl\charts" + f"\{self.i}" + "1min.png")
                image2 = Image.open(r"C:\Screener\tmp\pnl\charts" + f"\{self.i}" + "d.png")
                image3 = Image.open(r"C:\Screener\tmp\pnl\charts" + f"\{self.i}" + "h.png")
            except:
                pass
        table = []
        bar = self.df_traits.iat[self.i,2]
        maxsize = 0
        size = 0
        for k in range(len(bar)):
            shares = float(bar[k][2]) 
            size += shares
            if abs(size) > abs(maxsize):
                maxsize = size
        for k in range(len(bar)):
            startdate = datetime.datetime.strptime(bar[0][1], '%Y-%m-%d %H:%M:%S')
            date  = datetime.datetime.strptime(bar[k][1], '%Y-%m-%d %H:%M:%S')
            shares = (float(bar[k][2]))
            price = float(bar[k][3])
            try:
                size = f'{round(shares / maxsize * 100)} %'
            except:
                size = 'NA'
            timedelta = (date - startdate)
            if k == 0:
                percent = ""
            else:
                percent = round(float(bar[0][2])*((price / float(bar[0][3])) - 1) * 100 / abs(float(bar[0][2])),2)
            table.append([date,shares,price,percent,timedelta,size])
        table2 = [[],[]]
        for i in range(6,14):
            try:
                string = f'{round(self.df_traits.iat[self.i,i],2)} %'
            except:
                string = str(self.df_traits.iat[self.i,i])
            table2[0].append(string)
        for i in range(14,22):
            try:
                string = f'{round(self.df_traits.iat[self.i,i],2)} %'
            except:
                string = str(self.df_traits.iat[self.i,i])
            table2[1].append(string)
        bio1 = io.BytesIO()
        image1.save(bio1, format="PNG")
        bio2 = io.BytesIO()
        image2.save(bio2, format="PNG")
        bio3 = io.BytesIO()
        image3.save(bio3, format="PNG")
        self.window["-IMAGE1-"].update(data=bio1.getvalue())
        self.window["-IMAGE2-"].update(data=bio2.getvalue())
        self.window["-IMAGE3-"].update(data=bio3.getvalue())
        self.window["-number-"].update(str(f"{self.i + 1} of {len(self.df_traits)}"))
        self.window["-table2-"].update(table2)
        self.window["-table-"].update(table)
        
    def create(bar):
        i = bar[0]
        if (os.path.exists(r"C:\Screener\tmp\pnl\charts" + f"\{i}" + "1min.png") == False):
            df = bar[1]
            try:
                god = bar[2]
                tflist = ['d']
            except:
                tflist = ['1min','h','d']
            mc = mpf.make_marketcolors(up='g',down='r')
            s  = mpf.make_mpf_style(marketcolors=mc)
            if os.path.exists("C:/Screener/laptop.txt"): #if laptop
                fw = 22
                fh = 12
                fs = 1.95
            else:
                fw = 26
                fh = 13
                fs = 1.16
            ticker = df.iat[i,0]
            for tf in tflist:
                try:
                    string1 = str(i) + str(tf) + ".png"
                    p1 = pathlib.Path("C:/Screener/tmp/pnl/charts") / string1
                    datelist = []
                    colorlist = []
                    trades = []
                    for k in range(len(df.iat[i,2])):
                        date = datetime.datetime.strptime(df.iat[i,2][k][1], '%Y-%m-%d %H:%M:%S')
                        if tf == 'd':
                            date = date.date()
                        val = float(df.iat[i,2][k][2])
                        if val > 0:
                            colorlist.append('g')
                            add = pd.DataFrame({
                                    'Datetime':[df.iat[i,2][k][1]], 
                                    'Symbol':[df.iat[i,2][k][0]],
                                    'Action':"Buy",
                                    'Price':[float(df.iat[i,2][k][3])]
                                    })
                            trades.append(add)
                        else:
                            colorlist.append('r')
                        datelist.append(date)
                    god = bar[1].iloc[i]['arrows']
                    god = [list(x) for x in god]
                    dfall= pd.DataFrame(god, columns=['Datetime', 'Price', 'Color', 'Marker'])
                    dfall['Datetime'] = pd.to_datetime(dfall['Datetime'])
                    dfall = dfall.sort_values('Datetime')
                    colors = []
                    dfsByColor = []
                    for zz in range(len(dfall)):
                        if(dfall.iloc[zz]['Color'] not in colors):
                            colors.append(dfall.iloc[zz]['Color'])
                    for yy in range(len(colors)):
                        colordf = dfall.loc[dfall['Color'] == colors[yy]] 
                        dfsByColor.append(colordf)
                    df1 = data.get(ticker,tf,account = True)
                    startdate = dfall.iloc[0]['Datetime']
                    enddate = dfall.iloc[-1]['Datetime']
                    try:
                        l1 = data.findex(df1,startdate) - 50
                    except:
                        if 'd' in tf or 'w' in tf:
                            df1 = df1 = data.get(ticker,tf,account = False)
                            l1 = data.findex(df1,startdate) - 50
                        else:
                            raise Exception()
                    closed = df.iloc[i]['closed']
                    if closed:
                        r1 = data.findex(df1,enddate) + 50
                    else:
                        r1 = len(df1)
                    minmax = 300
                    if l1 < 0:
                        l1 = 0
                    df1 = df1[l1:r1]
                    times = df1.index.to_list()
                    timesdf = []
                    for _ in range(len(df1)):
                        nextTime = pd.DataFrame({ 
                            'Datetime':[df1.index[_]]
                            })
                        timesdf.append(nextTime)
                    mainindidf = pd.concat(timesdf).set_index('Datetime', drop=True)
                    apds = [mpf.make_addplot(mainindidf)]
                    for datafram in dfsByColor:
                        datafram['Datetime'] = pd.to_datetime(datafram['Datetime'])
                        tradelist = []
                        for t in range(len(datafram)): 
                            tradeTime = datafram.iloc[t]['Datetime']
                            for q in range(len(times)):
                                if(q+1 != len(times)):
                                    if(times[q+1] >= tradeTime):
                                        test = pd.DataFrame({
                                            'Datetime':[times[q]],
                                            'Marker':[datafram.iloc[t]['Marker']],
                                            'Price':[float(datafram.iloc[t]['Price'])]
                                            })
                                        tradelist.append(test)
                                        break
                                else:
                                    test = pd.DataFrame({
                                            'Datetime':[times[q]],
                                            'Marker':[datafram.iloc[t]['Marker']],
                                            'Price':[float(datafram.iloc[t]['Price'])]
                                            })
                                    tradelist.append(test)
                                    break
                        df2 = pd.concat(tradelist).reset_index(drop = True)
                        df2['Datetime'] = pd.to_datetime(df2['Datetime'])
                        df2 = df2.sort_values(by=['Datetime'])
                        df2['TradeDate_count'] = df2.groupby("Datetime").cumcount() + 1
                        newdf = (df2.pivot(index='Datetime', columns='TradeDate_count', values="Price")
                            .rename(columns="price{}".format)
                            .rename_axis(columns=None))
                        series = mainindidf.merge(newdf, how='left', left_index=True, right_index=True)[newdf.columns]
                        if series.isnull().values.all(axis=0)[0]:
                            pass
                        else: 
                            apds.append(mpf.make_addplot(series,type='scatter',markersize=300,alpha = .4,marker=datafram.iloc[0]['Marker'],edgecolors='black', color=datafram.iloc[0]['Color']))
                    if tf == 'h':
                        mav = (10,20,50)
                    elif tf == 'd':
                        mav = (5,10)
                    else:
                        mav = ()
                    fig, axlist = mpf.plot(df1, type='candle', volume=True  , 
                                           title=str(f'{ticker} , {tf}'), 
                                           style=s, warn_too_much_data=100000,returnfig = True,figratio = (fw,fh),
                                           figscale=fs, panel_ratios = (5,1), mav=mav, 
                                           tight_layout = True,
                                          addplot=apds)
                    ax = axlist[0]
                    ax.set_yscale('log')
                    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
                    plt.savefig(p1, bbox_inches='tight') 
                except Exception as e:
                    shutil.copy(r"C:\Screener\tmp\blank.png",p1)

    def calcaccount(df_pnl,df_log,startdate = None,tf = None,bars = None,account_type = None, df_traits = None):
        if account_type == 'Trade':
            df = traits.update('open',df_log,df_traits,df_pnl)
            return [df,tf,bars,account_type]
        df_log = df_log.sort_values(by='datetime', ascending = True)
        if startdate != None and not isinstance(startdate, str):
            startdate = str(startdate)[:-2] + '00'
            startdate = datetime.datetime.strptime(startdate, '%Y-%m-%d %H:%M:%S')
        conct = True
        if startdate == None:
            conct = False
        account = True
        df_aapl = data.get('NFLX','1min',account = account)
        if startdate != 'now' and startdate != None and startdate > df_aapl.index[-1]:
            return df_pnl
        if startdate != None:
            if startdate == 'now':
                df_pnl = df_pnl[:-1]
                startdate = df_pnl.index[-1]
                index = -1
            else:
                del_index = data.findex(df_pnl,startdate) 
                df_pnl = df_pnl[:del_index]
                index = data.findex(df_pnl,startdate)
            if index == None or index >= len(df_pnl):
                index = -1
            bar = df_pnl.iloc[index]
            pnl = bar['close']
            deposits = bar['deposits']
            positions = bar['positions'].split(',')
            shares = bar['shares'].split(',')
            pos = []
            for i in range(len(shares)):
                ticker = positions[i]
                if ticker != '':
                    share = float(shares[i])
                    df = data.get(ticker,'1min',account = account)
                    pos.append([ticker,share,df])
            gud = df_log.set_index('datetime')
            log_index = data.findex(gud,startdate) 
            if log_index != None and log_index < len(df_log):
                nex = df_log.iloc[log_index]['datetime']
            else:
                nex = datetime.datetime.now() + datetime.timedelta(days = 100)
            if nex < startdate:
                try:
                    log_index += 1
                    nex = df_log.iloc[log_index]['datetime']
                except:
                    nex = datetime.datetime.now() + datetime.timedelta(days = 100)
        else:
            startdate = df_log.iat[0,1] - datetime.timedelta(days = 1)
            pnl = 0
            deposits = 0
            pos = []
            index = 0
            log_index = 0
            nex = df_log.iat[0,1]
        start_index = data.findex(df_aapl,startdate)
        prev_date = df_aapl.index[start_index - 1]
        date_list = df_aapl[start_index:].index.to_list()
        df_list = []
        pbar = tqdm(total=len(date_list))
        for i in range(len(date_list)):
            date = date_list[i]
            if i > 0:
                prev_date = date_list[i-1]
            pnlvol = 0
            pnlo = pnl
            pnll = pnlo
            pnlh = pnlo
            while date > nex:
                remove = False
                ticker = df_log.iat[log_index,0]
                shares = df_log.iat[log_index,2]
                price = df_log.iat[log_index,3]
                if ticker == 'Deposit':
                    deposits += price
                else:
                    pos_index = None
                    for i in range(len(pos)):
                        if pos[i][0] == ticker:
                            if not isinstance(pos[i][2], pd.DataFrame):
                                prev_shares = pos[i][1]
                                avg = pos[i][2]
                                if shares / prev_shares > 0:
                                    pos[i][2] = ((avg*prev_shares) + (price*shares))/(prev_shares + shares)
                                else:
                                    gosh = (price - avg) * (-shares)
                                    pnl += gosh
                                    if gosh > 0:
                                        pnlh += gosh
                                    else:
                                        pnll += gosh
                            pos_index = i
                            pos[i][1] += shares
                            if pos[i][1] == 0:
                                remove = True
                    if pos_index == None:
                        pos_index = len(pos)
                        try:
                            df = data.get(ticker,'1min',account = account)
                            data.findex(df,date) + 1
                        except:
                            df = price
                        pos.append([ticker,shares,df])
                    df = pos[pos_index][2]
                    if isinstance(df, pd.DataFrame):
                        ind = data.findex(df,prev_date)
                        c1 = df.iat[ind,3]
                        gosh = (c1 - price)*shares
                        pnl += gosh
                        if gosh > 0:
                            pnlh += gosh
                        else:
                            pnll += gosh
                    pnlvol += abs(shares*price)
                    if remove:
                        del pos[pos_index]
                log_index += 1
                if log_index >= len(df_log):
                    nex = datetime.datetime.now() + datetime.timedelta(days=100)
                else:
                    nex = df_log.iat[log_index,1]
            positions = ""
            god_shares = ""
            for i in range(len(pos)):
                ticker = pos[i][0]
                shares = pos[i][1]
                df = pos[i][2]
                if isinstance(df, pd.DataFrame):
                    index = data.findex(df,date)
                    prev_index = data.findex(df,prev_date)
                    prevc = df.iat[prev_index,3]
                    c = df.iat[index,3] 
                    o = df.iat[index,0]
                    h = df.iat[index,1]
                    l = df.iat[index,2]
                    pnl += (c - prevc) * shares
                    ch = (h - prevc) * shares
                    cl = (l - prevc) * shares
                    if shares > 0:
                        pnll += cl
                        pnlh+= ch
                    else:
                        pnll += ch
                        pnlh += cl
                    pnlo += (o - prevc) * shares
                if i >= 1:
                    positions += "," + (str(ticker))
                    god_shares += "," + (str(shares))
                else:
                    positions += str(ticker)
                    god_shares += str(shares)
            add = pd.DataFrame({
                'datetime':[pd.Timestamp(date)],
                'open':[pnlo],
                'high':[pnlh],
                'low':[pnll],
                'close':[pnl],
                'volume':[pnlvol],
                'deposits':[deposits],
                'account':[deposits + pnl],
                'positions':[positions],
                'shares':[god_shares]
                })
            df_list.append(add)
            pbar.update(1)
        df = pd.concat(df_list)
        if conct:
            df = pd.concat([df_pnl.reset_index(),df])
        df = df.reset_index(drop = True)
        df = df.sort_values(by='datetime')
        df = df.set_index('datetime',drop = True)
        df.reset_index().to_feather(r"C:\Screener\sync\pnl.feather")
        if tf == None:
            return df 
        else:
            return [df,tf,bars,account_type]

    def account(self,date = None):
        if self.event == 'Trade' or 'Real':
            self.account_type = self.event
        if self.event == "Load" or self.event == "Recalc":
            tf = self.values['input-timeframe']
            bars = self.values['input-bars']
            if tf == "":
                tf = 'd'
            if bars == "":
                bars = 375
        else:
            tf = 'd'
            bars = 375
        if self.df_pnl.empty or self.event == "Recalc":
            df = Account.calcaccount(self.df_pnl,self.df_log,date)
            self.df_pnl = df
        if self.account_type == 'Trade':
            df = self.df_traits
        else:
            df = self.df_pnl
        bar = [df,tf,bars,self.account_type]
        Account.account_plot(bar)
        Account.plot_update(self)
    
    def account_plot(bar):
        try:
            df = bar[0]
            tf = bar[1]
            bars = int(bar[2])
            account_type = bar[3]
            if account_type == 'Trade':
                df = df.sort_values(by='datetime',ascending = True)
                df = df.set_index('datetime')
                df = df[['open','high','low','close','volume']]
                pc = 0
                for i in range(len(df)):
                    v = df.iat[i,4]
                    c = df.iat[i,3] + pc
                    o = pc
                    h = df.iat[i,1] + pc
                    l = df.iat[i,2] + pc
                    df.iloc[i] = [o,h,l,c,v]
                    pc = c
            else:
                if tf == '':
                    tf = 'd'
                if tf != "1min":
                    logic = {'open'  : 'first','high'  : 'max','low'   : 'min','close' : 'last','volume': 'sum' }
                    df = df.resample(tf).apply(logic).dropna()
                df = df[-bars:]
            mc = mpf.make_marketcolors(up='g',down='r')
            s  = mpf.make_mpf_style(marketcolors=mc)
            if os.path.exists("C:/Screener/laptop.txt"):
                fw = 30
                fh = 13.8
                fs = 3.4
            else:
                fw = 42
                fh = 18
                fs = 2.1
            string1 = "pnl.png"
            p1 = pathlib.Path("C:/Screener/tmp/pnl") / string1
            fig, axlist = mpf.plot(df, type='candle', volume=True, style=s, warn_too_much_data=100000,returnfig = True,figratio = (fw,fh),figscale=fs, panel_ratios = (5,1), mav=(10,20), tight_layout = True)
            plt.savefig(p1, bbox_inches='tight')
        except TimeoutError:
            pass
        
    def plot_update(self):
        bio1 = io.BytesIO()
        image1 = Image.open(r"C:\Screener\tmp\pnl\pnl.png")
        image1.save(bio1, format="PNG")
        self.window["-CHART-"].update(bio1.getvalue())

    def update(bars,df_log,df_traits,df_pnl):
        if not df_traits.empty:
            for bar in bars:
                ticker = bar[0]
                date = bar[1]
                df_traits = df_traits[df_traits['ticker'] != ticker]
        new_traits = Traits.get_list(df_log)
        old_traits = df_traits
        new = pd.concat([old_traits,new_traits]).drop_duplicates('datetime',keep=False)
        new = new.sort_values(by='datetime', ascending = False)
        df = Traits.calc(new,df_pnl)
        df_traits = pd.concat([df_traits,df])
        df_traits = df_traits.sort_values(by='datetime',ascending = False).reset_index(drop = True)
        df_traits.to_feather(r"C:\Screener\sync\traits.feather")
        return df_traits

    def get_list(df_log):
        pos = []
        df_traits = pd.DataFrame()
        for k in range(len(df_log)):
            row = df_log.iloc[k].to_list()
            ticker = row[0]
            if ticker != 'Deposit':
                shares = row[2]
                date = row[1]
                index = None
                for i in range(len(pos)):
                    if pos[i][0] == ticker:
                        index = i
                        break
                if index != None:
                    prev_share = pos[index][2]
                else:
                    prev_share = 0
                    pos.append([ticker,date,shares,[]])
                    index = len(pos) - 1
                pos[index][3].append([str(x) for x in row])
                shares = prev_share + shares
                if shares == 0:
                    for i in range(len(pos)):
                        if pos[i][0] == ticker:
                            index = i
                            bar = pos[i]
                            add = pd.DataFrame({
                            'ticker': [bar[0]],
                            'datetime':[bar[1]],
                            'trades': [bar[3]]
                            })
                            df_traits = pd.concat([df_traits,add])
                            del pos[i]
                            break
                else:
                    pos[index][2] = shares
        for i in range(len(pos)-1,-1,-1):
            index = i
            bar = pos[i]
            add = pd.DataFrame({
            'ticker': [bar[0]],
            'datetime':[bar[1]],
            'trades': [bar[3]]
            })
            df_traits = pd.concat([df_traits,add])
            del pos[i]
        df = df_traits
        return df

    def calc(df_traits,df_pnl):
        if not df_traits.empty:
            df_traits = df_traits.sort_values(by='datetime', ascending = False).reset_index(drop = True)
            df_vix = data.get('^VIX','d')
            df_qqq = data.get('QQQ','d')
            arg_list = []
            for i in range(len(df_traits)):
                bar = df_traits.iloc[i]
                arg_list.append([bar,df_pnl,df_vix,df_qqq])
            if len(arg_list) > 30:
                df_list = data.pool(Traits.trait_calc,arg_list)
            else:
                df_list = []
                for i in range(len(arg_list)):
                    df_list.append(Traits.trait_calc(arg_list[i]))
            try:
                df = pd.concat(df_list)
                df = df.reset_index(drop = True).sort_values(by='datetime',ascending = False)
            except:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        return df

    def trait_calc(pack):
        bar = pack[0]
        df_pnl = pack[1]
        df_vix = pack[2]
        df_qqq = pack[3]
        ticker = bar[0]
        date = bar[1]
        trades = bar[2]
        openprice = float(trades[0][3])
        lastdate = datetime.datetime.strptime(trades[-1][1],'%Y-%m-%d %H:%M:%S')
        if (datetime.datetime.now() - lastdate).days > 10:
            acnt = False
        else:
            acnt = True
        run = False
        try:
            df_1min = data.get(ticker,'1min',account = acnt)
            hourly = data.get(ticker,'h',account = acnt)
            daily = data.get(ticker,'d',account = acnt)
            startd = data.findex(daily,date)
            start = data.findex(hourly,date)
            open_date = daily.index[data.findex(daily,date)]
            recent_price = df_1min.iat[-1,3]
            if startd != None and start != None:
                run = True
        except:
            recent_price = float(trades[-1][3])
        if float(trades[0][2]) > 0:
            direction = 1
        else: 
            direction = -1
        trade_pnl = 0
        maxshares = sum([abs(float(s)) for d,d,s,d,d in trades])/2
        maxshares = maxshares * direction
        fb = float(trades[0][3])  *   maxshares
        pnl = -fb 
        buys = 0
        fs = None
        arrow_list = []
        trade_setup = 'None'
        size = 0
        maxsize = 0
        shg = 0
        total_size = 0
        pnl_high = -10000000000
        pnl_low = 1000000000000

        for i in range(len(trades)):
            sdate = trades[i][1]
            sh = float(trades[i][2])
            price = float(trades[i][3])
            setup = trades[i][4]
            dollars = price * sh
            total_size += abs(dollars)
            size += dollars
            shg += sh
            if abs(size) > abs(maxsize):
                maxsize = (size)
            if  setup != 'None' and "":
                trade_setup = setup
            if sh > 0:
                color = 'g'
                symbol = '^'
            else:
                color = 'r'
                symbol = 'v'
            arrow_list.append([str(sdate),str(price),str(color),str(symbol)])
            if sh*direction > 0:
                last_open_date = datetime.datetime.strptime(trades[i][1],'%Y-%m-%d %H:%M:%S')
            if dollars*direction < 0:
                if fs == None:
                    fs = price
                pnl -= dollars
            else:
                buys -= dollars
            trade_pnl -= dollars
            if not run:
                pnlc = trade_pnl + price * shg
                if pnlc < pnl_low:
                    pnl_low = pnlc
                if pnlc > pnl_high:
                    pnl_high = pnlc
        try:
            account_val = df_pnl.iloc[data.findex(df_pnl,date)]['account']
        except:
            try:
                account_val = df_pnl.iloc[-1]['account']
            except:
                account_val = 10000
        maxloss = -2
        if shg != 0:
            closed = False
            if not run:
                trade_pnl += recent_price * shg
                pnl_pcnt = ((trade_pnl / abs(maxsize)) ) *100
                pnl_account = (trade_pnl/ account_val ) * 100
        else:
            closed = True
            pnl_pcnt = ((trade_pnl / abs(maxsize)) ) *100
            pnl_account = (trade_pnl/ account_val ) * 100
            if pnl_pcnt < maxloss:
                maxloss = pnl_pcnt
        if closed:
            fbuy = (pnl/fb) * 100 * direction
            fsell = (fs*maxshares + buys)/maxsize * 100 * direction
            rfsell = fsell - pnl_pcnt
            rfbuy = fbuy - pnl_pcnt
        else:
            fbuy = pd.NA
            fsell = pd.NA
            rfsell = pd.NA
            rfbuy = pd.NA
        size = maxsize
        h10 = pd.NA
        h20 = pd.NA
        h50 = pd.NA
        d5 = pd.NA
        d10 = pd.NA
        h10time = pd.NA
        h20time = pd.NA
        h50time = pd.NA
        d5time = pd.NA
        d10time = pd.NA
        try:
            ivix = data.findex(df_vix,date)
            vix = df_vix.iat[ivix,0]
        except:
            vix = 0
        if run:
            iqqq = data.findex(df_qqq,date)
            ma = []
            for i in range(51):
                ma.append(df_qqq.iat[iqqq+i-50,3])
                if i == 49:
                    ma50 = statistics.mean(ma)
            m50 = (statistics.mean(ma[-50:])/ma50 - 1) * 100
            open_index = data.findex(df_1min,open_date)
            low = 1000000000*direction
            entered = False
            i = open_index
            stopped = False
            stop = ((maxloss/100)*direction + 1) * openprice
            stopdate = datetime.datetime.now() + datetime.timedelta(days = 100)
            max_days = 3
            before_max = True
            shares = 0
            nex = date
            trade_index = 0
            pnl = 0
            pnl_low = 10000000
            pnl_high = -1000000
            exit = False
            while True:
                if i >= len(df_1min) or (exit and not before_max and closed):
                    break
                if direction > 0:
                    clow = df_1min.iat[i,2] 
                else:
                    clow = df_1min.iat[i,1] 
                cdate = df_1min.index[i]
                if  (cdate - date).days > max_days or i - open_index > 390*max_days :
                    before_max = False
                if clow*direction < low*direction and before_max:
                    low = clow
                if cdate >= date and not entered:
                    entered = True
                    risk = (direction*(low - openprice))/openprice * 100
                    low = 1000000000*direction
                if cdate > last_open_date and  direction*clow < stop*direction and not stopped:
                    stopped = True
                    copen = df_1min.iat[i,0]
                    if direction*copen < direction*stop:
                        stop = copen
                    stopdate = df_1min.index[i+1]
                    arrow_list.append([str(stopdate),str(stop),'k',symbol]) 
                while cdate > nex:
                    sh = float(trades[trade_index][2])
                    price = float(trades[trade_index][3])
                    pnl += (df_1min.iat[i-1,3] - price)*sh
                    shares += sh
                    trade_index += 1
                    if trade_index >= len(trades):
                        nex = datetime.datetime.now() + datetime.timedelta(days=100)
                        exit = True
                    else:
                        nex = datetime.datetime.strptime(trades[trade_index][1],'%Y-%m-%d %H:%M:%S')
                index = data.findex(df_1min,cdate)
                prevc = df_1min.iat[index - 1,3]
                c = df_1min.iat[index,3] 
                h = df_1min.iat[index,1]
                l = df_1min.iat[index,2]
                pnlh =  pnl + (h - prevc) * shares
                pnll = pnl + (l - prevc) * shares
                pnl = pnl + (c - prevc) * shares
                if pnll < pnl_low:
                    pnl_low = pnll
                if pnlh > pnl_high:
                    pnl_high = pnlh
                i += 1
            if not closed:
                trade_pnl = pnl
                pnl_pcnt = ((trade_pnl / abs(maxsize)) ) *100
                pnl_account = (trade_pnl/ account_val ) * 100
            low = (direction*(low - openprice)/openprice) * 100
            if direction > 0:
                symbol = 'v'
            else:
                symbol = '^'
            prices = []
            for i in range(50):
                prices.append(hourly.iat[i + start - 50,3])
            i = 0
            try:
                while True:
                    close = hourly.iat[start+i,3]
                    cdate = hourly.index[start + i] + datetime.timedelta(hours = 1)

                    if cdate > stopdate:
                        if pd.isna(h10):
                            h10 = maxloss
                            h10time = ( cdate- date).total_seconds() / 3600
                        if pd.isna(h20):
                            h20 = maxloss
                            h20time = (hourly.index[start + i + 1] - date).total_seconds() / 3600
                        if pd.isna(h50):
                            h50 = maxloss
                            h50time = (hourly.index[start + i + 1] - date).total_seconds() / 3600
                    if (not pd.isna(h20) and not pd.isna(h10) and pd.isna(h50)):
                        break
                    if direction * close < direction * statistics.mean(prices[-10:]) and pd.isna(h10):
                        h10 = direction*(close/openprice - 1)*100
                        h10time = ((hourly.index[start + i] - date)+datetime.timedelta(hours=1)).total_seconds() / 3600
                        arrow_list.append([str(cdate),str(close),'m',str(symbol)])
                    if direction * close < direction * statistics.mean(prices[-20:]) and pd.isna(h20):
                        h20 = direction*(close/openprice - 1)*100
                        h20time = ((hourly.index[start + i] - date)+datetime.timedelta(hours=1)).total_seconds() / 3600
                        arrow_list.append([str(cdate),str(close),'b',str(symbol)])
                    if direction * close < direction * statistics.mean(prices[-50:]) and pd.isna(h50):
                        h50 = direction*(close/openprice - 1)*100
                        h50time = ((hourly.index[start + i] - date)+datetime.timedelta(hours=1)).total_seconds() / 3600
                        arrow_list.append([str(cdate),str(close),'c',str(symbol)]) 
                    i += 1
                    if i + start >= len(hourly):
                        break
                    prices.append(hourly.iat[start + i,3])
            except:
                pass
            start = startd 
            prices = []
            for i in range(10):
                prices.append(daily.iat[i + start - 10,3])
            i = 0
            try:
                while True:
                    close = daily.iat[start+i,3]
                    cdate = daily.index[start + i ] + datetime.timedelta(days = 1)
                    if cdate > stopdate:
                        if pd.isna(d5):
                            d5 = maxloss
                            d5time = (daily.index[start+i+1] - date).total_seconds() / 3600
                        if pd.isna(d10):
                            d10 = maxloss
                            d10time = (daily.index[start+i+1] - date).total_seconds() / 3600

                    if (not pd.isna(d10) and not pd.isna(d5)):
                        break
                    if direction * close < direction * statistics.mean(prices[-5:]) and pd.isna(d5):
                        d5 = direction*(close/openprice - 1)*100
                        d5time = ((daily.index[start+i] - date)+datetime.timedelta(days=1)).total_seconds() / 3600
                        arrow_list.append([str(cdate),str(close),'y',str(symbol)])
                    if direction * close < direction * statistics.mean(prices[-10:]) and pd.isna(d10):
                        d10 = direction*(close/openprice - 1)*100
                        d10time = ((daily.index[start+i] - date)+datetime.timedelta(days=1)).total_seconds() / 3600
                        arrow_list.append([str(cdate),str(close),'w',str(symbol)])

                    i += 1
                    if i + start + 1 >= len(daily):
                        break
                    prices.append(daily.iat[start+i,3])
            except:
                pass
            r10 = h10 - pnl_pcnt
            r20 = h20 - pnl_pcnt
            r50 = h50 - pnl_pcnt
            r5d = d5 - pnl_pcnt
            r10d = d10 - pnl_pcnt
            rfsell = fsell - pnl_pcnt
            rfbuy = fbuy - pnl_pcnt
        else:
            h10  = pd.NA
            h20 = pd.NA
            h50 = pd.NA
            d5 = pd.NA
            d10 = pd.NA
            r10 = pd.NA
            r20 = pd.NA
            r50 = pd.NA
            r5d = pd.NA
            r10d = pd.NA
            d10time = pd.NA
            h10time  = pd.NA
            h20time = pd.NA
            h50time = pd.NA
            d5time = pd.NA
            low  = pd.NA
            risk  = pd.NA
            m50 = pd.NA
        try:
            gudddd = risk
        except:
            risk = pd.NA
        add = pd.DataFrame({
        'ticker': [ticker],
        'datetime':[date],
        'trades': [trades],
        'setup':[trade_setup],
        'pnl':[trade_pnl],
        'account':[pnl_account],
        'percent':[pnl_pcnt],
        'fsell':[fsell],
        'fbuy':[fbuy],
        'p10':[h10],
        'p20':[h20],
        'p50':[h50],
        'p5d':[d5],
        'p10d':[d10],
        'rpercent':[0],
        'rfsell':[rfsell],
        'rfbuy':[rfbuy],
        'r10':[r10],
        'r20':[r20],
        'r50':[r50],
        'r5d':[r5d],
        'r10d':[r10d],
        'vix':[vix], 
        'm50':[m50],
        'arrows':[arrow_list],
        'closed':[closed],
        't10d':[d10time],
        't20':[h20time],
        't10':[h10time],
        't50':[h50time],
        't5d':[d5time],
        'min':[low],
        'risk':[risk],
        'open':[0],
        'high':[pnl_high],
        'low':[pnl_low],
        'close':[trade_pnl],
        'volume':[total_size],
        })
        return add

    def build_monthly(self):
        df = self.df_traits
        g = df.groupby(pd.Grouper(key='datetime', freq='M'))
        dfs = [group for _,group in g]
        god = []
        date = 'Overall'
        loss = df[df['account'] <= 0]
        avg_loss = loss['account'].mean()
        gain = df[df['account'] > 0]
        avg_gain = gain['account'].mean()
        wins = []
        for i in range(len(df)):
            if df.iloc[i]['account'] > 0:
                wins.append(1)
            else:
                wins.append(0)
        win = statistics.mean(wins) * 100
        pnl = ((df['account'] / 100) + 1).tolist()
        gud = 1
        for i in pnl:
            gud *= i
        pnl = gud
        trades = len(df)
        god.append([date,round(avg_gain,2), round(avg_loss,2), round(win,2), round(trades,2), round(pnl,2)])
        for df in dfs:
            date = str(df.iat[0,1])
            date = date[:-12]
            loss = df[df['account'] <= 0]
            avg_loss = loss['account'].mean()
            gain = df[df['account'] > 0]
            avg_gain = gain['account'].mean()
            wins = []
            for i in range(len(df)):
                if df.iloc[i]['account'] > 0:
                    wins.append(1)
                else:
                    wins.append(0)
            win = statistics.mean(wins) * 100
            trades = len(df)
            pnl = ((df['account'] / 100) + 1).tolist()
            gud = 1
            for i in pnl:
                gud *= i
            pnl = (gud - 1) *100
            god.append([date,round(avg_gain,2), round(avg_loss,2), round(win,2), round(trades,2), round(pnl,2)])
        return god

    def build_traits(self):
        traits = self.df_traits
        god = []
        traits_list = [6,7,8,9,10,11,12,13]
        for i in traits_list:
            t = traits.columns[i]
            tn = traits.columns[i+8]
            n = round(traits[t].mean(),2)
            r = round(traits[tn].mean(),2)
            god.append([t,n,r])
        self.traits_table = god
        return god

    def traits(self):
        inp = False
        if self.df_traits.empty or self.event == 'Recalc':
            self.df_traits = pd.DataFrame()
            self.df_log = self.df_log.sort_values(by='datetime', ascending = True)
            self.df_traits = Traits.update([],self.df_log,pd.DataFrame(),self.df_pnl)
            self.df_traits.to_feather(r"C:\Screener\sync\traits.feather")
        if '+CLICKED+' in self.event:
            if os.path.exists("C:/Screener/laptop.txt"): #if laptop
                size = (49,25)
            else:
                size = (25,10)
            c = self.event[2][1]
            if c == 0:
                return
            plt.clf()
            y = [p[5] for p in self.monthly[1:] if not np.isnan(p[c])]
            x = [p[c] for p in self.monthly[1:] if not np.isnan(p[c])]
            plt.scatter(x,y)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x,p(x),"r--")
            plt.gcf().set_size_inches(size)
            string1 = "traits.png"
            p1 = pathlib.Path("C:/Screener/tmp/pnl") / string1
            plt.savefig(p1,bbox_inches='tight')
            bio1 = io.BytesIO()
            image1 = Image.open(r"C:\Screener\tmp\pnl\traits.png")
            image1.save(bio1, format="PNG")
            self.window["-CHART-"].update(data=bio1.getvalue())
        elif self.event == '-table_traits-':
            i = self.values['-table_traits-'][0]
            inp = self.traits_table[i][0]
        elif self.event == '-table_gainers-' or self.event == '-table_losers-':
            if self.event == '-table_gainers-':
                df = self.gainers
                i = self.values['-table_gainers-'][0]
            else:
                df = self.losers
                i = self.values['-table_losers-'][0]
            bar = [i,df,1]
            if os.path.exists("C:/Screener/tmp/pnl/charts"):
                shutil.rmtree("C:/Screener/tmp/pnl/charts")
            os.mkdir("C:/Screener/tmp/pnl/charts")
            plot.create(bar)
            bio1 = io.BytesIO()
            image1 = Image.open(f'C:/Screener/tmp/pnl/charts/{i}d.png')
            image1.save(bio1, format="PNG")
            self.window["-CHART-"].update(data=bio1.getvalue())
        elif self.event == 'Traits':
            inp = 'account'
            gainers2 = self.df_traits.sort_values(by = ['pnl'])[:10].reset_index(drop = True)
            gainers = pd.DataFrame()
            gainers['#'] = gainers2.index + 1
            gainers['Ticker'] = gainers2['ticker']
            gainers['$'] = gainers2['pnl'].round(2)
            losers2 = self.df_traits.sort_values(by = ['pnl'] , ascending = False)[:10].reset_index(drop = True)
            losers = pd.DataFrame()
            losers['#'] = losers2.index + 1
            losers['Ticker'] = losers2['ticker']
            losers['$'] = losers2['pnl'].round(2)
            self.losers = losers2
            self.gainers = gainers2
            self.monthly = Traits.build_monthly(self)
            traits = Traits.build_traits(self)
            self.window["-table_gainers-"].update(gainers.values.tolist())
            self.window["-table_losers-"].update(losers.values.tolist())
            self.window["-table_traits-"].update(traits)
            self.window["-table_monthly-"].update(self.monthly)
        if inp != False:
            bins = 50
            if os.path.exists("C:/Screener/laptop.txt"): #if laptop
                size = (49,25)
            else:
                size = (25,10)
            if inp == "":
                inp = 'p10'
            try:
                plt.clf()
                if ':'  in inp:
                    inp = inp.split(':')
                    inp1 = inp[0]
                    inp2 = inp[1]
                    x = self.df_traits[inp1].to_list()
                    y = self.df_traits[inp2].to_list()
                    plt.scatter(x,y)
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    plt.plot(x,p(x),"r--")
                else:
                    fifty = self.df_traits[inp].dropna().to_list()
                    plt.hist(fifty, bins, alpha=1, ec='black',label='Percent') 
                plt.gcf().set_size_inches(size)
                string1 = "traits.png"
                p1 = pathlib.Path("C:/Screener/tmp/pnl") / string1
                plt.savefig(p1,bbox_inches='tight')
                
                bio1 = io.BytesIO()
                image1 = Image.open(r"C:\Screener\tmp\pnl\traits.png")
                image1.save(bio1, format="PNG")
                self.window["-CHART-"].update(data=bio1.getvalue())
            except:
                pass
      