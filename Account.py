


#filter out non used logs of used ticker to speed up recalc traits

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


class Run: 

    def run(self):
        with Pool(6) as self.pool:
            sg.theme('DarkGrey')
            try: self.df_log = pd.read_feather(r"C:\Stocks\local\account\log.feather").sort_values(by='datetime',ascending = False)
            except FileNotFoundError: self.df_log = pd.DataFrame()
            try: self.df_traits = pd.read_feather(r"C:\Stocks\local\account\traits.feather").sort_values(by='datetime',ascending = False)
            except FileNotFoundError: self.df_traits = pd.DataFrame()
            try: self.df_pnl = pd.read_feather(r"C:\Stocks\local\account\account.feather").set_index('datetime', drop = True)
            except FileNotFoundError: self.df_pnl = pd.DataFrame()
            self.init = True
            self.menu = 'Traits'
            Log.log(self)
            while True:
                self.event, self.values = self.window.read()
                if self.event == "Traits" or self.event == "Plot" or self.event == "Account" or self.event == "Log":
                    if self.df_log.empty: 
                        sg.Popup('Log is empty')
                        continue
                    if self.df_pnl.empty: Account.calc(self,self.df_log)
                    if self.df_traits.empty: Traits.calc(self,self.df_log)
                    self.menu = self.event
                    self.init = True
                    self.window.close()
                if self.menu == "Traits": Traits.traits(self)
                elif self.menu == "Plot": Plot.plot(self)
                elif self.menu == "Account": Account.account(self)
                elif self.menu == "Log": Log.log(self)

class Log:

    def recalc(self,updated_log):
        new_log = pd.concat([self.df_log, updated_log]).drop_duplicates(keep=False).sort_values(by='datetime', ascending = False)
        self.df_log = updated_log.sort_values(by='datetime', ascending = False).reset_index(drop = True)
        self.df_pnl = Account.recalc(self,new_log)
        self.df_traits = Traits.recalc(self,new_log)
        Log.update(self)
        self.df_log.to_feather(r"C:\Stocks\local\account\log.feather")

    def manual_log(self):
        ticker = self.values['input-ticker']
        shares = float(self.values['input-shares'])
        price = float(self.values['input-price'])
        setup = self.values['input-setup']
        try:
            dt = datetime.datetime.strptime(self.values['input-datetime'], '%Y-%m-%d %H:%M:%S')
            if ticker == ''  or shares == '' or price == '': raise TimeoutError
        except (TimeoutError, TypeError):
            sg.Popup('check inputs')
            return
        updated_log = self.df_log
        if self.index == None: self.index = len(self.df_log)
        updated_log.iat[self.index,0] = ticker
        updated_log.iat[self.index,1] = dt
        updated_log.iat[self.index,2] = shares
        updated_log.iat[self.index,3] = price
        updated_log.iat[self.index,4] = setup
        Log.recalc(self,updated_log)

    def log(self):
        if self.init:
            Log.update(self)
        elif self.event == 'Pull':
            Log.pull(self)
        elif self.event == 'Delete':
            updated_log = self.df_log.drop(self.index).reset_index(drop = True)
            Log.recalc(self,updated_log)
        elif self.event == 'Enter':
            Log.manual_log(self)
        elif self.event == '-table-' :
           self.log_index = self.values['-table-'][0]
        elif self.event == 'Clear': 
            if self.log_index == None: Log.update_inputs(self)
            else: self.log_index = None

    def update_inputs(self):
        if self.log_index == None: bar = ['','','','','']
        else: bar = self.df_log.iloc[self.log_index]
        self.window["-input_ticker-"].update(bar[0])
        self.window["-input_datetime-"].update(bar[1])
        self.window["-input_shares-"].update(bar[2])
        self.window["-input_price-"].update(bar[3])
        self.window["-input_setup-"].update(bar[4])
        self.window['-index-'].update(self.log_index)

    def update(self):
        if self.init:
            self.log_index = None
            ident = data.identify()
            if ident =='desktop':scale = 6
            elif ident =='laptop':scale = 4
            else: raise Exception('no account scale set')
            toprow = ['Ticker        ','Datetime        ','Shares ', 'Price   ','Setup    ']
            c1 = [  
            [(sg.Text("Ticker    ")),sg.InputText(key = '-input_ticker-')],
            [(sg.Text("Datetime")),sg.InputText(key = '-input_datetime-')],
            [(sg.Text("Shares   ")),sg.InputText(key = '-input_shares-')],
            [(sg.Text("Price     ")),sg.InputText(key = '-input_price-')],
            [(sg.Text("Setup    ")),sg.InputText(key = '-input_setup-')],
            [sg.Text("",key = '-index-')],
            [sg.Button('Delete'),sg.Button('Clear'),sg.Button('Enter')],
            [sg.Button('Pull')],
            [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            c2 = [[sg.Table([],headings=toprow,key = '-table-',auto_size_columns=True,num_rows = 30,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            layout = [
            [sg.Column(c1),
                sg.VSeperator(),
                sg.Column(c2),]]
            self.window = sg.Window('Log', layout,margins = (10,10),scaling=scale,finalize = True)
        Log.update_inputs(self)
        self.window['-table-'].update(self.df_log.values.tolist())
        self.window.maximize()
    
    def pull(self):
        ident = data.identify()
        if ident == 'desktop':
            host = "imap.gmail.com"
            username = "billingsandrewjohn@gmail.com"
            password = 'kqnrpkqscmvkrrnm'
            download_folder = "C:/Screener/tmp/pnl"
            if not os.path.isdir(download_folder):
                os.makedirs(download_folder, exist_ok=True)
            mail = Imbox(host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)
            dt = datetime.date.today() - datetime.timedelta(days = 1)
            messages = mail.messages(sent_from='noreply@email.webull.com',date__gt=dt)
            for (uid, message) in messages:
                mail.mark_seen(uid)
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
            updated_log =  log2
        elif ident == 'ben':
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
            #dfn.to_feather('C:/Screener/tmp/pnl/log.feather')
            updated_log =  dfn
        else:
            raise Exception('no pull method created')
        Log.recalc(self,updated_log)



















class Traits:

    def calc(self,new_log):
        dfs = []
        ticker_list = [*set(new_log['ticker'])]
        for ticker in ticker_list:
            ticker_logs = self.df_log[self.df_log['ticker'] == ticker]
            dfs.append(ticker_logs)
            self.df_traits = self.df_traits[self.df_traits['ticker'] != ticker]
        trades = Traits.get_trades(pd.concat([dfs]))
        traits = pd.concat(self.pool(Traits.calc,trades.values.tolist()))
        self.df_traits = pd.concat([self.df_traits,traits]).sort_values(by='datetime',ascending = False).reset_index(drop = True)
        self.df_traits.to_feather(r'C:\Stocks\local\account\traits.feather')

    def get_trades(df_log):
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

    def worker(pack):
        bar = pack[0]
        df_pnl = pack[1]
        ticker = bar[0]
        date = bar[1]
        trades = bar[2]
        openprice = float(trades[0][3])
        close_datetime = datetime.datetime.strptime(trades[-1][1],'%Y-%m-%d %H:%M:%S')
        run = False
        df_1min = data.get(ticker,'1min',close_datetime)
        open_date = daily.index[data.findex(daily,date)]
        recent_price = df_1min.iat[-1,3]
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
        size = maxsize

        open_index = data.findex(df_1min,open_date)
        low = 1000000000*direction
        entered = False
        i = open_index
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

        traits = pd.DataFrame({
        'ticker': [ticker],
        'datetime':[date],
        'trades': [trades],
        'setup':[trade_setup],
        'pnl':[trade_pnl],
        'account':[pnl_account],
        'percent':[pnl_pcnt],
        'arrows':[arrow_list],
        'closed':[closed],
        'min':[low],
        'risk':[risk],
        'open':[0],
        'high':[pnl_high],
        'low':[pnl_low],
        'close':[trade_pnl],
        'volume':[total_size],
        })
        return traits

    def build_rolling_traits(self):

        def calc(df,is_overall):
            if is_overall: date = str(df.iat[0,1])[:-12]
            else: date = 'Overall'
            avg_loss = df[df['account'] <= 0]['account'].mean()
            avg_gain = df[df['account'] > 0]['account'].mean()
            wins = []
            for i in range(len(df)):
                if df.iloc[i]['account'] > 0: wins.append(1)
                else: wins.append(0)
            win = statistics.mean(wins) * 100
            pnl = ((df['account'] / 100) + 1).tolist()
            gud = 1
            for i in pnl:
                gud *= i
            pnl = gud
            trades = len(df)
            return [date,round(avg_gain,2), round(avg_loss,2), round(win,2), round(trades,2), round(pnl,2)]
        g = df.groupby(pd.Grouper(key='datetime', freq='3M'))
        dfs = [group for _,group in g]
        rolling_traits = []
        rolling_traits.append(calc(self.df,True))
        for df in dfs:
            rolling_traits.append(calc(df))
        return rolling_traits

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
            Traits.calc(self,self.df_log)
        elif '+CLICKED+' in self.event:
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
      






    def update(self):
        scaletraits = 4

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
        self.winodw.maximize()








class Account:
    
    def account(self):
        if self.event == 'Recalc': Account.recalc(self,self.df_log)
        else: self.pnl_chart_type = self.event
        Account.update(self)


    def update(self):
        if self.init:
            ident = data.identify()
            if ident == 'laptop': scale = 4
            else: raise Exception('no scale defined')
            self.pnl_chart_type = 'Trade'
            layout =[
            [sg.Image(key = '-CHART-')],
            [sg.Button('Trade'),sg.Button('Real'),sg.Button('Recalc')],
            [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=scale,finalize = True)
        if self.pnl_chart_type == 'Trade':
            df = self.df_traits.set_index('datetime')[['open','high','low','close','volume']]
            pc = 0
            for i in range(len(df)):
                v = df.iat[i,4]
                c = df.iat[i,3] + pc
                o = pc
                h = df.iat[i,1] + pc
                l = df.iat[i,2] + pc
                df.iloc[i] = [o,h,l,c,v]
                pc = c
        elif self.event == 'Real':
            df = self.df_pnl
            logic = {'open'  : 'first','high'  : 'max','low'   : 'min','close' : 'last','volume': 'sum' }
            df = df.resample('d').apply(logic).dropna()
        mc = mpf.make_marketcolors(up='g',down='r')
        s  = mpf.make_mpf_style(marketcolors=mc)
        ident = data.identify
        if ident == 'laptop':
            fw = 30
            fh = 13.8
            fs = 3.4
        elif ident == 'desktop':
            fw = 42
            fh = 18
            fs = 2.1
        else:
            raise Exception('scale not defined')
        string1 = "pnl.png"
        p1 = pathlib.Path("C:/Stocks/local/account") / string1
        _,_ = mpf.plot(df, type='candle', volume=True, style=s, warn_too_much_data=100000,returnfig = True,figratio = (fw,fh),figscale=fs, panel_ratios = (5,1), mav=(10,20), tight_layout = True)
        plt.savefig(p1, bbox_inches='tight')
        bio1 = io.BytesIO()
        image1 = Image.open(r"C:\Screener\tmp\pnl\pnl.png")
        image1.save(bio1, format="PNG")
        scaleaccount = 5
        self.window["-CHART-"].update(bio1.getvalue())
        self.winodw.maximize()
  
    def calc(self,new_log):
        start_datetime = new_log.iloc[-1]['datetime']
        df = data.get('QQQ','1min',start_datetime)
        index = data.findex(df,start_datetime) - 1
        date_list = df[index:].index.to_list()
        index = data.findex(self.df_pnl,date_list[0])
        bar = self.df_pnl.iloc[index]
        self.df_pnl = self.df_pnl[:index]
        pos = bar['positions']
        df_log = self.df_log
        df_list = []
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
                            df = data.get(ticker,'1min',datetime.datetime.now())
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
        df = pd.concat(df_list)
        df = pd.concat([self.df_pnl.reset_index(drop = True),df])
        df = df.reset_index(drop = True)
        df = df.sort_values(by='datetime')
        df = df.set_index('datetime',drop = True)
        df.reset_index().to_feather(r"C:\Screener\sync\pnl.feather")
        self.df_pnl = df

class Plot:

    def update(self):
        toprow = ['Date             ','Shares   ','Price  ']
        toprow2 = ['pnl $ ','pnl %  ','pnl account  ', '  risk %  ','risk % account    ','rank    ']
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
        
        scaleplot = 4.5


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

        self.window.maximize()


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

if __name__ == '__main__':
    Run.run(Run)




    
        #startdate = data.format_date(new_log.index[-1])
        #conct = True
        #if startdate == None:
        #    conct = False
        #account = True
        #if startdate != 'now' and startdate != None and startdate > df_aapl.index[-1]:
        #    return df_pnl
        #if startdate != None:
        #    if startdate == 'now':
        #        df_pnl = df_pnl[:-1]
        #        startdate = df_pnl.index[-1]
        #        index = -1
        #    else:
        #        del_index = data.findex(df_pnl,startdate) 
        #        df_pnl = df_pnl[:del_index]
        #        index = data.findex(df_pnl,startdate)
        #    if index == None or index >= len(df_pnl):
        #        index = -1
        #    bar = df_pnl.iloc[index]
        #    pnl = bar['close']
        #    deposits = bar['deposits']
        #    positions = bar['positions'].split(',')
        #    shares = bar['shares'].split(',')
        #    pos = []
        #    for i in range(len(shares)):
        #        ticker = positions[i]
        #        if ticker != '':
        #            share = float(shares[i])
        #            df = data.get(ticker,'1min',account = account)
        #            pos.append([ticker,share,df])
        #    gud = df_log.set_index('datetime')
        #    log_index = data.findex(gud,startdate) 
        #    if log_index != None and log_index < len(df_log):
        #        nex = df_log.iloc[log_index]['datetime']
        #    else:
        #        nex = datetime.datetime.now() + datetime.timedelta(days = 100)
        #    if nex < startdate:
        #        try:
        #            log_index += 1
        #            nex = df_log.iloc[log_index]['datetime']
        #        except:
        #            nex = datetime.datetime.now() + datetime.timedelta(days = 100)
        #else:
        #    startdate = df_log.iat[0,1] - datetime.timedelta(days = 1)
        #    pnl = 0
        #    deposits = 0
        #    pos = []
        #    index = 0
        #    log_index = 0
        #    nex = df_log.iat[0,1]
        #start_index = data.findex(df_aapl,startdate)
        #prev_date = df_aapl.index[start_index - 1]
        #date_list = df_aapl[start_index:].index.to_list()
        #df_list = []
        #pbar = tqdm(total=len(date_list))


