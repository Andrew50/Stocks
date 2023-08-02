


#filter out non used logs of used ticker to speed up recalc traits


#periodic pnl plot

from Data import Data as data
import pandas as pd
import mplfinance as mpf
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
import PySimpleGUI as sg
import matplotlib.ticker as mticker
import datetime

import PIL
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
            try: self.df_traits = pd.read_feather(r"C:\Stocks\local\account\traits.feather")
            except FileNotFoundError: self.df_traits = pd.DataFrame()
            try: self.df_pnl = pd.read_feather(r"C:\Stocks\local\account\pnl.feather").set_index('datetime', drop = True)
            except FileNotFoundError: self.df_pnl = pd.DataFrame()
            self.init = True
            self.menu = 'Log'
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
        print(new_log)
        self.df_pnl = Account.calc(self,new_log)
        self.df_traits = Traits.calc(self,new_log)
        Log.update(self)
        self.df_log.to_feather(r"C:\Stocks\local\account\log.feather")

    def manual_log(self):
        ticker = self.values['-input_ticker-']
        shares = float(self.values['-input_shares-'])
        price = float(self.values['-input_price-'])
        setup = self.values['-input_setup-']
        try:
            dt = datetime.datetime.strptime(self.values['-input_datetime-'], '%Y-%m-%d %H:%M:%S')
            if ticker == ''  or shares == '' or price == '': raise TimeoutError
        except (TimeoutError, TypeError):
            sg.Popup('check inputs')
            return
        updated_log = self.df_log.copy()
        if self.log_index == None: self.log_index = len(self.df_log)
        updated_log.iat[self.log_index,0] = ticker
        updated_log.iat[self.log_index,1] = dt
        updated_log.iat[self.log_index,2] = shares
        updated_log.iat[self.log_index,3] = price
        updated_log.iat[self.log_index,4] = setup
        self.log_index = None
        self.window['-log_table-'].update(select_rows=[])
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
        elif self.event == 'Clear': 
            if self.log_index == None: Log.update_inputs(self)
            else: 
                self.log_index = None
                self.window['-log_table-'].update(select_rows=[])
        elif self.event == '-log_table-':
            try:
                self.log_index = self.values['-log_table-'][0]
                Log.update_inputs(self)
            except IndexError:
                pass

    def update_inputs(self):
        if self.log_index == None: bar = ['','','','','']
        else: bar = self.df_log.iloc[self.log_index]
        self.window["-input_ticker-"].update(bar[0])
        self.window["-input_datetime-"].update(bar[1])
        self.window["-input_shares-"].update(bar[2])
        self.window["-input_price-"].update(bar[3])
        self.window["-input_setup-"].update(bar[4])

    def update(self):
        if self.init:
            self.log_index = None
            ident = data.identify()
            if ident =='desktop':scale = 6
            elif ident =='laptop':scale = 4
            else: raise Exception('no account scale set')
            self.init = False
            toprow = ['Ticker        ','Datetime        ','Shares ', 'Price   ','Setup    ']
            c1 = [  
            [(sg.Text("Ticker    ")),sg.InputText(key = '-input_ticker-')],
            [(sg.Text("Datetime")),sg.InputText(key = '-input_datetime-')],
            [(sg.Text("Shares   ")),sg.InputText(key = '-input_shares-')],
            [(sg.Text("Price     ")),sg.InputText(key = '-input_price-')],
            [(sg.Text("Setup    ")),sg.InputText(key = '-input_setup-')],
            [sg.Button('Delete'),sg.Button('Clear'),sg.Button('Enter')],
            [sg.Button('Pull')],
            [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            c2 = [[sg.Table([],headings=toprow,key = '-log_table-',auto_size_columns=True,num_rows = 30,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            layout = [
            [sg.Column(c1),
                sg.VSeperator(),
                sg.Column(c2),]]
            self.window = sg.Window('Log', layout,margins = (10,10),scaling=scale,finalize = True)
        self.window['-log_table-'].update(self.df_log.values.tolist())
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
            dfnew = []
            for i in range(len(df)):
                if "FILLED" in df.iloc[i][1]:
                    if "-" not in df.iloc[i][0]:
                        dfnew.append(df.iloc[i])
            dfnew = pd.DataFrame(dfnew)
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
            if not self.df_traits.empty: self.df_traits = self.df_traits[self.df_traits['ticker'] != ticker]
        trades = Traits.get_trades(pd.concat(dfs))
        arglist = [[trades.iloc[i], self.df_pnl] for i in range(len(trades))]
        traits = pd.concat(self.pool.map(Traits.worker,arglist))
        self.df_traits = pd.concat([self.df_traits,traits]).sort_values(by='datetime',ascending = True).reset_index(drop = True)
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
                            bar[3].reverse()
                            add = pd.DataFrame({
                            'ticker': [bar[0]],
                            'datetime':[data.format_date(bar[1])],
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
            bar[3].reverse()
            add = pd.DataFrame({
            'ticker': [bar[0]],
            'datetime':[data.fomrat_date(bar[1])],
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
        trades = bar[2]
        open_datetime = data.format_date(trades[0][1])
        close_datetime = data.format_date(trades[-1][1])
        setup = trades[0][4]
        data_exists = True
        account_size = df_pnl.iloc[data.findex(df_pnl,open_datetime)]['account']
        try: 
            df_1min = data.get(ticker,'1min',close_datetime)
            time = datetime.time(9,30,0)
            open_datetime = datetime.datetime.combine(open_datetime.date(),time)
            index = data.findex(df_1min,open_datetime)
            df_1min = df_1min[index:]
        except (FileNotFoundError , IndexError): data_exists = False
        if float(trades[0][2]) > 0: direction = 1
        else:  direction = -1
        arrow_list = []
        current_size = 0
        total_size = 0
        high_dollars = float('-inf')
        low_dollars = float('inf')
        pnl_dollars = 0
        open_shares = 0
        size_dollars = 0
        total_size = 0
        current_pnl = 0
        prev_price = 0
        for i in range(len(trades)):
            date = trades[i][1]
            shares = float(trades[i][2])
            price = float(trades[i][3])
            dollars = price * shares
            total_size += abs(dollars)
            current_size += dollars
            
            pnl_dollars -= dollars
            if abs(current_size) > abs(size_dollars):
                size_dollars = abs(current_size)
            if shares > 0:
                color = 'g'
                symbol = '^'
            else:
                color = 'r'
                symbol = 'v'
            arrow_list.append([str(date),str(price),str(color),str(symbol)])
            if not data_exists:
                current_pnl += open_shares*(price - prev_price)
                if current_pnl < low_dollars: low_dollars = current_pnl
                if current_pnl > high_dollars: high_dollars = current_pnl
                open_shares += shares
                prev_price = price
        if open_shares != 0:
            pnl_dollars += open_shares * df_1min.iat[-1,3]
            closed = False

        else: closed = True
        if direction == 1: 
            low_col = 2
            high_col = 1
        else: 
            low_col = 1
            high_col = 2
        if data_exists:
            low_dollars = float('inf')
            high_dollars = float('-inf')
            lod_price = float('inf')
            min_percent = float('inf')
            open_shares = 0
            next_trade_date = data.format_date(trades[0][1])
            trade_index = 1
            opened = False
            prev_low = 0
            prev_high = 0
            current_low = 0
            current_high = 0
            size = 0
            for i in range(len(df_1min)):
                date = df_1min.index[i]
                low = df_1min.iat[i,low_col]
                high = df_1min.iat[i,high_col]
                current_low += open_shares * (low - prev_low)
                current_high += open_shares * (high - prev_high)
                prev_high = high
                prev_low = low
               
                current_low_percent = (current_low / abs(size) - 1) *100
                if current_low_percent < min_percent: min_percent = current_low_percent
                if not opened:
                    if price < lod_price:
                        lod_price = price
                    risk_percent = (lod_price / price - 1) * 100
                if date >= next_trade_date:
                    opened = True
                    shares = float(trades[trade_index][2])
                    price = float(trades[trade_index][3])
                    open_shares += shares
                    trade_index += 1
                    size += shares*price
                    current_low += shares * (prev_low - price)
                    current_high += shares * (prev_high - price)
                    try: next_trade_date = data.format_date(trades[trade_index][1])
                    except IndexError: next_trade_date = datetime.datetime.now()
                if current_low < low_dollars: low_dollars = current_low
                if current_high > high_dollars: high_dollars = current_high
            min_account = (low_dollars / account_size - 1)*100
        else:
            risk_percent = pd.NA
            min_percent = pd.NA
            min_account = pd.NA
        pnl_percent = (pnl_dollars / size_dollars - 1) * 100
        pnl_account = (pnl_dollars / account_size - 1) * 100
        size_percent = (size_dollars / account_size) * 100
        traits = pd.DataFrame({
        'ticker': [ticker],
        'datetime':[open_datetime],
        'trades': [trades],
        'setup':[setup],
        'pnl $':[pnl_dollars],
        'pnl %':[pnl_percent],
        'pnl a':[pnl_account],
        'size $':[size_dollars],
        'size %':[size_percent],
        'risk %':[risk_percent], #to lod
        'min %':[min_percent],
        'min a':[min_account],
        'arrow_list':[arrow_list],
        'closed':[closed],
        'open':[0],
        'high':[high_dollars],
        'low':[low_dollars],
        'close':[pnl_dollars],
        'volume':[total_size],
        })
        return traits



    def calc_trait_values(df,title):
        avg_loss = df[df['pnl a'] <= 0]['pnl a'].mean()
        avg_gain = df[df['pnl a'] > 0]['pnl a'].mean()
        wins = []
        for i in range(len(df)):
            if df.iloc[i]['pnl a'] > 0: wins.append(1)
            else: wins.append(0)
        win = statistics.mean(wins) * 100
        risk = df[df['risk %'] > 0]['risk %'].mean()
        size = df[df['size %'] > 0]['size %'].mean()
        pnl = ((df['pnl a'] / 100) + 1).tolist()
        gud = 1
        for i in pnl:
            gud *= i
        pnl = gud
        trades = len(df)
        return [title,round(avg_gain,2), round(avg_loss,2), round(win,2), round(risk,2), round(size,2), round(trades,2), round(pnl,2)]

    def build_rolling_traits_table(self):

        g = self.df_traits.groupby(pd.Grouper(key='datetime', freq='3M'))
        dfs = [group for _,group in g]
        rolling_traits = []
        rolling_traits.append(Traits.calc_trait_values(self.df_traits,True))
        for df in dfs:
            rolling_traits.append(Traits.calc_trait_values(df,str(df.iat[0,1])[:-12]))
        return rolling_traits

    def build_setup_traits_table(self):
        g = self.df_traits.groupby(pd.Grouper(key='setup',))
        dfs = [group for _,group in g]
        setup_traits = []
        for df in dfs:
            setup_traits.append(Traits.calc_trait_values(df,df.iloc[0]['setup']))
        return setup_traits
        
    def build_trades_table(self, winners):
        sorted_df = self.df_traits.sort_values(by = ['pnl %'], ascending = winners).reset_index(drop = True)
        df = pd.DataFrame()
        df['#'] = sorted_df.index + 1
        df['Ticker'] = sorted_df['ticker']
        df['% a'] = sorted_df['pnl %'].round(2)
    
    def traits(self):

        if self.df_traits.empty or self.event == 'Recalc': Traits.calc(self,self.df_log)

        elif self.event == '-table_losers-':  ####account
            i = self.values['-table_losers-'][0]
            t = self.values['-table_losers-']

        elif self.event == '-table_winners-': ####account
            i = self.values['-table_winners-'][0]
            t = self.values['-table_winners-']

        elif '+CLICKED+' in self.event:
            pass


        elif self.event == '-table_setups-':  ####in the chart
            i = self.values['-table_setups-'][0]
            t = self.values['-table_setups-']
            x = [v[0] for v in t]

        elif '+CLICKED+' in self.event: ###rolling traits
            pass

        Traits.update(self)

        #    c = self.event[2][1]



        #    size = (data.get_scale('Traits fw')*data.get_scale('Traits fs'),data.get_scal('Traits fh')*data.get_scale('Traits fs'))
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
      



    def update(self):

        if self.init:
            self.init = False
            scale = data.get_scale('Traits ui')
            rolling_traits_header = ['Date','Avg Gain','Avg Loss','Win %','Risk %','Size a','Trades','PnL a']
            biggest_trades_header = ['Rank','Ticker  ','% Account']
            setup_traits_header = ['Setup','Avg Gain','Avg Loss','Win %','Risk %','Size a','Trades','PnL a']
            c1 = [[sg.Table([],headings=biggest_trades_header,key = '-table_winners-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            c2 = [[sg.Table([],headings=biggest_trades_header,key = '-table_losers-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,selected_row_colors='red on yellow')]]
            c3 = [[sg.Table([],headings=setup_traits_header,key = '-table_setups-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,enable_click_events=True)]]
            c4 = [[sg.Table([],headings=rolling_traits_header,key = '-table_rolling_traits-',auto_size_columns=True,num_rows = 10,justification='left',enable_events=True,enable_click_events=True)]]
            c5 = [[sg.Button('Recalc')],
                [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            c6 = [[sg.Image(key = '-CHART-')]]
            layout = [
            [sg.Column(c1), sg.VSeperator(), sg.Column(c2), sg.VSeperator(), sg.Column(c3), sg.VSeperator(),
            sg.Column(c4), sg.VSeperator(),],[sg.Column(c5),sg.VSeperator(),sg.Column(c6),]]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=scale,finalize = True)
            self.window.maximize()
        self.window['-table_winners-'].update(Traits.build_trades_table(self,True))
        self.window['-table_losers-'].update(Traits.build_trades_table(self,False))
        self.window['-table_setups-'].update(Traits.build_setup_traits_table(self))
        self.window['-table_rolling_traits-'].update(Traits.build_rolling_traits_table(self))


class Account:
    
    def account(self):
        if self.event == 'Recalc': Account.recalc(self,self.df_log)
        else: 
            self.pnl_chart_type = self.event
        Account.update(self)

    def update(self):
        if self.init:
            ident = data.identify()
            if ident == 'laptop': scale = 4
            else: raise Exception('no scale defined')
            self.pnl_chart_type = 'Trade'
            self.init = False
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
        elif self.pnl_chart_type == 'Real':
            df = self.df_pnl
            logic = {'open'  : 'first','high'  : 'max','low'   : 'min','close' : 'last','volume': 'sum' }
            df = df.resample('d').apply(logic).dropna()
        mc = mpf.make_marketcolors(up='g',down='r')
        s  = mpf.make_mpf_style(marketcolors=mc)
        string1 = "pnl.png"
        p1 = pathlib.Path("C:/Stocks/local/account") / string1
        _,_ = mpf.plot(df, type='candle', volume=True, style=s, warn_too_much_data=100000,returnfig = True,figratio = (data.get_scale('Account fw'),data.get_scale('Account fh')),figscale=data.get_scale('Account fs'), panel_ratios = (5,1), mav=(10,20), tight_layout = True)
        plt.savefig(p1, bbox_inches='tight',dpi = data.get_scale('Account dpi'))
        bio1 = io.BytesIO()
        image1 = Image.open(r"C:\Stocks\local\account\pnl.png")
        image1.save(bio1, format="PNG")
        self.window["-CHART-"].update(bio1.getvalue())
        self.window.maximize()

  
    def calc(self,new_log):
        start_datetime = new_log.iloc[-1]['datetime']
        df = data.get(tf = '1min',dt = datetime.datetime.now())
        index = data.findex(df,start_datetime) - 1
        prev_date = df.index[index]
        date_list = df[index:].index.to_list()
        if self.df_pnl.empty:
            pos = []
            pnl = 0
            deposits = 0
        else:
            index = data.findex(self.df_pnl,date_list[0])
            bar = self.df_pnl.iloc[index]
            self.df_pnl = self.df_pnl[:index]
            open_positions = bar['positions'].split(',')
            open_shares = bar['shares'].split(',')
            pos = []
            for i in range(len(open_shares)):
                ticker = open_positions[i]
                if ticker != '':
                    shares = float(open_shares[i])
                    df = data.get(ticker,'1min',datetime.datetime.now())
                    pos.append([ticker,shares,df])
            pnl = bar['open']
            deposits = bar['deposits']
        df_log = self.df_log
        df_list = []
        next_trade_date = new_log.iloc[-1]['datetime']
        log_index = 0
        
        for i in range(len(date_list)):
            date = date_list[i]
            if i > 0:
                prev_date = date_list[i-1]
            pnlvol = 0
            pnlo = pnl
            pnll = pnlo
            pnlh = pnlo
            while date > next_trade_date:
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
                    next_trade_date = datetime.datetime.now() + datetime.timedelta(days=100)
                else:
                    next_trade_date = df_log.iat[log_index,1]
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
        df = pd.concat([self.df_pnl.reset_index(),df])
        df = df.reset_index(drop = True)
        df = df.sort_values(by='datetime')
        df.reset_index().to_feather(r"C:\Stocks\local\account\pnl.feather")
        self.df_pnl = df.set_index('datetime',drop = True)

class Plot:

    def sort(self):
        scan = self.df_traits
        sort_val = None
        if not self.init:
            sort = self.values['-input_sort-']
            reqs = sort.split('&')
            for req in reqs:
                if '^' in req:
                    sort_val = req.split('^')[1]
                    if sort_val not in scan.columns:
                        raise TimeoutError
                else:
                    req = req.split('|')
                    dfs = []
                    for r in req:
                        r = r.split('=')
                        trait = r[0]
                        if trait not in scan.columns or len(r) == 1:
                            raise TimeoutError
                        val = r[1]
                        if trait == 'annotation':
                            df = scan[scan[trait].str.contrains(val)]
                        else:
                            df = scan[scan[trait] == val]
                        dfs.append(df)
                    scan = pd.concat(dfs).drop_duplicates()
        if sort_val != None: scan = scan.sort_values(by = [sort_val], ascending = False)
        else:scan = scan.sample(frac = 1)
        self.sorted_traits = scan
        if os.path.exists("C:/Stocks/local/account/charts"):
            while True:
                try:
                    shutil.rmtree("C:/Stocks/local/account/charts")
                    break
                except:
                    pass
        os.mkdir("C:/Stocks/local/account/charts")

    def update(self):
        if self.init:
            ident = data.identify()
            if ident == 'laptop':scale = 4
            elif ident == 'desktop':scale = 4.5
            else: raise Exception('no scale given')
            Plot.sort(self)
            Plot.preload(self)
            
            c2 = [   [sg.Image(key = '-IMAGE3-')], [sg.Image(key = '-IMAGE1-')]]
            c1 = [[sg.Image(key = '-IMAGE2-')],
                [(sg.Text(key = '-number-'))], 
                [sg.Table([],num_rows = 2, key = '-trait_table-',auto_size_columns=True,justification='left', expand_y = False)],
                [sg.Table([],key = '-trade_table-',auto_size_columns=True,justification='left',num_rows = 5, expand_y = False)],
                [sg.Button('Prev'),sg.Button('Next'),sg.Button('Load'),sg.InputText(key = '-input_sort-')],
                [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            layout = [ [sg.Column(c1), sg.VSeperator(), sg.Column(c2)],]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=scale,finalize = True)
            Plot.preload
        trade_headings = ['Date             ','Shares   ','Price  ']
        trait_headings = ['pnl $','pnl %','pnl a', 'risk %','risk a','rank']
        bar = self.sorted_traits.iloc[self.i]
        trades = bar['trades']
        trade_table = [[datetime.datetime.strptime(trades[k][1], '%Y-%m-%d %H:%M:%S'),(float(trades[k][2])),float(trades[k][3])] for k in range(len(trades))]
        trait_table = [bar[trait] for trait in trait_headings]
        self.window["-number-"].update(str(f"{self.i + 1} of {len(self.df_traits)}"))
        self.window["-trait_table-"].update(trait_table,headings = trait_headings)
        self.window["-trade_table-"].update(trade_table, headings = trade_headings)
        for i in range(1,4):
            while True:
                try: 
                    image = Image.open(f'C:\Stocks\local\study\charts\{i}{self.i}.png')
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    self.window[f'-IMAGE{i}-'].update(data = bio.getvalue())
                except (PIL.UnidentifiedImageError, FileNotFoundError, OSError): pass
                else: break
        self.window.maximize()

    def preload(self):

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

    def plot(self):
        if self.event == 'Load':
            Plot.sort(self)
        elif self.event == 'Next' :
            if self.i == len(self.df_traits) - 1: self.i = 0
            else: self.i += 1
        elif self.event == 'Prev':
            if self.i == 0: self.i = len(self.df_traits) - 1
            else: self.i -= 1
        Plot.update(self)
        Plot.preload(self)
        

    def create(bar):
        i = bar[0]
        if (os.path.exists(r"C:\Screener\tmp\pnl\charts" + f"\{i}" + "1min.png") == False):
            df = bar[1]
            ticker = df.iat[i,0]
            tflist = ['1min','h','d']
            mc = mpf.make_marketcolors(up='g',down='r')
            s  = mpf.make_mpf_style(marketcolors=mc)
            fs = data.get_scale('Plot fs')
            fw = data.get_scale('Plot fw')
            fh = data.get_scale('Plot fh')
            dpi = data.get_scale('Plot dpi')
            for tf in tflist:
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
                if tf != '1min': mav = (10,20,50)
                else: mav = ()
                _, axlist = mpf.plot(df1, type='candle', volume=True  , 
                                        title=str(f'{ticker} , {tf}'), 
                                        style=s, warn_too_much_data=100000,returnfig = True,figratio = (fw,fh),
                                        figscale=fs, panel_ratios = (5,1), mav=mav, 
                                        tight_layout = True,
                                        addplot=apds)
                ax = axlist[0]
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
                plt.savefig(p1, bbox_inches='tight',dpi = dpi) 

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


