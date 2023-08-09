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

        with Pool(int(data.get_config('Data cpu_cores'))) as self.pool:
            sg.theme('DarkGrey')
            try: self.df_log = pd.read_feather(r"C:\Stocks\local\account\log.feather").sort_values(by='datetime',ascending = False)
            except FileNotFoundError: self.df_log = pd.DataFrame()
            try: self.df_traits = pd.read_feather(r"C:\Stocks\local\account\traits.feather")
            except FileNotFoundError: self.df_traits = pd.DataFrame()
            try: self.df_pnl = pd.read_feather(r"C:\Stocks\local\account\pnl.feather").set_index('datetime', drop = True)
            except FileNotFoundError: self.df_pnl = pd.DataFrame()
            self.init = True
            self.menu = 'Log'
            try: self.queued_recalcs = pd.read_feather('C:/Stocks/local/account/queued_recalcs.feather')
            except FileNotFoundError: self.queued_recalcs = pd.DataFrame()
            Log.log(self)
            while True:
                self.event, self.values = self.window.read()

                if self.event == "Traits" or self.event == "Plot" or self.event == "Account" or self.event == "Log":
                    if self.df_log.empty: 
                        sg.Popup('Log is empty')
                        continue
                    if self.df_pnl.empty: Account.calc(self,self.df_log)
                    if self.df_traits.empty: Traits.calc(self,self.df_log)
                    if not self.queued_recalcs.empty: 
                        Account.calc(self)
                        Traits.calc(self)
                        self.queued_recalcs = pd.DataFrame()
                        try: os.remove('C:/Stocks/local/account/queued_recalcs.feather')
                        except FileNotFoundError: pass
                    self.menu = self.event
                    self.init = True
                    self.window.close()
                if self.menu == "Traits": Traits.traits(self)
                elif self.menu == "Plot": Plot.plot(self)
                elif self.menu == "Account": Account.account(self)
                elif self.menu == "Log": Log.log(self)

class Log:

    def queue(self,updated_log):
        if not self.df_log.empty: 
            deposits = self.df_log[self.df_log['ticker'] == 'Deposit']
            new_log = pd.concat([self.df_log[self.df_log['ticker'] != 'Deposit'], updated_log]).drop_duplicates(keep=False).sort_values(by='datetime', ascending = False)
            self.df_log = pd.concat([updated_log,deposits]).sort_values(by='datetime', ascending = False).reset_index(drop = True)
            self.queued_recalcs = pd.concat([self.queued_recalcs,new_log]).reset_index(drop = True)
        else:
            self.df_log = updated_log
            self.queued_recalcs = updated_log
        self.queued_recalcs.to_feather('C:/Stocks/local/account/queued_recalcs.feather')
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
        Log.queue(self,updated_log)

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
            self.window = sg.Window('Log', layout,margins = (10,10),scaling=data.get_config('Log ui_scale'),finalize = True)
        self.window['-log_table-'].update(self.df_log.values.tolist())
        self.window.maximize()
    
    def pull(self):
        ident = data.get_config('Data identity')
        print(ident)
        if ident == 'desktop' or ident == 'laptop':
            host = "imap.gmail.com"
            username = "billingsandrewjohn@gmail.com"
            password = 'kqnrpkqscmvkrrnm'
            download_folder = "C:/Stocks/local/account"
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
            log = pd.read_csv(download_folder + '/Webull_Orders_Records.csv')
            log2 = pd.DataFrame()
            log2['ticker'] = log['Symbol'] 
            log2['datetime']  = pd.to_datetime(log['Filled Time'],format='mixed')
            log2['shares'] = log['Total Qty']
            log2['price'] = log['Avg Price']
            for i in range(len(log)):
                if log.at[i,'Side'] != 'Buy':
                    log2.at[i,'shares'] *= -1
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
        Log.queue(self,updated_log)


class Traits:

    def calc(self,new_log = pd.DataFrame()):
        if new_log.empty: new_log = self.queued_recalcs
        dfs = []
        ticker_list = [*set(new_log['ticker'])]
        for ticker in ticker_list:
            ticker_logs = self.df_log[self.df_log['ticker'] == ticker]
            dfs.append(ticker_logs)
            if not self.df_traits.empty: self.df_traits = self.df_traits[self.df_traits['ticker'] != ticker]
        trades = Traits.get_trades(pd.concat(dfs))
        print(self.df_pnl)
        arglist = [[trades.iloc[i], self.df_pnl] for i in range(len(trades))]
        traits = pd.concat(data.pool(Traits.worker,arglist))
        self.df_traits = pd.concat([self.df_traits,traits]).sort_values(by='datetime',ascending = True).reset_index(drop = True)
        self.df_traits.to_feather(r'C:\Stocks\local\account\traits.feather')
        

    def get_trades(df_log):
        pos = []
        df_log = df_log.sort_values(by='datetime',ascending = True).reset_index(drop = True)
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
                           # bar[3].reverse()
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
         #   bar[3].reverse()
            add = pd.DataFrame({
            'ticker': [bar[0]],
            'datetime':[data.format_date(bar[1])],
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
        first_trade = trades[0]
        last_trade = trades[-1]
        open_datetime = data.format_date(first_trade[1])
        close_datetime = data.format_date(last_trade[1])
        setup = first_trade[4]
        data_exists = True
        account_size = df_pnl.iloc[data.findex(df_pnl,open_datetime)]['account']
        try: 
            df_1min = data.get(ticker,'1min',close_datetime)
            time = datetime.time(9,30,0)
            rounded_open_datetime = datetime.datetime.combine(open_datetime.date(),time)
            index = data.findex(df_1min,rounded_open_datetime)
            df_1min = df_1min[index:]
        except (FileNotFoundError , IndexError) as e: 
            data_exists = False
        if float(first_trade[2]) > 0: direction = 1
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
        high_price = -100000000*direction
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
                if price*direction > high_price*direction: high_price = price
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
            next_trade_date = data.format_date(first_trade[1])
            trade_index = 0
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

                
                if size != 0:
                    current_low_percent = (current_low / abs(size) - 1) *100
                    if current_low_percent < min_percent: min_percent = current_low_percent
                if not opened:
                    if direction*low < direction*lod_price:
                        lod_price = low
                       
                while date > next_trade_date:
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
                current_low += open_shares * (low - prev_low)
                current_high += open_shares * (high - prev_high)
                if current_low < low_dollars: low_dollars = current_low
                if current_high > high_dollars: high_dollars = current_high
                prev_high = high
                prev_low = low
            min_account = (low_dollars / account_size - 1)*100
            risk_percent = ( float(trades[0][3]) / lod_price - 1) * 100 * direction
            if pnl_dollars > high_dollars: high_dollars = pnl_dollars
            if pnl_dollars < low_dollars: low_dollars = pnl_dollars
            if high*direction > high_price*direction: high_price = high
        else:
            risk_percent = pd.NA
            min_percent = pd.NA
            min_account = pd.NA
        pnl_percent = ((pnl_dollars / size_dollars)) * 100
        pnl_account = ((pnl_dollars / account_size)) * 100
        high_percent = ((high_price / float(first_trade[3])) - 1) * 100 * direction
        size_percent = (size_dollars / account_size) * 100

        def try_round(v):
            try: return round(v,2)
            except: return v
        traits = pd.DataFrame({
        'ticker': [ticker], 'datetime':[open_datetime], 'trades': [trades], 'setup':[setup], 'pnl $':[try_round(pnl_dollars)], 'pnl %':[try_round(pnl_percent)], 'pnl a':[try_round(pnl_account)], 'size $':[try_round(size_dollars)], 'size %':[try_round(size_percent)],  'high %':[try_round(high_percent)],
       'risk %':[try_round(risk_percent)], 'min %':[try_round(min_percent)],'min a':[try_round(min_account)], 'arrow_list':[arrow_list], 'closed':[try_round(closed)], 'open':[0], 'high':[try_round(high_dollars)], 'low':[try_round(low_dollars)], 'close':[try_round(pnl_dollars)],'volume':[try_round(total_size)]})
        return traits

    def calc_trait_values(df,title):
        avg_loss = df[df['pnl a'] <= 0]['pnl a'].mean()
        avg_gain = df[df['pnl a'] > 0]['pnl a'].mean()
        wins = []
        for i in range(len(df)):
            if df.iloc[i]['pnl $'] > 0: wins.append(1)
            else: wins.append(0)
        win = statistics.mean(wins) * 100
        high = df[df['high %'] > 0]['high %'].mean()
        risk = df[df['risk %'] > 0]['risk %'].mean()
        size = df[df['size %'] > 0]['size %'].mean()
        pnl = ((df['pnl a'] / 100) + 1).tolist()
        gud = 1
        for i in pnl:
            gud *= i
        pnl = gud
        trades = len(df)
        return [title,round(avg_gain,2), round(avg_loss,2), round(win,2),round(high,2) ,  round(risk,2), round(size,2), round(trades,2), round(pnl,2)]

    def build_rolling_traits_table(self):

        g = self.df_traits.groupby(pd.Grouper(key='datetime', freq='3M'))
        dfs = [group for _,group in g]
        rolling_traits = []
        rolling_traits.append(Traits.calc_trait_values(self.df_traits,'Overall'))
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
        sorted_df = self.df_traits.sort_values(by = ['pnl a'], ascending = winners).reset_index(drop = True)
        df = pd.DataFrame()
        df['#'] = sorted_df.index + 1
        df['Ticker'] = sorted_df['ticker']
        df['Date'] = sorted_df['datetime']
        df['% a'] = sorted_df['pnl a'].round(2)
        return df.values.tolist()
    
    def traits(self):
        if self.df_traits.empty or self.event == 'Recalc': 
            Traits.calc(self,self.df_log)
            self.queued_recalcs = pd.DataFrame()
            try: os.remove('C:/Stocks/local/account/queued_recalcs.feather')
            except FileNotFoundError: pass
        elif (self.event == '-table_losers-' or self.event =='-table_winners-') and len(self.values[self.event]) > 0:
            sorted_df = self.df_traits.sort_values(by = ['pnl a'], ascending = (self.event == '-table_winners-')).reset_index(drop = True)
            Plot.create([self.values[self.event][0],sorted_df,True])
        elif '+CLICKED+' in self.event and self.event[2][1] != 0:
            table = self.window[self.event[0]].Values
            x_labels = [b[0] for b in table]
            y = [b[self.event[2][1]] for b in table]
            x = [i + 1 for i in range(len(table))]
            plt.clf()
            plt.scatter(x,y,s = data.get_config('Traits market_size'))
            if self.event[0] == '-table_rolling_traits-':
                z = np.polyfit(x[1:], y[1:], 1)
                p = np.poly1d(z)
                plt.plot(x,p(x),"r--")
                plt.xticks(x, x_labels)
            plt.gcf().set_size_inches((data.get_config('Traits chart_size') * data.get_config('Traits chart_aspect_ratio') * 22.5,data.get_config('Traits chart_size') * 25.7))
            string1 = "trait.png"
            p1 = pathlib.Path("C:/Stocks/local/account") / string1
            plt.savefig(p1,bbox_inches='tight')
        Traits.update(self)
    def update(self):

        if self.init:
            self.init = False
            scale = data.get_config('Traits ui')
            rolling_traits_header = ['Date  ','Avg Gain','Avg Loss','Win %','High %','Risk %','Size a','Trades','PnL a']
            biggest_trades_header = ['Rank','Ticker','Date    ','% a ']
            setup_traits_header = ['Setup','Gain','Loss','Win %','High','Risk %','Size','Trades','PnL a']
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

        if os.path.exists('C:/Stocks/local/account/trait.png'):
            image = Image.open('C:/Stocks/local/account/trait.png')
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            self.window['-CHART-'].update(data = bio.getvalue())


class Account:
    
    def account(self):
        if self.event == 'Recalc': 
            Account.calc(self,self.df_log)
            self.queued_recalcs = pd.DataFrame()
            try: os.remove('C:/Stocks/local/account/queued_recalcs.feather')
            except FileNotFoundError: pass
        else:  self.pnl_chart_type = self.event
        Account.update(self)

    def update(self):
        if self.init:
            self.pnl_chart_type = 'Trade'
            self.init = False
            layout =[
            [sg.Image(key = '-CHART-')],
            [sg.Button('Trade'),sg.Button('Periodic Trade'),sg.Button('Real'),sg.Button('Periodic Real'),sg.Button('Recalc')],
            [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=data.get_config('Account ui_scale'),finalize = True)
        if 'Trade' in self.pnl_chart_type:
            
            df = self.df_traits.set_index('datetime')[['open','high','low','close','volume']]
            if 'Periodic' not in self.pnl_chart_type:
                pc = 0
                for i in range(len(df)):
                    v = df.iat[i,4]
                    c = df.iat[i,3] + pc
                    o = pc
                    h = df.iat[i,1] + pc
                    l = df.iat[i,2] + pc
                    df.iloc[i] = [o,h,l,c,v]
                    pc = c
        elif 'Real' in self.pnl_chart_type:
            df = self.df_pnl
            logic = {'open'  : 'first','high'  : 'max','low'   : 'min','close' : 'last','volume': 'sum' }
            df = df.resample('d').apply(logic).dropna()
            if 'Periodic' in self.pnl_chart_type:
                pc = 0
                for i in range(len(df)):
                    c = df.iat[i,3] - pc
                    v = df.iat[i,4]
                    o = 0
                    h = df.iat[i,1] - pc
                    l = df.iat[i,2] - pc
                    df.iloc[i] = [o,h,l,c,v]
                    pc += c
        mc = mpf.make_marketcolors(up='g',down='r')
        s  = mpf.make_mpf_style(marketcolors=mc)
        string1 = "pnl.png"
        p1 = pathlib.Path("C:/Stocks/local/account") / string1
        _,_ = mpf.plot(df, type='candle', volume=True, style=s, warn_too_much_data=100000,returnfig = True,figratio = (data.get_config('Account chart_aspect_ratio'),1),figscale=data.get_config('Account chart_size'), panel_ratios = (5,1), mav=(10,20), tight_layout = True)
        plt.savefig(p1, bbox_inches='tight',dpi = data.get_config('Account chart_dpi'))
        bio1 = io.BytesIO()
        image1 = Image.open(r"C:\Stocks\local\account\pnl.png")
        image1.save(bio1, format="PNG")
        self.window["-CHART-"].update(bio1.getvalue())
        self.window.maximize()

  
    def calc(self, new_log = pd.DataFrame()):
        if new_log.empty: new_log = self.queued_recalcs
        new_log = new_log.sort_values(by='datetime',ascending = True).reset_index(drop = True)
        start_datetime = new_log.iloc[0]['datetime']
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
            self.df_pnl = self.df_pnl[:index]

            bar = self.df_pnl.iloc[-1]
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
        df_list = []
        next_trade_date = new_log.iloc[1]['datetime']
        log_index = 0
        pbar = tqdm(total=len(date_list))
        for i in range(len(date_list)):
            date = date_list[i]
            if i > 0: prev_date = date_list[i-1]
            pnlvol = 0
            pnlo = pnl
            pnll = pnlo
            pnlh = pnlo
            while date > next_trade_date:
                remove = False
                ticker = new_log.iat[log_index,0]
                shares = new_log.iat[log_index,2]
                price = new_log.iat[log_index,3]
                if ticker == 'Deposit': deposits += price
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
                        except :
                            df = price
                        pos.append([ticker,shares,df])
                    df = pos[pos_index][2]
                    if isinstance(df, pd.DataFrame):
                        ind = data.findex(df,prev_date)
                        c1 = df.iat[ind,3]
                        gosh = (c1 - price)*shares
                        pnl += gosh
                        if gosh > 0: pnlh += gosh
                        else: pnll += gosh
                    pnlvol += abs(shares*price)
                    if remove: del pos[pos_index]
                log_index += 1
                if log_index >= len(new_log): next_trade_date = datetime.datetime.now() + datetime.timedelta(days=100)
                else: next_trade_date = new_log.iat[log_index,1]
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
                }).set_index('datetime')
            df_list.append(add)
            pbar.update(1)
        pbar.close()
        df = pd.concat(df_list)
        df = pd.concat([self.df_pnl,df])
        df = df.sort_values(by='datetime', ascending = True)
        df.reset_index().to_feather(r"C:\Stocks\local\account\pnl.feather")
        self.df_pnl = df
        print(df)

class Plot:

    def sort(self):
        try:
            scan = self.df_traits
            sort_val = None
            if not self.init:
                sort = self.values['-input_sort-']
                reqs = sort.split('&')
                if sort != "":
                    for req in reqs :
                        if '^' in req:
                            sort_val = req.split('^')[1]
                            if sort_val not in scan.columns and sort_val != 'r':
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
                if scan.empty: raise TimeoutError
            if sort_val != None:
                if sort_val == 'r': scan = scan.sample(frac = 1)
                else: scan = scan.sort_values(by = [sort_val], ascending = False)
            else:scan = scan.sort_values(by = 'datetime', ascending = True)
        except TimeoutError: sg.Popup('no setups found')
        self.sorted_traits = scan
        self.i = 0
        if os.path.exists("C:/Stocks/local/account/charts"):
            while True:
                try:
                    shutil.rmtree("C:/Stocks/local/account/charts")
                    break
                except:
                    pass
        os.mkdir("C:/Stocks/local/account/charts")
        Plot.preload(self)

    def update(self):
        trade_headings = ['Date             ','Shares   ','Price  ']
        trait_headings = ['setup','pnl $','pnl %','pnl a', 'high %','risk %']
        if self.init:
            Plot.sort(self)
            
            
            c2 = [   [sg.Image(key = '-IMAGE2-')], [sg.Image(key = '-IMAGE0-')]]
            c1 = [[sg.Image(key = '-IMAGE1-')],
                [(sg.Text(key = '-number-'))], 
                [sg.Table([],headings = trait_headings,num_rows = 2, key = '-trait_table-',auto_size_columns=True,justification='left', expand_y = False)],
                [sg.Table([],headings = trade_headings, key = '-trade_table-',auto_size_columns=True,justification='left',num_rows = 5, expand_y = False)],
                [sg.Button('Prev'),sg.Button('Next'),sg.Button('Load'),sg.InputText(key = '-input_sort-')],
                [sg.Button('Account'), sg.Button('Log'),sg.Button('Traits'),sg.Button('Plot')]]
            layout = [ [sg.Column(c1), sg.VSeperator(), sg.Column(c2)],]
            self.window = sg.Window(self.menu, layout,margins = (10,10),scaling=data.get_config('Plot ui_scale'),finalize = True)
            self.init = False
        
        bar = self.sorted_traits.iloc[self.i]
        trades = bar['trades']
        trade_table = [[datetime.datetime.strptime(trades[k][1], '%Y-%m-%d %H:%M:%S'),(float(trades[k][2])),float(trades[k][3])] for k in range(len(trades))]
        trait_table = [[bar[trait] for trait in trait_headings]]
        self.window["-number-"].update(str(f"{self.i + 1} of {len(self.df_traits)}"))
        self.window["-trait_table-"].update(trait_table)
        self.window["-trade_table-"].update(trade_table)
        for i in range(3):
            while True:
                try: 
                    image = Image.open(f'C:/Stocks/local/account/charts/{i}_{self.i}.png')
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    self.window[f'-IMAGE{i}-'].update(data = bio.getvalue())
                except (PIL.UnidentifiedImageError, FileNotFoundError, OSError, SyntaxError) as e: pass
                else: break
        self.window.maximize()

    def preload(self):
        helper_list = list(range(len(self.sorted_traits))) + list(range(len(self.sorted_traits)))
        if self.i == 0: index_list = [0,1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9]
        else: index_list = [self.i + 9, self.i - 9]
        
        index_list = [helper_list[i] for i in index_list]
        arglist = []
        for i in index_list: arglist.append([i,self.sorted_traits,False])
        self.pool.map_async(Plot.create,arglist)

         

    def plot(self):
        if self.event == 'Load':
            Plot.sort(self)
        elif self.event == 'Next' :
            if self.i == len(self.df_traits) - 1: self.i = 0
            else: self.i += 1
            Plot.preload(self)
        elif self.event == 'Prev':
            if self.i == 0: self.i = len(self.df_traits) - 1
            else: self.i -= 1
            Plot.preload(self)
        Plot.update(self)

        

    def create(bar):
        i = bar[0]
        df = bar[1]
        from_traits = bar[2]
        if from_traits: 
            tflist = ['d']
            source = 'Traits'
        else: 
            
            tflist = ['1min','h','d']
            source = 'Plot'

        trait_bar = df.iloc[i]
        ticker = trait_bar['ticker']
        dt = trait_bar['datetime']
        for ii in range(len(tflist)):
            if not from_traits: 
                p = pathlib.Path("C:/Stocks/local/account/charts") / (str(ii) + '_' + str(i)  + ".png")
                if os.path.exists(p): return
            else: p = 'C:/Stocks/local/account/trait.png'

            tf = tflist[ii]
                

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
            god = bar[1].iloc[i]['arrow_list']
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
            startdate = dfall.iloc[0]['Datetime']
            enddate = dfall.iloc[-1]['Datetime']
            df1 = data.get(ticker,tf,dt,100,50)
            if df1.empty: 
                shutil.copy(r"C:\Stocks\sync\files\blank.png",p)
                continue
                
            minmax = 300
            
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

            mc = mpf.make_marketcolors(up='g',down='r')
            s  = mpf.make_mpf_style(marketcolors=mc)
            _, axlist = mpf.plot(df1, type='candle', volume=True  , 
                                    title=str(f'{ticker} , {tf}'), 
                                    style=s, warn_too_much_data=100000,returnfig = True,figratio = (data.get_config(f'{source} chart_aspect_ratio'),1),
                                    figscale=data.get_config(f'{source} chart_size'), panel_ratios = (5,1), mav=mav, 
                                    tight_layout = True,axisoff = True,
                                    addplot=apds)
            ax = axlist[0]
            ax.set_yscale('log')
            ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
            plt.savefig(p, bbox_inches='tight',dpi = data.get_config(f'{source} chart_dpi')) 

if __name__ == '__main__':
    Run.run(Run)



