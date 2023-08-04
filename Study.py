
#automatically add typed in custom sub setup to sub setup list
#retype
import PySimpleGUI as sg
import os 
import pandas as pd
import pathlib
import mplfinance as mpf
import math
from PIL import Image
import PIL
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import io
from Data import Data as data
import shutil
from multiprocessing.pool import Pool
    

class Study:

    def run(self,current = False):
        self.preload_amount = 10
        self.sub_setups_list = {
            'd_EP':['true EP','range EP','low EP','volume EP','other EP'],
            'd_NEP':['backside NEP','range NEP','pivot NEP','high NEP', 'other NEP'],
            'd_F':['bull F','breakdown F','other F'],
            'd_NF':['bear F','breakout NF','other F'],
            'd_MR':['nep MR','straight MR','parabolic MR','extended MR','other MR'],
            'd_PS':['microcap PS','largecap PS','other PS'],
            'd_P':['strong P','weak P','range P','pocket P','other P'],
            'd_NP':['strong NP','weak NP','range NP','other P']}
        with Pool(int(data.get_config('Data cpu_cores'))) as self.pool:
            self.current = current
            self.init = True
            self.previ = None
            self.lookup(self)
            while True:
                self.event, self.values = self.window.read()
                if self.event == 'Yes' or self.event == 'No':
                    if self.event == 'Yes': val = 1
                    else: val = 0
                    self.event = 'Next'
                    bar = self.setups_data.iloc[self.i]
                    ticker = bar['ticker']
                    date = bar['datetime']
                    setup = bar['setup']
                    data.add_setup(ticker,date,setup,val,1)
                if self.event == 'Next' and (self.i < len(self.setups_data) - 1 or ((self.i < len(self.setups_data) + .5) and not self.current)): 
                    self.previ = self.i
                    if current: self.i += 1
                    else: self.i += .5
                    self.update(self)
                    self.window.refresh()
                    self.preload(self)
                elif self.event == 'Prev' and self.i > 0: 
                    self.previ = self.i
                    if current: self.i -= 1
                    else: self.i -= .5
                    self.update(self)
                elif self.event == 'Load':
                    self.previ = self.i
                    self.update(self) #save the annotation before lookup
                    self.previ = self.i
                    self.lookup(self)
                if self.event == sg.WIN_CLOSED:
                    self.window.close()

    def preload(self):
        if self.i == 0:
            if self.current: index_list = [i for i in range(self.preload_amount)]
            else: index_list = [i/2 for i in range(self.preload_amount*2)]
        else: index_list = [self.preload_amount + self.i - 1]
        arglist = [[self.setups_data,i,self.current] for i in index_list if i < len(self.setups_data)]
        self.pool.map(self.plot,arglist)
        
    def lookup(self):
        try:
            if self.current:
                scan = pd.read_feather(r"C:\Screener\local\study\todays_setups.feather").sort_values(by=['z'], ascending=False)
            else:
                scan = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
                sort_val = None
                if not self.init:
                    sort = self.values['-input_sort-']
                    reqs = sort.split('&')
                    if sort != "":
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
                                    if 'annotation' in trait:
                                        df = scan[scan[trait].str.contrains(val)]
                                    else:
                                        df = scan[scan[trait] == val]
                                    dfs.append(df)
                                scan = pd.concat(dfs).drop_duplicates()
                if sort_val != None: scan = scan.sort_values(by = [sort_val], ascending = False)
                else:scan = scan.sample(frac = 1)
            if scan.empty:
                raise TimeoutError
        except TimeoutError:
            sg.Popup('no setups found')
        else:
            self.setups_data = scan
            self.i = 0.0
            if os.path.exists("C:/Stocks/local/study/charts"):
                while True:
                    try:
                        shutil.rmtree("C:/Stocks/local/study/charts")
                        break
                    except:
                        pass
            os.mkdir("C:/Stocks/local/study/charts")
            self.preload(self)
            self.update(self)

    def plot(bar):
        setups_data = bar[0]
        print(setups_data)
        i = bar[1]
        #print(i)
        if int(i) != i: revealed = True
        else: revealed = False
        current = bar[2]
        date = (setups_data.iloc[math.floor(i)][1])
        ticker = setups_data.iloc[math.floor(i)][0]
        setup = setups_data.iloc[math.floor(i)][2]
        z= setups_data.iloc[math.floor(i)][3]
        tf= setup.split('_')[0]
        tf_list = []
        if 'w' in tf or 'd' in tf or 'h' in tf:
            intraday = False
            req_tf = ['1min','h','d','w']
            for t in req_tf:
                if t in tf: tf_list.append(tf)
                else: tf_list.append(t)
        else:
            intraday == True
            if tf == '1min': tf_list = ['d','h','5min','1min']
            else: tf_list = ['d','h',tf,'1min']
        plt.rcParams.update({'font.size': 30})
        mc = mpf.make_marketcolors(up='g',down='r')
        s  = mpf.make_mpf_style(marketcolors=mc)
        ii = len(tf_list)
        first_minute_high = 1
        first_minute_low = 1
        first_minute_close = 1
        first_minute_volume = 0
        for tf in tf_list:
            p = pathlib.Path("C:/Stocks/local/study/charts") / f'{ii}{i}.png'
            try:
                chart_size = 100
                if 'min' in tf: chart_offset = chart_size - 1
                else: chart_offset = 20
                if not revealed: chart_offset = 0
                df = data.get(ticker,tf,date,chart_size,chart_offset)
                if df.empty:
                    raise TimeoutError
                if not revealed and not intraday:
                    if tf == '1min':
                        open = df.iat[-1,0]
                        first_minute_high = df.iat[-1,1]/open
                        first_minute_low = df.iat[-1,2]/open
                        first_minute_close = df.iat[-1,3]/open
                        first_minute_volume = df.iat[-1,4]
                    else:
                        open = df.iat[-1,0]
                        df.iat[-1,1] = open * first_minute_high
                        df.iat[-1,2] = open * first_minute_low
                        df.iat[-1,3] = open * first_minute_close
                        df.iat[-1,4] = first_minute_volume
                if (current or revealed) and ii == 1: title = f'{ticker} {setup} {z} {tf}' 
                else: title = str(tf)
                if revealed: _, axlist = mpf.plot(df, type='candle', axisoff=True,volume=True, style=s, returnfig = True, title = title, figratio = (data.get_config('Study chart_aspect_ratio'),1),figscale=data.get_config('Study chart_size'), panel_ratios = (5,1), mav=(10,20), tight_layout = True,vlines=dict(vlines=[date], alpha = .25))
                else: _, axlist =  mpf.plot(df, type='candle', volume=True,axisoff=True,style=s, returnfig = True, title = title, figratio = (data.get_config('Study chart_aspect_ratio'),1),figscale=data.get_config('Study chart_size'), panel_ratios = (5,1), mav=(10,20), tight_layout = True)#, hlines=dict(hlines=[pmPrice], alpha = .25))
                ax = axlist[0]
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
                plt.savefig(p, bbox_inches='tight',dpi = data.get_config('Study chart_dpi'))
            except TimeoutError:
                shutil.copy(r"C:\Stocks\sync\files\blank.png",p)
            ii -= 1

    def update(self):
        if self.init:
            sg.theme('DarkGrey')
            layout = [  
            [sg.Image(key = '-IMAGE1-'),sg.Image(key = '-IMAGE2-')],
            [sg.Image(key = '-IMAGE3-'),sg.Image(key = '-IMAGE4-')],
            [(sg.Text( key = '-number-'))]]
            if self.current:
                layout += [[sg.Button('Prev'), sg.Button('Next'), sg.Button('Yes'),sg.Button('No')]]
            else:
                layout += [[sg.Multiline(size=(150, 5), key='-annotation-')],
                [sg.Combo([],key = '-sub_setup-', size = (20,10))],
                [sg.Button('Prev'), sg.Button('Next'),sg.Button('Load'),sg.InputText(key = '-input_sort-')]]
            self.window = sg.Window('Study', layout,margins = (10,10),scaling = data.get_config('Study ui_scale'),finalize = True)
            self.init = False
        for i in range(1,5):
            while True:
                try: 
                    image = Image.open(f'C:\Stocks\local\study\charts\{i}{self.i}.png')
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    self.window[f'-IMAGE{i}-'].update(data = bio.getvalue())
                except (PIL.UnidentifiedImageError, FileNotFoundError, OSError): pass
                else: break
        self.window['-number-'].update(str(f"{math.floor(self.i + 1)} of {len(self.setups_data)}"))
        if not self.current:
            if self.previ != None:
                df = pd.read_feather(r"C:\Stocks\local\study\historical_setups.feather")
                annotation = self.values["-annotation-"]
                sub_setup = self.values['-sub_setup-']
                if int(self.previ) == self.previ: col = 'pre_annotation'
                else: col = 'post_annotation'
                index = self.setups_data.index[math.floor(self.previ)]
                self.setups_data.at[index, col] = annotation
                df.at[index, col] = annotation
                self.setups_data.at[index,'sub_setup'] = sub_setup
                df.at[index,'sub_setup'] = sub_setup
                df.to_feather(r"C:\Stocks\local\study\historical_setups.feather")
            if int(self.i) == self.i: col = 'pre_annotation'
            else: col = 'post_annotation'
            bar = self.setups_data.iloc[math.floor(self.i)]
            self.window["-annotation-"].update(bar[col])
            ss = list(self.sub_setups_list[bar['setup']])
            self.window['-sub_setup-'].update(values = ss, value = bar['sub_setup'])
        self.window.maximize()

if __name__ == "__main__":
    Study.run(Study,False)