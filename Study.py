

import PySimpleGUI as sg
import os 
import pandas as pd
import pathlib
import mplfinance as mpf
import math
from PIL import Image
from matplotlib import pyplot as plt
import datetime
import matplotlib.ticker as mticker
import io
from Data import Data as data
import shutil
from multiprocessing.pool import Pool
    

class Study:

    def __init__(self,name):
        self.preload_amount = 10

    def run(self,current = False):
        self.current = current
        self.init = True
        self.previ = None
        with Pool(data.get_nodes()) as self.pool:
            self.lookup(self)
            self.update(self,True,None,0)
            while True:
                self.event, self.values = self.window.read()
                if self.event == 'Yes' or self.event == 'No':
                    self.event = 'Next'
                    data.add_setup(ticker,date,setup)
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
        arglist = [[self.setups_data,i,self.current] for i in index_list]
        self.pool.map_async(self.plot,arglist)
        
    def lookup(self):
        if self.historical:
            scan = pd.read_feather(r"C:\Stocks\local\study\setups.feather")
            sort = self.values('-input_sort-')
            reqs = sort.split('&')
            sort = None
            for req in reqs:
                if '^' in req:
                    sort = req[1:]
                else:
                    req = req.split('|')
                    dfs = []
                    for r in req:
                        r = r.split('=')
                        trait = r[0]
                        val = r[1]
                        if trait == 'annotation':
                            df = scan[scan[trait].str.contrains(val)]
                        else:
                            df = scan[scan[trait] == val]
                        dfs.append(df)
                    scan = pd.concat(dfs).drop_duplicates()
            if sort != None:
                scan = scan.sort_values(by = [sort], ascending = False)
        else:
            scan = pd.read_feather(r"C:\Screener\local\study\todays_setups.feather").sort_values(by=['z'], ascending=False)
        if scan.empty:
            sg.Popup('no setups found')
        else:
            self.setups_data = scan
            self.i = 0.0
            self.init = True
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
        i = bar[1]
        current = bar[2]
        date = (setups_data.iloc[i][1])
        ticker = setups_data.iloc[i][0]
        setup = setups_data.iloc[i][2]
        z= setups_data.iloc[i][3]
        tf= setup.split('_')[0]
        if int(i) != i:
            revealed = True
        elif current == True:
            return #if its current then no need to create unrevealed charts as they wont be used
        i = math.floor(i)
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
        ident = data.identify()
        if ident == 'laptop':
            if current:
                fs = .49
                fw = 41
                fh = 18
                dpi = 330
            else:
                raise Exception('undefined figure scales')
        elif ident == 'desktop':
            if current:
                fs = 1.08
                fw = 15
                fh = 7
            else:
                fw = 20
                fh = 7
                fs = .8
        elif ident == 'tae':
            if current:
                fs = .75
                fw = 41
                fh = 18
            else:
                fs = .6
                fw = 41
                fh = 18
        elif ident == 'ben':
            raise Exception('undefined figure scales')
        plt.rcParams.update({'font.size': 30})
        mc = mpf.make_marketcolors(up='g',down='r')
        s  = mpf.make_mpf_style(marketcolors=mc)
        ii = len(tf_list)
        for tf in tf_list:
            ii -= 1
            p4 = pathlib.Path("C:/Screener/tmp/charts") / f'{ii}{i}.png'
            try:
                chart_size = 150
                if 'min' in tf:
                    chart_offset = chart_size - 1
                else:
                    chart_offset = 30
                if not revealed: chart_offset = 0
                df = data.get(ticker,tf,date,chart_size,chart_offset)
                
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
                if revealed:
                    _, axlist = mpf.plot(df, type='candle', axisoff=True,volume=True, style=s, returnfig = True, figratio = (fw,fh),figscale=fs, panel_ratios = (5,1), mav=(10,20), tight_layout = True,vlines=dict(vlines=[d4], alpha = .25))
                else:
                    _, axlist =  mpf.plot(df, type='candle', volume=True,axisoff=True,style=s, returnfig = True, figratio = (fw,fh),figscale=fs, panel_ratios = (5,1), mav=(10,20), tight_layout = True)#, hlines=dict(hlines=[pmPrice], alpha = .25))
                ax = axlist[0]
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
                plt.savefig(p4, bbox_inches='tight',dpi = dpi)
            except TimeoutError:
                shutil.copy(r"C:\Screener\tmp\blank.png",p4)
        
    def update(self):
        image1 = None
        image2 = None
        image3 = None
        image4 = None

        gosh = 1
        start = datetime.datetime.now()
        while True:
            if (datetime.datetime.now() - start).seconds < gosh or init:
                if image1 == None:
                    try:
                        image1 = Image.open(r"C:\Screener\tmp\charts\1" + str(self.i) + ".png")
                    except:
                        pass
                if image2 == None:
                    try:
                        image2 = Image.open(r"C:\Screener\tmp\charts\2" + str(self.i) + ".png")
                    except :
                        pass
                if image3 == None:
                    try:
                        image3 = Image.open(r"C:\Screener\tmp\charts\3" + str(self.i) + ".png")
                    except:
                        pass
                if image4 == None:
                    try:
                        image4 = Image.open(r"C:\Screener\tmp\charts\4" + str(self.i) + ".png")
                    except:
                        pass

                if image1 != None and image2 != None and image3 != None and image4 != None:
                    break
            else:
                self.preload(self,self.i)
                start = datetime.datetime.now()
                print('reloading image')
                gosh = 10
        
        bio1 = io.BytesIO()

        image1.save(bio1, format="PNG")
        bio2 = io.BytesIO()
        image2.save(bio2, format="PNG")
        bio3 = io.BytesIO()
        image3.save(bio3, format="PNG")
        bio4 = io.BytesIO()
        image4.save(bio4, format="PNG")
        



        if self.init:
            sg.theme('DarkGrey')
            if data.identify() == 'tae': scale = 1.1
            else: scale = 2.5
            layout = [  
            [sg.Image(bio1.getvalue(),key = '-IMAGE-'),sg.Image(bio2.getvalue(),key = '-IMAGE2-')],
            [sg.Image(bio3.getvalue(),key = '-IMAGE3-'),sg.Image(bio4.getvalue(),key = '-IMAGE4-')],
            [(sg.Text( key = '-number-'))]]
            if self.current:
                layout += [[sg.Button('Prev'), sg.Button('Next'), sg.Button('Yes'),sg.Button('No')]]
            else:
                layout += [[sg.Multiline(size=(150, 5), key='-annotation-')],
                [sg.InputText(key = '-input_sort-')],
                [sg.Button('Prev'), sg.Button('Next'),sg.Button('Load')]]
                self.window = sg.Window('Study', layout,margins = (10,10),finalize = True)
            self.window["-IMAGE-"].update(data=bio1.getvalue())
            self.window["-IMAGE2-"].update(data=bio2.getvalue())
            self.window["-IMAGE3-"].update(data=bio3.getvalue())
            self.window["-IMAGE4-"].update(data=bio4.getvalue())
            self.window['-number-'].update(str(f"{round(self.i + 1)} of {len(self.setups_data)}"))
            if not self.current:
                if self.previ != None:
                df = pd.read_feather(r"C:\Stocks\local\study\setups.feather")
                annotation = self.values["-annotation-"]
                
                if int(self.previ) == self.previ: col = 'pre_annotation'
                else: col = 'post_annotation'
                previ = math.floor(previ)
                index = self.setups_data.index[math.floor(previ)]
                self.setups_data.at[index, col] = annotation
                df.at[index, col] = annotation
                df.to_feather(r"C:\Stocks\local\study\historical_setups.feather")
            if int(self.i) == self.i: col = 'pre_annotation'
            else: col = 'post_annotation'
            self.window["annotation"].update(self.setups_data.at[math.floor(self.i),col])
        self.window.maximize()

if __name__ == "__main__":
    UI.loop(UI,True)




          g1 = [[sg.Text("true EP")],
                      [sg.Text("range EP")],
                      [sg.Text("pivot EP")],
                      [sg.Text("low EP")],
                      [sg.Text("volume EP")]
                      ]
                g2 = [     [sg.Text("backside NEP")],
                      [sg.Text("range NEP")],
                      [sg.Text("pivot NEP")],
                      [sg.Text("high NEP")]]

                g3 = [    [sg.Text("strong P")],
                      [sg.Text("weak P")],
                      [sg.Text("range P")],
                      [sg.Text("pocket P")]]

                g4 = [ [sg.Text("strong NP")],
                      [sg.Text("weak NP")],
                      [sg.Text("range NP")]]
                      


                      
                g5 =   [[sg.Text("nep MR")],
                      [sg.Text("parabolic MR")],
                      [sg.Text("straight MR")],
                      [sg.Text("extended MR")]]

                g6 = [     [sg.Text("bull F")],
                      [sg.Text("bear F")],
                      [sg.Text("breakdown F")],
                      [sg.Text("breakout F")]]