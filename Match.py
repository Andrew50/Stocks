import pathlib, io, shutil, os, math, random, PIL
import pandas as pd
import mplfinance as mpf
import PySimpleGUI as sg
from Data import Data as data
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Screener import Screener as screener
from multiprocessing.pool import Pool
import time

class Match:
    
    def match(ticker, tf = 'd', dt = None):
        start_time = time.time()
        for i in range(1):
            df = data.get(ticker, tf, dt, bars=40)
            print(df)
        print("--- %s seconds ---" % (time.time() - start_time))
        
if __name__ == "__main__":

    Match.match('IOT', 'd', '2023-06-02')