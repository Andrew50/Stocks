from locale import normalize
from Screener import Screener as screener
from multiprocessing.pool import Pool
from Data import Data as data
import numpy as np
import datetime
from Screener import Screener as screener
from scipy.spatial.distance import euclidean, cityblock
from sfastdtw import sfastdtw
import time
from Test import Data,Get
import os
import numpy as np
from sklearn import preprocessing
import mplfinance as mpf
import pyts

from pyts.approximation import SymbolicAggregateApproximation



