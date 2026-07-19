#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
from collections import deque
import datetime
import time
import ta
from enum import Enum
from os import listdir
from os.path import isfile, join
import sys
from operator import itemgetter


path = '/Users/uwe.muller/Hope/data/final/EURUSD/result/'


files = [f for f in listdir(path) if isfile(join(path, f)) and f[:11] == 'rl-results-']
ff = [[f, int(f[11:f.find('.')])] for f in files]
ff = sorted(ff, key=itemgetter(1))
files = [f[0] for f in ff]
print(files)

#df = pd.DataFrame()
cnt = []
for f in files:
    dfi = pd.read_csv(path + f)
    #df = df.append(dfi)
    npos = len(dfi[dfi["Avg-Total-Reward"] >= 0])
    avg_profit = np.mean(dfi["Avg-Total-Reward"])
    avg_hold = np.mean(dfi["Avg-Hold-Action"])
    avg_buy = np.mean(dfi["Avg-Buy-Action"])
    avg_sell = np.mean(dfi["Avg-Sell-Action"])
    avg_profit_long = np.mean(dfi["Long-Reward"])
    avg_profit_short = np.mean(dfi["Short_Reward"])
    avg_nlong = np.mean(dfi["#Long_Trades"])
    avg_nshort = np.mean(dfi["#Short_Trades"])

    cnt.append([int(f[11:-4]), 
                npos, 
                avg_profit, 
                avg_profit_long,
                avg_profit_short,
                avg_hold, 
                avg_buy, 
                avg_sell,
                avg_nlong,
                avg_nshort])

#df.to_csv(path + 'all-rl-results.csv')
pd.DataFrame(cnt, columns=["Iteration", 
                           "Cnt+Profit", 
                           "Avg-Profit",
                           "Avg-Profit-Long",
                           "Avg-Profit-Short",
                           "Avg-Hold",
                           "Avg-Buy",
                           "Avg-Sell",
                           "Avg-#Long-Trades",
                           "Avg-#Short-Trades"]).dropna().to_csv(path + 'cnt-pos.csv')
 
    



