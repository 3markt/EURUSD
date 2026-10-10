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
from enum import Enum
from os import listdir
from os.path import isfile, join
import sys
from operator import itemgetter
#import matplotlib
#matplotlib.use('QtAgg')  # Nutzt das frisch installierte PyQt6-Fenstersystem
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8)) # Schön groß machen!

path = '/home/uwe/Hope/data/final/EURUSD/result_50_1/'
#path = '/home/uwe/Hope/data/final/EURUSD/result_20260929/'

files = [f for f in listdir(path) if isfile(join(path, f)) and f[:11] == 'rl-results-']
#files = [f for f in files if int(f.split("-")[2].split(".")[0]) < 1000000]
ff = [[f, int(f[11:f.find('.')])] for f in files]
ff = sorted(ff, key=itemgetter(1))
files = [f[0] for f in ff]
print(len(files), files[-1])

#df = pd.DataFrame()
cnt = []
for f in files:
    dfi = pd.read_csv(path + f)
    npos = len(dfi[dfi["Avg-Total-Reward"] >= 0])
    avg_profit = np.mean(dfi["Avg-Total-Reward"])
    avg_hold = np.mean(dfi["Avg-Hold-Action"])
    avg_buy = np.mean(dfi["Avg-Buy-Action"])
    avg_sell = np.mean(dfi["Avg-Sell-Action"])
    avg_profit_long = np.mean(dfi["Long-Reward"])
    avg_profit_short = np.mean(dfi["Short_Reward"])
    avg_nlong = np.mean(dfi["#Long_Trades"])
    avg_nshort = np.mean(dfi["#Short_Trades"])
    avg_nhold = np.mean(dfi["#Hold_Trades"])
    td_err_mean = np.mean(dfi["td_err Mean"])
    td_err_median = np.mean(dfi["td_err Median"])
    td_err_min = np.mean(dfi["td_err Min"])
    td_err_max = np.mean(dfi["td_err Max"])
    td_err_p10 = np.mean(dfi["td_err Pct10"])
    td_err_p90 = np.mean(dfi["td_err Pct90"])

    cnt.append([int(f[11:-4]), 
                npos, 
                avg_profit, 
                avg_profit_long,
                avg_profit_short,
                avg_hold, 
                avg_buy, 
                avg_sell,
                avg_nlong,
                avg_nshort,
                avg_nhold,
                td_err_mean,
                td_err_median,
                td_err_min,
                td_err_max,
                td_err_p10,
                td_err_p90])

df = pd.DataFrame(cnt, columns=["Iteration", 
                                "Cnt+Profit", 
                                "Avg-Profit",
                                "Avg-Profit-Long",
                                "Avg-Profit-Short",
                                "Avg-Hold",
                                "Avg-Buy",
                                "Avg-Sell",
                                "Avg-#Long-Trades",
                                "Avg-#Short-Trades",
                                "Avg-#Hold-Trades",
                                "Avg-TD-Error-Mean",
                                "Avg-TD-Error-Median",
                                "Avg-TD-Error-Min",
                                "Avg-TD-Error-Max",
                                "Avg-TD-Error-P10",
                                "Avg-TD-Error-P90"]).dropna()
df['smoothC'] = df['Cnt+Profit'].rolling(100).mean()
df['smoothP'] = df['Avg-Profit'].rolling(100).mean()
df.to_csv(path + 'cnt-pos.csv')

x = df['Iteration']
plt.xlabel('Iteration')
plt.rcParams["figure.figsize"] = (12, 7)

plt.title('# positiver Profit')
var = 'Cnt+Profit'
y = df[var]
plt.ylabel(var)
plt.scatter(x, y, label=var, linewidth=0.1)
var = 'smoothC'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
plt.grid()
plt.show()

plt.title('Avg Profit Gesamt')
var = 'Avg-Profit'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
var = 'smoothP'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
plt.legend(loc='best')
plt.grid()
plt.show()

plt.title('Avg Profit per Type')
var = 'Avg-Profit-Long'
y = df[var]
plt.scatter(x, y, label=var, linewidth=1)
var = 'Avg-Profit-Short'
y = df[var]
plt.scatter(x, y, label=var, linewidth=1)
plt.legend(loc='best')
plt.grid()
plt.show()

plt.title('Anzahl Actions per Typ')
var = 'Avg-Hold'
l1 = 'Avg-Hold-Actions'
y = df[var]
plt.scatter(x, y, label=l1, linewidth=0.1)
var = 'Avg-Buy'
l2 = 'Avg-Buy-Actions'
y = df[var]
plt.scatter(x, y, label=l2, linewidth=0.1)
var = 'Avg-Sell'
l3 = 'Avg-Sell-Actions'
y = df[var]
plt.scatter(x, y, label=l3, linewidth=0.1)
plt.legend(loc='best')
plt.grid()
plt.show()

plt.title('Anzahl Trades per Typ')
y1 = df['Avg-#Short-Trades']
l1 = '#-Short-Trades'
plt.scatter(x, y1, label=l1, linewidth=0.1)
y2 = df['Avg-#Long-Trades']
l2 = '#-Long-Trades'
plt.scatter(x, y2, label=l2, linewidth=0.1)
y2 = df['Avg-#Hold-Trades']
l2 = '#-Hold-Trades'
plt.scatter(x, y2, label=l2, linewidth=0.1)
plt.legend(loc='best')
plt.grid()
plt.show()

plt.title('Durchschnittliche Trade-Länge per Typ')
y1 = df['Avg-Sell']/df['Avg-#Short-Trades']
l1 = 'Avg-Short-Length'
plt.scatter(x, y1, label=l1, linewidth=0.1)
y2 = df['Avg-Buy']/df['Avg-#Long-Trades']
l2 = 'Avg-Long-Length'
plt.scatter(x, y2, label=l2, linewidth=0.1)
y2 = df['Avg-Hold']/df['Avg-#Hold-Trades']
l2 = 'Avg-HoldLength'
plt.scatter(x, y2, label=l2, linewidth=0.1)
plt.legend(loc='best')
plt.grid()
plt.show()

plt.title('TD Error Statistics')
var = 'Avg-TD-Error-Mean'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
var = 'Avg-TD-Error-Median'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
var = 'Avg-TD-Error-Min'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
var = 'Avg-TD-Error-Max'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
var = 'Avg-TD-Error-P10'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
var = 'Avg-TD-Error-P90'
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
plt.legend(loc='best')
plt.grid()
plt.show()