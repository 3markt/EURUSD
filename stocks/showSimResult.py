#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import datetime
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


path = '/Users/uwe.muller/Hope/model/DAXLongShort/'
name = 'simulation.csv'

df = pd.read_csv(path+name)

longWin = df['Win-Long'].sum()
longLoss = df['Loss-Long'].sum()
longMed = df['Medium-Long'].sum()
longNum = df['#Trades-Long'].sum()

shortWin = df['Win-Short'].sum()
shortLoss = df['Loss-Short'].sum()
shortMed = df['Medium-Short'].sum()
shortNum = df['#Trades-Short'].sum()

print('All LONG: win=%5i loss=%5i med=%6.2f #=%5i SHORT: win=%5i loss=%5i med=%6.2f #=%5i' % \
     (longWin, -longLoss, longMed, longNum, shortWin, -shortLoss, shortMed, shortNum))
    
symbls = list(set(df['Symbol'].values.tolist()))
symbls.sort()

for symbl in symbls:
    lWin = df.loc[df['Symbol'] == symbl, 'Win-Long'].sum()
    lLoss = -df.loc[df['Symbol'] == symbl, 'Loss-Long'].sum()
    lMed = df.loc[df['Symbol'] == symbl, 'Medium-Long'].sum()
    lNum = df.loc[df['Symbol'] == symbl, '#Trades-Long'].sum()
    sWin = df.loc[df['Symbol'] == symbl, 'Win-Short'].sum()
    sLoss = -df.loc[df['Symbol'] == symbl, 'Loss-Short'].sum()
    sMed = df.loc[df['Symbol'] == symbl, 'Medium-Short'].sum()
    sNum = df.loc[df['Symbol'] == symbl, '#Trades-Short'].sum()


    print('%8s LONG: win=%5i loss=%5i med=%6.2f #=%5i SHORT: win=%5i loss=%5i med=%6.2f #=%5i' % \
          (symbl, lWin, lLoss, lMed, lNum, sWin, sLoss, sMed, sNum))
