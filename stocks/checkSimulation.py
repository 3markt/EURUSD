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
import ta
import numpy as np



basePath = '/Users/uwe.muller/Hope/'
name = 'DAX_bestMyAccu.csv'

df = pd.read_csv(basePath + 'simulation/' + name)

allWinLong = df['Win-Long'].sum()
allMedLong = df['Medium-Long'].sum()
allLossLong = df['Loss-Long'].sum()
allNumTradesLong = df['#Trades-Long'].sum()
allTotalLong = allWinLong+allMedLong-allLossLong

allWinShort = df['Win-Short'].sum()
allMedShort = df['Medium-Short'].sum()
allLossShort = df['Loss-Short'].sum()
allNumTradesShort = df['#Trades-Short'].sum()
allTotalShort = allWinShort+allMedShort-allLossShort

allTotal = allTotalLong + allTotalShort
allNumTrades = allNumTradesLong + allNumTradesShort

allPct = 100*allTotal/allNumTrades

print('TotalTrades=%6i TotalWin=%6.2f TotalPct=%6.2f TotalWinLong=%6.2f/%5i TotalWinShort=%6.2f/%5i' % \
      (allNumTrades, allTotal, allPct, allTotalLong, allNumTradesLong, allTotalShort, allNumTradesShort))
