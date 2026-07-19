# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import yfinance as yf
import pandas as pd

path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'

all_symbl = pd.read_csv(path + "IEXSymbls.csv")

symbl = "RYN"

aktie = yf.Ticker(symbl)
print(aktie.info["website"])
print(aktie.earnings/1000000)
ts = aktie.history(period="max")
ts['day'] = ts.index.day
ts['GV'] = 100*(ts['Close'].shift(-1) - ts['Close'])/ts['Close']
ts.to_csv(path + symbl + ".csv")

meanDays = ts[['day', 'GV']].groupby(by='day').mean()
meanDays.rename({'GV': 'meanGV'}, inplace=True, axis=1)
medDays = ts[['day', 'GV']].groupby(by='day').median()
medDays.rename({'GV': 'medianGV'}, inplace=True, axis=1)
stdDays = ts[['day', 'GV']].groupby(by='day').std()
stdDays.rename({'GV': 'stdGV'}, inplace=True, axis=1)
sumDays = ts[['day', 'GV']].groupby(by='day').sum()
sumDays.rename({'GV': 'sumGV'}, inplace=True, axis=1)
cntDays = ts[['day', 'GV']].groupby(by='day').count()
cntDays.rename({'GV': 'cntGV'}, inplace=True, axis=1)
tsDays = pd.concat([meanDays, medDays, stdDays, sumDays, cntDays], axis = 1)
print(tsDays)
