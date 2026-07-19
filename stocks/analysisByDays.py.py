# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import yfinance as yf
import pandas as pd


def posVals(x):
    
    cntPos = 0
    for s in x:
        if (s > 0):
            cntPos += 1
    return int(cntPos)
    
    
    
path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'

all_symbl = pd.read_csv(path + "IEXSymbls.csv")

symbl = "AAPL"

aktie = yf.Ticker(symbl)
print(aktie.info["website"])
print(aktie.earnings/1000000)
ts = aktie.history(period="max")
ts['day'] = ts.index.day
ts['weekDay'] = ts.index.weekday
ts['month'] = ts.index.month
ts['GV'] = 100*(ts['Close'].shift(-1) - ts['Close'])/ts['Close']
ts.to_csv(path + symbl + ".csv")

ts = ts[-1300:]
meanDays = ts[['day', 'GV']].groupby(by='day').mean()
meanDays.rename({'GV': 'meanGV'}, inplace=True, axis=1)
medDays = ts[['day', 'GV']].groupby(by='day').median()
medDays.rename({'GV': 'medianGV'}, inplace=True, axis=1)
cntPosDays = ts[['day', 'GV']].groupby(by='day').agg(posVals)
cntPosDays.rename({'GV': 'cntPosDays'}, inplace=True, axis=1)
stdDays = ts[['day', 'GV']].groupby(by='day').std()
stdDays.rename({'GV': 'stdGV'}, inplace=True, axis=1)
sumDays = ts[['day', 'GV']].groupby(by='day').sum()
sumDays.rename({'GV': 'sumGV'}, inplace=True, axis=1)
cntDays = ts[['day', 'GV']].groupby(by='day').count()
cntDays.rename({'GV': 'cntGV'}, inplace=True, axis=1)
tsDays = pd.concat([meanDays, medDays, stdDays, sumDays, cntDays, cntPosDays], axis = 1)
tsDays['posFract'] = tsDays['cntPosDays']/tsDays['cntGV']
print(tsDays)

meanWDays = ts[['weekDay', 'GV']].groupby(by='weekDay').mean()
meanWDays.rename({'GV': 'meanGV'}, inplace=True, axis=1)
medWDays = ts[['weekDay', 'GV']].groupby(by='weekDay').median()
medWDays.rename({'GV': 'medianGV'}, inplace=True, axis=1)
cntPosWDays = ts[['weekDay', 'GV']].groupby(by='weekDay').agg(posVals)
cntPosWDays.rename({'GV': 'cntPosWDays'}, inplace=True, axis=1)
stdWDays = ts[['weekDay', 'GV']].groupby(by='weekDay').std()
stdWDays.rename({'GV': 'stdGV'}, inplace=True, axis=1)
sumWDays = ts[['weekDay', 'GV']].groupby(by='weekDay').sum()
sumWDays.rename({'GV': 'sumGV'}, inplace=True, axis=1)
cntWDays = ts[['weekDay', 'GV']].groupby(by='weekDay').count()
cntWDays.rename({'GV': 'cntGV'}, inplace=True, axis=1)
tsWDays = pd.concat([meanWDays, medWDays, stdWDays, sumWDays, cntWDays, cntPosWDays], axis = 1)
tsWDays['posFract'] = tsWDays['cntPosWDays']/tsWDays['cntGV']
print(tsWDays)

meanMonth = ts[['month', 'GV']].groupby(by='month').mean()
meanMonth.rename({'GV': 'meanGV'}, inplace=True, axis=1)
medMonth = ts[['month', 'GV']].groupby(by='month').median()
medMonth.rename({'GV': 'medianGV'}, inplace=True, axis=1)
cntPosMonth = ts[['month', 'GV']].groupby(by='month').agg(posVals)
cntPosMonth.rename({'GV': 'cntPosMonth'}, inplace=True, axis=1)
stdMonth = ts[['month', 'GV']].groupby(by='month').std()
stdMonth.rename({'GV': 'stdGV'}, inplace=True, axis=1)
sumMonth = ts[['month', 'GV']].groupby(by='month').sum()
sumMonth.rename({'GV': 'sumGV'}, inplace=True, axis=1)
cntMonth = ts[['month', 'GV']].groupby(by='month').count()
cntMonth.rename({'GV': 'cntGV'}, inplace=True, axis=1)
tsMonth = pd.concat([meanMonth, medMonth, stdMonth, sumMonth, cntMonth, cntPosMonth], axis = 1)
tsMonth['posFract'] = tsMonth['cntPosMonth']/tsMonth['cntGV']
print(tsMonth)


