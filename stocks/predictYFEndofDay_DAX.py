#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam
from sklearn.utils.class_weight import compute_class_weight

from os import listdir
from os.path import isfile, join
import datetime
from dateutil.relativedelta import relativedelta
import sys
import math
import gc
import ta
import warnings
import yfinance as yf


if not sys.warnoptions:
    warnings.simplefilter("ignore")






def high_lowCandle(df):
    
    df['highCandle'] = (df['High'] - df['Close'])/df['Close']    
    df['lowCandle'] = (df['Open'] - df['Low'])/df['Open']
    df.loc[(df['Close'] - df['Open'] < 0), 'highCandle'] = (df['Low'] - df['Close'])/df['Close']
    df.loc[(df['Close'] - df['Open'] < 0), 'lowCandle'] = (df['Open'] - df['High'])/df['Open']
    
    return df
    




def calcVola(df, shift, volaName):
        
    df[volaName] = 0.0

    for i in range(1, shift+1):
        df['Win'] = 100*(df['High'].shift(i) - df['Close'].shift(i-1))/df['Close'].shift(i-1)
        df['Loss'] = 100*(df['Low'].shift(i) - df['Close'].shift(i-1))/df['Close'].shift(i-1)
        df[volaName] += df['Win'].abs() + df['Loss'].abs()
        
    df[volaName] = np.log(df[volaName]/shift)
    return df    




def transPercentile(df, low, high):
    
    df.loc[(df > df.quantile(high))] = df.quantile(high)
    df.loc[(df < df.quantile(low))] = df.quantile(low)
    
    return df





def prepFeaturesRaw(symbl, path, df, colHead, features, bs, ts):
    
    stdLow = 0.01
    stdHigh = 0.99
    
    df['Volume'] = transPercentile(df['Volume'], stdLow, stdHigh)

    df['WL'] = 100*(df['Close'].shift(1) - df['Close'])/df['Close']
    df['WL5'] = 100*(df['Close'].shift(5) - df['Close'])/df['Close']

    df = high_lowCandle(df)
    df['highCandle'] = transPercentile(df['highCandle'], stdLow, stdHigh)
    df['lowCandle'] = transPercentile(df['lowCandle'], stdLow, stdHigh)
                                                           
    df['MA5'] = ta.trend.SMAIndicator(df['Close'], 5).sma_indicator()            
    df['cMA5'] = 100*(df['Close'] - df['MA5'])/df['MA5']        
    df['cMA5'] = transPercentile(df['cMA5'], stdLow, stdHigh)

    df = ta.add_volume_ta(df, "High", "Low", "Close", "Volume")

    df = ta.add_trend_ta(df, "High", "Low", "Close", "Volume")
    df['ichimoku_conv'] = 100*(df['trend_ichimoku_conv'] - df['Close'])/df['Close']
    df['ichimoku_a'] = 100*(df['trend_ichimoku_a'] - df['Close'])/df['Close']
    df['ichimoku_conv'] = transPercentile(df['ichimoku_conv'], stdLow, stdHigh)
    df['ichimoku_a'] = transPercentile(df['ichimoku_a'], stdLow, stdHigh)
                
    df = ta.add_momentum_ta(df, "High", "Low", "Close", "Volume")
 
    df['DAXWL'] = 100*(df['Schlusskurs'].shift(1) - df['Schlusskurs'])/df['Schlusskurs']
    df['DAXWL5'] = 100*(df['Schlusskurs'].shift(5) - df['Schlusskurs'])/df['Schlusskurs']
    df['WL2DAX'] = df['WL'] - df['DAXWL']
    df['WL2DAX5'] = df['WL5'] - df['DAXWL5']
    df['WL'] = transPercentile(df['WL'], stdLow, stdHigh)
    df['WL2DAX'] = transPercentile(df['WL2DAX'], stdLow, stdHigh)
    df['WL2DAX5'] = transPercentile(df['WL2DAX5'], stdLow, stdHigh)
    
    df = calcVola(df, 5, 'Vola')
    df['Vola'] = transPercentile(df['Vola'], stdLow, stdHigh)

    df = df[colHead + features]
    
    df[-bs-ts:].to_csv(path + symbl + '.csv', index=False)
                    
    return df



def fitScaler(df, features):
    
    sc = MinMaxScaler(feature_range=(0,1))
    xData = df[features].dropna().values
    
    return sc.fit(xData)




def getPredictions(model, df, sc, features, bs, ts):
    
    tradeDecision = {0:'NoTrade', 1:'Long', 2:'Short'}
    
    df = df[-bs-ts:]
    df = df.reset_index(drop=True)
    x = df[features].values
    x_unpacked = sc.transform(x)

    x_packed = np.array([])
    for i in range(ts+1, bs+ts+1):
        x_packed = np.append(x_packed, x_unpacked[i-ts:i])
    
    x_packed = x_packed.reshape(bs, ts, len(features))
        
    p = model.predict(x_packed, batch_size=bs)
    
    prediction = np.argmax(p[-1])
    
    yHat = tradeDecision[prediction]
    
    return yHat
    
    
    
    
    


basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
name = 'dax.csv'

df = pd.read_csv(path + name, sep=';')

inpath = basePath + 'Data/Aktienkurse/aktuell/'
outpath = basePath + 'Data/Aktienkurse/EndOfDayPredictions/'

modelPath = 'DAX/'
modelName = '/DAXAvgTar2020-12-31_bestMyAccuModel.hdf'

bs = 50
ts = 10

symbls = df['Symbl'].values.tolist()

colHead = ['date', 'Open', 'Close', 'High', 'Low']

features = ['WL', 'WL5', 'cMA5', 'WL2DAX', 'WL2DAX5', 'Vola', 
            'highCandle', 'lowCandle', 'ichimoku_conv', 'ichimoku_a', 
            'momentum_uo', 'volume_cmf'
            ]

model = load_model(basePath + 'model/' + modelPath + modelName)

#
# process DAX data
#
# read DAX History
dfDAX = pd.read_csv(inpath + 'DAX_WL.csv', parse_dates=['date'], infer_datetime_format=True)
del dfDAX['Volumen']

# get DAX current data from Yahoo Finance
dax = yf.Ticker('^GDAXI')
daxOpen = dax.info['regularMarketOpen']
daxHigh = dax.info['regularMarketDayHigh']
daxLow = dax.info['regularMarketDayLow']
daxClose = float(input("Enter daxClose:"))
# enter current manually (not available in Yahoo Finance)
print("daxClose is: ", daxClose)
todayDt = datetime.date.today()
todayDt = datetime.datetime.strptime(todayDt.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
daxData = [{'date':todayDt, 'Erster':daxOpen, 'Hoch':daxHigh, 'Tief': daxLow, 'Schlusskurs':daxClose}]

dfDAX = dfDAX.append(daxData)
dfDAX = dfDAX.reset_index(drop=True)

# generate symbols for Yahoo Finance
ysymbls = []
for symbl in symbls:
    ysymbl = symbl.split('.')[0] + '.DE'
    ysymbls.append(ysymbl)

tradeRules = []
for symbl in symbls:
    # read historical data    
    df = pd.read_csv(inpath + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.sort_values('date')
    # add current data from Yahoo Finance
    ticker = yf.Ticker(ysymbls[symbls.index(symbl)])
    Open = ticker.info['regularMarketOpen']
    High = ticker.info['regularMarketDayHigh']
    Low = ticker.info['regularMarketDayLow']
    Close = (ticker.info['bid'] + ticker.info['ask'])/2
    Volume = ticker.info['regularMarketVolume']
    todyData =[{'date':todayDt, 'Open':Open, 'High':High, 'Low':Low, 'Close':Close, 'Volume':Volume}]
    df = df.append(todyData)
    df = df.reset_index(drop=True)
    # merge DAX historical data
    df = df.merge(dfDAX, on='date')   
    # prepare features    
    df = prepFeaturesRaw(symbl, outpath, df, colHead, features, bs, ts)
    lastClose = df['Close'].iloc[-1]
    # fit scaler
    sc = fitScaler(df, features)
    # get predictions
    yHat = getPredictions(model, df, sc, features, bs, ts)
    print('TradeRule for %10s = %7s ' % (symbl, yHat))
    tradeRules.append([symbl, yHat])

pd.DataFrame(tradeRules, columns=['Symbol', 'TradeRule']).to_csv(outpath + 'tradRules_' + todayDt.strftime('%Y-%m-%d') + '.csv', index=False)

    
