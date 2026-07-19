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

    df['HL'] = 100*(df['High'] - df['Low'])/df['Low']
    df['HL'] = transPercentile(df['HL'], stdLow, stdHigh)

    df['coGap'] = 100*(df['Open'].shift(-1) - df['Close'])/df['Close']
    df['coGap'] = transPercentile(df['coGap'], stdLow, stdHigh)

    df = high_lowCandle(df)
    df['highCandle'] = transPercentile(df['highCandle'], stdLow, stdHigh)
    df['lowCandle'] = transPercentile(df['lowCandle'], stdLow, stdHigh)
                                                           
    df['MA5'] = ta.trend.SMAIndicator(df['Close'], 5).sma_indicator()            
    df['MA20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()    
    df['cMA5'] = 100*(df['Close'] - df['MA5'])/df['MA5']        
    df['cMA20'] = 100*(df['Close'] - df['MA20'])/df['MA20']                
    df['cMA5'] = transPercentile(df['cMA5'], stdLow, stdHigh)
    df['cMA20'] = transPercentile(df['cMA20'], stdLow, stdHigh)

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
    
#    df.to_csv(path + symbl + '.csv', index=False)
                    
    return df



def fitScaler(df, features):
    
    sc = MinMaxScaler(feature_range=(0,1))
    xData = df[features].dropna().values
    
    return sc.fit(xData)




def getPredictions(model, df, sc, features, coGaps, bs, ts):
    
    tradeDecision = {0:'NoTrade', 1:'Long', 2:'Short'}
    yHat = []
    
    for coGap in coGaps:
        gap = coGap
        minGap = df['coGap'].min()
        maxGap = df['coGap'].max()
        if (gap < minGap):
            gap = minGap
        if (gap > maxGap):
            gap = maxGap
            
        df['coGap'].iloc[-1] = gap
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
        
        yHat.append(tradeDecision[prediction])
    
    return yHat
    
    
    
    
    


basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
name = 'dax.csv'

df = pd.read_csv(path + name, sep=';')

inpath = basePath + 'Data/Aktienkurse/aktuell/'
outpath = basePath + 'Data/Aktienkurse/EndOfDayPredictions/'

modelPath = 'DAX/'
modelName = 'DAX_2020-12-31_bestMyAccuModel.hdf'

bs = 50
ts = 10

symbls = df['Symbl'].values.tolist()

colHead = ['date', 'Open', 'Close', 'High', 'Low']
features = ['WL', 'HL', 'coGap', 'cMA5', 'cMA20', 'WL2DAX', 'WL2DAX5', 
            'Vola', 'highCandle', 'lowCandle', 'ichimoku_conv', 
            'ichimoku_a', 'momentum_rsi', 'momentum_uo', 'volume_cmf'
            ]

coGaps = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0,
          0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0    ]

outCols = ['Symbol', '-2.0', '-1.75', '-1.5', '-1.25', '-1.0', '-0.75', 
           '-0.5', '-0.25', '0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0']

model = load_model(basePath + 'model/' + modelPath + modelName)

allDecisions = []
allNextOpen = []
tradeTable = []

for symbl in symbls:
    print('... processing ' + symbl)
    df = pd.read_csv(inpath + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
        
    dfIndex = pd.read_csv(inpath + 'DAX_WL.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.merge(dfIndex, on='date')
    df = prepFeaturesRaw(symbl, outpath, df, colHead, features, bs, ts)
    lastClose = df['Close'].iloc[-1]
    
    sc = fitScaler(df, features)
    tradeDecision = getPredictions(model, df, sc, features, coGaps, bs, ts)
    tradeDecision = [symbl] + tradeDecision
    allDecisions.append(tradeDecision)
    
    nextOpen = [lastClose + lastClose*x/100 for x in coGaps]
    nextOpen = [symbl] + nextOpen
    allNextOpen.append(nextOpen)
    
    trTable = [symbl]
    for i in range(1, len(coGaps)+1):
        if (tradeDecision[i] == 'Long'):
            trTable.append(str(round(nextOpen[i], 2)))
        elif (tradeDecision[i] == 'Short'):
            trTable.append(str(round(-nextOpen[i], 2)))
        else:
            trTable.append(str(0))
            
    tradeTable.append(trTable)

    
outDF0 = pd.DataFrame(allDecisions, columns=outCols)
outDF0.to_csv(outpath + 'TradeDecisions.csv', index=False)

outDF1 = pd.DataFrame(allNextOpen, columns=outCols)
outDF1.to_csv(outpath + 'NextOpen.csv', index=False)

outDF2 = pd.DataFrame(tradeTable, columns=outCols)
outDF2.to_csv(outpath + 'TradeTable.csv', index=False)
