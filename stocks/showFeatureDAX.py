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

from os import listdir
from os.path import isfile, join
import datetime
import sys
import math
import gc
import ta
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")





def calcWinLoss(pHat, y, yMedium, limit, nDays):
    
    winL = 0.
    winS = 0.
    medL = 0.
    medS = 0.
    lossL = 0.
    lossS = 0.
    nTradeL = 0
    nTradeS = 0
    n = len(y)
    for i in range(n-nDays, n):
        tradeDecision = np.argmax(pHat[i])
        if (tradeDecision == 1):
# ... it's a Long Trade   
            nTradeL += 1
            if (y[i] == 1):
                winL += limit
            elif (y[i] == 2):
                lossL += limit
            else:
                medL += yMedium[i]
        elif (tradeDecision == 2):
# ... it's a Short Trade   
            nTradeS += 1
            if (y[i] == 2):
                winS += limit
            elif (y[i] == 1):
                lossS += limit
            else:
                medS -= yMedium[i]
            
    return winL, medL, lossL, nTradeL, winS, medS, lossS, nTradeS





def calcTarget(df, shift, limit, tarName):
        
    df[tarName] = 0
    df['tarFac'] = 0.5
    df.loc[df['Vola'] > 0.5, 'tarFac'] = df['Vola']
    for i in range(1, shift+1):
        df['Win'+str(i)] = 100*(df['High'].shift(-i) - df['Close'])/(df['Close']*df['tarFac']*df['tarFac'])
        df['Loss'+str(i)] = 100*(df['Low'].shift(-i) - df['Close'])/(df['Close']*df['tarFac']*df['tarFac'])


        df.loc[(df['Win'+str(i)] > limit) & 
               (df['Win'+str(i)].abs() > df['Loss'+str(i)].abs()) &
               (df[tarName] == 0), tarName] = 1

        df.loc[(df['Loss'+str(i)] < (-1)*limit) & 
               (df['Loss'+str(i)].abs() > df['Win'+str(i)].abs()) &
               (df[tarName] == 0), tarName] = 2
        
        
    return df    


def calcVola(df, shift, volaName):
        
    df[volaName] = 0.0

    for i in range(1, shift+1):
        df['Win'] = 100*(df['High'].shift(i) - df['Close'].shift(i-1))/df['Close'].shift(i-1)
        df['Loss'] = 100*(df['Low'].shift(i) - df['Close'].shift(i-1))/df['Close'].shift(i-1)
        df[volaName] += df['Win'].abs() + df['Loss'].abs()
        
    df[volaName] = np.log(df[volaName]/shift)
    return df    



def calcTargetLen(df, shift):
    
    df['TarLong'] = 0
    df['TarShort'] = 0
    df.loc[df['Target'] == 1, 'TarLong'] = 1
    df.loc[df['Target'] == 2, 'TarShort'] = 1
            
    df['TarLong'] = df['TarLong'].rolling(shift).sum()    
    df['TarShort'] = df['TarShort'].rolling(shift).sum()    
    
    return df




def calcTargetMedium(df, shift, tarName):
    
    df[tarName] = 100*(df['Close'].shift(-shift) - df['Open'].shift(-1))/df['Open'].shift(-1)
    
    return df  




def high_lowCandle(df):
    
    df['highCandle'] = (df['High'] - df['Close'])
    df.loc[(df['Close'] - df['Open'] < 0), 'highCandle'] = (df['Open'] - df['High'])/df['Open']
    
    df['lowCandle'] = (df['Open'] - df['Low'])
    df.loc[(df['Close'] - df['Open'] < 0), 'highCandle'] = (df['High'] - df['Close'])/df['Close']
    
    return df
    


def maxHL(df, period):
    
    df['HL'+str(period)] = -999999.99
    for i in range(1, period+1):
        df['HLtemp'] = (df['High'] - df['Low'].shift(i))/df['Low'].shift(i)
        df['LHtemp'] = (df['Low'] - df['High'].shift(i)/df['High'].shift(i))
        df['maxHL'] = df[['HLtemp', 'LHtemp']].max(axis=1)
        
        df.loc[df['maxHL'] > df['HL'+str(period)], 'HL'+str(period)] = df['maxHL']
        
    return df
        



def transPercentile(df, low, high):
    
    df.loc[(df > df.quantile(high))] = df.quantile(high)
    df.loc[(df < df.quantile(low))] = df.quantile(low)
    
    return df




def prepData(path, symbl, features, bs, ts, ff, toDt, numB, limit, nDays):
    
    
    scaleCols = ['Open', 'High', 'Low', 'Close', 'Volume']
    initCols = ['date', 'Target'] + scaleCols
    headerCols = ['date', 'Target', 'Open', 'High', 'Low', 'Close']

    stdLow = 0.01
    stdHigh = 0.99

    df = pd.read_csv(path + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.sort_values('date')
    df = df.reset_index(drop=True)

    df = calcVola(df, 5, 'Vola')
    df = calcTarget(df, 5, 2.0, 'Target')
    
    df = calcTargetLen(df, 10)
    
    df['Current'] = df['Open'].shift(-1)

    df['Volume'] = transPercentile(df['Volume'], stdLow, stdHigh)

    df['WL'] = (df['Close'] - df['Open'])/df['Open']
    df['WL5'] = 100*(df['Close'].shift(5) - df['Close'])/df['Close']
    df['WL10'] = 100*(df['Close'].shift(10) - df['Close'])/df['Close']
    df['WL20'] = 100*(df['Close'].shift(20) - df['Close'])/df['Close']
    df['WL60'] = 100*(df['Close'].shift(60) - df['Close'])/df['Close']

    df['HL'] = (df['High'] - df['Low'])/df['Low']
    df['HL'] = transPercentile(df['HL'], stdLow, stdHigh)
    df['coGap'] = (df['Close'] - df['Open'].shift(-1))/df['Open'].shift(-1)
    df['coGap'] = transPercentile(df['coGap'], stdLow, stdHigh)
            
    df = high_lowCandle(df)
    df['highCandle'] = transPercentile(df['highCandle'], stdLow, stdHigh)
    df['lowCandle'] = transPercentile(df['lowCandle'], stdLow, stdHigh)
                           
    df['MA5'] = ta.trend.SMAIndicator(df['Close'], 5).sma_indicator()            
    df['MA10'] = ta.trend.SMAIndicator(df['Close'], 10).sma_indicator()
    df['MA20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
    df['MA60'] = ta.trend.SMAIndicator(df['Close'], 60).sma_indicator()
    
    df['cMA5'] = 100*(df['Close'] - df['MA5'])/df['MA5']        
    df['cMA10'] = 100*(df['Close'] - df['MA10'])/df['MA10']        
    df['cMA20'] = 100*(df['Close'] - df['MA20'])/df['MA20']        
    df['cMA60'] = 100*(df['Close'] - df['MA60'])/df['MA60']
    
    df['MA5_10'] = 100*(df['MA5'] - df['MA10'])/df['MA10']
    df['MA5_20'] = 100*(df['MA5'] - df['MA20'])/df['MA20']
    df['MA5_60'] = 100*(df['MA5'] - df['MA60'])/df['MA60']
    df['MA10_20'] = 100*(df['MA10'] - df['MA20'])/df['MA20']
    df['MA10_60'] = 100*(df['MA10'] - df['MA60'])/df['MA60']
    df['MA20_60'] = 100*(df['MA20'] - df['MA60'])/df['MA60']
    
    df['cMA5'] = transPercentile(df['cMA5'], stdLow, stdHigh)
    df['cMA10'] = transPercentile(df['cMA10'], stdLow, stdHigh)
    df['cMA20'] = transPercentile(df['cMA20'], stdLow, stdHigh)
    df['cMA60'] = transPercentile(df['cMA60'], stdLow, stdHigh)
    df['MA5_10'] = transPercentile(df['MA5_10'], stdLow, stdHigh)
    df['MA5_20'] = transPercentile(df['MA5_20'], stdLow, stdHigh)
    df['MA5_60'] = transPercentile(df['MA5_60'], stdLow, stdHigh)
    df['MA10_20'] = transPercentile(df['MA10_20'], stdLow, stdHigh)
    df['MA10_60'] = transPercentile(df['MA10_60'], stdLow, stdHigh)
    df['MA20_60'] = transPercentile(df['MA20_60'], stdLow, stdHigh)
    
    df = ta.add_volume_ta(df, "High", "Low", "Close", "Volume")

    df = ta.add_trend_ta(df, "High", "Low", "Close", "Volume")
    df['ichimoku_conv'] = (df['trend_ichimoku_conv'] - df['Close'])/df['Close']
    df['ichimoku_conv'] = transPercentile(df['ichimoku_conv'], stdLow, stdHigh)
    df['ichimoku_base'] = (df['trend_ichimoku_base'] - df['trend_ichimoku_conv'])/df['trend_ichimoku_conv']
    df['ichimoku_base'] = transPercentile(df['ichimoku_base'], stdLow, stdHigh)

    df['ichimoku_a'] = (df['trend_ichimoku_a'] - df['Close'])/df['Close']
    df['ichimoku_a'] = transPercentile(df['ichimoku_a'], stdLow, stdHigh)
    df['ichimoku_b'] = (df['trend_ichimoku_b'] - df['trend_ichimoku_a'])/df['trend_ichimoku_a']
    df['ichimoku_b'] = transPercentile(df['ichimoku_b'], stdLow, stdHigh)
    
    
    df = ta.add_momentum_ta(df, "High", "Low", "Close", "Volume")
    df = ta.add_volatility_ta(df, "High", "Low", "Close", "Volume")
    df = ta.add_others_ta(df, "Close")
 
    dfDAX = pd.read_csv(path + 'DAX_WL.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.merge(dfDAX, on='date')

    dfVDAX = pd.read_csv(path + 'VDAXNew.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.merge(dfVDAX, on='date')

    df['DAXWL'] = 100*(df['Schlusskurs'].shift(1) - df['Schlusskurs'])/df['Schlusskurs']
    df['DAXWL5'] = 100*(df['Schlusskurs'].shift(5) - df['Schlusskurs'])/df['Schlusskurs']
    df['DAXWL10'] = 100*(df['Schlusskurs'].shift(10) - df['Schlusskurs'])/df['Schlusskurs']
    df['DAXWL20'] = 100*(df['Schlusskurs'].shift(20) - df['Schlusskurs'])/df['Schlusskurs']
    df['DAXWL60'] = 100*(df['Schlusskurs'].shift(60) - df['Schlusskurs'])/df['Schlusskurs']

    df['WL2DAX'] = df['WL'] - df['DAXWL']
    df['WL2DAX5'] = df['WL5'] - df['DAXWL5']
    df['WL2DAX10'] = df['WL10'] - df['DAXWL10']
    df['WL2DAX20'] = df['WL20'] - df['DAXWL20']
    df['WL2DAX60'] = df['WL60'] - df['DAXWL60']

    df['WL'] = transPercentile(df['WL'], stdLow, stdHigh)
    df['WL5'] = transPercentile(df['WL5'], stdLow, stdHigh)
    df['WL10'] = transPercentile(df['WL10'], stdLow, stdHigh)
    df['WL20'] = transPercentile(df['WL20'], stdLow, stdHigh)
    df['WL60'] = transPercentile(df['WL60'], stdLow, stdHigh)

    df['DAXWL'] = transPercentile(df['DAXWL'], stdLow, stdHigh)
    df['DAXWL5'] = transPercentile(df['DAXWL5'], stdLow, stdHigh)
    df['DAXWL10'] = transPercentile(df['DAXWL10'], stdLow, stdHigh)
    df['DAXWL20'] = transPercentile(df['DAXWL20'], stdLow, stdHigh)
    df['DAXWL60'] = transPercentile(df['DAXWL60'], stdLow, stdHigh)
    
    df['WL2DAX'] = transPercentile(df['WL2DAX'], stdLow, stdHigh)
    df['WL2DAX5'] = transPercentile(df['WL2DAX5'], stdLow, stdHigh)
    df['WL2DAX10'] = transPercentile(df['WL2DAX10'], stdLow, stdHigh)
    df['WL2DAX20'] = transPercentile(df['WL2DAX20'], stdLow, stdHigh)
    df['WL2DAX60'] = transPercentile(df['WL2DAX60'], stdLow, stdHigh)
    
    df['DAXMA5'] = ta.trend.SMAIndicator(df['Schlusskurs'], 5).sma_indicator()            
    df['DAXMA10'] = ta.trend.SMAIndicator(df['Schlusskurs'], 10).sma_indicator()
    df['DAXMA20'] = ta.trend.SMAIndicator(df['Schlusskurs'], 20).sma_indicator()
    
    df['MA5_DAX'] =100*(df['MA5'] - df['DAXMA5'])/df['DAXMA5']
    df['MA10_DAX'] =100*(df['MA10'] - df['DAXMA10'])/df['DAXMA10']
    df['MA20_DAX'] =100*(df['MA20'] - df['DAXMA20'])/df['DAXMA20']

    df['MA5_DAX'] = transPercentile(df['MA5_DAX'], stdLow, stdHigh)
    df['MA10_DAX'] = transPercentile(df['MA10_DAX'], stdLow, stdHigh)
    df['MA20_DAX'] = transPercentile(df['MA20_DAX'], stdLow, stdHigh)

    df = df[headerCols + features].dropna()
    df = df.reset_index(drop=True)
    
    x_data = df[features].values
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)
            
    df.loc[:,6:6+len(features)] = x_data

    df.to_csv(path + 'test/' + symbl + '.csv', index=False)


###### START #############################################
############### Update this section - all ################
##########################################################

basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
indexName = 'dax.csv'

df = pd.read_csv(path + indexName, sep=';')

symbls = df['Symbl'].values.tolist() 

numB = 1

bs = 50
ts = 10
ff = 19

limit = 5

nDays = 20

nameModel = 'DAXLong'

##########################################################
############### Update this section - all ################
###### END ###############################################

features = ['Volume', 'WL', 'HL', 'coGap', 'momentum_rsi', 'cMA5', 
            'MA5_10', 'MA10_20', 'WL2DAX', 'WL2DAX5', 'WL2DAX10', 
            'WL2DAX20', 'Vola', 'TarLong', 'TarShort'
            ]




toDate = '2008-01-31'
    
toDt = datetime.datetime.strptime(toDate, '%Y-%m-%d')
path = basePath + 'Data/Aktienkurse/aktuell/'
for symbl in symbls:
    prepData(path, symbl, features, bs, ts, ff, toDt, numB, limit, nDays)
    
            


