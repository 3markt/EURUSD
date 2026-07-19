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



"""
def calcTarget(df, limit):
    
    df['Today'] = (df['Low'] + df['High'] + df['Open'].shift(-1))/3
    df['Tomorrow'] = (df['Low'].shift(-1) + df['High'].shift(-1) + df['Close'].shift(-1))/3
    
    df['WLTarget'] = 100*(df['Tomorrow'] - df['Today'])/df['Today'] 
    
    df['Target'] = 1
    df.loc[df['WLTarget'] > limit, 'Target'] = 0
    df.loc[df['WLTarget'] < (-1)*limit, 'Target'] = 2
            
    return df


def calcTarget(df, limit):
    
    df['Today'] = (df['Low']+df['High']+df['Close']+df['Open'])/4
    df['Tomorrow'] = (df['Low'].shift(-1) + df['High'].shift(-1) + df['Close'].shift(-1) + df['Open'].shift(-1))/4
    
    df['WLTarget'] = 100*(df['Tomorrow'] - df['Today'])/df['Today'] 
    
    df['Target'] = 0
    df.loc[df['WLTarget'] > limit, 'Target'] = 1
    df.loc[df['WLTarget'] < (-1)*limit, 'Target'] = 2
            
    return df

"""

def calcTargetLong(df, limit, name):

    df[name] = 0
    df.loc[100*(df['High'].shift(-1) - df['Open'].shift(-1))/df['Open'].shift(-1) > limit, name] = 1
            
    return df


def calcTargetShort(df, limit, name):
    
    df[name] = 0
    df.loc[100*(df['Low'].shift(-1) - df['Open'].shift(-1))/df['Open'].shift(-1) < limit, name] = 1
            
    return df





def high_lowCandle(df):
    
    df['highCandle'] = 100*(df['High'] - df['Close'])/df['Close']    
    df['lowCandle'] = 100*(df['Open'] - df['Low'])/df['Open']
    df.loc[(df['Close'] - df['Open'] < 0), 'highCandle'] = 100*(df['Low'] - df['Close'])/df['Close']
    df.loc[(df['Close'] - df['Open'] < 0), 'lowCandle'] = 100*(df['Open'] - df['High'])/df['Open']
    
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





def prepFeatures(path, df, colHead, features, bs, ts):
    
    stdLow = 0.01
    stdHigh = 0.99
    
    df['Volume'] = transPercentile(df['Volume'], stdLow, stdHigh)

    df['Current'] = df['Close']
    df['WL'] = 100*(df['Current'].shift(1) - df['Current'])/df['Current']
    df['WL'] = transPercentile(df['WL'], stdLow, stdHigh)
    df['WL5'] = 100*(df['Current'].shift(5) - df['Current'])/df['Current']
    df['WL5'] = transPercentile(df['WL5'], stdLow, stdHigh)

    df['HL'] = 100*(df['High'] - df['Low'])/df['Low']
    df['HL'] = transPercentile(df['HL'], stdLow, stdHigh)

    df['coGap'] = 100*(df['Open'].shift(-1) - df['Close'])/df['Close']
    df['coGap'] = transPercentile(df['coGap'], stdLow, stdHigh)

    df = high_lowCandle(df)
    df['highCandle'] = transPercentile(df['highCandle'], stdLow, stdHigh)
    df['lowCandle'] = transPercentile(df['lowCandle'], stdLow, stdHigh)
                                                           
    df['MA5'] = ta.trend.SMAIndicator(df['Current'], 5).sma_indicator()            
    df['cMA5'] = 100*(df['Current'] - df['MA5'])/df['MA5']        
    df['cMA5'] = transPercentile(df['cMA5'], stdLow, stdHigh)

    df = ta.add_volume_ta(df, "High", "Low", "Close", "Volume")

    df = ta.add_trend_ta(df, "High", "Low", "Close", "Volume")
    df['ichimoku_conv'] = 100*(df['trend_ichimoku_conv'] - df['Current'])/df['Current']
    df['ichimoku_a'] = 100*(df['trend_ichimoku_a'] - df['Current'])/df['Current']
    df['ichimoku_conv'] = transPercentile(df['ichimoku_conv'], stdLow, stdHigh)
    df['ichimoku_a'] = transPercentile(df['ichimoku_a'], stdLow, stdHigh)
                
    df = ta.add_momentum_ta(df, "High", "Low", "Close", "Volume")
 
    df = ta.add_volatility_ta(df, "High", "Low", "Close", "Volume")
    df['bbh'] = 100*(df['volatility_bbh'] - df['Current'])/df['Current']
    df['bbh'] = transPercentile(df['bbh'], stdLow, stdHigh)
    df['bbl'] = 100*(df['volatility_bbl'] - df['Current'])/df['Current']
    df['bbl'] = transPercentile(df['bbl'], stdLow, stdHigh)

    df['WLco'] = 100*(df['Current'] - df['Open'])/df['Open']
    df['WL5co'] = 100*(df['Current'].shift(5) - df['Current'])/df['Current']
    df['DAXWL'] = 100*(df['Schlusskurs'].shift(1) - df['Schlusskurs'])/df['Schlusskurs']
    df['DAXWL5'] = 100*(df['Schlusskurs'].shift(5) - df['Schlusskurs'])/df['Schlusskurs']
    df['WL2DAX'] = df['WLco'] - df['DAXWL']
    df['WL2DAX5'] = df['WL5co'] - df['DAXWL5']
    df['WL2DAX'] = transPercentile(df['WL2DAX'], stdLow, stdHigh)
    df['WL2DAX5'] = transPercentile(df['WL2DAX5'], stdLow, stdHigh)
    
    df = calcVola(df, 5, 'Vola')
    df['Vola'] = transPercentile(df['Vola'], stdLow, stdHigh)
    
    df = df[colHead + features].dropna()
    n = len(df)
    n_bs = n//bs - 1
    if (n_bs + ts > n):
        n_bs -= 1
    df = df[-n_bs*bs-ts:]
    df = df.reset_index(drop=True)
                
    return df




basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
name = 'daxTrain.csv'

df = pd.read_csv(path + name, sep=';')

inpath = basePath + 'Data/Aktienkurse/aktuell/'
outpath = basePath + 'Data/Aktienkurse/abt/'

bs = 50
ts = 10

symbls = df['Symbl'].values.tolist()
startDate = df['StartDate'].values.tolist() 
endDate = df['EndDate'].values.tolist() 
    
startDate = [datetime.datetime.strptime(x, '%Y-%m-%d')+relativedelta(months=-6) for x in startDate]
endDate = [datetime.datetime.strptime(x, '%Y-%m-%d')+relativedelta(months=-3) for x in endDate]

colHead = ['date', 'Open', 'Close', 'High', 'Low', 'TargetLong', 'TargetLongTPR', 'TargetShort', 'TargetShortTPR']

features = ['WL', 'WL5', 'HL', 'coGap', 'cMA5', 'WL2DAX', 'WL2DAX5', 'Vola', 
            'highCandle', 'lowCandle', 'ichimoku_conv', 'ichimoku_a', 
            'momentum_rsi', 'bbh', 'bbl'
            ]

longTarget = 0
shortTarget = 0
bothTarget = 0
N = 0

for symbl in symbls:
    df = pd.read_csv(inpath + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.sort_values('date')
    df = df.loc[(df['date'] > startDate[symbls.index(symbl)]) & (df['date'] < endDate[symbls.index(symbl)])]
    df = df.reset_index(drop=True)
    
    df = calcTargetLong(df, 1.0, 'TargetLong')
    df = calcTargetLong(df, 2.0, 'TargetLongTPR')
    df = calcTargetShort(df, -1.0, 'TargetShort')
    df = calcTargetShort(df, -2.0, 'TargetShortTPR')
    n = len(df)
    longT = len(df.loc[df['TargetLong'] == 1])
    shortT = len(df.loc[df['TargetShort'] == 1])
    bothT = len(df.loc[(df['TargetLong'] == 1) & (df['TargetShort'] == 1)])
    
    dfIndex = pd.read_csv(inpath + 'DAX_WL.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.merge(dfIndex, on='date')
    df = prepFeatures(outpath, df, colHead, features, bs, ts)
    
    print('%8s n=%5i Long=%6.2f Short=%6.2f Both=%6.2f' % \
         (symbl, n, longT/n, shortT/n, bothT/n))
        
    longTarget += longT
    shortTarget += shortT
    bothTarget += bothT
    N += n
            
    if (len(df) >= bs+ts):
        df.to_csv(outpath + symbl + '.csv', index=False)

print('Total N=%5i Long=%6.2f Short=%6.2f Both=%6.2f' % (N, longTarget/N, shortTarget/N, bothTarget/N))


