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





def calcVola(df, shift, volaName):
        
    df[volaName] = 0.0

    for i in range(1, shift+1):
        df['Win'] = 100*(df['High'].shift(i) - df['Close'].shift(i-1))/df['Close'].shift(i-1)
        df['Loss'] = 100*(df['Low'].shift(i) - df['Close'].shift(i-1))/df['Close'].shift(i-1)
        df[volaName] += df['Win'].abs() + df['Loss'].abs()
        
    df[volaName] = df[volaName]/(2*shift)
    df['ln'+volaName] = np.log(df[volaName])

    return df    




def high_lowCandle(df):
    
    df['highCandle'] = (df['High'] - df['Close'])/df['Close']    
    df['lowCandle'] = (df['Open'] - df['Low'])/df['Open']
    df.loc[(df['Close'] - df['Open'] < 0), 'highCandle'] = (df['Low'] - df['Close'])/df['Close']
    df.loc[(df['Close'] - df['Open'] < 0), 'lowCandle'] = (df['Open'] - df['High'])/df['Open']
    
    return df
    



def transPercentile(df, low, high):
    
    df.loc[(df > df.quantile(high))] = df.quantile(high)
    df.loc[(df < df.quantile(low))] = df.quantile(low)
    
    return df


"""
def calcTarget(df, limit):
    
    df['Today'] = (df['Low']+df['High']+df['Close'])/3
    df['Tomorrow'] = (df['Low'].shift(-1) + df['High'].shift(-1) + df['Close'].shift(-1))/3
    
    df['WLTarget'] = 100*(df['Tomorrow'] - df['Today'])/df['Today'] 
    
    df['Target'] = 0
    df.loc[df['WLTarget'] > limit, 'Target'] = 1
    df.loc[df['WLTarget'] < (-1)*limit, 'Target'] = 2
            
    return df
"""    
    

def calcTargetLong(df, limit):
    
#    df['Today'] = (df['Low'] + df['High'] + df['Open'].shift(-1))/3
#    df['Tomorrow'] = (df['Low'].shift(-1) + df['High'].shift(-1) + df['Close'].shift(-1))/3

    df['TargetLong'] = 0
    df.loc[100*(df['High'].shift(-1) - df['Open'].shift(-1))/df['Open'].shift(-1) > limit, 'TargetLong'] = 1
            
    return df


def calcTargetShort(df, limit):
    
    df['TargetShort'] = 0
    df.loc[100*(df['Low'].shift(-1) - df['Open'].shift(-1))/df['Open'].shift(-1) < limit, 'TargetShort'] = 1
            
    return df
    

def prepData(path, symbls, features, bs, ts):
    
    allData = pd.DataFrame()

    stdLow = 0.01
    stdHigh = 0.99
    for symbl in symbls:
        df = pd.read_csv(path + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
        if (len(df) >= 200):

            df = df.sort_values('date')
            df = df.reset_index(drop=True)
            df = calcTargetLong(df, 2.0)
            df = calcTargetShort(df, -2.0)
            df = calcVola(df, 5, 'Vola')
            
            df['Volume'] = transPercentile(df['Volume'], stdLow, stdHigh)
            df['Current'] = df['Open'].shift(-1)
            df['WL'] = (df['Current'] - df['Open'])/df['Open']
            df['WL5'] = 100*(df['Current'].shift(5) - df['Current'])/df['Current']
            df['WL10'] = 100*(df['Current'].shift(10) - df['Current'])/df['Current']
            df['WL20'] = 100*(df['Current'].shift(20) - df['Current'])/df['Current']

            df['HL'] = (df['High'] - df['Low'])/df['Low']
            df['HL'] = transPercentile(df['HL'], stdLow, stdHigh)
            df['coGap'] = 100*(df['Open'].shift(-1) - df['Close'])/df['Close']
            df['coGap'] = transPercentile(df['coGap'], stdLow, stdHigh)
                    
            df = high_lowCandle(df)
            df['highCandle'] = transPercentile(df['highCandle'], stdLow, stdHigh)
            df['lowCandle'] = transPercentile(df['lowCandle'], stdLow, stdHigh)
                                   
            df['MA5'] = ta.trend.SMAIndicator(df['Current'], 5).sma_indicator()            
            df['MA10'] = ta.trend.SMAIndicator(df['Current'], 10).sma_indicator()
            df['MA20'] = ta.trend.SMAIndicator(df['Current'], 20).sma_indicator()
            
            df['cMA5'] = 100*(df['Current'] - df['MA5'])/df['MA5']        
            df['cMA10'] = 100*(df['Current'] - df['MA10'])/df['MA10']        
            df['cMA20'] = 100*(df['Current'] - df['MA20'])/df['MA20']        
            
            df['MA5_10'] = 100*(df['MA5'] - df['MA10'])/df['MA10']
            df['MA5_20'] = 100*(df['MA5'] - df['MA20'])/df['MA20']
            df['MA10_20'] = 100*(df['MA10'] - df['MA20'])/df['MA20']
            
            df['cMA5'] = transPercentile(df['cMA5'], stdLow, stdHigh)
            df['cMA10'] = transPercentile(df['cMA10'], stdLow, stdHigh)
            df['cMA20'] = transPercentile(df['cMA20'], stdLow, stdHigh)
            df['MA5_10'] = transPercentile(df['MA5_10'], stdLow, stdHigh)
            df['MA5_20'] = transPercentile(df['MA5_20'], stdLow, stdHigh)
            df['MA10_20'] = transPercentile(df['MA10_20'], stdLow, stdHigh)
            
            df = ta.add_volume_ta(df, "High", "Low", "Close", "Volume")

            df = ta.add_trend_ta(df, "High", "Low", "Close", "Volume")
            df['ichimoku_conv'] = (df['trend_ichimoku_conv'] - df['Current'])/df['Current']
            df['ichimoku_conv'] = transPercentile(df['ichimoku_conv'], stdLow, stdHigh)
            df['ichimoku_base'] = (df['trend_ichimoku_base'] - df['trend_ichimoku_conv'])/df['trend_ichimoku_conv']
            df['ichimoku_base'] = transPercentile(df['ichimoku_base'], stdLow, stdHigh)

            df['ichimoku_a'] = (df['trend_ichimoku_a'] - df['Current'])/df['Current']
            df['ichimoku_a'] = transPercentile(df['ichimoku_a'], stdLow, stdHigh)
            df['ichimoku_b'] = (df['trend_ichimoku_b'] - df['trend_ichimoku_a'])/df['trend_ichimoku_a']
            df['ichimoku_b'] = transPercentile(df['ichimoku_b'], stdLow, stdHigh)
            
            
            df = ta.add_momentum_ta(df, "High", "Low", "Close", "Volume")
            df = ta.add_volatility_ta(df, "High", "Low", "Close", "Volume")
            df = ta.add_others_ta(df, "Close")

            df['bbh'] = 100*(df['volatility_bbh'] - df['Current'])/df['Current']
            df['bbh'] = transPercentile(df['bbh'], stdLow, stdHigh)
            df['bbl'] = 100*(df['volatility_bbl'] - df['Current'])/df['Current']
            df['bbl'] = transPercentile(df['bbl'], stdLow, stdHigh)

            dfDAX = pd.read_csv(path + 'DAX_WL.csv', parse_dates=['date'], infer_datetime_format=True)
            df = df.merge(dfDAX, on='date')

            dfVDAX = pd.read_csv(path + 'VDAXNew.csv', parse_dates=['date'], infer_datetime_format=True)
            df = df.merge(dfVDAX, on='date')

            df['DAXWL'] = 100*(df['Schlusskurs'].shift(1) - df['Schlusskurs'])/df['Schlusskurs']
            df['DAXWL5'] = 100*(df['Schlusskurs'].shift(5) - df['Schlusskurs'])/df['Schlusskurs']
            df['DAXWL10'] = 100*(df['Schlusskurs'].shift(10) - df['Schlusskurs'])/df['Schlusskurs']
            df['DAXWL20'] = 100*(df['Schlusskurs'].shift(20) - df['Schlusskurs'])/df['Schlusskurs']

            df['WLco'] = 100*(df['Current'] - df['Open'])/df['Open']
            df['WL5co'] = 100*(df['Current'].shift(5) - df['Current'])/df['Current']
            df['WL10co'] = 100*(df['Current'].shift(10) - df['Current'])/df['Current']
            df['WL20co'] = 100*(df['Current'].shift(20) - df['Current'])/df['Current']
    
            df['WL2DAX'] = df['WLco'] - df['DAXWL']
            df['WL2DAX5'] = df['WL5co'] - df['DAXWL5']
            df['WL2DAX10'] = df['WL10co'] - df['DAXWL10']
            df['WL2DAX20'] = df['WL20co'] - df['DAXWL20']

            df['WL'] = transPercentile(df['WL'], stdLow, stdHigh)
            df['WL5'] = transPercentile(df['WL5'], stdLow, stdHigh)
            df['WL10'] = transPercentile(df['WL10'], stdLow, stdHigh)
            df['WL20'] = transPercentile(df['WL20'], stdLow, stdHigh)

            df['DAXWL'] = transPercentile(df['DAXWL'], stdLow, stdHigh)
            df['DAXWL5'] = transPercentile(df['DAXWL5'], stdLow, stdHigh)
            df['DAXWL10'] = transPercentile(df['DAXWL10'], stdLow, stdHigh)
            df['DAXWL20'] = transPercentile(df['DAXWL20'], stdLow, stdHigh)
            
            df['WL2DAX'] = transPercentile(df['WL2DAX'], stdLow, stdHigh)
            df['WL2DAX5'] = transPercentile(df['WL2DAX5'], stdLow, stdHigh)
            df['WL2DAX10'] = transPercentile(df['WL2DAX10'], stdLow, stdHigh)
            df['WL2DAX20'] = transPercentile(df['WL2DAX20'], stdLow, stdHigh)
            
            df['DAXMA5'] = ta.trend.SMAIndicator(df['Schlusskurs'], 5).sma_indicator()            
            df['DAXMA10'] = ta.trend.SMAIndicator(df['Schlusskurs'], 10).sma_indicator()
            df['DAXMA20'] = ta.trend.SMAIndicator(df['Schlusskurs'], 20).sma_indicator()
            
            df['MA5_DAX'] =100*(df['MA5'] - df['DAXMA5'])/df['DAXMA5']
            df['MA10_DAX'] =100*(df['MA10'] - df['DAXMA10'])/df['DAXMA10']
            df['MA20_DAX'] =100*(df['MA20'] - df['DAXMA20'])/df['DAXMA20']

            df['MA5_DAX'] = transPercentile(df['MA5_DAX'], stdLow, stdHigh)
            df['MA10_DAX'] = transPercentile(df['MA10_DAX'], stdLow, stdHigh)
            df['MA20_DAX'] = transPercentile(df['MA20_DAX'], stdLow, stdHigh)
            
            print(symbl, len(df))
            df.dropna(inplace=True)
            print(symbl, len(df))
                    
            allData = allData.append(df)
                    
    n = len(allData)
    longData = allData.loc[allData['TargetLong'] == 1]
    shortData = allData.loc[allData['TargetShort'] == 1]
    bothData = allData.loc[(allData['TargetShort'] == 1) & (allData['TargetLong'] == 1)]
    nullData = allData.loc[(allData['TargetShort'] == 0) & (allData['TargetLong'] == 0)]
    
    nL = len(longData)
    nS = len(shortData)
    nB = len(bothData)
    print('-------> n = %6i Target Long = %1.4f Short = %1.4f Both = %1.4f' % (n, nL/n, nS/n, nB/n))

    """
    longData = longData.append(nullData.sample(nL))    
    longData.to_csv(path + 'explore/longData.csv', index=False)

    shortData = shortData.append(nullData.sample(nS))    
    shortData.to_csv(path + 'explore/shortData.csv', index=False)
    """    

#    allData.to_csv(path + 'explore/shortData.csv', index=False)

    




###### START #############################################
############### Update this section - all ################
##########################################################

path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'
name = 'daxTrain.csv'

df = pd.read_csv(path + name, sep=';')

symbls = df['Symbl'].values.tolist() 

features = ['WL', 'WL5', 'cMA5', 'WL2DAX', 'WL2DAX5', 'Vola', 
            'highCandle', 'lowCandle', 'ichimoku_conv', 'ichimoku_a', 
            'momentum_uo', 'volume_cmf'
            ]

##########################################################
############### Update this section - all ################
###### END ###############################################
path = '/Users/uwe.muller/Hope/Data/Aktienkurse/aktuell/'
prepData(path, symbls, features, 50, 5)



