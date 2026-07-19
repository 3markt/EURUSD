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





def high_lowCandle(df):
    
    df['highCandle'] = (df['High'] - df['Close'])/df['Close']
    df.loc[(df['Close'] - df['Open'] < 0), 'highCandle'] = (df['Open'] - df['High'])/df['Open']
    
    df['lowCandle'] = (df['Open'] - df['Low'])/df['Open']
    df.loc[(df['Close'] - df['Open'] < 0), 'highCandle'] = (df['High'] - df['Close'])/df['Close']
    
    return df
    




def prepData(path, symbl, featuresLong, featuresShort, toDt, toDate, nDays):
    
    
    df = pd.read_csv(path + 'aktuell/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
    scaleCols = ['Open', 'High', 'Low', 'Close', 'Volume']
    supplCols = ['date']
    features = list(set(featuresLong + featuresShort))        
        
    df = df[supplCols + scaleCols]
    sc = MinMaxScaler(feature_range=(0,1))
    df.loc[:,1:6] = sc.fit_transform(df[scaleCols].values)

    df['Current'] = df['Open'].shift(-1)

    df['Volume'] = df['Volume'].rank()                        
    df['WL'] = df['Close'] - df['Open']
    df['WL'] = df['WL'].rank()                        
    df['WL5'] = df['Close'] - df['Open'].shift(5)
    df['WL5'] = df['WL5'].rank()
    df['HL'] = df['High'] - df['Low']
    df['HL'] = df['HL'].rank()
    df['HL5'] = (df['High'] - df['Low'].shift(5))/df['Low'].shift(5)
    df['HL5'] = df['HL5'].rank()
    df['GD10Diff'] = (df['Close'].rolling(10).mean() - df['Current'])/df['Current']
    df['GD10Diff'] = df['GD10Diff'].rank()
    df['GD60Diff'] = (df['Close'].rolling(60).mean() - df['Current'])/df['Current']
    df['GD60Diff'] = df['GD60Diff'].rank()
    df['coGap'] = 100*((df['Close'] - df['Open'].shift(-1))/df['Close'])
    df['coGap'] = df['coGap'].rank()
            
    df = high_lowCandle(df)
    df['highCandle'] = df['highCandle'].rank()
    df['lowCandle'] = df['lowCandle'].rank()
                           
                          
    df = ta.add_volume_ta(df, "High", "Low", "Close", "Volume")
    df['volume_obv'] = df['volume_obv'].rank()

    df = ta.add_trend_ta(df, "High", "Low", "Close", "Volume")
    df['trend_macd'] = df['trend_macd'].rank()
    df['trend_cci'] = df['trend_cci'].rank()
    df['ichi_convcurrent'] = (df['trend_ichimoku_conv'] - df['Current'])/df['Current']
    df['ichi_convcurrent'] = df['ichi_convcurrent'].rank()
    df['ichi_basecurrent'] = (df['trend_ichimoku_base'] - df['Current'])/df['Current']
    df['ichi_basecurrent'] = df['ichi_basecurrent'].rank()
    df['ichi_acurrent'] = (df['trend_ichimoku_a'] - df['Current'])/df['Current']
    df['ichi_acurrent'] = df['ichi_acurrent'].rank()
    df['ichi_bcurrent'] = (df['trend_ichimoku_b'] - df['Current'])/df['Current']
    df['ichi_bcurrent'] = df['ichi_bcurrent'].rank()
     
    df = ta.add_momentum_ta(df, "High", "Low", "Close", "Volume")
 
    df = ta.add_volatility_ta(df, "High", "Low", "Close", "Volume")
    df['volatility_bbw'] = df['volatility_bbw'].rank()

    df = ta.add_others_ta(df, "Close")
    df['others_dr'] = df['others_dr'].rank()
    
    dfIndex = pd.read_csv(path + 'aktuell/DAX_WL.csv', parse_dates=['date'], infer_datetime_format=True)
    df = df.merge(dfIndex, on='date')
    
    df = df[supplCols + features].dropna()
    df = df.reset_index(drop=True)
                    
    df = df[df['date'] > toDt]
    df = df.iloc[:25]
    df[supplCols + features].to_csv(path + 'aktuell/test/' + symbl + '_' + toDate + '.csv', index=False)



###### START #############################################
############### Update this section - all ################
##########################################################

basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
indexName = 'dax.csv'

df = pd.read_csv(path + indexName, sep=';')

symbls = df['Symbl'].values.tolist() 
print(symbls)
nDays = 20

toDates = ['2015-01-31', '2015-02-28', '2015-03-31', '2015-04-30', '2015-05-31', 
           '2015-06-30', '2015-07-31', '2015-08-31', '2015-09-30', '2015-10-31', 
           '2015-11-30', '2015-12-31', 
           '2016-01-31', '2016-02-29', '2016-03-31', '2016-04-30', '2016-05-31', 
           '2016-06-30', '2016-07-31', '2016-08-31', '2016-09-30', '2016-10-31', 
           '2016-11-30', '2016-12-31',
           '2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30', '2017-05-31', 
           '2017-06-30', '2017-07-31', '2017-08-31', '2017-09-30', '2017-10-31', 
           '2017-11-30', '2017-12-31', 
           '2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30', '2018-05-31', 
           '2018-06-30', '2018-07-31', '2018-08-31', '2018-09-30', '2018-10-31', 
           '2018-11-30', '2018-12-31',
           '2019-01-31', '2019-02-28', '2019-03-31', '2019-04-30', '2019-05-31', 
           '2019-06-30', '2019-07-31', '2019-08-31', '2019-09-30', '2019-10-31', 
           '2019-11-30', '2019-12-31', 
           '2020-01-31', '2020-02-28', 
           '2020-03-31', '2020-04-30', '2020-05-31', '2020-06-30']


##########################################################
############### Update this section - all ################
###### END ###############################################

featuresLong = ['Volume', 'WL', 'WL5', 'HL', 'HL5', 'GD10Diff', 'GD60Diff', 'coGap', 
                'highCandle', 'lowCandle', 'AllWL1', 'trend_macd', 'trend_cci', 
                'ichi_convcurrent', 'ichi_basecurrent', 'ichi_acurrent', 
                'ichi_bcurrent', 'momentum_rsi', 'volatility_bbw', 'others_dr'
                ]

featuresShort = ['Volume', 'WL', 'WL5', 'HL', 'HL5', 'GD10Diff', 'GD60Diff', 'coGap', 
                 'highCandle', 'lowCandle', 'AllWL1', 'trend_macd', 'trend_cci', 
                 'ichi_convcurrent', 'ichi_basecurrent', 'ichi_acurrent', 
                 'ichi_bcurrent', 'momentum_rsi', 'volatility_bbw', 'others_dr'
                 ]


toDate = '2017-08-31'
    
toDt = datetime.datetime.strptime(toDate, '%Y-%m-%d')
for symbl in symbls:
    prepData(path, symbl, featuresLong, featuresShort, toDt, toDate, nDays)
    
