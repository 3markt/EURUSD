#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:06:43 2021

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
import datetime as dt
from os import listdir
from os.path import isfile, join
import sys
import matplotlib.pyplot as plt
import ta





def prepABT(df):
    

    df['WL'] = 10000*(df['close'] - df['close'].shift(1))/df['close']
    
    BB = ta.volatility.BollingerBands(df['close'], 
                                      window = 10, 
                                      window_dev = 2)
    df['bb_h'] = (BB.bollinger_hband() - df['close'])/df['close']
    df['bb_l'] = (BB.bollinger_lband() - df['close'])/df['close']
    df['bb_m'] = (BB.bollinger_mavg() - df['close'])/df['close']
    
    MACD = ta.trend.MACD(df['close'],
                         window_slow = 26, 
                         window_fast = 12, 
                         window_sign = 9)
    df['macd'] = MACD.macd()
    df['macd_d'] = MACD.macd_diff()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], 
                                               df['low'], 
                                               df['close'], 
                                               window=14).average_true_range()
    
    KELTNER = ta.volatility.KeltnerChannel(df['high'], 
                                           df['low'], 
                                           df['close'],
                                           window=20, 
                                           window_atr=10)
    df['keltner_h'] = (KELTNER.keltner_channel_hband() - df['close'])/df['close']
    df['keltner_l'] = (KELTNER.keltner_channel_lband() - df['close'])/df['close']
    df['keltner_m'] = (KELTNER.keltner_channel_mband() - df['close'])/df['close']
    
    return df






fext = '-202105.csv'
from_dt = '2021-05-01 00:00:00'
basepath = '/Users/uwe.muller/Hope/'
path = basepath + 'data/forex/'

tsfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[:6] == 'EURUSD' \
           and f[7:14] >= '2021-04']
tsfiles.sort()
print(tsfiles)

abt5Min = pd.DataFrame()
abt15Min = pd.DataFrame()
abt1H = pd.DataFrame()
abt4H = pd.DataFrame()
abt1D = pd.DataFrame()
for f in tsfiles:
    if (f[7:14] in ['2019-10', '2019-11', '2019-12']):
        dfTS = pd.read_csv(path + f, compression='gzip', header=None)
    else:
        dfTS = pd.read_csv(path + f, compression='zip', header=None)
        
    dfTS.rename(columns={1: 'Time', 2: 'Bid', 3: 'Ask'}, inplace=True)
    dfTS.set_index(pd.DatetimeIndex(dfTS.Time), inplace=True)
    dfTS = dfTS.sort_index()
    dfTS['Kurs'] = (dfTS['Bid'] + dfTS['Ask'])/2
    dfTS = dfTS.dropna()
    
    ts5Min = dfTS['Kurs'].resample('5Min', label='right', closed='right').ohlc()
    ts5Min = ts5Min.dropna()
    abt5Min = abt5Min.append(ts5Min)

    ts15Min = dfTS['Kurs'].resample('15Min', label='right', closed='right').ohlc()
    ts15Min = ts15Min.dropna()
    abt15Min = abt15Min.append(ts15Min)

    ts1H = dfTS['Kurs'].resample('1H', label='right', closed='right').ohlc()
    ts1H = ts1H.dropna()
    abt1H = abt1H.append(ts1H)

    ts4H = dfTS['Kurs'].resample('4H', label='right', closed='right').ohlc()
    ts4H = ts4H.dropna()
    abt4H = abt4H.append(ts4H)

    print('--> ' + f + ' with lenDF %7i len5Min %4i len15Min %4i len1H %3i len4H %3i' % (len(dfTS), 
                                                                                         len(ts5Min),
                                                                                         len(ts15Min),
                                                                                         len(ts1H),
                                                                                         len(ts4H)))


abt5Min = abt5Min[~abt5Min.index.duplicated(keep='first')]
abt5Min = prepABT(abt5Min)
abt5Min = abt5Min.rename(columns={'open':'Open', 'high':'High', 'low':'Low','close':'Close', })
abt5Min = abt5Min.dropna()
abt5Min = abt5Min.loc[abt5Min.index >= from_dt]
abt5Min.to_csv(basepath + 'data/abt/eurusd5Min' + fext)

abt15Min = abt15Min[~abt15Min.index.duplicated(keep='first')]
abt15Min = prepABT(abt15Min)
abt15Min = abt15Min.rename(columns={'open':'Open', 'high':'High', 'low':'Low','close':'Close', })
abt15Min = abt15Min.dropna()
abt15Min = abt15Min.loc[abt15Min.index >= from_dt]
abt15Min.to_csv(basepath + 'data/abt/eurusd15Min' + fext)
        
abt1H = abt1H[~abt1H.index.duplicated(keep='first')]
abt1H = prepABT(abt1H)
abt1H = abt1H.rename(columns={'open':'Open', 'high':'High', 'low':'Low','close':'Close', })
abt1H = abt1H.dropna()
abt1H = abt1H.loc[abt1H.index >= from_dt]
abt1H.to_csv(basepath + 'data/abt/eurusd1H' + fext)

abt4H = abt4H[~abt4H.index.duplicated(keep='first')]
abt4H = prepABT(abt4H)
abt4H = abt4H.rename(columns={'open':'Open', 'high':'High', 'low':'Low','close':'Close', })
abt4H = abt4H.dropna()
abt4H = abt4H.loc[abt4H.index >= from_dt]
abt4H.to_csv(basepath + 'data/abt/eurusd4H' + fext)