#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import sys
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
from pickle import dump




def readABT(path):
    
    tsfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[:13] == 'eurusdRewards'] 
#               and f[16] == '6']
    tsfiles.sort()
    df = pd.DataFrame()
    for f in tsfiles:
        dfi = pd.read_csv(path + f, parse_dates=['Time'], infer_datetime_format=True)
        print(f, len(dfi))
        df = df.append(dfi)
    
    sys.stdout.flush()
    df.sort_values(by='Time', inplace=True)
#    df = df.loc[df['Time'] > '2015-03-01 00:00:00']
    df['minute'] = df['Time'].dt.minute.apply([lambda x: str(x).rjust(2, '0')])
    df['hour'] = df['Time'].dt.hour.apply([lambda x: str(x)])
    df['index'] = (df['hour'] + '.' + df['minute']).astype(float)
    df.reset_index(drop=True, inplace=True)
    
    return df


def train_and_saveScaler(path, df, features2scale):
    
    df['close_scaled'] = df['Close']
    scaler = StandardScaler()
    scaler.fit(df[features2scale])
    
    dump(scaler, open(path, 'wb'))



features2scale = ['index', 'close_scaled', 'WL', 
                  'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                  'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                  'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                  'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                  ]


#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'

path = basepath + 'data/abt/'

tsABT = readABT(path)
tsABT['Position'] = tsABT['Position'] * 0.5
spath = basepath + 'scaler/scaler-eurusd.pkl'
tsABT = train_and_saveScaler(spath, tsABT, features2scale)
