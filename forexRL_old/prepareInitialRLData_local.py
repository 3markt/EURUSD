#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np




def packRLData(df, bs, ts, feature_list, target):
    
    
#    all_data = np.array([])
    all_data = []
        
    if (len(df) > bs + ts):
        x_data = df[feature_list].values
        y_data = df[target].values
        dt_data = df['Time'].values.astype(str)
        close_data = df['Close'].values
        
        n = len(df)
        n_bs = n//(bs+ts)
        
        print(n, n/(bs+ts), n_bs)
        nbTrain = 0
        for b in range(n_bs):
            if (b%100 == 0):
                print('%4i-te Iteration' % (b))
            x = np.array([])
            y = np.array([])    
            dtd = np.array([])
            close = np.array([])
            nbTrain += 1
            for i in range(ts+b*(bs+ts), (b+1)*(bs+ts)):
                x = np.append(x, x_data[i-ts:i])
                y = np.append(y, y_data[i-1])
                dtd = np.append(dtd, dt_data[i-1])
                close = np.append(close, close_data[i-1])
            x = x.reshape(bs, ts, len(feature_list))
            y = y.reshape(bs, 3)
            dtd = dtd.reshape(bs, 1)
            close = close.reshape(bs, 1)
            all_data.append((b, dtd, close, x, y))
                    
        return all_data
    else:
        raise Exception('Not enough training data available')
        
    
    
feature_list = ['Position', 'index', 'close_scaled', 'WL', 
                'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                ]

target = ['Reward-Hold', 'Reward-Buy', 'Reward-Sell']

    
ts = 60
bs = 192

path = '/home/chitlom/data/rl/'
dataName = 'eurusd-Data'

df = pd.read_csv(path + dataName + '-ABT.csv', parse_dates=['Time'], infer_datetime_format=True)
print(len(df)/(ts+bs))
df = df.loc[df['Time'] >= '2020-03-30 00:00:00']
print(len(df)/(ts+bs))
all_data = packRLData(df, bs, ts, feature_list, target)
np.save(path + dataName + '-All-Test.npy', all_data)
