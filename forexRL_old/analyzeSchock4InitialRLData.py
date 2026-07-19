#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
    
    
feature_list = ['Position', 'index', 'close_scaled', 'WL', 
                'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                ]

target = ['Reward-Hold', 'Reward-Buy', 'Reward-Sell']

    
ts = 60
bs = 192

path = '/Users/uwe.muller/Hope/Data/rl/'
dataName = 'eurusd-Data'

df = pd.read_csv(path + dataName + '-ABT.csv', parse_dates=['Time'], infer_datetime_format=True)
print(len(df)/(ts+bs))
df = df.loc[df['Time'] >= '2010-03-04 00:00:00']
print(len(df)/(ts+bs))

WLmax = round(df['WL'].max(), 2)
WLmin = round(df['WL'].min(), 2)
WL05 = round(df['WL'].quantile(q=0.05), 2)
WL10 = round(df['WL'].quantile(q=0.10), 2)
WL25 = round(df['WL'].quantile(q=0.25), 2)
WL50 = round(df['WL'].quantile(q=0.50), 2)
WL75 = round(df['WL'].quantile(q=0.75), 2)
WL90 = round(df['WL'].quantile(q=0.90), 2)
WL95 = round(df['WL'].quantile(q=0.95), 2)
print(WLmin, WL05, WL10, WL25, WL50, WL75, WL90, WL95, WLmax)
df95 = df[df['WL'] >= WL95]['Time']
print(df95)

