#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
from enum import Enum
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from os import listdir
from os.path import isfile, join, isdir
import sys


def readAndMergeABT(spath, path, feature_cols, reward_cols):
    
    reward_f = [f for f in listdir(spath) if isfile(join(spath, f)) and f[:16] == 'eurusdMaxRewards'] 
    feature_f = [f for f in listdir(path) if isfile(join(path, f)) and f[:14] == 'eurusdFeatures'] 


    reward_f.sort()
    feature_f.sort()
    i = 0
    for f in reward_f:
        df_result = pd.DataFrame()
        df_reward = pd.read_csv(spath + f, parse_dates=['Time'], infer_datetime_format=True)
        df_reward = df_reward[reward_cols]
        df_reward.sort_values(by='Time', inplace=True)
        print(f, len(df_reward))

        df_feature= pd.read_csv(path + feature_f[i], parse_dates=['Time'], infer_datetime_format=True)
        df_feature = df_feature[feature_cols]
        df_feature.sort_values(by='Time', inplace=True)
        print(feature_f[i], len(df_feature))
        
        df_result = pd.merge_ordered(df_reward, df_feature, how='inner', on='Time')
        print('Result-Length=', len(df_result))
        df_result.to_csv(path + f, index=False)
        i += 1

    



feature_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'WL', 
                'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                ]

reward_cols = ['Time', 'Reward-Hold', 'Reward-Buy', 'Reward-Sell', 'Position']


ts = 60
bs = 192
numLSTM1 = 110
nEpochs = 10
lr = 0.001

basepath = '/Users/uwe.muller/Hope/'

path = basepath + 'data/abt/'
spath = basepath + 'data/abt/save/'

readAndMergeABT(spath, path, feature_cols, reward_cols)


