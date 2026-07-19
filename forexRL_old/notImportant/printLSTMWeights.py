#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import gym
from collections import deque
import datetime
import time
from keras.models import load_model



feature_list = ['Position', 'index', 'close_scaled', 'WL', 
                'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                ]

target = ['Reward-Hold', 'Reward-Buy', 'Reward-Sell']

ts = 60
bs = 192
numLSTM1 = 120
nEpochs = 10
lr = 0.000284

#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'
currency = 'eurusd'

model = load_model(basepath + 'model/saveModel/eurusdModel-24ff-120-192-60_202105')
weights = model.get_weights()

print(len(weights))
for ww in weights:
    if isinstance(ww, np.ndarray):
        print('1st-level-len', len(ww))
    else:
        print('1st-level-w',ww)
    for w in ww:
        if isinstance(w, np.ndarray):
            print('2nd-level-len', len(w))                
        else:
            print('2nd-level-w',w)


