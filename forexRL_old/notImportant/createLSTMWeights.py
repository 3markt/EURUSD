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

baseModel = str(len(feature_list)) + 'ff-' + str(numLSTM1) + '-' + str(bs) + '-' + str(ts)
dataName = currency + 'Data-' + str(len(feature_list)) + 'ff-' + str(bs) + '-' + str(ts)

weightName = currency + 'Weights-' + baseModel
modelName = currency + 'Model-' + baseModel

path = basepath + 'data/forex/'
apath = basepath + 'data/abt/'
mpath = basepath + 'model/' + currency + '-' + baseModel + '/'
rpath = basepath + 'results/'

model = load_model(basepath + 'model/save/' + modelName)
model.save_weights(mpath + weightName)



