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
import ta
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.models import Model, load_model, Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam
import random


def init_model(path):
    
    # define LSTM-Model
    model = Sequential()
    model.add(LSTM(units=120, 
                   recurrent_dropout=0.3,
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01),
                   kernel_initializer='truncated_normal',
                   batch_input_shape=(1, 60, 24)))
    
    model.add(Dense(units=3, 
                    kernel_initializer='truncated_normal'))    
    model.compile(loss='mean_squared_error',
                  optimizer="Adam",
                  metrics=['mae'])
    
    model.load_weights(path)
    
    return model


feature_list = ['Position', 'index', 'close_scaled', 'WL', 
                'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                ]

#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'

mpath = basepath + 'model/eurusd-24ff-120-192-60/eurusdWeights-24ff-120-192-60'
abt = pd.read_csv(basepath + 'data/forex/duka/aabt.csv')

abt = abt[feature_list]

model = init_model(mpath)

print(model.predict(abt.values.reshape(1, 60, 24)))
print(abt['Position'].iat[-1])
print()
abt['Position'] = 1.0
print(model.predict(abt.values.reshape(1, 60, 24)))
print(abt['Position'].iat[-1])
print()

abt['Position'] = 0.5
abt['Position'].iat[-1] = 1.0
print(model.predict(abt.values.reshape(1, 60, 24)))
print(abt['Position'].iat[-1])
print()

p = [0.0, 0.5, 1.0]
for i in range(59):
    abt['Position'].iat[i] = random.choice(p)
print(model.predict(abt.values.reshape(1, 60, 24)))
print(abt['Position'].iat[-1])


