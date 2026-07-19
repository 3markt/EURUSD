#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import gym
import random
from collections import deque
import datetime
import time
import ta
from enum import Enum
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam
import gym
from gym.envs.registration import register
from forexTrader.envs import forexEnv



def create_model(mpath):

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
                  optimizer=Adam(lr=0.005),
                  metrics=['mae'])
    model.load_weights(mpath)
    
    return model



#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'

apath = basepath + 'data/rl/'
mpath = basepath + 'model/eurusd-Inititial/'
all_data = np.load(apath + 'eurusd-Data-All.npy', allow_pickle=True)
print(all_data[0, 0], all_data[0, 1][0])
n = len(all_data)
ts = 60
bs = 192
start_b = 2814
print('%6i Batches' % (n))
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'forex-v0' in env:
        del gym.envs.registration.registry.env_specs[env]

register(id='forex-v0', entry_point='forexTrader.envs.forexEnv:ForexEnv')
env = gym.make(id='forex-v0', 
               data=all_data, 
               window_size=ts, 
               batch_size=bs, 
               n_batch=n,
               start_batch=start_b)

model = create_model(mpath + 'eurusdWeights-Inititial')
curr_state = env.reset()
total_reward = 0
for i in range((start_b + 1)*bs, (start_b + 11)*bs):
    action = np.argmax(model.predict(curr_state.reshape(1,60,24)))
    new_state, reward, done, info = env.step(action)
    if done:
        total_reward += info["total_reward"]
        print('batch-id = %4i datetime = %16s daily-reward=%6.2f total-reward=%6.2f' % \
              (info["batch_id"], info["datetime"], info["total_reward"], total_reward))
        curr_state = env.reset()
    else:
        curr_state = new_state
            
