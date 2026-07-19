#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:17:28 2021

@author: uwe.mueller
"""


import pandas as pd
import numpy as np
import sys
from os import listdir
from os.path import isfile, join, isdir

import gym
from gym.envs.registration import register
from forexTrader.envs import forexEnv
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam

"""

def set_model(weight_path):
    
    bs = 1
    ts = 60
    ff = 24
    learning_rate = 0.0005
#    old_model = load_model(model_path)
#    weights = old_model.get_weights()
    
    model = Sequential()
    
    model.add(LSTM(units=72, 
                   recurrent_dropout=0.3,
                   return_sequences=True,
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01),
                   kernel_initializer='truncated_normal',
                   batch_input_shape=(bs, 
                                      ts, 
                                      ff)))
      
    model.add(LSTM(units=48, 
                   recurrent_dropout=0.2,
                   return_sequences=True,
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01),
                   kernel_initializer='truncated_normal',
                   batch_input_shape=(bs, 
                                      ts, 
                                      ff)))
    
    model.add(LSTM(units=24, 
                   recurrent_dropout=0.2,
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01),
                   kernel_initializer='truncated_normal',
                   batch_input_shape=(bs, 
                                      ts, 
                                      ff)))
            
    model.add(Dense(units=3, 
                    kernel_initializer='truncated_normal'))
        
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['mae'])
    
    model.load_weights(weight_path)
    
#    model.set_weights(weights)

    return model

"""    

def set_model(weight_path):
    
    bs = 1
    ts = 60
    ff = 24
    learning_rate = 0.0005

    model = Sequential()
    
    model.add(LSTM(units=120, 
                   recurrent_dropout=0.3,
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01),
                   kernel_initializer='truncated_normal',
                   batch_input_shape=(bs, 
                                      ts, 
                                      ff)))
      
    model.add(Dense(units=3, 
                    kernel_initializer='truncated_normal'))
        
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['mae'])
    
    model.load_weights(weight_path)
    
    return model



feature_list = ['Position', 'index', 'close_scaled', 'WL', 
                'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                ]


#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'

apath = basepath + 'data/rl/'
mpath = basepath + 'model/eurusd-Initial/eurusdModel-Initial/'
#wpath = basepath + 'model/eurusd-ongoing/Professional1/Professional1'
wpath = basepath + 'model/eurusd-Inititial/eurusdWeights-Inititial'
all_data = np.load(apath + 'eurusd-Data-All-Test.npy', allow_pickle=True)
n = len(all_data)
ts = 60
bs = 192
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
               start_batch=0,
               batch_replay=0)

print('observation_space', env.observation_space.shape)

model = set_model(wpath)

i = 0
avgReward = 0
out = []
while (i < n):
    total_reward = 0    
    state = env.reset()
    avg_action = 0
    while True:
        y = model.predict(state.reshape(1, 60, 24))
        action = np.argmax(y[-1])
        print(y.shape, action)
        avg_action += action
        assert action in [0, 1, 2]
        state, reward, done, info = env.step(action)
        out.append(info)
        
        if done:
            print('Batch %4i with Total-Reward = %5.2f avg-action = %3.2f' \
                  % (i, info["total_reward"], avg_action/bs))
            avgReward += info["total_reward"]
            if (i%100 == 0):
                pd.DataFrame(out).to_csv(apath + 'simulation_result.csv', index=False)
                
            break
    
    i += 1
    
    sys.stdout.flush()

avgReward /= n
print('Average Total-Reward = %6.3f' % (avgReward))
pd.DataFrame(out).to_csv(apath + 'simulation_result.csv', index=False)

