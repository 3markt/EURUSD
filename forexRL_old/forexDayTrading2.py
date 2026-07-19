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
from forexTrader.agent import ForexDayTrader 
from keras.models import Model, load_model, Sequential
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam


def rate_scheduler(epoch, rate):
    
    return rate * np.exp(-0.00005*rate)


def setIterRates(trader, i):
    global dr, exploration_rate
    
    exploration_rate = rate_scheduler(i, exploration_rate)
    dr = rate_scheduler(i, dr)
    dreaming_rate = 0.84 - dr
    trader.set_rates(dreaming_rate, exploration_rate)



#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'

apath = basepath + 'data/rl/'
mpath = 'model/eurusd-ongoing/'

all_data = np.load(apath + 'eurusd-Data-All.npy', allow_pickle=True)
n = len(all_data)
ts = 60
bs = 192
name = 'myTraderMaxAvgTotalReward2'
maxAvgTotalReward = -999.99

#######################################
### change re-start parameter here ####
dr = 0.0478
exploration_rate = 0.0494
init_weight_name = 'myTraderMaxAvgTotalReward2'
start = 386001
end = 800000
### change re-start parameter here ####
#######################################

if (isfile(basepath + mpath + 'maxAvgTotalReward.npy')):
    maxAvgTotalReward = np.load(basepath + mpath + 'maxAvgTotalReward.npy')

print('%6i Batches' % (n))
print(all_data[0,3].shape)
print('Maximum Total Reward = %5.2f' % (maxAvgTotalReward))
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'forex-v0' in env:
        del gym.envs.registration.registry.env_specs[env]

register(id='forex-v0', entry_point='forexTrader.envs.forexEnv:ForexEnv')
env = gym.make(id='forex-v0', 
               data=all_data, 
               window_size=ts, 
               batch_size=bs, 
               n_batch=n)


trader = ForexDayTrader(env, basepath, mpath, init_weight_name)
result = []
for i in range(start, end):
    
    setIterRates(trader, i)    
    action_hold = 0
    action_buy = 0
    action_sell = 0
    buy_reward = 0.
    sell_reward = 0.
    curr_state = env.reset().reshape(1,60,24)
    for j in range(bs):
        
        # batch loop
        action = trader.act(curr_state)
        new_state, reward, done, info = env.step(action)
        new_state = new_state.reshape(1,60,24)
        trader.remember(curr_state, action, reward, new_state, done)

        if (action == 0):
            action_hold += 1
        elif (action == 1):
            action_buy += 1
        elif (action == 2):
            action_sell += 1
            
        if done:
            trader.day_dream(info["datetime"], 
                             info["batch_id"], 
                             info["total_reward"])
        else:
            curr_state = new_state
            
    trader.dream()
    
    result.append([i, 
                   info["datetime"][:10], 
                   info["batch_id"], 
                   info["total_reward"],
                   info["total_reward_long"],
                   info["total_reward_short"],
                   info["n_long_trades"],
                   info["n_short_trades"],
                   action_hold,
                   action_buy,
                   action_sell])

    trader.replay(i)
    
    
    if (i % 1000 == 0):
        df = pd.DataFrame(result, columns=['Iteration',
                                           'DateTime',
                                           'Batch-ID',
                                           'Avg-Total-Reward',
                                           'Long-Reward',
                                           'Short_Reward',
                                           '#Long_Trades',
                                           '#Short_Trades',
                                           'Avg-Hold-Action',
                                           'Avg-Buy-Action',
                                           'Avg-Sell-Action'])
        
        df.to_csv(apath + 'result/rl-results-' + str(i) + '.csv', index=False)
        result = []

        avgTotalReward = np.mean(df["Avg-Total-Reward"])
        trader.train_target()
        print('Average Total Reward = %5.2f Max Total Reward = %5.2f' % (avgTotalReward, maxAvgTotalReward))
        print('exploration_rate = %5.4f dr = %5.4f' % (exploration_rate, dr))
        if (avgTotalReward > maxAvgTotalReward):
        # only save model with maximum Average Total Reward
            print("saving model with avgTotalReward = %5.2f to %32s" % (avgTotalReward, name))
            trader.save_model(name)
            maxAvgTotalReward = avgTotalReward
            np.save(basepath + mpath + 'maxAvgTotalReward.npy', maxAvgTotalReward)
                    
                