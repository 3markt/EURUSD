#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""
import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.registration import register
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from forexTrader.envs import forexEnv
from forexTrader.agent import ForexDayTrader


def evalOneDay(trader, day):
    #
    # Methode berechnet den Profit für einen Tage
    #
    trade_fee = 1.0
    pip = 10000
    batch_id = day[0]
    close = day[2].flatten().tolist()
    x = day[3].astype(np.float32)
    action = trader.predictOneDay(x).tolist()
    num_hold_action = sum([1 for x in action if x == 0])
    num_buy_action = sum([1 for x in action if x == 1])
    num_sell_action = sum([1 for x in action if x == -1])
    profit = 0.
    profit_pos = 0.
    profit_neg = 0.
    position = 0
    last_i = None
    i = 0
    num_trades = 0
    num_short_trades = 0
    num_long_trades = 0
    for a in action:
        if (a == 0):
            if (position in [-1, 1]):
                profit += position * pip * (close[i] - close[last_i]) - trade_fee
                num_trades += 1
                if (position == 1):
                    num_long_trades += 1
                    profit_pos += position * pip * (close[i] - close[last_i]) - trade_fee
                elif (position == -1):
                    num_short_trades += 1
                    profit_neg += position * pip * (close[i] - close[last_i]) - trade_fee
                last_i = None
        elif (a == 1):
            if (position == -1):
                profit += position * pip * (close[i] - close[last_i]) - trade_fee
                profit_neg += position * pip * (close[i] - close[last_i]) - trade_fee
                last_i = i
                num_trades += 1
                num_short_trades += 1
            elif (position == 0):
                last_i = i
        elif (a == -1):
            if (position == 1):
                profit += position * pip * (close[i] - close[last_i]) - trade_fee
                profit_pos += position * pip * (close[i] - close[last_i]) - trade_fee
                last_i = i
                num_trades += 1
                num_long_trades += 1
            elif (position == 0):
                last_i = i
        position = a
        i += 1

    return [batch_id,
            profit,
            profit_pos,
            profit_neg,
            num_hold_action,
            num_buy_action,
            num_sell_action,
            num_trades,
            num_long_trades,
            num_short_trades]



bs = 252
ts = 36
ff = 65

final_path = '/home/uwe/Hope/data/final/EURUSD/'

env_gym = gym.envs.registration.registry.copy()
for env in env_gym:
    if 'forex-v0' in env:
        del gym.envs.registry['forex-v0']

register(id='forex-v0', entry_point='forexTrader.envs.forexEnv:ForexEnv')

env = gym.make(id='forex-v0',
               data_path=final_path,
               window_size=ts,
               batch_size=bs)

trader = ForexDayTrader(env, final_path, 'bestModel', bs, ts, ff)

newDay = env.unwrapped.getRL_Data('2018-01-26')

result = evalOneDay(trader, newDay)

print(result)