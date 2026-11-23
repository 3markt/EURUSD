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

bs = 252
ts = 36
ff = 65

final_path = '/home/uwe/Hope/data/final/EURUSD/'
data_path = final_path + 'RL-data/'
file = 'day-2015-12-03.npy'

env_gym = gym.envs.registration.registry.copy()
for env in env_gym:
    if 'forex-v0' in env:
        del gym.envs.registry['forex-v0']

register(id='forex-v0', entry_point='forexTrader.envs.forexEnv:ForexEnv')

env = gym.make(id='forex-v0',
               data_path=data_path,
               window_size=ts,
               batch_size=bs)

trader = ForexDayTrader(env, final_path, 'bestModel', bs, ts, ff)