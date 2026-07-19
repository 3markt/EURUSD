#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import gym
from enum import Enum


path = '/Users/uwe.muller/Hope/data/abt/'
file = 'eurusdMaxRewards.csv'

cols = ['Time', 'Position', 'Reward-Hold', 'Reward-Buy', 'Reward-Sell']


df = pd.read_csv(path + file,
                 parse_dates=['Time'], infer_datetime_format=True)

df[cols].to_csv(path + 'maxRewards.csv')

