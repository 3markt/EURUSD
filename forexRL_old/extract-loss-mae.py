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
import datetime as dt
from os.path import isfile, join, isdir
import random
import re


path = '/Users/uwe.muller/Hope/data/rl/'
filename = path + 'nohup.out'
train_info = []
with open(filename) as file:
    for line in file:
        if (line[:4] == 'Fit:'):
            train_info.append(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line))
            
pd.DataFrame(train_info, columns=["Iteration", "Train-Cnt", "Loss", "Mae"]) \
    .to_csv(path + "rl-results/loss-mae.csv", index=False)


