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

path = '/home/chitlom/data/rl/longterm_memory/'

ind = np.load(path + 'index.npy')
print(ind)

l = []
for i in range(ind):
    ll = np.load(path + 'ltmem' + str(i) + '.npy', allow_pickle=True).tolist()
    l += ll
    print(len(l))
    