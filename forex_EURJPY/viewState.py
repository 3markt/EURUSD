#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import math
import pickle


path = '/Users/uwe.muller/Hope/data/final/EURJPY/longterm_memory/'
mem = np.load(path + 'ltmem110.npy', allow_pickle=True).tolist()

# 0 - dt 
# 1 - batch_id, 
# 2 - total_reward,
# 3 - action,
# 4 - state, 
# 5 - target

dayind = 20
ind5m = 130
print('Date:', mem[dayind][0])
print('Batch-ID:', mem[dayind][1])
print('Total Reward:', mem[dayind][2])
print('Actions:', mem[dayind][3][ind5m-1])
for i in range(12):
    print(i, mem[dayind][4][ind5m+i][-1-i][0:6])

for i in range(276):
    print(mem[dayind][5][i])
