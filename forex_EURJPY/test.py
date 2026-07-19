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


reward = np.full((10,3), 0.0)
for i in range(10-1): 
    reward[i, 1] += 0.1
    reward[i, 2] += 1

position = np.array(10*[0.]).astype(float)
currProfit = np.array(10*[0.]).astype(float)
        
rw = np.c_[position, currProfit, reward]
print(rw)

df = pd.DataFrame(rw, 
             columns=['Position',
                      'currProfit',
                      'Reward-Hold', 
                      'Reward-Buy',
                      'Reward-Sell'])
print(df)
