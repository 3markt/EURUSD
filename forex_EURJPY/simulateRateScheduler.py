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


def rate_LRscheduler(i, rate): 
    
   # rate = rate*np.exp((-0.525/(10 + 1*i)))
    rate = rate * np.exp(-(1/(100000 + i)))
    return rate
        

def rate_DRscheduler(i, rate): 
    
   # rate = rate*np.exp((-0.525/(10 + 1*i)))
    rate = rate * np.exp(-(1/(750000 + i)))
    return rate

def rate_ERscheduler(i, rate): 
    
   # rate = rate*np.exp((-0.525/(10 + 1*i)))
    rate = rate * np.exp(-(1/(250000 + i)))
    return rate

        

lr = 0.1
dr = 0.5
base = 0.4
er = 0.9
init_epochs = 1

for i in range(2000000): 
    lr = rate_LRscheduler(i, lr)
    dr = rate_DRscheduler(i, dr)
    er = rate_ERscheduler(i, er)
    if (i % 100000 == 0):
        print('Iter = %6i, lr = %5.4f dr = %5.4f er = %5.4f' % (i, lr, base+dr, er))


