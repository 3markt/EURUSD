#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd


def setIterRates(rate, i):
    
    rate = rate * np.exp(-(1/(64000 + i)))
    
    return rate


lr = 0.001

for i in range(1500000):
    lr = setIterRates(lr, i)
print('learning_rate = %5.8f' % (lr))

