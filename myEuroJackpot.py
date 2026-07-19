#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import numpy as np


mySeed = 2005*11*29
np.random.seed(seed=mySeed)

for i in range(9):
    a = np.random.choice(range(1,51), 5, replace=False)
    b = np.random.choice(range(1,13), 2, replace=False)
    a.sort()
    b.sort()
    print(a, b)
    
