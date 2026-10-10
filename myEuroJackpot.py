#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import numpy as np


for i in range(8):
    match i:
        case 0:
            mySeed = 29112005
            np.random.seed(seed=mySeed)
        case 2:
            mySeed = 27071995
            np.random.seed(seed=mySeed)
        case 3:
            mySeed = 13031994
            np.random.seed(seed=mySeed)
        case 4:
            mySeed = 9071972
            np.random.seed(seed=mySeed)
        case 6:
            mySeed = 1021959
            np.random.seed(seed=mySeed)
    a = np.random.choice(range(1,51), 5, replace=False)
    b = np.random.choice(range(1,13), 2, replace=False)
    a.sort()
    b.sort()
    print(a, b)
    
