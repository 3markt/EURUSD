#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random

a = np.array([0, 3, 0, 2, 0, 1])
print(np.percentile(a, 95))

#path = '/home/uwe/Hope/data/final/EURUSD/longterm_memory/'

#ltmem = np.load(path + 'ltmem636.npy', allow_pickle=True).tolist()

#print(ltmem[1][5].shape)

#ltind = random.sample(range(641), 1)[0]
#print(ltind)

