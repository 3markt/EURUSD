#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random

path = '/home/uwe/Hope/data/final/EURUSD/longterm_memory/'

ltmem = np.load(path + 'ltmem636.npy', allow_pickle=True).tolist()

print(ltmem[1][0])

#ltind = random.sample(range(641), 1)[0]
#print(ltind)

