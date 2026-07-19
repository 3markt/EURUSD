#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd

path = '/Users/uwe.muller/Hope/data/final/EURUSD/longterm_memory/'


d = np.load(path + 'ltmem10.npy', allow_pickle=True)

x = d[20, 4]
xs = x[0, :, :4]
tmp_xs = xs[1:,:2]
xs = x[144, :, :4]
xs[:-1, :2] = tmp_xs
print(xs)

print(x[0:5, :, :4])