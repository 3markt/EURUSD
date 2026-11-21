#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""
import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8)) # Schön groß machen!


path = '/home/uwe/Hope/data/final/EURUSD/RL-data/'
file = 'day-2015-12-03.npy'

y = np.load(path + file, allow_pickle=True)[2].flatten()
print(y.shape, len(y))
x = range(len(y))

plt.scatter(x, y, linewidth=0.1)
plt.grid()
plt.show()

