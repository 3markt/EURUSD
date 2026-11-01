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

path = '/home/uwe/Hope/data/final/EURUSD/RL-data/'
file = 'day-2017-12-22.npy'

x = np.load(path + file, allow_pickle=True)
print(x[0])