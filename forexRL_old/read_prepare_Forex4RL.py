#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from os import listdir
from os.path import isfile, join
import sys
import matplotlib.pyplot as plt
import ta

currency = 'EURJPY'
basepath = '/Users/uwe.muller/Hope/Data/'
rpath = basepath + 'forex/' + currency + '/'
apath = basepath + 'abt/' + currency + '/'

min_files = [f for f in listdir(rpath) if isfile(join(rpath, f)) and f[:13] == currency + '_Candle']
min_files.sort()
print(min_files)

dateparse = lambda dates: [datetime.strptime(d, '%d.%m.%Y %H:%M:%S.%f %Z%z').replace(tzinfo=None) for d in dates]

df_min = pd.DataFrame()
for f in min_files:
    df = pd.read_csv(rpath + f, parse_dates=['Local time'], date_parser = dateparse)
    print('read file ' + f)
    df.rename(columns={'Local time':'Time'}, inplace=True)
    df_min = df_min.append(df)
    
print(df_min)
