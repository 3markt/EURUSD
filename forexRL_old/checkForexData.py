#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
import joblib
from os import listdir, makedirs
from os.path import isfile, join, isdir


path = '/Users/uwe.muller/Hope/Data/abt/'

tsfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[:13] == 'eurusdRewards'
           and f[13] in ['1', '2', '3', '4', '5', '6', '7']]

tsfiles.sort()
df = pd.DataFrame()

for f in tsfiles:
    dfi = pd.read_csv(path + f, parse_dates=['Time'], infer_datetime_format=True)
    print(f, len(dfi))
    df = df.append(dfi)

wl = df['WL'].values
wl01 = np.percentile(wl, 0.1)
wl99 = np.percentile(wl, 99.9)
print(wl01, wl99)

df_low = df[df['WL'] <= wl01]
df_low = df_low[['Time', 'WL']]

low_h_obs = np.sort(df_low['Time'].dt.hour)
llow_h = np.sort(list(set(low_h_obs)))
for l in llow_h:
    print(l, np.count_nonzero(low_h_obs == l))

low_m_obs = np.sort(df_low['Time'].dt.minute)
llow_m = np.sort(list(set(low_m_obs)))
for l in llow_m:
    print(l, np.count_nonzero(low_m_obs == l))


df_high = df[df['WL'] >= wl99]
df_high = df_high[['Time', 'WL']]


