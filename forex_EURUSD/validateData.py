#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
import datetime as dt
from os import listdir, makedirs
from os.path import isfile, join, isdir


rawpath = '/Users/uwe.muller/Hope/data/raw/'
currency = 'EURUSD'
ppath = '/Users/uwe.muller/Hope/data/processed/' + currency + '/forexCandle/'

"""
cpath = rawpath + 'forexCandle/'
cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f)) and \
            f[:6] == currency]
cfiles.sort()

df = pd.DataFrame()
dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')
for f in cfiles:
    print(f)
    dfi = pd.read_csv(cpath + f, parse_dates=['Gmt time'], date_parser=dateparse)
    dfi.rename({'Gmt time':'ttime'}, inplace=True, axis=1)
    dfi = dfi[['ttime', 'Open', 'High', 'Low', 'Close', ]]
    df = df.append(dfi)

"""
df = pd.read_csv(ppath + 'df_all.csv', parse_dates=['time'])
df.set_index(pd.DatetimeIndex(df.time), inplace=True)
print(len(df))

"""
df = df[df['time'].dt.dayofweek != 6]
print(len(df))
df = df[(df['ttime'].dt.month != 1) | (df['ttime'].dt.day != 1)]
df = df[(df['ttime'].dt.month != 12) | (df['ttime'].dt.day != 25)]
df = df[(df['ttime'].dt.year != 2008) | (df['ttime'].dt.month != 4) | \
        (df['ttime'].dt.day != 4)]
df = df[(df['ttime'].dt.year != 2014) | (df['ttime'].dt.month != 7) | \
        (df['ttime'].dt.day != 4)]
print(len(df))
"""

dff = df.groupby(df.index.date).count()
dff = dff[dff['Close'] < 252]
dff.to_csv(ppath + 'weekendTest.csv')
