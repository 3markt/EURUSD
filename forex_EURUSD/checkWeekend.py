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


cpath = rawpath + 'forexCandle/'
cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f)) and \
            f[:6] == currency]
cfiles.sort()

df = pd.DataFrame()
dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')
for f in cfiles[-1:]:
    print(f)
    dfi = pd.read_csv(cpath + f, parse_dates=['Gmt time'], date_parser=dateparse)
    dfi.rename({'Gmt time':'ttime'}, inplace=True, axis=1)
    dfi = dfi[['ttime', 'Open', 'High', 'Low', 'Close', ]]
    df = df.append(dfi)

df = df.sort_values(by='ttime')
df.reset_index(drop=True, inplace=True)
#df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
#df.drop('ttime', inplace=True, axis=1)
print(df.index)
df['ttime'] = df['ttime'] + pd.to_timedelta(5, 'min')
df.loc[df['ttime'].dt.dayofweek == 6, 'ttime'] = df['ttime'] \
                                                    - pd.to_timedelta(2, 'D')
print(len(df))
df.drop_duplicates(subset=['ttime'], 
                   keep='first', 
                   inplace=True, 
                   ignore_index=True)
print(len(df))

df = df.loc[df['ttime'] >= '2008-01-01 00:05:00']
df = df[(df['ttime'].dt.month != 1) | (df['ttime'].dt.day != 1)]
df = df[(df['ttime'].dt.month != 12) | (df['ttime'].dt.day != 24)]
df = df[(df['ttime'].dt.month != 12) | (df['ttime'].dt.day != 25)]
df = df[(df['ttime'].dt.month != 12) | (df['ttime'].dt.day != 31)]

print(len(df))

df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
df.drop('ttime', inplace=True, axis=1)
df = df.asfreq('5min', method='ffill')
print(len(df))

df = df[(df.index.weekday != 5)]
df = df[(df.index.weekday != 6)]
print(len(df))


df.to_csv(ppath + 'df_final.csv')

dff = df.groupby(df.index.date).count()
print(len(dff))
dff = dff[dff['Close'] < 288]
dff.to_csv(ppath + 'weekendTest.csv')

