#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import gym
from enum import Enum


bs = 228
ts = 60

path = '/Users/uwe.muller/Hope/data/abt/'
file = 'eurusd5Min.csv'

df = pd.read_csv(path + file,
                 parse_dates=['Time'], infer_datetime_format=True)

df = df[df['Time'] >= '2010-01-01']
df = df.loc[df['Time'].dt.hour <= 20]

n = len(df)
print(n, n/(bs+ts))

cnt= df.set_index(pd.DatetimeIndex(df.Time))
cnt = pd.DataFrame(df['Time'].groupby(cnt.index.date).count())
cnt = cnt.rename(columns={'Time':'Count'})
cnt['Time'] = cnt.index.copy()
cnt['Time'] = pd.to_datetime(cnt['Time'])
cnt.reset_index(drop=True, inplace=True)
cnt.to_csv(path + 'cnt5Min.csv')

"""
df = df.loc[(df['Time'] < '2017-12-25') | (df['Time'] >= '2017-12-26')]
df = df.loc[(df['Time'] < '2015-05-31') | (df['Time'] >= '2015-06-01')]

n = len(df)
print(n, n/(bs+ts))
df.to_csv(path + file)

"""
