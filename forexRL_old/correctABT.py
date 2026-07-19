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
from datetime import datetime

def createCnt(df):
    
    cnt= df.set_index(pd.DatetimeIndex(df.Time))
    cnt = pd.DataFrame(df['Time'].groupby(cnt.index.date).count())
    cnt = cnt.rename(columns={'Time':'Count'})
    cnt['Time'] = cnt.index.copy()
    cnt['Time'] = pd.to_datetime(cnt['Time'])
    cnt.reset_index(drop=True, inplace=True)
    cnt = cnt[['Time', 'Count']]
    return cnt


bt = 5

path = '/Users/uwe.muller/Hope/data/abt/'
#file = 'eurusd-5-15-1H-4H.csv'
file = 'eurusd4H.csv'

df = pd.read_csv(path + file,
                 parse_dates=['Time'], infer_datetime_format=True)


df = df.loc[df['Time'].dt.strftime('%H:%M:%S') <= '20:00:00']
df = df.loc[df['Time'].dt.strftime('%H:%M:%S') > '00:00:00']

n = len(df)
print(n, n/bt)

cnt = createCnt(df)
print(cnt.loc[cnt['Count'] != bt])
delListDF = cnt.loc[cnt['Count'] != bt]
delListDF = delListDF['Time']
delListDF = delListDF.apply(lambda x: x.strftime("%Y-%m-%d"))
delList = delListDF.values.tolist()

print(len(delList))

df['Date'] = df['Time'].apply(lambda x: x.strftime("%Y-%m-%d"))
df = df.loc[~df['Date'].isin(delList)]
n = len(df)
print(n, n/bt)
df.to_csv(path + file, index=False)


