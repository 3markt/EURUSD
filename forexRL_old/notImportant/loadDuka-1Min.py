#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import datetime as dt
import time


def confDT(row):
    
    try:
        return dt.datetime.strptime(row[:19], '%d.%m.%Y %H:%M:%S')
    except:
        print(row[:19])



fext = '-202106.csv'
basepath = '/Users/uwe.muller/Hope/'
path = basepath + 'data/forex/duka/'

df = pd.read_csv(path + 'EURUSD-1Min-01.06.2021.csv')
df.rename(columns={'Local time': 'Time'}, inplace=True)
df['Time'] = df['Time'].apply(confDT)
df.set_index(pd.DatetimeIndex(df.Time), inplace=True)

print(len(df))

ts5Min = df.resample('5T', label='right').agg({'Open': 'first',
                                               'Close': 'last',
                                               'Low': 'min',
                                               'High': 'max'})

ts5Min.to_csv(basepath + 'data/abt/eurusd5Min' + fext)

ts15Min = df.resample('15T', label='right').agg({'Open': 'first',
                                                 'Close': 'last',
                                                 'Low': 'min',
                                                 'High': 'max'})

ts15Min.to_csv(basepath + 'data/abt/eurusd15Min' + fext)

ts1H = df.resample('60T', label='right').agg({'Open': 'first',
                                              'Close': 'last',
                                              'Low': 'min',
                                              'High': 'max'})

ts1H.to_csv(basepath + 'data/abt/eurusd1H' + fext)

ts4H = df.resample('240T', label='right').agg({'Open': 'first',
                                               'Close': 'last',
                                               'Low': 'min',
                                               'High': 'max'})

ts4H.to_csv(basepath + 'data/abt/eurusd4H' + fext)

