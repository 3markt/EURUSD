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


path = '/Users/uwe.muller/Hope/data/abt/'
file1 = 'maxRewards-201503.csv'
file3 = 'maxRewards-202103.csv'
file2 = 'eurusd-5-15-1H-4H-1D.csv'



df1 = pd.read_csv(path + file1,
                 parse_dates=['Time'], infer_datetime_format=True)

df1.drop(df1.columns[df1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(len(df1))

df3 = pd.read_csv(path + file3,
                 parse_dates=['Time'], infer_datetime_format=True)
df3.drop(df3.columns[df3.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(len(df3))

df1 = pd.concat([df1, df3])
print(len(df1))

df2 = pd.read_csv(path + file2,
                 parse_dates=['Time'], infer_datetime_format=True)

df2.drop(df2.columns[df2.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(df2.dtypes)


print('Len Rewards = %6i, len eurusd5Min= %6i' % (len(df1), len(df2)))

abt = pd.merge_ordered(df1, df2, 
                       how='inner', on='Time')


print(len(abt))

abt.to_csv(path + 'eurusdMaxRewards-5-15-1H-4H-1D.csv', index=False)

