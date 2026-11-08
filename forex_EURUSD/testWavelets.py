#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))



path = '/home/uwe/Hope/data/raw/'

df = pd.read_csv(path + 'wavelets.csv')
df.set_index(pd.DatetimeIndex(df.time), inplace=True)
df = df[df.index.date == dt.date(2022, 1, 13)]
x = df.index
var = 'we64_D1'
y = df[var]
plt.scatter(x, y, label=var)
var = 'we32_D1'
y = df[var]
plt.scatter(x, y, label=var)
var = 'we32_A2'
y = df[var]
plt.scatter(x, y, label=var)
plt.legend(loc='best')
plt.grid()
plt.show()


