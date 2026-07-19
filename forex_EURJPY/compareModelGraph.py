#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


var = 'profit'
ext1 = '_neu'
ext2 = '_alt'

file = '/Users/uwe.muller/Hope/Data/final/EURUSD/testResult/torch_no_shuffle_35_7.csv'
df1 = pd.read_csv(file)
df1 = df1[df1['Iteration'] >= 1000]
df1['trueR'] = df1['truePred']/df1['cntPos']
df1['MA'+var] = df1[var].rolling(window=30).mean()

file = '/Users/uwe.muller/Hope/Data/final/EURUSD/testResult/r075-ts12-ff63-u142-u221.csv'
df2 = pd.read_csv(file)
df2 = df2[df2['Iteration'] >= 1000]
df2['RMSE-Test'] = df2['val_mae']
df2['trueR'] = df2['truePred']/df2['cntPos']
df2['MA'+var] = df2[var].rolling(window=30).mean()

x_label = 'Iteration'


df = pd.merge_ordered(df1, df2, 
                      how='left', on='Iteration', 
                      suffixes=(ext1, ext2))
df = df[df['Iteration'] > 1020]

x = df[x_label]
y1 = df[var + ext1]
y2 = df[var + ext2]
y1 = df['MA' + var + ext1]
y2 = df['MA' + var + ext2]

plt.rcParams["figure.figsize"] = (12, 7)
plt.plot(x, y1, label=var+ext1, linewidth=1)
plt.plot(x, y2, label=var+ext2, linewidth=1)
plt.xlabel(x_label)
plt.ylabel(var)
plt.title('Modell-Vergleich')
plt.legend(loc='best')
plt.grid()
plt.show()

y = y1 - y2
plt.plot(x, y, label='Diff' + var, linewidth=1)
plt.grid()
plt.show()

