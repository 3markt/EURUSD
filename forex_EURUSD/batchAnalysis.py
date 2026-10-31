#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from operator import itemgetter
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8)) # Schön groß machen!

path = '/home/uwe/Hope/data/final/EURUSD/result/'

files = [f for f in listdir(path) if isfile(join(path, f)) and f[:11] == 'rl-results-']
#files = [f for f in files if int(f.split("-")[2].split(".")[0]) < 1000000]
ff = [[f, int(f[11:f.find('.')])] for f in files]
ff = sorted(ff, key=itemgetter(1))
files = [f[0] for f in ff]
print(len(files), files[-1])

df = pd.DataFrame()
for f in files:
    dfi = pd.read_csv(path + f)
    df = pd.concat([df, dfi], ignore_index=True)

limit = 2.0
print(len(df.loc[df['td_err Abs Mean'] > limit]), 'von', len(df))

y = df.loc[df['td_err Abs Mean'] > limit, ['Avg-Total-Reward']]
x = df.loc[df['td_err Abs Mean'] > limit, ['Iteration']]
plt.plot(x, y)
plt.grid()
plt.ylabel('Total Revenue')
plt.scatter(x,y)
plt.show()

