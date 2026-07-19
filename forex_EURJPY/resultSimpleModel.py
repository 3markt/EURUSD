#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import math
import matplotlib.pyplot as plt

timescale = 'min5'
path = '/Users/uwe.muller/Hope/data/final/EURJPY/testResult/'
f = 'test_' + timescale + '.csv'
df = pd.read_csv(path + f)

x = df['Iteration']
plt.xlabel('Iteration')
plt.rcParams["figure.figsize"] = (12, 7)

var = 'RMSE-Train'
#y = df['truePosPred']/(df['truePosPred'] + df['falsePosPred'])
y = df[var]
plt.ylabel(var)
plt.title(timescale)
plt.scatter(x, y, label=var, linewidth=0.1)
plt.grid()
#plt.show()

var = 'RMSE-Test'
#y = df['cntTrueNegPred']/(df['cntTrueNegPred'] + df['cntFalseNegPred'])
y = df[var]
plt.scatter(x, y, label=var, linewidth=0.1)
plt.legend(loc='best')

plt.grid()
plt.show()
