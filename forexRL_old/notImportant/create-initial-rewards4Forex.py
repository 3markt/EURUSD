#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:03:32 2021

@author: uwe.mueller
"""
import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir
import sys

        

def calcReward(i, action):
    global bt, tradeFee, gamma, price, diffs, pip
    
    r = 0
    if (action == 1):
# reward for opening a Long position
        for j in range(bt-2, i, -1):
            r = gamma*(r + diffs[j])
        r += diffs[i] - tradeFee
    elif (action == 2):
# reward for opening a short position
        for j in range(bt-2, i, -1):
            r = gamma*(r - diffs[j])
        r += -diffs[i] - tradeFee
            
    return r
        


bt = 252
tradeFee = 1.5
gamma = 0.9
pip = 10000

#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'
path = basepath + 'data/abt/'

tsfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[:14] == 'eurusdFeatures'
           and f[14] in ['1', '2', '3', '4', '5', '6', '7']]

tsfiles.sort()
for f in tsfiles:
    df = pd.read_csv(path + f, parse_dates=['Time'], infer_datetime_format=True)
    n = len(df)
    print(f, n, n/bt)
    outDF = pd.DataFrame()
    
    position = 0
    k = 0
    while ((k+1)*(bt) <= n):
        iDF = df[k*bt:(k+1)*bt]
        iDF.reset_index(drop=True, inplace=True)
        price = iDF['Close'].values
        diffs = np.diff(price, append=price[-1])*pip
    
        sys.stdout.flush()
        reward = np.full((bt, 1, 3), 0.)
        for i in range(59, bt-1):
            reward[i, position, 1] += calcReward(i, 1)
            reward[i, position, 2] += calcReward(i, 2)
    
        rw = []
        for i in range(bt):
            tmp = np.append(diffs[i], reward[i, position])
            tmp = np.append(position, tmp)
            rw.append(tmp)
        iDF = pd.concat([iDF, pd.DataFrame(rw, columns=['Position',
                                                        'diffs',
                                                        'Reward-Hold', 
                                                        'Reward-Buy',
                                                        'Reward-Sell'])], axis=1)
    
        outDF = outDF.append(iDF)
        k += 1
    
    outDF.reset_index(drop=True, inplace=True)
    outDF.to_csv(path + 'eurusdRewards' + f[14:], index=False)
    


