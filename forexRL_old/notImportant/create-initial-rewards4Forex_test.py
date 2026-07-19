#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:03:32 2021

@author: uwe.mueller
"""
import pandas as pd
import numpy as np
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
        r -= diffs[i] - tradeFee
            
    return r
        

"""
batch_id = sys.argv[1]
if (batch_id not in ['1', '2', '3', '4', '5', '6', '7']):
    exit('Wrong batch id ' + batch_id)

print('Batch with id ' + batch_id + ' started ...')
"""

batch_id = '-202105'

bt = 252
tradeFee = 1.5
gamma = 0.90
pip = 10000

basepath = '/Users/uwe.muller/Hope/'
#basepath = '/home/chitlom/'
apath = basepath + 'data/abt/'

df = pd.read_csv(apath + 'eurusdFeatures' + batch_id + '.csv', parse_dates=['Time'], infer_datetime_format=True)

print(len(df))
n = len(df)
print(n, n/bt)
outDF = pd.DataFrame()

position = 0
k = 0
while ((k+1)*(bt) <= n):
    iDF = df[k*bt:(k+1)*bt]
    iDF.reset_index(drop=True, inplace=True)
    price = iDF['Close'].values
    diffs = np.diff(price, append=price[-1])*pip
    print('batch id = ' + batch_id + ' processing ' + str(k) + ' with Start Datetime ' + str(iDF['Time'].iloc[0]))

    sys.stdout.flush()
    reward = np.full((bt, 1, 3), 0.)
    for i in range(59, bt-1):
        reward[i, position, 1] += calcReward(i, 1)
        reward[i, position, 2] += calcReward(i, 2)

    rw = []
    for i in range(bt):
        rw.append(np.append(position,reward[i, position]))
    iDF = pd.concat([iDF, pd.DataFrame(rw, columns=['Position',
                                                    'Reward-Hold', 
                                                    'Reward-Buy',
                                                    'Reward-Sell'])], axis=1)

    outDF = outDF.append(iDF)
    k += 1

outDF.reset_index(drop=True, inplace=True)
outDF.to_csv(apath + 'eurusdRewards' + batch_id + '.csv', index=False)



