#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:03:32 2021

@author: uwe.mueller
"""
import pandas as pd
import numpy as np
import sys

        

def calcReward(i, lastChangeTick, position, action):
    global bt, tradeFee, price, diffs, pip, atr
    
    r = 0
    if (position == 0 and action == 1):
# reward for opening a Long position
        r = diffs[i+1] - tradeFee
    elif (position == 0 and action == 2):
# reward for opening a short position
        r = -diffs[i+1] - tradeFee
    elif (position == 1 and action == 0):
# reward for closing a long position
        r = pip*(price[i] - price[lastChangeTick]) - tradeFee
    elif (position == 1 and action == 1):
# reward for holding a long position
        r = pip*(price[i] - price[lastChangeTick]) + diffs[i+1] - tradeFee
    elif (position == 1 and action == 2):
# reward for closing a long position and opening a short one
        r1 = pip*(price[i] - price[lastChangeTick])
        r2 = -diffs[i+1]
        r = r1 + r2 - tradeFee
    elif (position == 2 and action == 0):
# reward for closing a short position
        r = -pip*(price[i] - price[lastChangeTick]) - tradeFee
    elif (position == 2 and action == 1):
# reward for closing a short position and opening a long one
        r1 = -pip*(price[i] - price[lastChangeTick])
        r2 = diffs[i+1]
        r = r1 + r2 - tradeFee
    elif (position == 2 and action == 2):
# reward for holding a short position
        r = -pip*(price[i] - price[lastChangeTick]) - diffs[i+1] - tradeFee
            
    return r
        


batch_id = sys.argv[1]
#batch_id = '6'
if (batch_id not in ['1', '2', '3', '4', '5', '6', '7']):
    exit('Wrong batch id ' + batch_id)

print('Batch with id ' + batch_id + ' started ...')

#batch_id = '-202105'

bt = 252
tradeFee = 1.5
pip = 10000

#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'
apath = basepath + 'data/abt/'

df = pd.read_csv(apath + 'eurusdFeatures' + batch_id + '.csv', parse_dates=['Time'], infer_datetime_format=True)

print(len(df))
n = len(df)
print(n, n/bt)
outDF = pd.DataFrame()

k = 0
while ((k+1)*(bt) <= n):
    iDF = df[k*bt:(k+1)*bt]
    iDF.reset_index(drop=True, inplace=True)
    price = iDF['Close'].values
    diffs = np.diff(price, prepend=price[0])*pip
    atr = iDF['atr'].values
    print('batch id = ' + batch_id + ' processing ' + str(k) + ' with Start Datetime ' + str(iDF['Time'].iloc[0]))
    sys.stdout.flush()
    reward = np.full((bt, 3, 3), 0.)
#
# Simulation of all possible rewards over all possible position/action combinations
#
    for j in range(1000):
        lastChangeTick = 0
        position = 0
        for i in range(bt):
            action = np.random.choice([0, 1, 2])

            if (i == bt-1):
# end of batch - force to close trade
                action = 0                
            
            reward[i, position, action] += calcReward(i, lastChangeTick, position, action)
            
            if (action != position):
                position = action
                lastChangeTick = i
#
# pick the position with maximum reward and store it as a feature             
#
    reward /= j
    rw = []
    for i in range(bt):
        if (i < 59):
            position = 0
        else:
            position = int(np.argmax(reward[i])/3)
        rw.append(np.append(position,reward[i, position]))
    iDF = pd.concat([iDF, pd.DataFrame(rw, columns=['Position',
                                                    'Reward-Hold', 
                                                    'Reward-Buy',
                                                    'Reward-Sell'])], axis=1)
    outDF = outDF.append(iDF)
    k += 1

outDF.reset_index(drop=True, inplace=True)
outDF.to_csv(apath + 'eurusdRewards-V2-' + batch_id + '.csv', index=False)



