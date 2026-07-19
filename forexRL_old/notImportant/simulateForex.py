#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:03:32 2021

@author: uwe.mueller
"""
import pandas as pd
import numpy as np



def getBatch(df, i):
    global bs, ts
    
    startTick = i*(bs+ts)
    endTick = (i+1)*(bs+ts)
    
    return df[startTick:endTick]


def calcReward(price, position):
    global tradeFee
    
    if (position == 1):
        return 10000*(price[-1] - price[0] - tradeFee)
    elif (position == 2):
        return -10000*(price[-1] - price[0] + tradeFee)



bs = 192
ts = 60
tradeFee = 0.00001

basepath = '/Users/uwe.muller/Hope/'
df = pd.read_csv(basepath + 'program/forexRL/gym_anytrading/datasets/data/eurusd.csv')
df = df.drop(columns='Unnamed: 0', axis=1)
n = len(df)
print(n, n/(bs+ts))
outDF = pd.DataFrame()

k = 0
while ((k+1)*bs+ts < n):
    iDF = getBatch(df, k)
    iDF.reset_index(drop=True, inplace=True)
    price = iDF['Close'].values
    print('processing ' + str(k))
    reward = np.full((len(price), 2), 0.)
    for j in range(10000):
        startTradeTick = None
        endTradeTick = None
        position = 0
        for i in range(ts-1, bs+ts+1):
            action = np.random.choice([0, 1, 2, 3])
            if (position == 0 and action in [1, 2]):
                startTradeTick = i
                if (action == 1):
                    position = 1
                else:
                    position = 2
                    
            elif (position in [1, 2] and action == 3):
                endTradeTick = i
                reward[startTradeTick:endTradeTick, position-1] += calcReward(price[startTradeTick:endTradeTick], position)
                position = 0
    
    iDF = pd.concat([iDF, pd.DataFrame(reward/j, columns=['Reward-Long', 'Reward-Short'])], axis=1)
    print(iDF)
    outDF = outDF.append(iDF)

outDF.reset_index(drop=True, inplace=True)
outDF.to_csv(basepath + 'program/forexRL/gym_anytrading/datasets/data/eurusdABT.csv', index=False)



