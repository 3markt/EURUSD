#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""
import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, isdir

def calculateRisk(close, window):

    pip = 10000
    risk = np.zeros_like(close, dtype=float)
    risk[:-window] = np.abs(pip * ((close[window:] - close[:-window]) / close[:-window]))

    return round(np.sum(risk), 2)


path = '/home/chitlom/data/final/EURUSD/RL-data/'
# path = '/home/uwe/Hope/data/final/EURUSD/RL-data/'
opath = '/home/chitlom/data/final/EURUSD/'
# opath = '/home/uwe/Hope/data/final/EURUSD/'

dfiles = [f for f in listdir(path) if isfile(join(path, f)) \
               and f[:3] == 'day']
dfiles.sort()
crossX = []
for fn in dfiles:
    rl = np.load(path + fn, allow_pickle=True)
    day = fn[4:14]
    batch_id = rl[0]
    close = rl[2]
    risk_60 = calculateRisk(close, window=12)
    crossX.append([batch_id, day, risk_60])

pd.DataFrame(crossX, columns=['batch_id',
                              'day',
                              'risk-60 Min']).sort_values(by=['batch_id']).to_csv(opath + 'rl-data-crossX.csv', index=False)
