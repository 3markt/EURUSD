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

    pip = 10
    risk = np.zeros_like(close, dtype=float)
    risk[window:] = np.abs(pip * np.log(close[window:] / close[:-window]))

    return np.cumsum(risk)


# path = '/home/chitlom/data/final/EURUSD/RL-data/'
path = '/home/uwe/Hope/data/final/EURUSD/RL-data/'
# opath = '/home/chitlom/data/final/EURUSD/'
opath = '/home/uwe/Hope/data/final/EURUSD/'

dfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[:3] == 'day']
dfiles.sort()
risksum = np.zeros(252, dtype=float)
risksum2 = np.zeros(252, dtype=float)
for fn in dfiles:
    rl = np.load(path + fn, allow_pickle=True)
    day = fn[4:14]
    batch_id = rl[0]
    close = rl[2].flatten()
    risk_60 = calculateRisk(close, window=12)
    risksum += risk_60
    risksum2 += risk_60**2


print(risksum, risksum2)