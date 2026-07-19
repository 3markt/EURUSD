#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
import ta
from os import listdir
from os.path import isfile, join
from datetime import datetime



def calcTarget1(df, shift, limit, tarName):
        
    df[tarName] = 0
    for i in range(1, shift+1):
        df.loc[(100*(df['Close'].shift(-i) - df['Open'].shift(-1))/df['Open'].shift(-1) > limit)
               & (df[tarName] == 0), tarName] = 1
                
    return df    



dj = ['AAPL',         # Apple DJ
      'AMZN',         # Amazon DJ
      'AXP',          # American Express DJ
      'BA',           # Boeing DJ
      'CAT',          # Caterpillar
      'CSCO',         # Cisco Systems
      'CVX',          # Chevron
      'DIS',          # Walt Disney
      'DOW',          # Dow Inc.
      'GS',           # Goldman Sachs
      'IBM',          # IBM
      'INTC',         # Intel Corporation
      'JNJ',          # Johnson & Johnson
      'JPM',          # JP Morgan Chase
      'KO',           # Coca Cola
      'MCD',          # McDonal's Corporation
      'MMM',          # 3M Company
      'MRK',          # Merck & Co
      'MSFT',         # Microsoft
      'NKE',          # Nike Inc
      'PFE',          # Pfizer
      'PG',           # Procter & Gamble
      'RTX',          # Raytheon Technologie
      'TRV',          # Travelers Companies
      'UNH',          # United Health Group
      'V',            # VISA Inc.
      'VZ',           # Verizon Communications
      'WBA',          # Walgreens Boots Alliance
      'WMT',          # Wal-Mart
      'XOM'          # Exxon Mobile
      ]

path = '/Users/uwe.muller/Hope/Data/Aktienkurse/aktuell/'

symbls = dj

for symbl in symbls:
    df = pd.read_csv(path + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    if (len(df) >= 1000):
        print(symbl,len(df))
        df = df.sort_values('date')
        df.set_index(pd.DatetimeIndex(df['date']), inplace=True)        
        
        df = calcTarget1(df, 5, 1.0, 'Target')
        df['prevTar50'] = df['Target'].shift(1).rolling(50).sum()
        
        if (symbls.index(symbl) == 0):
            rankDF = df[['date', 'prevTar50']]
        else:
            rankDF = rankDF.join(df[['date', 'prevTar50']], how='outer', rsuffix=symbl)

cols = [c for c in rankDF.columns if c.lower()[:4] != 'date']
rankDF = rankDF[cols]
rankDF = rankDF.rank(axis=1, ascending=False)
rankDF.to_csv(path + 'explore/ranks1.csv')

        
        
        
        
   
