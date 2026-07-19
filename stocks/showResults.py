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
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


name1 = 'result_fitAllStocks_v3'
name2 = 'result_fitAllStocks_v4'
path = '/Users/uwe.muller/Hope/Data/Aktienkurse/aktuell/results/'#
files1 = [f for f in listdir(path) if isfile(join(path, f)) and f[:len(name1)] == name1]
files1.sort()
print(files1)
files2 = [f for f in listdir(path) if isfile(join(path, f)) and f[:len(name2)] == name2]
files2.sort()
print(files2)

symbls = ['ADS.DEX',      # Adidas DAX
       'ALV.DEX',      # Allianz DAX
       'BAS.DEX',      # BASF DAX
       'BAYN.DEX',     # Bayer DAX
       'BEI.DEX',      # Beiersdorf DAX
       'BMW.DEX',      # BMW DAX
       'CON.DEX',      # Continental DAX
       '1COV.DEX',     # Covestro
       'DAI.DEX',      # Daimler
       'DBK.DEX',      # Dt. Bank
       'DB1.DEX',      # Dt. Boerse
       'DPW.DEX',      # Dt. Post
       'DTE.DEX',      # Dt. Telekom
       'EOAN.DEX',     # EO.N
       'FME.DEX',      # Fresenius Medical Care
       'FRE.DEX',      # Fresenius
       'HEI.DEX',      # HeidelbergCement
       'HEN3.DEX',     # Henkel Vz.
       'IFX.DEX',      # Infineon
       'LIN.DEX',      # Linde PLC
       'LHA.DEX',      # Lufthansa
       'MRK.DEX',      # Merck KGaA
       'MTX.DEX',      # MTU Aero Engines
       'MUV2.DEX',     # Munich RE
       'RWE.DEX',      # RWE
       'SAP.DEX',      # SAP
       'SIE.DEX',      # Siemens
       'VOW.DEX',      # VW Vz.
       'VNA.DEX',      # Vonovia
       'WDI.DEX'      # Wirecard          
       ]

iter = 20
df1 = pd.DataFrame()
for f in files1:
    df1 = df1.append(pd.read_csv(path + f))
    df1 = df1[df1['iteration'] >= iter]


iter = 20
df2 = pd.DataFrame()
for f in files2:
    df2 = df2.append(pd.read_csv(path + f))
    df2 = df2[df2['iteration'] >= iter]

for symbl in symbls:
    ddf1 = df1[df1['symbol'] == symbl]
    ddf2 = df2[df2['symbol'] == symbl]
    print(symbl, round(ddf1['tst-acc'].mean(), 3), \
          round(ddf2['tst-acc'].mean(), 3))

"""
plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (14,8))
plt.plot(ddf['iteration'].values, ddf['tst-acc'].values, 'o')
plt.show()
"""