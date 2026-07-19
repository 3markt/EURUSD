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
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join



def calcTarget(df, shift, limit):
        
    df['Target'] = 0
    for i in range(1, shift+1):
        df['CH'+str(i)] = 100*(df['High'].shift(-i) - df['Close'])/df['Close']
        df['CL'+str(i)] = 100*(df['Low'].shift(-i) - df['Close'])/df['Close']
        df.loc[(df['CH'+str(i)] > (-1)*df['CL'+str(i)]) 
               & (df['CH'+str(i)] > limit)
               & (df['Target'] == 0), 'Target'] = 1
        df.loc[(df['CH'+str(i)] < (-1)*df['CL'+str(i)]) 
               & (df['CL'+str(i)] < (-1)*limit)
               & (df['Target'] == 0), 'Target'] = 2
        
    return df    




def plotX(x):
    
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize = (14,8))
    plt.plot(x, 'o')
    plt.show()



def prepData(path, symbls, bs, ts, ff):

    x_all = []  
    y_all = []
    symbls_nb = []
    for symbl in symbls[:1]:
        df = pd.read_csv(path + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
        df = df.sort_values('date')
#        df = df[df['date'] >= datetime.strptime('01-01-2015', '%d-%m-%Y')]
#        print(len(df))
        df = df.reset_index(drop=True)
        df.rename(inplace = True,
                             columns={'date':'Date',
                                      '1. open':'Open',
                                      '2. high':'High',
                                      '3. low':'Low',
                                      '4. close':'Close',
                                      '5. volume':'Volume'})
    
        df['Current'] = df['Open'].shift(-1)
        df = calcTarget(df, 2, 1.0)
    
#        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'h-shift', 'l-shift', 'Target']]
        df.dropna(inplace=True)
        n = len(df)
        n_bs = n//bs - 1
        if (n_bs + ts > n):
            n_bs -= 1
        df = df[-n_bs*bs-ts:]
        df = df.reset_index(drop=True)
#        print(len(df))        
        df.to_csv(path + 'abt/' + symbl + '.csv')
        
#        plotX(df['Close'].values)

        x_data = df[['Open','High', 'Low', 'Close']].values
        y_data = df['Target'].values
        sc = MinMaxScaler(feature_range=(-1,1))
        x_data = sc.fit_transform(x_data)

# prepare training data        
        symbl_x = np.array([])
        symbl_y = np.array([])
        for b in range(n_bs):
            batch_x = np.array([])
            batch_y = np.array([])
            for i in range(b*bs+ts, (b+1)*bs+ts):
                batch_x = np.append(batch_x, x_data[i-ts:i])
                batch_y = np.append(batch_y, y_data[i-1])
            symbl_x = np.append(symbl_x, batch_x)
            symbl_y = np.append(symbl_y, batch_y)
        x_all.append(symbl_x.reshape((n_bs)*bs, ts, ff))
        y_all.append(symbl_y.reshape((n_bs)*bs, 1, 1))
        symbls_nb.append([symbl, n_bs])

    return x_all, y_all, symbls_nb







bs = 50
ts = 10
ff = 4

stocks = ['ADS.DEX',      # Adidas
          'ALV.DEX',      # Allianz
          'BAS.DEX',      # BASF
          'BAYN.DEX',     # Bayer
          'BEI.DEX',      # Beiersdorf
          'BMW.DEX',      # BMW
          'CON.DEX',      # Continental
          '1COV.DEX',     # Covestro
          'DAI.DEX',      # Daimler
          'DBK.DEX',      # Dt. Bank
          'DB1.DEX',      # Dt. Boerse
          'DPW.DEX',      # Dt. Post
          'DTE.DEX',      # Dt. Telekom               
          'EOAN.DEX',     # EO.N
          'FRE.DEX',      # Fresenius
          'FME.DEX',      # Fresenius Medical Care
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
          'WDI.DEX'       # Wirecard        
                ]

path = '/Users/uwe.muller/Hope/Data/Aktienkurse/aktuell/'

x_all, y_all, symbls = prepData(path, stocks, bs, ts, ff)

