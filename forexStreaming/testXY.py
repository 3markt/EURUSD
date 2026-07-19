#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import psycopg2
import pandas as pd
import os
import ta
import datetime as dt
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir



featureList = ['Position', 'zeitIndex', 'scaledClose', 'round_lot', 'WL', 
               'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi', 'macd_30Min', 
               'macd_d_30Min', 'bb_h_30Min', 'bb_l_30Min', 'rsi_30Min', 
               'macd_6H', 'macd_d_6H', 'bb_h_6H', 'bb_l_6H', 'rsi_6H',
               'CPI_US' , 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
               'Zins_US', 'Zins_EU', 'Schock_Preis_US', 'Schock_Preis_EU', 
               'Verbraucherstimmung_US', 'Verbraucherstimmung_EU', 'GfK', 
               'Schock_Nachfrage_US', 'Schock_Nachfrage_EU', 
               'PMI_Manufacturing_US', 'PMI_Service_US', 
               'PMI_Manufacturing_EU', 'PMI_Service_EU', 
               'Schock_Produktion_US', 'Schock_Produktion_DE', 
               'JoblessInitial_US', 'UnemploymentRate_US', 
               'UnemploymentRate_EU', 'Schock_Arbeitsmarkt_US', 
               'Schock_Arbeitsmarkt_EU', 'GDP_US', 'GDP_EU', 
               'TradeBalance_US', 'TradeBalance_EU', 
               'Schock_Konjunktur_US', 'Schock_Konjunktur_EU']

    

#basepath = '/home/chitlom/data/'
basepath = '/Users/uwe.muller/Hope/data/'
path = basepath + 'final/EURUSD/RL-data/'
tpath = basepath + 'test/'
xpath = basepath + 'final/EURUSD/X/'
xb = np.load(xpath + 'x-2008-12-18.npy', allow_pickle=True)
x = np.array([])
for i in range(xb.shape[1]-1, 215):
    print(i,xb[i, 71, :].shape)
    x = np.append(x, xb[i, 71, :])
print(x.shape)

"""
files = [f for f in listdir(path) if isfile(join(path, f)) and f[:4] == 'day-']
files.sort()
#print(files)
find = [f[4:14] for f in files]
#print(find)
days = np.random.choice(range(len(files)), size=100, replace=False)

for f in files[:20]:
    data = np.load(path + f, allow_pickle=True)
    print(f, data[0], data[1][0])
    
    for i in range(batch.shape[0]):
        df_x = pd.DataFrame(batch[i], columns=featureList)
        df_x.to_csv(tpath + 'x' + str(i) + '.csv')
    """
        
