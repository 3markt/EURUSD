#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import datetime as dt
import ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


ecoCols = ['CPI_US', 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
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

path = '/Users/uwe.muller/Hope/data/processed/EURUSD/forexCandle/'

df = pd.read_csv(path + 'abtEcoDataRF_all.csv')

df_x = df[ecoCols]
df_y = df['TT36']

x_train, x_test, y_train, y_test = train_test_split(df_x,
                                                    df_y,
                                                    test_size = 0.25,
                                                    train_size =0.75)
print(len(x_train), len(x_test))

model = RandomForestRegressor(n_estimators=100,
                              min_samples_leaf=10)

fitted = model.fit(x_train, y_train)
scored_train = model.score(x_train, y_train)
scored_test = model.score(x_test, y_test)
print(scored_train, scored_test)
for e in ecoCols:
    print(e, round(model.feature_importances_[ecoCols.index(e)], 4))

