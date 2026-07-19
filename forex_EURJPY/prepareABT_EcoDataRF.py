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


def cleanEcoDate(row):
    #
    # apply-aufruf mit axis=0
    # Entfernen der Klammern im Datum
    #
    row = row.split('(')[0]
    
    return row


def cleanEcoPct(row):
    #
    # apply-aufruf mit axis=0
    #Entfernen der %-Zeichen und Umwandeln nach float
    #
    row = row.replace(',', '')
    row = float(row.strip('%KMB'))
    
    return row



def missingEcoData(row):
    #
    # apply-Aufruf mit axis=1
    # Fehlende Werte im Forecast werden mit der Veraenderung gegenueber 
    # Vorperiode aufgefuellt
    #
    if pd.isna(row['Forecast'] or row['Forecast'].strip() == ''):
        row['Forecast'] = row['Previous']

    
    return row




def prepareEcoData(path, fn):
    #
    # Einlesen der Daten
    df = pd.read_csv(path + fn,
                     dtype={'Actual': str, 
                            'Forecast': str,
                            'Previous': str})
    df.rename({'Release Date':'Date'}, inplace=True, axis=1)
    #
    # Bereinigen der Daten
    df['Date'] = df['Date'].apply(cleanEcoDate)
    df = df.apply(missingEcoData, axis = 1)
    df['Actual'] = df['Actual'].apply(cleanEcoPct)
    df['Forecast'] = df['Forecast'].apply(cleanEcoPct)
    #
    # Zusammenfassen von Date und Time
    df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    #
    # Umrechung von lokaler auf GMT-Zeiten
    df['time'] = df['time'] + pd.to_timedelta(5, unit='h')
    #
    # Zeit-Index Festlegen
    df.set_index(pd.DatetimeIndex(df.time), inplace=True)
    #
    # Löschen der ueberfluessigen Spalten
    df.drop(['Date', 'Time', 'Previous', 'time'], axis=1, inplace=True)
    #
    # zur Sicherheit nochmals sortieren
    df = df.sort_values(by='time')
    
    return df
   
   
def mergeEcoFeature(df_all, df_eco, name):

    df_eco.rename({'Actual':name}, axis=1, inplace=True)
    df_eco.drop('Forecast', axis=1, inplace=True)
    df = pd.merge_ordered(df_all, df_eco,
                          how='left', on='time', 
                          suffixes=('', name))

    df.rename({'time':'ttime'}, axis=1, inplace=True)
    df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
    df.drop('ttime', axis=1, inplace=True)
    df = df.fillna(method='ffill')
    
    return df



def mergeEcoSchock(df_all, df_schock, name, fn):
    
    # calculate Mean and STD for standardization
    s_mean = (df_schock['Actual'] - df_schock['Forecast']).mean()
    s_std = (df_schock['Actual'] - df_schock['Forecast']).std()
    # merge Schock mit df_all
    df = pd.merge_ordered(df_all, df_schock, 
                          how='left', on='time', 
                          suffixes=('', name))
    df.rename({'time':'ttime'}, axis=1, inplace=True)
    df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
    #
    # Berechnen der Schocks
    df['schock'] = ((df['Actual'] - df['Forecast'] - s_mean)/s_std) \
                     .round(decimals=2)
    #
    # addiere die Schocks zum Schock-Feature
    df.loc[df['schock'].notna(), name] = df[name] + df['schock']
    #
    # behalte alle Zeilen mit einem Schock > 0
    df.loc[df['schock'].notna(), 'keep'] = 'j'    
    #
    # und zum Schluß Löschen der Hilfsspalten
    df = df.drop(['ttime', 'schock',  'Actual', 'Forecast'], axis=1)
  
    return df
    



def addEcoData(df_all):
    
    epath = '/Users/uwe.muller/Hope/data/processed/EURUSD/kalender/'
    
    df_cal = pd.read_csv(epath + 'Kalender_Features.csv')
    #
    # Feature Name und Dateiname in eine Liste
    features = df_cal[['FeatureName', 'FileName']].values.tolist()
    features = [[f[0].strip(), f[1].strip()] for f in features]
    #
    # 
    df_all['keep'] = 'n'
    #
    # Schleife über der Feature-Liste
    for feature in features:
        print('processing Feature', feature)
        if (feature[0][:7] != 'Schock_'):
            #
            # Verarbeitung der normalen Feature
            df_feature = prepareEcoData(epath, feature[1])
            df_all = mergeEcoFeature(df_all, df_feature, feature[0])
        else:
            #
            # Verarbeitung der Schocks
            fn_l = feature[1].split(',')
            df_all[feature[0]] = 0
            for fn in fn_l:
                print('processing Schock', fn)
                df_schock = prepareEcoData(epath, fn.strip())
                df_all = mergeEcoSchock(df_all,
                                        df_schock, 
                                        feature[0],
                                        fn.strip())
                
            print(feature[0], 
                  df_all[feature[0]].min(), 
                  df_all[feature[0]].max())
    #
    # Behalte alle Zeilen mit einem Schock
    df_all = df_all[df_all['keep'] == 'j']
    df_all.dropna(inplace=True, axis=0)
    df_all.drop(['Close', 'keep'], inplace=True, axis=1)

    return df_all




path = '/Users/uwe.muller/Hope/data/processed/EURUSD/forexCandle/'
c = 'df_all_scaled.csv'
df_all= pd.read_csv(path + c, 
                    parse_dates=['time'], 
                    date_format='%Y-%m-%d %H:%M:%S')

df_all = df_all[['time', 'Close']]

df_all['T3'] = 10000*(df_all['Close'].shift(-3) - df_all['Close'])/df_all['Close']

df_all['T12'] = 10000*(df_all['Close'].shift(-12) - df_all['Close'])/df_all['Close']

df_all['T36'] = 10000*(df_all['Close'].shift(-36) - df_all['Close'])/df_all['Close']

df_all['AvgOpen'] = (df_all['Close'].shift(3) + df_all['Close'].shift(2) + \
                     df_all['Close'].shift(1))/3
    
df_all['AvgClose'] = (df_all['Close'] + df_all['Close'].shift(-1) + \
                      df_all['Close'].shift(-2))/3
    
df_all['TT3'] = 10000*(df_all['AvgClose'].shift(-2) - df_all['AvgOpen'])/ \
                        df_all['AvgOpen']

df_all['TT12'] = 10000*(df_all['AvgClose'].shift(-11) - df_all['AvgOpen'])/ \
                        df_all['AvgOpen']

df_all['TT36'] = 10000*(df_all['AvgClose'].shift(-35) - df_all['AvgOpen'])/ \
                        df_all['AvgOpen']

df_all.rename({'time':'ttime'}, axis=1, inplace=True)
df_all.set_index(pd.DatetimeIndex(df_all.ttime, name='time'), 
                 inplace=True)
df_all.drop('ttime', axis=1, inplace=True)
df_all = addEcoData(df_all)

df_all.to_csv(path + 'abtEcoDataRF_all.csv')



"""
df = df_all[df_all.index < '2008-01-03']
plt.rcParams["figure.figsize"] = (12, 7)
x = df.index
plt.plot(x, df['WL'])
plt.plot(x, df['MA36'])
"""
