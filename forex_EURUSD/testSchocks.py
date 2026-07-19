#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
import datetime as dt
from os import listdir, makedirs
from os.path import isfile, join, isdir




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



def stdSchock(row):
    
    if (row['Actual'] >= 0.):
        if (row['Actual'] < 1.):
            row['Actual'] += 1.
            row['Forecast'] += 1.
        row['Schock'] = (row['Actual'] - row['Forecast'])/row['Actual']
    else:
        if (row['Actual'] > -1.):
            row['Actual'] -= 1.
            row['Forecast'] -= 1.
        row['Schock'] = (row['Forecast'] - row['Actual'])/row['Actual']
    
    
    return row

        

    


currency = 'EURUSD'
path = '/Users/uwe.muller/Hope/data/processed/' + currency + '/kalender/'
cpath = '/Users/uwe.muller/Hope/data/processed/' + currency + '/forexCandle/'
tpath = '/Users/uwe.muller/Hope/data/test/kalender/'

df_all = pd.read_csv(cpath + 'df_all.csv')

df_all = df_all[['time', 'Close']]
df_all.set_index(pd.DatetimeIndex(df_all.time), drop=True, inplace=True)
df_all.drop('time', axis = 1, inplace = True)
df_all['WL3'] = 10000*(df_all['Close'] - df_all['Close'].shift(3))/df_all['Close']
df_all['WL6'] = 10000*(df_all['Close'] - df_all['Close'].shift(6))/df_all['Close']
df_all['WL12'] = 10000*(df_all['Close'] - df_all['Close'].shift(9))/df_all['Close']
df_all['WL24'] = 10000*(df_all['Close'] - df_all['Close'].shift(15))/df_all['Close']
df_all['WL48'] = 10000*(df_all['Close'] - df_all['Close'].shift(24))/df_all['Close']
df_all['WL96'] = 10000*(df_all['Close'] - df_all['Close'].shift(39))/df_all['Close']
df_all = df_all.dropna()
df_all.drop('Close', inplace = True, axis = 1)

sfiles = [f for f in listdir(path) if isfile(join(path, f)) \
          and f != 'Kalender_Features.csv' and f[-4:] == '.csv']

sfiles.sort()
rr = []
for s in sfiles:
    df = pd.read_csv(path + s,
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
    # Berechnen des Schocks
    s_mean = (df['Actual'] - df['Forecast']).mean()
    s_std = (df['Actual'] - df['Forecast']).std()
    df['schock'] = (df['Actual'] - df['Forecast'] - s_mean)/s_std
    df['Schock'] = 0.
    df = df.apply(stdSchock, axis = 1)
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
    df.drop(['Date', 'Time', 'Previous', 'time', 'Actual', 'Forecast'], 
            axis=1, inplace=True)
    #
    # zur Sicherheit nochmals sortieren
    df = df.sort_values(by='time')
    #
    # 
    df = pd.merge_ordered(df, df_all,
                          how='inner', on='time')


    x = df['schock'].values
    y3 = df['WL3'].values
    r3 = np.corrcoef(x, y3)[0,1]
    y6 = df['WL6'].values
    r6 = np.corrcoef(x, y6)[0,1]
    y12 = df['WL12'].values
    r12 = np.corrcoef(x, y12)[0,1]
    y24 = df['WL24'].values
    r24 = np.corrcoef(x, y24)[0,1]
    y48 = df['WL48'].values
    r48 = np.corrcoef(x, y48)[0,1]
    y96 = df['WL96'].values
    r96 = np.corrcoef(x, y96)[0,1]

    print('%s: %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f' % \
          (s, r3, r6, r12, r24, r48, r96))
    rr.append([s, r3, r6, r12, r24, r48, r96])
    
pd.DataFrame(rr, columns = ['Schock', 'Corr3', 'Corr6', 
                            'Corr12', 'Corr24', 'Corr48',
                            'Corr96']).to_csv(tpath + 'SchockCorrelation.csv')



