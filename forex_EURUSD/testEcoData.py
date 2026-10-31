#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:07:49 2026

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random



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
    # Entfernen der %-Zeichen und Umwandeln nach float
    #
    row = row.replace('.', '')
    row = float(row.strip('%KMB').replace(',', '.'))

    return row



def missingEcoData(row):
    #
    # apply-Aufruf mit axis=1
    # Fehlende Werte im Forecast werden mit dem Actual befüllt
    #
    if pd.isna(row['Forecast'] or row['Forecast'].strip() == ''):
        row['Forecast'] = row['Actual']

    return row



def readAndCleanEcoFeatureFile(path, fn):
    #
    # Liest und bereinigt die Eco Feature Date
    #
    df = pd.read_csv(path + fn,
                     dtype={'Actual': str,
                            'Forecast': str,
                            'Previous': str})
    df.rename({'Release Date': 'Date'}, inplace=True, axis=1)
    #
    # Bereinigen der Daten und Auffüllen fehlender Werte
    df['Date'] = df['Date'].apply(cleanEcoDate)
    if ('Forecast' in df.columns):
        df = df.apply(missingEcoData, axis=1)
        df['Forecast'] = df['Forecast'].apply(cleanEcoPct)
    if ('Previous' in df.columns):
        df['Previous'] = df['Previous'].apply(cleanEcoPct)
    df['Actual'] = df['Actual'].apply(cleanEcoPct)
    #
    # Zusammenfassen von Date und Time
    df['ttime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.strip(), dayfirst=True)
    #
    # Zeit-Index Festlegen und sortieren
    df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
    df.drop(['ttime'], axis=1, inplace=True)
    df = df.sort_index()
    #
    # Umwandeln von Europe/Berlin auf utc - 1 Stunde abziehen
    df.index = df.index.tz_localize('Europe/Berlin', ambiguous='infer').tz_convert('UTC')

    return df



def processNormalEcoFeature(path, df_eco, ecoFeatureName, fn, delta):
    #
    # Verarbeitung eines normalen Eco-Features und Anhängen an den DataFrame
    #
    df = readAndCleanEcoFeatureFile(path, fn)
    #
    # Benamung des Eco Features
    df.rename({'Actual': ecoFeatureName}, inplace=True, axis=1)
    #
    # Löschen der ueberfluessigen Spalten
    for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
        if (s in df.columns):
            df.drop(s, axis=1, inplace=True)
    #
    # Verarbeitung der Deltas
    dd = [int(i) for i in delta.split(',')]
    if (dd[0] > 0):
        for d in dd:
            delta_name = 'delta' + str(d) + '_' + ecoFeatureName
            df[delta_name] = ((df[ecoFeatureName] \
                               - df[ecoFeatureName].shift(d)).div(df[ecoFeatureName].replace(0, np.nan))).round(3)
            df[delta_name] = df[delta_name].fillna(0)

            df.dropna(inplace=True)
    #
    # Verknüpfen mit dem Eco Features dataframe
    if (len(df_eco) > 0):
        df_eco = pd.merge_ordered(df_eco.reset_index(),
                                  df.reset_index(),
                                  on='time',
                                  how='outer',
                                  fill_method='ffill').set_index('time')
    else:
        df_eco = pd.concat([df_eco, df], axis=1, sort=True)
        df_eco = df_eco.ffill()

    return df_eco



def processSpreadFeature(path, df_eco, ecoFeatureName, fn, delta):
    #
    # Verarbeitung eines Spread Features - spread zwischen zwei unterschiedlichen Eco Feature
    #
    fn1 = fn.split(',')[0].strip()
    fn2 = fn.split(',')[1].strip()
    #
    # Einlesen der beiden Dateien
    df1 = readAndCleanEcoFeatureFile(path, fn1, )
    for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
        if (s in df1.columns):
            df1.drop(s, axis=1, inplace=True)

    df2 = readAndCleanEcoFeatureFile(path, fn2)
    for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
        if (s in df2.columns):
            df2.drop(s, axis = 1, inplace = True)

    #
    # da nicht alle Daten den gleichen Zeitstempel haben, müssen wir die beiden dataframes mit merge order verbinden
    df = pd.merge_ordered(df1, df2, on='time', how='outer', fill_method='ffill', suffixes=('_1', '_2'))
    #
    # Index auf datetime
    df.set_index(pd.DatetimeIndex(df.time), inplace=True)
    #
    # berechnen des spread
    df[ecoFeatureName] = (df['Actual_1'] - df['Actual_2']).round(3)
    #
    # Bereinigen des Ergebnisses
    df.drop(['time', 'Actual_1', 'Actual_2'], axis=1, inplace=True)
    df.dropna(inplace=True)
    #
    # Verarbeitung der Deltas
    dd = [int(i) for i in delta.split(',')]
    if (dd[0] > 0):
        for d in dd:
            delta_name = 'delta' + str(d) + '_' + ecoFeatureName
            df[delta_name] = ((df[ecoFeatureName] \
                               - df[ecoFeatureName].shift(d)).div(df[ecoFeatureName].replace(0, np.nan))).round(3)
            df[delta_name] = df[delta_name].fillna(0)
            df.dropna(inplace=True)
    #
    # Verknüpfen mit dem Eco Features dataframe
    if (len(df_eco) > 0):
        df_eco = pd.merge_ordered(df_eco.reset_index(),
                                  df.reset_index(),
                                  on='time',
                                  how='outer',
                                  fill_method='ffill').set_index('time')
    else:
        df_eco = pd.concat([df_eco, df], axis=1, sort=True)
        df_eco = df_eco.ffill()

    return df_eco



def processSchockFeature(path, df_eco, ecoSchockName, fn, schockMeanStd):
    #
    # Verarbeitung eines Schock-Features
    #
    fn = fn.strip()
    df = readAndCleanEcoFeatureFile(path, fn)
    #
    # Berechne den Schock und löschen aller unnötigen Spalten
    df['schock'] = ((df['Actual'] - df['Forecast'])).round(3)
    for s in ['Date', 'Time', 'Forecast', 'Previous', 'Actual']:
        if (s in df.columns):
            df.drop(s, axis=1, inplace=True)
    #
    # calculate Mean and STD for standardization
    s_mean = df['schock'].mean()
    s_std = df['schock'].std()
    if (s_std == 0):
        s_std = 1
    df['schock'] = (df['schock'] - s_mean)/s_std
    schockMeanStd.append([fn, s_mean, s_std])
    #
    # Verknüpfen mit dem Eco Features dataframe
    if (len(df_eco) > 0):
        df_eco = pd.merge_ordered(df_eco.reset_index(),
                                  df.reset_index(),
                                  on='time',
                                  how='outer').set_index('time')
    else:
        df_eco = pd.concat([df_eco, df], axis=1, sort=True)



    #
    # Addieren des Schocks - erst prüfen, ob es den ecoSchock bereits gibt
    if (ecoSchockName not in df_eco.columns):
        df_eco[ecoSchockName] = 0
    #
    # nach dem merge_ordered haben wir etliche nan-werte, die in den schocks mit 0 befüült werden müssen
    for c in df_eco.columns:
        if (c.split('_')[0] == 'schock'):
            df_eco[c] = df_eco[c].fillna(0)
        else:
            df_eco[c] = df_eco[c].ffill()

    # für alle restlichen eco-features werden die Lücken mit ffill befüllt



    df_eco[ecoSchockName] = (df_eco[ecoSchockName] + df_eco['schock']).round(3)
    df_eco.drop(['schock'], axis=1, inplace=True)

    return df_eco



path = '/home/uwe/Hope/data/processed/EURUSD/kalender/neu/'
fn = 'kalender-features.csv'

df = pd.read_csv(path + fn,
                 dtype={'Feature': str,
                        'Datei': str,
                        'Delta': str})

ecoFeatures = df.values.tolist()
print(ecoFeatures)
df_eco = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
schockMeanStd = []
for ef in ecoFeatures:
    print(ef)
    if (ef[0].split('_')[0] == 'spread'):
        df_eco = processSpreadFeature(path, df_eco, ef[0].strip(), ef[1].strip(), ef[2].strip())
    elif (ef[0].split('_')[0] == 'schock'):
        df_eco = processSchockFeature(path, df_eco, ef[0].strip(), ef[1].strip(), schockMeanStd)
    else:
        df_eco = processNormalEcoFeature(path, df_eco, ef[0].strip(), ef[1].strip(), ef[2].strip())

pd.DataFrame(schockMeanStd, columns=['schock', 'mean', 'std']).to_csv(path + 'mean-std-schock.csv')
df_eco.to_csv(path + 'test.csv')