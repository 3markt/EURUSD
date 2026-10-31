#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:07:49 2026

@author: uwe.mueller
"""
import pandas as pd
import numpy as np



def cleanDeNum(row):
    #
    # apply-aufruf mit axis=0
    # Entfernen der Klammern im Datum
    #
    row = float(row.replace(',', '.'))

    return row



def makeDeNum(row):
    #
    # apply-aufruf mit axis=0
    # umwandeln vom internationalen ins deutsche zahlenformat
    #
    row = row.replace('.', ',')

    return row



path = '/home/uwe/Hope/data/processed/EURUSD/kalender/neu/'
fn0 = 'EU-Euribor Futures Historische Daten.csv'
fn1 = 'EU-Euribor Futures Historische Daten1.csv'
fn2 = 'EU-Euro-STR Futures.csv'

df0 = pd.read_csv(path + fn0)
df0['Zuletzt'] = df0['Zuletzt'].apply(cleanDeNum)
df0['Actual'] = (100 - df0['Zuletzt']).round(3)
#
# Behandlung der Uhrzeit in Abhängigkeit von Winter- und Sommerzeit
#
# 1. Sicherstellen, dass die Spalte ein echtes Datum ist
df0['Datum'] = pd.to_datetime(df0['Datum'], dayfirst=True)
df0['ttime'] = pd.to_datetime(df0['Datum'].dt.strftime('%Y-%m-%d') + " 17:30:00").dt.tz_localize('Europe/Berlin')
df0.set_index(pd.DatetimeIndex(df0.ttime, name='time'), inplace=True)
# die Daten stehen für alle bereits um 17:30 Uhr für denselben Tag zur Verfügung
#df0.index = df0.index + pd.Timedelta(days=1)
df0.drop(['Datum', 'ttime', 'Zuletzt', 'Eröffn.', 'Hoch', 'Tief', 'Vol.', '+/- %'], axis=1, inplace=True)
df0 = df0.sort_index()
print(df0)

df1 = pd.read_csv(path + fn1)
df1['Zuletzt'] = df1['Zuletzt'].apply(cleanDeNum)
df1['Actual'] = (100 - df1['Zuletzt']).round(3)
#
# Behandlung der Uhrzeit in Abhängigkeit von Winter- und Sommerzeit
#
# 1. Sicherstellen, dass die Spalte ein echtes Datum ist
df1['Datum'] = pd.to_datetime(df1['Datum'], dayfirst=True)
df1['ttime'] = pd.to_datetime(df1['Datum'].dt.strftime('%Y-%m-%d') + " 17:30:00").dt.tz_localize('Europe/Berlin')
df1.set_index(pd.DatetimeIndex(df1.ttime, name='time'), inplace=True)
# die Daten stehen für alle bereits um 17:30 Uhr für denselben Tag zur Verfügung
#df1.index = df1.index + pd.Timedelta(days=1)
df1.drop(['Datum', 'ttime', 'Zuletzt', 'Eröffn.', 'Hoch', 'Tief', 'Vol.', '+/- %'], axis=1, inplace=True)
df1 = df1.sort_index()
print(df1)

df2 = pd.read_csv(path + fn2)
df2['Actual'] = (100 - df2['Price']).round(3)
#
# Behandlung der Uhrzeit in Abhängigkeit von Winter- und Sommerzeit
#
# 1. Sicherstellen, dass die Spalte ein echtes Datum ist
df2['Date'] = pd.to_datetime(df2['Date'])
df2['ttime'] = pd.to_datetime(df2['Date'].dt.strftime('%Y-%m-%d') + " 17:30:00").dt.tz_localize('Europe/Berlin')
df2.set_index(pd.DatetimeIndex(df2.ttime, name='time'), inplace=True)
# die Daten stehen für alle bereits um 17:30 Uhr für denselben Tag zur Verfügung
#df2.index = df2.index + pd.Timedelta(days=1)
df2.drop(['Date', 'ttime', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1, inplace=True)
df2 = df2.sort_index()
print(df2)

df = pd.concat([df0, df1, df2], axis=0).sort_index()
df['Release Date'] = df.index.strftime('%d.%m.%Y')
df['Time'] = df.index.strftime("%H:%M")
df["Actual"] = df["Actual"].astype(str)
df['Actual'] = df['Actual'].apply(makeDeNum)
df.to_csv(path + 'Zinserwartungen-3M-EU.csv', index=False)