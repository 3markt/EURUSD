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
    # umwandeln vom deutschen ins internationale zahlenformat
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
fn0 = 'Brent Öl Futures Historische Daten1.csv'
fn1 = 'Brent Öl Futures Historische Daten2.csv'

df0 = pd.read_csv(path + fn0)
df0['ttime'] = pd.to_datetime(df0['Datum'] + ' 06:00:00', dayfirst=True).dt.tz_localize('Europe/Berlin')
df0.set_index(pd.DatetimeIndex(df0.ttime, name='time'), inplace=True)
# es muss ein tag dazu addiert werden, da die Daten für den Vortag gelten und erst am nächsten Tag veröffentlicht werden
df0.index = df0.index + pd.Timedelta(days=1)
df0.drop(['Datum', 'ttime', 'Eröffn.', 'Hoch', 'Tief', 'Vol.', '+/- %'], axis=1, inplace=True)
df0 = df0.sort_index()
print(df0)

df1 = pd.read_csv(path + fn1)
df1['ttime'] = pd.to_datetime(df1['Datum'] + ' 06:00:00', dayfirst=True).dt.tz_localize('Europe/Berlin')
df1.set_index(pd.DatetimeIndex(df1.ttime, name='time'), inplace=True)
# es muss ein tag dazu addiert werden, da die Daten für den Vortag gelten und erst am nächsten Tag veröffentlicht werden
df1.index = df1.index + pd.Timedelta(days=1)
df1.drop(['Datum', 'ttime', 'Eröffn.', 'Hoch', 'Tief', 'Vol.', '+/- %'], axis=1, inplace=True)
df1 = df1.sort_index()
print(df1)

df = pd.concat([df0, df1], axis=0).sort_index()
df['Release Date'] = df.index.strftime('%d.%m.%Y')
df['Time'] = df.index.strftime("%H:%M")
df.rename({'Zuletzt': 'Actual'}, axis=1, inplace=True)
df["Actual"] = df["Actual"].astype(str)
df.to_csv(path + 'Brent-Oil-Futures-Daily.csv', index=False)