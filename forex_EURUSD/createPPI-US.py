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
    if (len(row.split('.')) == 1):
        row = row + '.0'
    row = row.replace('.', ',')

    return row



def missingEcoData(row):
    #
    # apply-Aufruf mit axis=1
    # Fehlende Werte im Forecast werden mit dem Actual befüllt
    #
    if pd.isna(row['Forecast'] or row['Forecast'].strip() == ''):
        row['Forecast'] = row['Actual']

    return row




path = '/home/uwe/Hope/data/processed/EURUSD/kalender/neu/'
fn = 'PPI-US-V1.csv'

df = pd.read_csv(path + fn, dtype={'Actual': str,
                                   'Forecast': str,
                                   'Previous': str})

df['ttime'] = pd.to_datetime(df['Release Date'] + ' ' + df['Time'])
df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
df = df.apply(missingEcoData, axis=1)
df['Actual'] = df['Actual'].apply(makeDeNum)
df['Forecast'] = df['Forecast'].apply(makeDeNum)
df['Previous'] = df['Previous'].apply(makeDeNum)
print(df)

df['Release Date'] = df.index.strftime('%d.%m.%Y')
df['Time'] = df.index.strftime("%H:%M")
df.drop(['ttime'], axis=1, inplace=True)
df.to_csv(path + 'PPI-US.csv', index=False)