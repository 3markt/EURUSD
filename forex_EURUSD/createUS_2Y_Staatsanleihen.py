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
fn = 'US-10y-Staatsanleihen-DGS10_1.csv'

df = pd.read_csv(path + fn, dtype={'Release Date': str, 'Actual': str})
df = df[~df['Actual'].isna()]
df['Actual'] = df['Actual'].apply(makeDeNum)
df['ttime'] = pd.to_datetime(df['Release Date'] + ' 22:15:00').dt.tz_localize('Europe/Berlin')
df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
df.drop(['ttime'], axis=1, inplace=True)
df['Release Date'] = df.index.strftime('%d.%m.%Y')
df['Time'] = df.index.strftime("%H:%M")
df.to_csv(path + 'US-10y-Staatsanleihen-DGS10.csv', index=False)