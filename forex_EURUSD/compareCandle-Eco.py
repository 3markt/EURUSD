#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import pandas as pd
import datetime as dt
from os import listdir, makedirs
from os.path import isfile, join, isdir


def mergeC_E(dfc, dfe, suffix):

    df = pd.merge_ordered(dfc, dfe, 
                          how='left', on='time', 
                          suffixes=('', suffix))
    
    df = df.fillna(method='ffill')
    
    return df



def readEco(fn, suffix):
    
    df = pd.read_csv(epath + fn)
    df = df.sort_values(by='time')
    df.rename({'time':'ttime',
               'Acual':suffix + '_Actual',
               'Schock':suffix + '_Schock'}, axis=1, inplace=True)
    
    df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
    df.drop('ttime', inplace=True, axis=1)

    return df



base = '/Users/uwe.muller/Hope/Data/'
epath = base + 'processed/kalender/'
cpath = base + 'raw/forexCandle/'
apath = base + 'analysis/'

#
# Einlesen und mergen der forexCandle-Daten
#
cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f)) and \
            f[:6] == 'EURUSD']
    
print(cfiles)

df = pd.DataFrame()
for c in cfiles[1:]:
    dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')
    dfi = pd.read_csv(cpath + c, parse_dates=['Gmt time'], date_parser=dateparse)
    dfi.rename({'Gmt time':'ttime'}, inplace=True, axis=1)
    dfi = dfi[['ttime', 'Close']]
    df = df.append(dfi)

df = df.sort_values(by='ttime')
df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
df.drop('ttime', inplace=True, axis=1)
print(df)

df_ppi_de = readEco('PPI-DE.csv', 'PPI-DE')
df = mergeC_E(df, df_ppi_de, 'PPI-DE')

print(df)
df.to_csv(apath + 'analysis.csv')
