#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import plotly.graph_objects as go
import pandas as pd
import datetime as dt



def convertTF(df, tf):
    
    dfs = df.resample(tf, label='right', closed='left')
    #dfs = df.resample(tf, kind='timestamp')
    df_new = dfs['Open'].first().to_frame()
    df_new['High'] = dfs['High'].max()
    df_new['Low'] = dfs['Low'].min()
    df_new['Close'] = dfs['Close'].last()
    df_new = df_new.dropna()
    
    return df_new


basepath = '/Users/uwe.muller/Hope/data/raw/'
infile = basepath + 'forexCandle/EURUSD_Candlestick_5_m_BID_01.01.2022-19.02.2023.csv'
outpath= basepath + '/test/'


dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')
df5m = pd.read_csv(infile, parse_dates=['Gmt time'], date_parser=dateparse)
df5m.rename({'Gmt time':'ttime'}, inplace=True, axis=1)
#df5m.set_index(pd.DatetimeIndex(df5m.time), inplace=True)
df5m.set_index(pd.DatetimeIndex(df5m.ttime, name='time'), inplace=True)
df5m.drop('ttime', inplace=True, axis=1)
print(df5m)

df30m = convertTF(df5m, '30min')
print(df30m)
df30m.to_csv(basepath + 'test/EURUSD-30m-BID_01.01.2022-19.02.2023.csv')


df6h = convertTF(df5m, '6H')
print(df6h)
df6h.to_csv(basepath + 'test/EURUSD-6H-BID_01.01.2022-19.02.2023.csv')


dfc = pd.merge_ordered(df5m, df30m, 
                       how='left', on='time', 
                       suffixes=('', '_30Min'))

dfc = dfc.fillna(method='ffill')

dfc = pd.merge_ordered(dfc, df6h, 
                       how='left', on='time', 
                       suffixes=('', '_6h'))

dfc = dfc.fillna(method='ffill')


dfc.to_csv(basepath + 'test/EURUSD-Combined-BID_01.01.2022-19.02.2023.csv')

