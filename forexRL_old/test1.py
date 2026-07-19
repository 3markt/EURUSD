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


basepath = '/Users/uwe.muller/Hope/data/'
infile = basepath + 'forexCandle/EURUSD_Candlestick_5_m_BID_01.01.2022-19.02.2023.csv'
outpath= basepath + '/test/'


dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')

df5m = pd.read_csv(infile, parse_dates=['Gmt time'], date_parser=dateparse)

df5m.rename({"Gmt time":"time"}, inplace=True, axis=1)
df5m.set_index(pd.DatetimeIndex(df5m.time), inplace=True)
print(df5m)

df1h = convertTF(df5m, '1H')
print(df1h)
df1h.to_csv(basepath + 'test/EURUSD-1H-BID_01.01.2022-19.02.2023.csv')

df6h = convertTF(df5m, '6H')
print(df6h)
df6h.to_csv(basepath + 'test/EURUSD-6H-BID_01.01.2022-19.02.2023.csv')

