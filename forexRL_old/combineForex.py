#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler



def createCnt(df):
    
    cnt= df.set_index(pd.DatetimeIndex(df.Time))
    cnt = pd.DataFrame(df['Time'].groupby(cnt.index.date).count())
    cnt = cnt.rename(columns={'Time':'Count'})
    cnt['Time'] = cnt.index.copy()
    cnt['Time'] = pd.to_datetime(cnt['Time'])
    cnt.reset_index(drop=True, inplace=True)
    cnt = cnt[['Time', 'Count']]
    return cnt



bt = 252
fext = '-202105.csv'


cols = ['Time', 'Open', 'High', 'Low', 'Close', 'WL',
        'macd', 'macd_d', 'bb_h', 'bb_m', 'bb_l', 'rsi', 'atr', 
        'keltner_h', 'keltner_m', 'keltner_l',
        'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_m_15Min', 'bb_l_15Min', 'rsi_15Min', 'atr_15Min',
        'keltner_h_15Min', 'keltner_m_15Min', 'keltner_l_15Min',
        'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_m_1H', 'bb_l_1H', 'rsi_1H', 'atr_1H',
        'keltner_h_1H', 'keltner_m_1H', 'keltner_l_1H',
        'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_m_4H', 'bb_l_4H', 'rsi_4H', 'atr_4H',
        'keltner_h_4H', 'keltner_m_4H', 'keltner_l_4H'
        ]

#start_date = '2009-05-25 00:00:00'

basepath = '/Users/uwe.muller/Hope/'

abt5Min = pd.read_csv(basepath + 'data/abt/eurusd5Min' + fext,
                      parse_dates=['Time'], infer_datetime_format=True)
#abt5Min = abt5Min.loc[abt5Min['Time'] >= start_date]

abt15Min = pd.read_csv(basepath + 'data/abt/eurusd15Min' + fext,
                    parse_dates=['Time'], infer_datetime_format=True)
#abt15Min = abt15Min.loc[abt15Min['Time'] >= start_date]

abt1H = pd.read_csv(basepath + 'data/abt/eurusd1H' + fext,
                    parse_dates=['Time'], infer_datetime_format=True)
#abt1H = abt1H.loc[abt1H['Time'] >= start_date]

abt4H = pd.read_csv(basepath + 'data/abt/eurusd4H' + fext,
                    parse_dates=['Time'], infer_datetime_format=True)
#abt4H = abt4H.loc[abt4H['Time'] >= start_date]

abt = pd.merge_ordered(abt5Min, abt15Min, 
                       how='left', on='Time', 
                       suffixes=('', '_15Min'))

abt = pd.merge_ordered(abt, abt1H, 
                       how='left', on='Time', 
                       suffixes=('', '_1H'))

abt = pd.merge_ordered(abt, abt4H, 
                       how='left', on='Time', 
                       suffixes=('', '_4H'))

abt = abt.fillna(method='ffill')
abt = abt.loc[abt['Time'].dt.strftime('%H:%M:%S') <= '21:00:00']
abt = abt.loc[abt['Time'].dt.strftime('%H:%M:%S') > '00:00:00']
print(len(abt))
abt = abt.dropna()
n = len(abt)
print(n, n/bt)

cnt = createCnt(abt)
delListDF = cnt.loc[cnt['Count'] != bt]
delListDF = delListDF['Time']
delListDF = delListDF.apply(lambda x: x.strftime("%Y-%m-%d"))
delList = delListDF.values.tolist()
print(len(delList))
print(delList)
abt['Date'] = abt['Time'].apply(lambda x: x.strftime("%Y-%m-%d"))

abt = abt.loc[~abt['Date'].isin(delList)]
n = len(abt)
print(n, n/bt)

"""
print('Gesamt:', len(abt))
outabt = abt.loc[(abt['Time'] >= '2009-05-11 00:00:00') & (abt['Time'] < '2012-01-01 00:00:00')]
print('Bis 2011:', len(outabt))
outabt[cols].to_csv(basepath + 'data/abt/eurusdFeatures.csv', index=False)

outabt = abt.loc[(abt['Time'] >= '2012-01-01 00:00:00') & (abt['Time'] < '2014-01-01 00:00:00')]
print('Bis 2013:', len(outabt))
outabt[cols].to_csv(basepath + 'data/abt/eurusdFeatures2.csv', index=False)

outabt = abt.loc[(abt['Time'] >= '2014-01-01 00:00:00') & (abt['Time'] < '2016-01-01 00:00:00')]
print('Bis 2015:', len(outabt))
outabt[cols].to_csv(basepath + 'data/abt/eurusdFeatures3.csv', index=False)

outabt = abt.loc[ (abt['Time'] >= '2016-01-01 00:00:00') & (abt['Time'] < '2018-01-01 00:00:00')]
print('Bis 2017:', len(outabt))
outabt[cols].to_csv(basepath + 'data/abt/eurusdFeatures4.csv', index=False)

outabt = abt.loc[(abt['Time'] >= '2018-01-01 00:00:00') & (abt['Time'] < '2020-01-01 00:00:00')]
print('Bis 2019:', len(outabt))
outabt[cols].to_csv(basepath + 'data/abt/eurusdFeatures5.csv', index=False)
"""

outabt = abt.loc[(abt['Time'] >= '2020-01-01 00:00:00')]
print('N:', len(outabt))
outabt[cols].to_csv(basepath + 'data/abt/eurusdFeatures' + fext, index=False)
