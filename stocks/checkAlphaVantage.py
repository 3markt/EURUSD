#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
indexName = 'dax.csv'

df = pd.read_csv(path + indexName, sep=';')

symbls = df['Symbl'].values.tolist() 
#startDate = df['StartDate'].values.tolist() 
#endDate = df['EndDate'].values.tolist() 
    
#startDate = [dt.datetime.strptime(x, '%Y-%m-%d')+relativedelta(months=-6) for x in startDate]
#endDate = [dt.datetime.strptime(x, '%Y-%m-%d')+relativedelta(months=-3) for x in endDate]

l = 0
for symbl in symbls:
    df = pd.read_csv(path + 'tmp/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
#    df = df.loc[(df['date'] > startDate[symbls.index(symbl)]) & (df['date'] < endDate[symbls.index(symbl)])]
    df = df.reset_index(drop=True)
    l += len(df)
    prevDate = df['date'].loc[0]
    cnt = 0
    for index, row in df.iterrows():
        diffDays = prevDate - row['date']
        if (diffDays > dt.timedelta(3)):
            cnt += 1
#            print(symbl, diffDays, row['date'], prevDate)
        prevDate = row['date']
    print(symbl, cnt, len(df))
print(l) 