#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd



path = '/Users/uwe.muller/Hope/data/processed/EURUSD/forexCandle/'
file = 'df_all_scaled.csv'

df_all= pd.read_csv(path + file, 
                    parse_dates=['time'], 
                    date_format='%Y-%m-%d %H:%M:%S')

df_all.set_index(pd.DatetimeIndex(df_all.time, name='time'), 
                 inplace=True, drop=True)

df_all = df_all['Close']
#
# Erzeugen der Tage
dff = df_all.groupby(df_all.index.date).count()
print('Anzahl Tage insgesamt:', len(dff))
df_days = dff.index
days = sorted(df_days.values.tolist())

x = []
for day in days:
    price = df_all[df_all.index.date == day].values
    diffs = 10000*np.diff(price, append=price[-1])
    
    for i in range(288):
        d = int(abs(np.random.normal(6, 12)))
        if (i+d >= 288):
            x.append(sum(diffs[i:288]) - 1.5)
        else:
            x.append(sum(diffs[i:i+d]) - 1.5)

currProfit_mean = np.mean(x)
currProfit_std = np.std(x)
print(currProfit_mean, currProfit_std)

pd.DataFrame([['Mean', currProfit_mean], ['Std', currProfit_std]], columns=['Name', 'Value']) \
    .to_csv(path + 'currentProfitMeanStd.csv', index=False)


    