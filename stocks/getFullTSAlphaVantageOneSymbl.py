# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time



path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'

ts = TimeSeries(key='JW22M5RPKKUT9ZX3', output_format='pandas', indexing_type='date')

symbl = 'DB1.DEX'
print('... reading ' + symbl)

df, meta = ts.get_daily_adjusted(symbl, outputsize='full') 

df.rename(inplace = True, 
          columns={'1. open':'Open',
                   '2. high':'High',
                   '3. low':'Low',
                   '4. close':'Close',
                   '6. volume':'Volume',
                   '8. split coefficient':'Split'                   
                   })
    
df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Split']]
   


split = 1.0
for i, row in df.iterrows():
    df.at[i, 'Open'] = row['Open']/split
    df.at[i, 'High'] = row['High']/split
    df.at[i, 'Low'] = row['Low']/split
    df.at[i, 'Close'] = row['Close']/split
    df.at[i, 'Volume'] = row['Volume']*split
    prev_split = split
    split *= row['Split']*row['Split']
    if (prev_split != split):
        print('... split = ', split)
  
df.to_csv(path + 'neu/' + symbl + '.csv')
print('File ' + symbl + '.csv written to path ' + path)

