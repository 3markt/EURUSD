# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries
import pandas as pd



path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'


#df = web.get_data_alphavantage('aapl', api_key='JW22M5RPKKUT9ZX3')
ts = TimeSeries(key='JW22M5RPKKUT9ZX3', output_format='pandas', indexing_type='date')
#ts = TimeSeries(key='JW22M5RPKKUT9ZX3')

info = ts.get_symbol_search('PMI')
df = pd.DataFrame(info[0])
print(df)
df.to_csv(path + 'symbol_info.csv', index=False)