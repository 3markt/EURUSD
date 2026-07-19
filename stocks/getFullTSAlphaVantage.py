# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import datetime as dt



path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'

dax = pd.read_csv(path + 'daxTrain.csv', sep=';')
tecdax = pd.read_csv(path + 'tecdax.csv', sep=';')
dj = pd.read_csv(path + 'dowjones.csv', sep=';')
nasdaq = pd.read_csv(path + 'nasdaq100.csv', sep=';')

"""
symbls = dax['Symbl'].values.tolist() + tecdax['Symbl'].values.tolist() + \
         dj['Symbl'].values.tolist() + nasdaq['Symbl'].values.tolist()
"""
symbls = dax['Symbl'].values.tolist()
         
#symbls = list(set(symbls))


#symbls = ['FOX', 'LBTYA']

ts = TimeSeries(key='JW22M5RPKKUT9ZX3', output_format='pandas', indexing_type='date')

nSymbl = 0
for symbl in symbls:
    print('... reading ' + symbl)
    if (nSymbl >= 5):
        time.sleep(61)
        nSymbl = 0
    
    try:
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
        
        print(df.dtypes)
        if (symbl == 'DB1.DEX' and df.loc[dt.datetime.strptime('2007-06-11', '%Y-%m-%d'), 'Split'] == 1):
            df.loc[dt.datetime.strptime('2007-06-11', '%Y-%m-%d'), 'Split'] = 2
            
        split = 1.0
        for i, row in df.iterrows():
            df.at[i, 'Open'] = row['Open']/split
            df.at[i, 'High'] = row['High']/split
            df.at[i, 'Low'] = row['Low']/split
            df.at[i, 'Close'] = row['Close']/split
            df.at[i, 'Volume'] = row['Volume']*split
            prev_split = split
            split *= row['Split']
            if (prev_split != split):
                if (symbl == 'HEN3.DEX' or symbl == 'BAS.DEX'):
                    print('Henkel korrektur: doppelter Split')
                    split *= row['Split']
                
                print('... split = ', split)
        
        df.to_csv(path + 'neu/' + symbl + '.csv')
        print('File ' + symbl + '.csv with ' + str(len(df)) + ' records written to path ' + path)
        
        
    except:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!ERROR reading AlphaVantage for Symbl ' + symbl)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
    nSymbl += 1
    
