#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import datetime as dt
import time
import psycopg2
import sys



def confDT(row):
    
    try:
        return dt.datetime.strptime(row[:19], '%d.%m.%Y %H:%M:%S')
    except:
        print(row[:19])



def log(msg):
    global currency
    
    real_dt = time.strftime('%Y-%m-%d %H:%M:%S', dt.datetime.now())
    print(real_dt + ': ' + msg)
    sys.stdout.flush()



def update_db(forex_values):
    
    insert_sql = """INSERT INTO forex_raw (currency, date_time, open, low, high, close, cnt_diff_vals) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(insert_sql, forex_values)
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))




basepath = '/home/chitlom/'
#basepath = '/Users/uwe.muller/Hope/'
path = basepath + 'data/forex/duka/'

df = pd.read_csv(path + 'EURUSD-1Min-01.06.2021.csv')
df.rename(columns={'Local time': 'Time'}, inplace=True)
df['Time'] = df['Time'].apply(confDT)
df.set_index(pd.DatetimeIndex(df.Time), inplace=True)

print(len(df))

df5Min = df.resample('5T', label='right').agg({'Open': 'first',
                                               'Close': 'last',
                                               'Low': 'min',
                                               'High': 'max'})


df5Min['Time'] = df5Min.index.strftime('%Y-%m-%d %H:%M')
print(df5Min)

i = 0
for index, row in df5Min.iterrows():
    forex_values = ('EUR/USD', row[4], row[0], row[2], row[3], row[1], 333)
    update_db(forex_values)
    i += 1

log('%4i values added to forex_raw for EUR/USD' % (i))
