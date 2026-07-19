#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:46:46 2018

@author: uwe.mueller
"""


import pandas as pd
import time
import sys
import psycopg2
import datetime as dt


def log(msg):
    global currency
    
    real_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
    print(real_dt + ' ' + currency + ': ' + msg)
    sys.stdout.flush()



def update_db(forex_values):
    
    insert_sql = """INSERT INTO forex_raw 
                (currency, date_time, open, low, high, close, cnt_diff_vals) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)"""
                
    delete_sql = """DELETE FROM forex_raw WHERE date_time = %s"""      
    
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(delete_sql, (forex_values[1], ))
        db_cursor.execute(insert_sql, forex_values)
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))


#basepath = '/Users/uwe.muller/Hope/data/'
basepath = '/home/chitlom/data/'
rawpath = basepath + 'raw/forexCandle/'

dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')

df = pd.read_csv(rawpath + 'EURUSD_Candlestick_5_M_BID_19.02.2023-08.04.2023.csv',
                 parse_dates=['Gmt time'], date_parser=dateparse)

df.rename({'Gmt time': 'time'}, axis=1, inplace=True)
for row in df.iterrows():
    currency = 'EURUSD'
    datetime = (row[1]['time'] + pd.Timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M')
    curr_open = row[1]['Open']
    curr_high = row[1]['High']
    curr_low = row[1]['Low']
    curr_close = row[1]['Close']
    cnt_diff_kurs = 999

    update_db((currency, 
               datetime, 
               curr_open, 
               curr_low,
               curr_high, 
               curr_close, 
               cnt_diff_kurs))
