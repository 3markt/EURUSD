#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import psycopg2
import datetime as dt
import time
import sys
import ta


def log(msg):
    global currency
    
    real_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
    print(real_dt + ' ' + currency + ': ' + msg)
    sys.stdout.flush()



def select_min5(curr):
    
    select_sql = "select date_time, close from forex_raw where currency='{}' \
                  order by date_time desc limit 4320" \
        .format(curr)
    
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(select_sql)
        
        df = pd.DataFrame(db_cursor.fetchall(), columns=['time', 'Close'])
        df = df.sort_values(by='time')
        df.reset_index(drop=True, inplace=True)
        
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))

    finally:
        # closing database connection.
        if db_connect:
            db_cursor.close()
            db_connect.close()

    return df



def createTechInd(df):

    df['scaledClose'] = df['Close']
    
    df['WL'] = 10000*(df['Close'] - df['Close'].shift(1))/df['Close']
    
    #df['h'] = df['time'].apply(lambda x:float(x[11:13]))
    #df['min'] = df['time'].apply(lambda x:float(x[14:16]))
    
    #df['zeitIndex'] = df['h'] + df['min']/60
    df['zeitIndex'] = df.index.hour + df.index.minute/60
   
    df['round_lot'] = 10000*(df['Close'] - df['Close'].round(decimals=2))
    
    BB = ta.volatility.BollingerBands(df['Close'], 
                                      window = 10, 
                                      window_dev = 2)
    df['bb_h'] = (BB.bollinger_hband() - df['Close'])/df['Close']
    df['bb_l'] = (BB.bollinger_lband() - df['Close'])/df['Close']
    
    MACD = ta.trend.MACD(df['Close'],
                         window_slow = 26, 
                         window_fast = 12, 
                         window_sign = 9)
    df['macd'] = MACD.macd()
    df['macd_d'] = MACD.macd_diff()

    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df = df.dropna()

    return df
    


def delete_from_table(tbl, currency):

    if (tbl == '5min'):
        delete_sql = """DELETE FROM min5_candle WHERE currency = %s""" 
    elif (tbl == '30min'):
        delete_sql = """DELETE FROM min30_candle WHERE currency = %s""" 
    elif (tbl == '6h'):
        delete_sql = """DELETE FROM h6_candle WHERE currency = %s""" 
    else:
        sys.exit("Flascher Tabellenname")
        
        
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(delete_sql, (currency, ))
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error deleteing from {} dreimarktDB Error: ' + str(error))

    finally:
        # closing database connection.
        if db_connect:
            db_cursor.close()
            db_connect.close()


    
def insert_into_table(tbl, values):
    
    if (tbl == '5min'):
        insert_sql = """INSERT INTO min5_candle 
                    (currency, date_time, close, scaled_close, wl, zeit_index,
                     round_lot, bb_h, bb_l, macd, macd_d, rsi) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    elif (tbl == '30min'):
        insert_sql = """INSERT INTO min30_candle 
                    (currency, date_time, wl, bb_h, bb_l, macd, macd_d, rsi) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
    elif (tbl == '6h'):
        insert_sql = """INSERT INTO h6_candle 
                    (currency, date_time, wl, bb_h, bb_l, macd, macd_d, rsi) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""    
    else:
        sys.exit("Flascher Tabellenname")
                
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(insert_sql, values)
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))

    finally:
        # closing database connection.
        if db_connect:
            db_cursor.close()
            db_connect.close()


def convertCandle(df, freq):
    
    dfs = df.resample(freq, label='right', closed='right')
    df_new = dfs['Close'].last().to_frame()
    df_new = df_new.dropna()
    
    return df_new
   


currency = 'EURUSD'
path = '/home/chitlom/data/final/EURUSD/'
df = select_min5(currency)
df['time'] = pd.to_datetime(df['time'])
df.to_csv(path + 'df_vorShift.csv')
#
# Verschieben der Eröffnungstransaktionen vom Sonntag auf den Freitag
df.loc[df['time'].dt.dayofweek == 6, 'time'] = df['time'] - pd.to_timedelta(2, 'D')
df.to_csv(path + 'df_nachShift.csv')
#
# Beim Wechsel von Sommer- auf Winterzeit gibt es Überschneidungen,
# die gelöscht werden müssen
df.drop_duplicates(subset=['time'], 
                   keep='first', 
                   inplace=True, 
                   ignore_index=True)
#
# Sortieren und Date-Timeindex setzen
df = df.sort_values(by='time')
df.set_index(pd.DatetimeIndex(df.time, name='time'), inplace=True)
df.drop('time', axis=1, inplace=True)
end_dt = df.index[-1]
print(end_dt)
#
# Auffüllen von fehlenden Werte: aufgrund vom Wechsel von Winter-
# auf Sommerzeit (es fehlt dann 1 Stunde), oder aufgrund von System-
# ausfällen
df = df.asfreq('5min', method='ffill')
#
# Löschen der Samstage und Sonntage
df = df[(df.index.weekday != 5)]
df = df[(df.index.weekday != 6)]
print(len(df))

df = createTechInd(df)
print(len(df))

df_min30 = convertCandle(df, '30min')
df_min30.drop(df_min30[df_min30.index.weekday == 5].index, inplace=True)
df_min30.to_csv(path + 'df_30min.csv')
print(len(df_min30))
df_min30 = createTechInd(df_min30)
print(len(df_min30))
end30_dt = df_min30.index[-1]
print(end30_dt)
if (end30_dt > end_dt):
    df_min30.drop(df_min30.tail(1).index,inplace=True)
df_min30.reset_index(inplace=True)

df_h6 = convertCandle(df, '6h')
df_h6.drop(df_h6[df_h6.index.weekday == 5].index, inplace=True)
df_h6.to_csv(path + 'df_h6.csv')
print(len(df_h6))
df_h6 = createTechInd(df_h6)
print(len(df_h6))
end6_dt = df_h6.index[-1]
print(end6_dt)
if (end6_dt > end_dt):
    df_h6.drop(df_h6.tail(1).index,inplace=True)
df_h6.reset_index(inplace=True)
df.reset_index(inplace=True)

#
# Bestimmen des kleinsten verfügbaren Datums
mdate = df_h6['time'].min().date() + dt.timedelta(days=1)
min_date = np.datetime64(mdate)

#
# Begrenzen auf das kleinste verfügbare Datum
df = df[df['time'] >= min_date]
df_min30 = df_min30[df_min30['time'] >= min_date]
df_h6 = df_h6[df_h6['time'] >= min_date]

#
# Löschen der bisherigen min5-Einträge
delete_from_table('5min', currency)
# ... und hinzufügen der neuen min5-Einträge
for index, row in df.iterrows():
    insert_into_table('5min', 
                      (currency,
                       row['time'].strftime('%Y-%m-%d %H:%M'),
                       row['Close'],
                       row['scaledClose'],
                       row['WL'],
                       row['zeitIndex'],
                       row['round_lot'],
                       row['bb_h'],
                       row['bb_l'],
                       row['macd'],
                       row['macd_d'],
                       row['rsi']))
    

#
# Löschen der bisherigen min30-Einträge
delete_from_table('30min', currency)
# ... und hinzufügen der neuen min30-Einträge
for index, row in df_min30.iterrows():
    insert_into_table('30min',
                      (currency,
                       row['time'].strftime('%Y-%m-%d %H:%M'),
                       row['WL'],
                       row['bb_h'],
                       row['bb_l'],
                       row['macd'],
                       row['macd_d'],
                       row['rsi']))
    
#
# Löschen der bisherigen h6-Einträge
delete_from_table('6h', currency)
# ... und hinzufügen der neuen h6-Einträge
for index, row in df_h6.iterrows():
    insert_into_table('6h',
                      (currency,
                       row['time'].strftime('%Y-%m-%d %H:%M'),
                       row['WL'],
                       row['bb_h'],
                       row['bb_l'],
                       row['macd'],
                       row['macd_d'],
                       row['rsi']))
 

