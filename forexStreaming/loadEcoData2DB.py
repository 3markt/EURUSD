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



def feature2db(values):
    
    insert_sql = """INSERT INTO eco_feature 
                (currency, date_time, name, actual, forecast) 
                VALUES (%s, %s, %s, %s, %s)"""
                
    delete_sql = """DELETE FROM eco_feature WHERE currency = %s and 
                    date_time = %s and name = %s"""      
    
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(delete_sql, (values[0], values[1], values[2], ))
        db_cursor.execute(insert_sql, values)
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))



def schock2db(values):
    
    insert_sql = """INSERT INTO eco_schock
                (currency, date_time, name, feed_name, actual, forecast) 
                VALUES (%s, %s, %s, %s, %s, %s)"""
                
    delete_sql = """DELETE FROM eco_schock WHERE currency = %s and 
                    date_time = %s and name = %s and feed_name = %s"""      
    
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(delete_sql, (values[0], 
                                       values[1], 
                                       values[2], 
                                       values[3]))
        db_cursor.execute(insert_sql, values)
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))



def cleanEcoDate(row):
    #
    # apply-aufruf mit axis=0
    # Entfernen der Klammern im Datum
    #
    row = row.split('(')[0]
    
    return row


def cleanEcoPct(row):
    #
    # apply-aufruf mit axis=0
    #Entfernen der %-Zeichen und Umwandeln nach float
    #
    row = row.replace(',', '')
    row = float(row.strip('%KMB'))
    
    return row



def missingEcoData(row):
    #
    # apply-Aufruf mit axis=1
    # Fehlende Werte im Forecast werden mit der Veraenderung gegenueber 
    # Vorperiode aufgefuellt
    #
    if pd.isna(row['Forecast'] or row['Forecast'].strip() == ''):
        row['Forecast'] = row['Previous']

    
    return row



def prepareEcoData(path, fn):
    #
    # Einlesen der Daten
    df = pd.read_csv(path + fn,
                     dtype={'Actual': str, 
                            'Forecast': str,
                            'Previous': str})
    df.rename({'Release Date':'Date'}, inplace=True, axis=1)
    #
    # Bereinigen der Daten
    df['Date'] = df['Date'].apply(cleanEcoDate)
    df = df.apply(missingEcoData, axis = 1)
    df['Actual'] = df['Actual'].apply(cleanEcoPct)
    df['Forecast'] = df['Forecast'].apply(cleanEcoPct)
    #
    # Zusammenfassen von Date und Time
    df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    #
    # Umrechung von lokaler auf GMT-Zeiten
    df['time'] = df['time'] + pd.to_timedelta(5, unit='h')
    #
    # auftunden auf die nächsten 5 Minuten
    df['time'] = df['time'].dt.ceil("5min")
    #
    # Zeit-Index Festlegen
    #df.set_index(pd.DatetimeIndex(df.time), inplace=True)
    #
    # Löschen der ueberfluessigen Spalten
    df.drop(['Date', 'Time', 'Previous'], axis=1, inplace=True)
    #
    # zur Sicherheit nochmals sortieren
    df = df.sort_values(by='time')
    
    return df
   


#basepath = '/Users/uwe.muller/Hope/data/'
basepath = '/home/chitlom/data/'
path = basepath + 'processed/EURUSD/kalender/'

dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')

df_cal = pd.read_csv(path + 'Kalender_Features.csv')
#
# Feature Name und Dateiname in eine Liste
features = df_cal[['FeatureName', 'FileName']].values.tolist()
features = [[f[0].strip(), f[1].strip()] for f in features]
#
# Schleife über der Feature-Liste
for feature in features:
    if (feature[0][:7] != 'Schock_'):
        #
        # Verarbeitung der normalen Feature
        df_feature = prepareEcoData(path, feature[1])
        print('Loading ' + feature[0])
        for row in df_feature.iterrows():
            currency = 'EURUSD'
            datetime = row[1]['time'].strftime('%Y-%m-%d %H:%M')
            name = feature[0]
            actual = row[1]['Actual']
            forecast = row[1]['Forecast']
        
            feature2db((currency, 
                        datetime, 
                        name, 
                        actual,
                        forecast))
    else:
        #
        # Verarbeitung der Schocks
        fn_l = feature[1].split(',')
        for fn in fn_l:
            print('loading Schock', fn)
            df_schock = prepareEcoData(path, fn.strip())
            for row in df_schock.iterrows():
                currency = 'EURUSD'
                datetime = row[1]['time'].strftime('%Y-%m-%d %H:%M')
                name = feature[0]
                feed_name = fn.split('.csv')[0].strip()
                actual = row[1]['Actual']
                forecast = row[1]['Forecast']
            
                schock2db((currency, 
                           datetime, 
                           name, 
                           feed_name,
                           actual,
                           forecast))
