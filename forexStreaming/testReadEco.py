#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import psycopg2
import pandas as pd
import os
import time


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
    """
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
    """
    return df
   
    
   
path = '/Users/uwe.muller/Hope/data/processed/EURUSD/forexCandle/'

df = pd.read_csv(path + 'schockStandardize.csv')
df['name'] = df['name'].apply(lambda x:x.replace('.csv', ''))
df.set_index('name', inplace=True, drop=True)
print(df.loc['CPI_US', 'mean'])

