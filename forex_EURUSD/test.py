#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd


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
    row = row.replace('.', '')
    row = float(row.strip('%KMB').replace(',', '.'))
    
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
   



path = '/Users/uwe.mueller/Hope/data/processed/EURUSD/kalender/'
fn = 'UnemploymentRate_US.csv'
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
df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
#
# Umrechung von lokaler auf GMT-Zeiten
df['time'] = df['time'] + pd.to_timedelta(-1, unit='h')
#
# Zeit-Index Festlegen
df.set_index(pd.DatetimeIndex(df.time), inplace=True)
#
# Löschen der ueberfluessigen Spalten
df.drop(['Date', 'Time', 'Previous', 'time'], axis=1, inplace=True)
#
# zur Sicherheit nochmals sortieren
df = df.sort_values(by='time')

print(df['Actual'].std())
