#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
import joblib
from os import listdir, makedirs
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio



def cleanDate(row):
    
    row = row.split('(')[0]
    
    return row


def perc2num(row):
    
    row = float(row.strip('%'))
    
    return row

def missingForecast(row):
    
    if pd.isna(row['Forecast']):
        row['Forecast'] = row['Actual'] - row['Previous']

    return row


base = '/Users/uwe.muller/Hope/Data/'
rawpath = base + 'raw/kalender/'
procpath = base + 'processed/kalender/'

ecofiles = [f for f in listdir(rawpath) if isfile(join(rawpath, f)) and \
            f.split('.')[1] == 'csv']

print(ecofiles)

for e in ecofiles:
    df = pd.read_csv(rawpath + e, sep=';')
    df.rename({'Release Date':'Date'}, inplace=True, axis=1)
    
    # Entfernen der Klammerausdruecke aus Datum
    df['Date'] = df['Date'].apply(cleanDate)
    # Zusammensetzen von Date+Time
    df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    # Umrechung auf GMT-Zeiten
    df['time'] = df['time'] + pd.to_timedelta(5, unit='h')
    # Zeit-Index Festlegen
    df.set_index(pd.DatetimeIndex(df.time), inplace=True)
    # Prozentzeichen entfernen
    df['Actual'] = df['Actual'].apply(perc2num)
    df['Forecast'] = df['Forecast'].apply(perc2num)
    # Fehlende Werte ersetzen
    df = df.apply(missingForecast, axis=1)
    # Aus 'Forecast' wird 'Schock'
    df.rename({'Forecast':'Schock'}, axis=1, inplace=True)
    #Löschen der ueberfluessigen Spalten
    df.drop(['Date', 'Time', 'Previous', 'time'], axis=1, inplace=True)
    # zur Sicherheit nochmals sortieren
    df = df.sort_values(by='time')
    # ... und zum Schluss als csv speichern
    df.to_csv(procpath + e)
