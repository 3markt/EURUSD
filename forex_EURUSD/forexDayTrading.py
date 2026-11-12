#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:07:49 2023

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
import ta
import sys
import gc
from os import listdir, makedirs
from os.path import isfile, join, isdir
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import gym
from gym.envs.registration import register
from gym.envs.registration import register
# import gymnasium as gym
# from gymnasium.envs.registration import register
# from gymnasium.envs.registration import register
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import pywt
from forexTrader.envs import forexEnv
from forexTrader.agent import ForexDayTrader 



class myLSTM(nn.Module):
    
    def __init__(self, inputsize):
        
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=inputsize, 
                            hidden_size=64,
                            num_layers=1, 
                            batch_first=True)
        
        self.linear = nn.Linear(64, 3)
        
        
        
    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x = self.linear(x)
        
        return x



class forexRLDayTrading:

    techCols = ['zeitIndex', 'round_lot', 'WL', 'v32_zscore', 'v64_zscore', 'we32_D1', 'we32_total', 'we64_D2',
                'we64_total', 'we128_D3', 'we128_D4', 'we128_total', 'we256_A4', 'we256_total']

    ecoCols = ['brent_oil', 'delta1_brent_oil', 'delta2_brent_oil', 'spread_3m_expected_zins',
               'delta1_spread_3m_expected_zins', 'delta2_spread_3m_expected_zins', 'us_fed_zins', 'spread_us_eu_zins',
               'delta1_spread_us_eu_zins', 'spread_us_de_2y_zins', 'delta1_spread_us_de_2y_zins',
               'spread_us_eu_10y_zins', 'delta1_spread_us_eu_10y_zins', 'spread_ppi', 'delta1_spread_ppi', 'spread_cpi',
               'delta1_spread_cpi', 'us_ism_pmi_manufact', 'delta1_us_ism_pmi_manufact', 'us_ism_pmi_service',
               'delta1_us_ism_pmi_service', 'us_unemployment', 'delta1_us_unemployment', 'us_jobless_initial',
               'delta1_us_jobless_initial', 'us_employment_change', 'delta1_us_employment_change', 'us_offene_stellen',
               'delta1_us_offene_stellen', 'eu_unemployment', 'delta1_eu_unemployment', 'us_einzelhandelsumsatz',
               'de_zew_ausblick', 'delta1_de_zew_ausblick', 'de_ifo', 'delta1_de_ifo',
               'schock_us_fed_zins', 'schock_eu_ecb_zins', 'schock_preis_us', 'schock_preis_eu',
               'schock_us_ism_pmi_manufact', 'schock_us_ism_pmi_service', 'schock_us_unemployment',
               'schock_eu_unemployment', 'schock_us_jobless_initial', 'schock_us_employment_change',
               'schock_us_offene_stellen', 'schock_us_einzelhandelsumsatz']

    scaledCols = techCols + ecoCols

    target = ['Reward-Hold', 'Reward-Buy', 'Reward-Sell']

    featureList = ['Position', 'currProfit', 'currTradeLength'] + techCols + ecoCols
    
    def __init__(self, currency):
        
        self.currency = currency
        self.basepath = '/home/chitlom/data/'
        self.rawpath = self.basepath + 'raw/'
        self.processedpath = self.basepath + 'processed/' + self.currency + '/'
        self.finalpath = self.basepath + 'final/' + self.currency + '/'
        self.schockStandardize = []
        self.bt = 288
        self.ts = 36
        self.bs = self.bt - self.ts
        self.ff = len(self.featureList)
        self.trade_fee = 1.0
        self.pip = 10000
        self.gamma = 0.75
        self.learning_rate = 0.1
        self.best_model = self.finalpath + 'model/bestModel/bestModel'
        self.last_best_upd = 0



    def cleanEcoDate(self, row):
        #
        # apply-aufruf mit axis=0
        # Entfernen der Klammern im Datum
        #
        row = row.split('(')[0]
        
        return row


    def cleanEcoPct(self, row):
        #
        # apply-aufruf mit axis=0
        #Entfernen der %-Zeichen und Umwandeln nach float
        #
        row = row.replace('.', '')
        row = float(row.strip('%KMB').replace(',', '.'))
        
        return row



    def missingEcoData(self, row):
        #
        # apply-Aufruf mit axis=1
        # Fehlende Werte im Forecast werden mit der Veraenderung gegenueber 
        # Vorperiode aufgefuellt
        #
        if pd.isna(row['Forecast'] or row['Forecast'].strip() == ''):
            row['Forecast'] = row['Previous']

        return row
    
    
        
    def createTechInd(self, df):

        df['scaledClose'] = df['Close']
        
        df['WL'] = 10000*(df['Close'] - df['Close'].shift(1))/df['Close']
        
        df['zeitIndex'] = df.index.hour + df.index.minute/60
        
        df['round_lot'] = 10000*(df['Close'] - df['Close'].round(decimals=2))

        return df



    def calculate_optimized_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet ein schlankes, gematchtes Feature-Set aus EURUSD 5-Minuten-Kursen.
        Verhindert Daten-Redundanz und schützt das LSTM vor Overfitting.
        """
        close = df['Close'].to_numpy()
        returns = np.log(df['Close'] / df['Close'].shift(1)).to_numpy()

        # Vorbereitung des Ziel-DataFrames
        feature_dict = {}

        # Schleife über alle Zeitschritte (wir starten ab Index 258 wegen des größten Fensters)
        start_idx = 258
        length = len(df)

        # Arrays für die Rohdaten-Sammlung initialisieren
        # (Wir befüllen Dicts mit Listen und konvertieren am Ende für maximale Performance)
        target_columns = [
            'we32_D1', 'we32_total',
            'we64_D2', 'we64_total',
            'we128_D3', 'we128_D4', 'we128_total',
            'we256_A4', 'we256_total'
        ]
        for col in target_columns:
            feature_dict[col] = np.full(length, np.nan)

        for t in range(start_idx, length):

            # --- WINDOW 32 (Level 1 reicht hier völlig aus für D1) ---
            x32 = returns[t - 32 + 1: t + 1]
            v32 = df['Volume'].to_numpy()[t - 32 + 1: t + 1]
            mean_v32 = np.mean(v32)
            std_v32 = np.std(v32)
            if std_v32 > 0:
                # Wie viele Standardabweichungen weicht das aktuelle Volumen vom 32er-Schnitt ab?
                df.loc[df.index[t], 'v32_zscore'] = (v32[-1] - mean_v32) / std_v32
            else:
                df.loc[df.index[t], 'v32_zscore'] = 0.0

            if np.isfinite(x32).all():
                c32 = pywt.wavedec(x32, 'db4', mode='periodization', level=1)
                e32 = np.array([np.sum(c ** 2) for c in c32])
                sum_e32 = e32.sum()
                if sum_e32 > 0:
                    feature_dict['we32_D1'][t] = e32[1] / sum_e32  # cD1 ist bei Level 1 am Index 1
                    feature_dict['we32_total'][t] = np.sum(x32 ** 2)

            # --- WINDOW 64 (Level 2 für D2) ---
            x64 = returns[t - 64 + 1: t + 1]
            v64 = df['Volume'].to_numpy()[t - 64 + 1: t + 1]
            mean_v64 = np.mean(v64)
            std_v64 = np.std(v64)
            if std_v64 > 0:
                # Wie viele Standardabweichungen weicht das aktuelle Volumen vom 64er-Schnitt ab?
                df.loc[df.index[t], 'v64_zscore'] = (v64[-1] - mean_v64) / std_v64
            else:
                df.loc[df.index[t], 'v64_zscore'] = 0.0

            if np.isfinite(x64).all():
                c64 = pywt.wavedec(x64, 'db4', mode='periodization', level=2)
                e64 = np.array([np.sum(c ** 2) for c in c64])
                sum_e64 = e64.sum()
                if sum_e64 > 0:
                    feature_dict['we64_D2'][t] = e64[1] / sum_e64  # cD2 ist bei Level 2 am Index 1
                    feature_dict['we64_total'][t] = np.sum(x64 ** 2)

            # --- WINDOW 128 (Level 4 für D3 und D4) ---
            x128 = returns[t - 128 + 1: t + 1]
            if np.isfinite(x128).all():
                c128 = pywt.wavedec(x128, 'db4', mode='periodization', level=4)
                e128 = np.array([np.sum(c ** 2) for c in c128])
                sum_e128 = e128.sum()
                if sum_e128 > 0:
                    feature_dict['we128_D3'][t] = e128[2] / sum_e128  # cD3 ist bei Level 4 an Index 2
                    feature_dict['we128_D4'][t] = e128[1] / sum_e128  # cD4 ist bei Level 4 an Index 1
                    feature_dict['we128_total'][t] = np.sum(x128 ** 2)

            # --- WINDOW 256 (Level 4 für A4) ---
            x256 = returns[t - 256 + 1: t + 1]
            if np.isfinite(x256).all():
                c256 = pywt.wavedec(x256, 'db4', mode='periodization', level=4)
                e256 = np.array([np.sum(c ** 2) for c in c256])
                sum_e256 = e256.sum()
                if sum_e256 > 0:
                    feature_dict['we256_A4'][t] = e256[0] / sum_e256  # cA4 ist immer an Index 0
                    feature_dict['we256_total'][t] = np.sum(x256 ** 2)

        # In DataFrame gießen
        features_df = pd.DataFrame(feature_dict, index=df.index)
        return pd.concat([df, features_df], axis=1).dropna()



    def scale_total_energies(self, df):
        total_cols = ['we32_total', 'we64_total', 'we128_total', 'we256_total']

        # 1. Logarithmieren, um die winzigen Werte linear zu strecken
        for col in total_cols:
            df[col] = -np.log(df[col])

        # 2. MinMaxScaler anwenden (Train fitten -> Test transformieren)
        for col in total_cols:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[col] = scaler.fit_transform(df[[col]])

        return df



    def processCandle(self):

        #
        # Erzeugen der Dateiliste
        cpath = self.rawpath + 'forexCandle/'
        path = self.processedpath + '/forexCandle/'
        cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f)) and \
                    f[:6] == self.currency]
        cfiles.sort()
        for fl in cfiles:
            print(fl)
        #
        # Einlesen der Rohdaten
        df = pd.DataFrame()
        for c in cfiles:
            if (c == 'EURUSD_M5_20230219_20260731.csv'):
                dfirst = False
            else:
                dfirst = True
            dfi = pd.read_csv(cpath + c, parse_dates=[0], dayfirst=dfirst)
            dfi.rename({'Gmt time': 'ttime'}, inplace=True, axis=1)
            dfi = dfi[['ttime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = pd.concat([df, dfi])
        #
        # re-initialisieren des Index
        df.reset_index(drop=True, inplace=True)
        #
        # Sicherstellen, dass das Datetime-Format aktiv ist
        df['ttime'] = pd.to_datetime(df['ttime'], utc=True,  format='mixed', dayfirst=True)
        #
        # und vorsichtshalber nochmals sortieren
        df = df.sort_values(by='ttime').reset_index(drop=True)
        #
        # Addiere 5 Minuten zur Zeit, um das Label auf die rechte Seite zu bekommen
        df['ttime'] = df['ttime'] + pd.Timedelta(minutes=5)
        #
        # und jetzt der Index auf datetime
        df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
        df.drop('ttime', inplace=True, axis=1)
        print('Anzahl Rohdaten', len(df))
        #
        # Löschen der Daten für Freitag ab 22:00 Uhr, Samstag komplett und Sonntage bis 21:00 Uhr
        df = df[~((df.index.weekday == 4) & (df.index.hour > 21))]
        #
        # ... und Sonntag bis 21:00 Uhr
        df = df[~((df.index.weekday == 6) & (df.index.hour < 21))]
        #
        # Korrektur der Winterzeiten (Sonntage ab 22 Uhr - ein kleiner Fehler bleibt)
        df = df[~(df.index.month.isin([1, 2, 3, 11, 12]) & (df.index.weekday == 6) & (df.index.hour < 22))]
        #
        # Korrektur der Sommerzeiten (Freitage bis 21 Uhr - ein kleiner Fehler bleibt)
        df = df[~(df.index.month.isin([4, 5, 6, 7, 8, 9, 10]) & (df.index.weekday == 4) & (df.index.hour > 20))]
        #
        # ... und jetzt verschieben wir die Daten vom Sonntag auf den Freitag
        df.index = df.index.where(df.index.weekday != 6, df.index - pd.DateOffset(days=2))
        #
        print('Anzahl Daten nach Löschen von Samstag und Transferieren von Sonntag auf Freitag', len(df))
        #
        # Löschen aller Duplikate, insbesondere beim Übergang von Sommerzeit auf Winterzeit
        df = df[~df.index.duplicated(keep='first')]
        #
        # Auffüllen von fehlenden Werte: aufgrund vom Wechsel von Winter-
        # auf Sommerzeit (es fehlt dann 1 Stunde), oder aufgrund von System-
        # ausfällen
        df = df.sort_index()
        #
        # Erzeugen der techCols
        df = self.createTechInd(df)
        #
        # Erzeugen der waveletCols und skalieren der total energies
        df = self.calculate_optimized_wavelet_features(df)
        df = self.scale_total_energies(df)
        #
        # Fehlende Werte müssen nach Vorne aufgefüllt werden
        df = df.ffill()
        #
        # Fehlende Werte am Anfang werden gelöscht
        df = df.dropna()
        #
        # und zum Schluss speichern in zentralem Dataframe
        self.df_all = df[['Close'] + self.techCols]



    def readAndCleanEcoFeatureFile(self, path, fn):
        #
        # Liest und bereinigt die Eco Feature Date
        #
        df = pd.read_csv(path + fn,
                         dtype={'Actual': str,
                                'Forecast': str,
                                'Previous': str})
        df.rename({'Release Date': 'Date'}, inplace=True, axis=1)
        #
        # Bereinigen der Daten und Auffüllen fehlender Werte
        df['Date'] = df['Date'].apply(self.cleanEcoDate)
        if ('Forecast' in df.columns):
            df = df.apply(self.missingEcoData, axis=1)
            df['Forecast'] = df['Forecast'].apply(self.cleanEcoPct)
        if ('Previous' in df.columns):
            df['Previous'] = df['Previous'].apply(self.cleanEcoPct)
        df['Actual'] = df['Actual'].apply(self.cleanEcoPct)
        #
        # Zusammenfassen von Date und Time
        df['ttime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.strip(), dayfirst=True)
        #
        # Zeit-Index Festlegen und sortieren
        df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
        df.drop(['ttime'], axis=1, inplace=True)
        df = df.sort_index()
        #
        # Umwandeln von Europe/Berlin auf utc - 1 Stunde abziehen
        df.index = df.index.tz_localize('Europe/Berlin', ambiguous='infer').tz_convert('UTC')

        return df



    def processNormalEcoFeature(self, path, ecoFeatureName, fn, delta):
        #
        # Verarbeitung eines normalen Eco-Features und Anhängen an den DataFrame
        #
        df = self.readAndCleanEcoFeatureFile(path, fn)
        #
        # Benamung des Eco Features
        df.rename({'Actual': ecoFeatureName}, inplace=True, axis=1)
        #
        # Löschen der ueberfluessigen Spalten
        for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
            if (s in df.columns):
                df.drop(s, axis=1, inplace=True)
        #
        # Verarbeitung der Deltas
        dd = [int(i) for i in delta.split(',')]
        if (dd[0] > 0):
            for d in dd:
                delta_name = 'delta' + str(d) + '_' + ecoFeatureName
                df[delta_name] = ((df[ecoFeatureName] \
                                   - df[ecoFeatureName].shift(d)).div(df[ecoFeatureName].replace(0, np.nan))).round(3)
                df[delta_name] = df[delta_name].fillna(0)

                df.dropna(inplace=True)
        #
        # Verknüpfen der eco features mit df_all
        df_all = pd.merge_ordered(self.df_all.reset_index(),
                                  df.reset_index(),
                                  on='time',
                                  how='left',
                                  fill_method='ffill').set_index('time')

        return df_all



    def processSpreadFeature(self, path, ecoFeatureName, fn, delta):
        #
        # Verarbeitung eines Spread Features - spread zwischen zwei unterschiedlichen Eco Feature
        #
        fn1 = fn.split(',')[0].strip()
        fn2 = fn.split(',')[1].strip()
        #
        # Einlesen der beiden Dateien
        df1 = self.readAndCleanEcoFeatureFile(path, fn1, )
        for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
            if (s in df1.columns):
                df1.drop(s, axis=1, inplace=True)

        df2 = self.readAndCleanEcoFeatureFile(path, fn2)
        for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
            if (s in df2.columns):
                df2.drop(s, axis=1, inplace=True)

        #
        # da nicht alle Daten den gleichen Zeitstempel haben, müssen wir die beiden dataframes mit merge order verbinden
        df = pd.merge_ordered(df1, df2, on='time', how='outer', fill_method='ffill', suffixes=('_1', '_2'))
        #
        # Index auf datetime
        df.set_index(pd.DatetimeIndex(df.time), inplace=True)
        #
        # berechnen des spread
        df[ecoFeatureName] = (df['Actual_1'] - df['Actual_2']).round(3)
        #
        # Bereinigen des Ergebnisses
        df.drop(['time', 'Actual_1', 'Actual_2'], axis=1, inplace=True)
        df.dropna(inplace=True)
        #
        # Verarbeitung der Deltas
        dd = [int(i) for i in delta.split(',')]
        if (dd[0] > 0):
            for d in dd:
                delta_name = 'delta' + str(d) + '_' + ecoFeatureName
                df[delta_name] = ((df[ecoFeatureName] \
                                   - df[ecoFeatureName].shift(d)).div(df[ecoFeatureName].replace(0, np.nan))).round(3)
                df[delta_name] = df[delta_name].fillna(0)
                df.dropna(inplace=True)
        #
        # Verknüpfen der eco features mit df_all
        df_all = pd.merge_ordered(self.df_all.reset_index(),
                                  df.reset_index(),
                                  on='time',
                                  how='left',
                                  fill_method='ffill').set_index('time')

        return df_all



    def processSchockFeature(self, path, ecoSchockName, fn):
        #
        # Verarbeitung eines Schock-Features
        #
        fn = fn.strip()
        df = self.readAndCleanEcoFeatureFile(path, fn)
        #
        # Berechne den Schock und löschen aller unnötigen Spalten
        df['schock'] = ((df['Actual'] - df['Forecast'])).round(3)
        for s in ['Date', 'Time', 'Forecast', 'Previous', 'Actual']:
            if (s in df.columns):
                df.drop(s, axis=1, inplace=True)
        #
        # calculate Mean and STD for standardization
        s_mean = df['schock'].mean()
        s_std = df['schock'].std()
        if (s_std == 0):
            s_std = 1
        df['schock'] = (df['schock'] - s_mean) / s_std
        self.schockStandardize.append([fn, s_mean, s_std])
        #
        # Verknüpfen der eco features mit df_all
        df_all = pd.merge_ordered(self.df_all.reset_index(),
                                  df.reset_index(),
                                  on='time',
                                  how='left').set_index('time')

        #
        # Addieren des Schocks - erst prüfen, ob es den ecoSchock bereits gibt
        if (ecoSchockName not in df_all.columns):
            df_all[ecoSchockName] = 0
        #
        # nach dem merge_ordered haben wir etliche nan-werte, die in den schocks mit 0 befüllt werden müssen
        for c in df_all.columns:
            if (c.split('_')[0] == 'schock'):
                df_all[c] = df_all[c].fillna(0)
            else:
                # für alle restlichen eco-features werden die Lücken mit ffill befüllt
                df_all[c] = df_all[c].ffill()

        df_all[ecoSchockName] = (df_all[ecoSchockName] + df_all['schock']).round(3)
        df_all.drop(['schock'], axis=1, inplace=True)

        return df_all



    def addEcoFeatures(self):
        #
        # Methode fügt den Daten die ökonomischen Features hinzu
        #
        epath = self.processedpath + 'kalender/neu/'
        df_eco_feature = pd.read_csv(epath + 'kalender-features.csv',
                                     dtype={'Feature': str,
                                            'Datei': str,
                                            'Delta': str})

        ecoFeatures = df_eco_feature.values.tolist()
        for ef in ecoFeatures:
            print(ef)
            if (ef[0].split('_')[0] == 'spread'):
                self.df_all = self.processSpreadFeature(epath, ef[0].strip(), ef[1].strip(), ef[2].strip())
            elif (ef[0].split('_')[0] == 'schock'):
                self.df_all = self.processSchockFeature(epath, ef[0].strip(), ef[1].strip())
            else:
                self.df_all = self.processNormalEcoFeature(epath, ef[0].strip(), ef[1].strip(), ef[2].strip())



    def formatFinalVersion(self,
                           startDt = '2005-02-01 00:00:00',
                           endDt = '2026-06-30 23:55:00'):
        #
        # einige Cleanups müssen noch gemacht werden
        cpath = self.processedpath + 'forexCandle/'

        print('Anzahl Beobachtungen vor Cleanup', len(self.df_all))
        sys.stdout.flush()
        #
        # fehlende Werte müssen über ffill befüllt werden
        self.df_all = self.df_all.asfreq('5min', method='ffill')
        #
        # Löschen der Samstage und Sonntage
        self.df_all = self.df_all[(self.df_all.index.weekday != 5)]
        self.df_all = self.df_all[(self.df_all.index.weekday != 6)]
        # Weihnachten, Sylvester und Neujahr müssen weg
        self.df_all = self.df_all[(self.df_all.index.month != 12) | (self.df_all.index.day != 24)]
        self.df_all = self.df_all[(self.df_all.index.month != 12) | (self.df_all.index.day != 25)]
        self.df_all = self.df_all[(self.df_all.index.month != 12) | (self.df_all.index.day != 31)]
        self.df_all = self.df_all[(self.df_all.index.month != 1) | (self.df_all.index.day != 1)]
        print('Anzahl Beobachtungen nach Löschen von Feiertagen', len(self.df_all))
        #
        # und sicherheitshalber nochmals sortieren
        self.df_all = self.df_all.sort_index()
        #
        # Selektion der relevanten Features und Beobachtungen
        self.df_all = self.df_all[['Close'] + self.techCols + self.ecoCols]
        self.df_all = self.df_all.loc[self.df_all.index >= startDt]
        self.df_all = self.df_all.loc[self.df_all.index <= endDt]

        print('Anzahl Beobachtungen nach Cleanup', len(self.df_all))
        for c in self.df_all.columns:
            n_nan = self.df_all[c].isnull().sum()
            print('Fehlende Werte in Spalte ' + c, n_nan)

        self.df_all.to_csv(cpath + 'df_all.csv')

        #
        # Berechne Mittelwert und Std für Skalierung und Speichere diese
        # als pickle-files
        df_mean = self.df_all[self.scaledCols].mean()
        df_mean.to_pickle(cpath + 'df_all_mean.pkl')
        df_mean.to_csv(cpath + 'df_all_mean.csv')
        df_std = self.df_all[self.scaledCols].std()
        df_std.to_pickle(cpath + 'df_all_std.pkl')
        df_std.to_csv(cpath + 'df_all_std.csv')
            
        pd.DataFrame(self.schockStandardize, columns=['name', 'mean', 'std']) \
                .to_csv(cpath + 'schockStandardize.csv', index=False)

        #
        # Skalieren der Daten           
        self.df_all[self.scaledCols] = (self.df_all[self.scaledCols] - df_mean)/df_std
        self.df_all.to_csv(cpath + 'df_all_scaled.csv')
 
        #
        # und zum Schluß noch ein Check, ob jeder Tag 288 Einträge hat
        dff = self.df_all.groupby(self.df_all.index.date).count()
        print('Anzahl Tage insgesamt:', len(dff))
        sys.stdout.flush()
        df_days = dff.index
        self.days = df_days.values.tolist()
        dff = dff[dff['Close'] < self.bt]
        dff['Close'].to_csv(cpath + 'FullDayCheck.csv')
       
        
       
    def calcSingleReward(self, rewards, transcost, i, action):
        #
        # Berechnet den Reward an der Stelle i im Tag
        #
        r = 0
        if (action == 1):
            # reward for opening a Long position
            r = action*rewards[i] - transcost[i]
        elif (action == -1):
            # reward for opening a short position
            r = action*rewards[i] - transcost[i]
                
        return r
        
        

    def calcInitReward(self, df_day):
        #
        # berechnet die initiale Belohnung pro Tag
        #
        df_day.reset_index(inplace=True)
        if (len(df_day) != self.bt):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht {}'.format(self.bt))
        #
        # Berechnen der Differenzen
        price = df_day['Close'].values
        rewards = self.pip * np.log(np.roll(price, -1) / price)
        rewards[-1] = self.pip * np.log(price[-1] / price[-2])
        rewards = np.clip(rewards, -5, 5)
        transcost = self.pip * np.log(1 + ((self.trade_fee / self.pip) / price))
        #
        # Schleife über den Tag
        reward = np.full((self.bt, 3), 0.)
        for i in range(self.bt-1):
            reward[i, 1] += self.calcSingleReward(rewards, transcost, i, 1)
            reward[i, 2] += self.calcSingleReward(rewards, transcost, i, -1)
        
        position = np.array(self.bt*[0.]).astype(float)
        currProfit = np.array(self.bt*[0.]).astype(float)
        currTradeLength = np.array(self.bt * [0.]).astype(float)
        
        rw = np.c_[position, currProfit, currTradeLength, reward]

        df_day = pd.concat([df_day, 
                            pd.DataFrame(rw, 
                                         columns=['Position',
                                                  'currProfit',
                                                  'currTradeLength',
                                                  'Reward-Hold', 
                                                  'Reward-Buy',
                                                  'Reward-Sell'])], 
                           axis=1)
                             
        cpath = self.processedpath + 'forexCandle/'
        df_day.to_csv(cpath + 'rewards.csv')
        
        return df_day
        


    def packInitXY(self, df, bs, ts, feature_list, target):
        
        
        x_data = df[feature_list].values
        y_data = df[target].values

        if (len(df) != self.bt):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht {}'.format(self.bt))
            
        x = np.array([])
        y = np.array([])    
        for i in range(ts+1, (bs+ts+1)):
            x = np.append(x, x_data[i-ts:i])
            y = np.append(y, y_data[i-ts:i])

        x = x.reshape(bs, ts, len(feature_list))
        y = y.reshape(bs, ts, 3)
        
        return x, y
    


    def packRLXY(self, df, batch_id, bs, ts, feature_list, target):
        
        x_data = df[feature_list].values
        y_data = df[target].values
        dt_data = df.index.values.astype(str)
        close_data = df['Close'].values
        
        if (len(df) != self.bt):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht {}'.format(self.bt))
            
        x = np.array([])
        y = np.array([])    
        dtd = np.array([])
        close = np.array([])
        for i in range(ts+1, (bs+ts+1)):
            x = np.append(x, x_data[i-ts:i])
            y = np.append(y, y_data[i-ts:i])
            dtd = np.append(dtd, dt_data[i-1])
            close = np.append(close, close_data[i-1])

        x = x.reshape(bs, ts, len(feature_list))
        y = y.reshape(bs, ts, 3)
        dtd = dtd.reshape(bs, 1)
        close = close.reshape(bs, 1)
        all_data = np.array((batch_id, dtd, close, x, y), dtype=object)

        return all_data
    


    def createABT(self):
        #
        # erzeugt die Analytical Base Table pro Tag indem
        #   1. die initialen Rewards mit den initialen Positionen berechnet
        #      werden, und
        #   2. die Daten in der für die LSTM-Modellierung notwendigen 
        #      Zeitreihen-Struktur gepackt werden
        #
        # Berechne initiale Rewards mit initialen Positionen pro Tag, packe
        # und speichere die Daten
        for day in self.days:
            #
            # extrahiere den Tag
            df_day = self.df_all[self.df_all.index.date == day]
            #
            # Berechne die initiale Belohnung
            df_day = self.calcInitReward(df_day)

            #
            # packe die Daten in eine Sequenz der Länge 72 für initiales Training
            x, y = self.packInitXY(df_day,
                                   self.bs,
                                   self.ts,
                                   self.featureList,
                                   self.target)
                
            np.save(self.finalpath + 'X/x-' + day.strftime('%Y-%m-%d') + '.npy', x)
            np.save(self.finalpath + 'Y/y-' + day.strftime('%Y-%m-%d') + '.npy', y)
            #
            # packe die Daten in eine Sequenz der Länge 72 für RL Training
            all_data = self.packRLXY(df_day,
                                     self.days.index(day),
                                     self.bs,
                                     self.ts,
                                     self.featureList,
                                     self.target)
                
            np.save(self.finalpath + 'RL-data/day-' + \
                        day.strftime('%Y-%m-%d') + '.npy', all_data)

    
    def initLR_scheduler(self, i, rate): 
        rate = rate*np.exp((-0.6/(10 + 1*i)))
        return rate


    
    def fitInitialModel(self):
        #
        # trainiert das initial RL-Modell
        #
        # sicherstellen, dass Model-Pfad existiert
        if (not isdir(self.finalpath + 'model/initModel/')):
            makedirs(self.finalpath + 'model/initModel/')

        # Lesen der verfügbaren xy-Daten
        xpath = self.finalpath + 'X/'
        xfiles = [f for f in listdir(xpath) if isfile(join(xpath, f)) \
                  and f[0] == 'x']
        xfiles.sort()
            
        ypath = self.finalpath + 'Y/'
        yfiles = [f for f in listdir(ypath) if isfile(join(ypath, f)) and \
                  f[0] == 'y']
        yfiles.sort()
        #xyfiles = np.c_[xfiles, yfiles]
        #
        # definiere das LSTM-Model
        ff = len(self.featureList)
        nbatch = len(xfiles)
        print('Anzahl Features =', ff, 'Anzahl Beobachtunge =', nbatch)
        
        model = myLSTM(ff)
        lr = 0.05
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        niter = 5000
        ntrain = 500
        ntest = 200
        nepoch = 1
        i = 1
        for i in range(ntrain+3000, niter, 500):
            x_train = np.array([])
            y_train = np.array([])
            x_test = np.array([])
            y_test = np.array([])
            days = np.random.choice(range(i), size=ntrain, replace=False)
            for day in days:
                x_train = np.append(x_train, np.load(xpath + xfiles[day]))
                y_train = np.append(y_train, np.load(ypath + yfiles[day]))

            for i in range(i, i+ntest):
                x_test = np.append(x_test, np.load(xpath + xfiles[i]))
                y_test = np.append(y_test, np.load(ypath + yfiles[i]))
                               
            x_train = torch.tensor(x_train.reshape(ntrain*self.bs, 
                                                   self.ts, 
                                                   ff)).type(torch.float)
            
            y_train = torch.tensor(y_train.reshape(ntrain*self.bs, 
                                                   self.ts, 
                                                   3)).type(torch.float)
#
            x_test = torch.tensor(x_test.reshape(ntest*self.bs, 
                                                 self.ts, 
                                                 ff)).type(torch.float)
            
            y_test = torch.tensor(y_test.reshape(ntest*self.bs, 
                                                 self.ts, 
                                                 3)).type(torch.float)
#      
            train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), 
                                           shuffle=False, 
                                           batch_size=23,
                                           drop_last=True)
            
            for ii in range(nepoch):
                model.train()
                for X, Y in train_loader:
                    Y_pred = model(X)
                    loss = loss_fn(Y_pred, Y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    Y_pred = model(x_train)
                    rmse = np.sqrt(loss_fn(Y_pred, y_train).detach().item())  
                    y_pred = model(x_test)
                    val_rmse = np.sqrt(loss_fn(y_pred, y_test).detach().item())
                    
                lr = self.initLR_scheduler(i+ii, lr)
                optimizer.param_groups[0]['lr'] = lr

                print("Iteration %d: lr %.6f train RMSE %.4f, test RMSE %.4f" \
                      % (i, lr, rmse, val_rmse))

                sys.stdout.flush()
            
            torch.save(model.state_dict(), 
                       self.finalpath + 'model/initModel/initModel.pt')



    def add_business_day(self, date_string, date_format="%Y-%m-%d"):
        # 1. String in ein Datumsobjekt umwandeln
        date_obj = datetime.strptime(date_string, date_format)

        # 2. Immer mindestens 1 Tag addieren
        date_obj += timedelta(days=1)

        # 3. Wenn Samstag (5), addiere 2 Tage extra -> Montag
        if date_obj.weekday() == 5:
            date_obj += timedelta(days=2)
        # Wenn Sonntag (6), addiere 1 Tag extra -> Montag
        elif date_obj.weekday() == 6:
            date_obj += timedelta(days=1)

        # 4. Zurück in einen String konvertieren
        return date_obj.strftime(date_format)


       
    def rate_LRscheduler(self, i, rate): 
        
        if (self.last_best_upd < 500):
            if (rate > 0.001):
                # make learning rate smaller
                rate = rate * np.exp(-(1/(950000 + i)))
            else:
                rate = 0.001
        else:
            if (rate < 0.01):
                # make learning rate bigger
                rate = rate * np.exp((1/(950000 + i)))
            else:
                rate = 0.01


        return rate
            
    
    def rate_DRscheduler(self, i, rate): 
        
        rate = rate * np.exp(-(1/(950000 + i)))
        return rate
    
    def rate_ERscheduler(self, i, rate): 
        
        if (rate > 0.01):
            rate = rate * np.exp(-(1 / (64000 + i)))
        #    rate = rate * np.exp(-(1/(12000 + i)))
        else:
            rate = 0.01
            
        return rate
    
            
             
    def setIterRates(self, trader, i):
        
        self.exploration_rate = self.rate_ERscheduler(i, self.exploration_rate)
        #self.dr = self.rate_DRscheduler(i, self.dr)
        #self.dreaming_rate = 0.25 + self.dr
        #self.learning_rate = self.rate_LRscheduler(i, self.learning_rate)
            
        trader.set_rates(self.dreaming_rate,
                         self.exploration_rate,
                         self.learning_rate)
    


    def evalOneDay(self, trader, day):
        #
        # Methode berechnet den Profit für einen Tage
        #
        batch_id = day[0]
        close = day[2].flatten().tolist()
        x = day[3].astype(np.float32)
        action = trader.predictOneDay(x).tolist()
        profit = 0.
        position = 0
        last_i = None
        i = 0
        num_trades = 0
        for a in action:
            if (a == 0):
                if (position in [-1, 1]):
                    profit += position * (close[i] - close[last_i]) - self.trade_fee
                    last_i = None
                    num_trades += 1
            elif (a == 1):
                if (position == -1):
                    profit += position * (close[i] - close[last_i]) - self.trade_fee
                    last_i = i
                    num_trades += 1
                elif (position == 0):
                    last_i = i
            elif (a == -1):
                if (position == 1):
                    profit += position * (close[i] - close[last_i]) - self.trade_fee
                    last_i = i
                    num_trades += 1
                elif (position == 0):
                    last_i = i
            position = a
            i += 1

        return batch_id, profit, sum(abs(x) for x in action), num_trades


 
    def fitForexDayTrader(self):
                  
        nameBestModel = 'bestModel'
        nameTmpModel = 'tmpModel'
        pathBestModel = self.finalpath + 'model/' + nameBestModel + '/'
        maxAvgTotalReward = -999.99
        # sicherstellen, dass Model-Pfad existiert
        if (not isdir(pathBestModel)):
            makedirs(pathBestModel)
        #
        #######################################
        ### change re-start parameter here ####
        #self.dr = 0.75
        self.dreaming_rate = 1.0
        self.exploration_rate = 0.99
        self.learning_rate = 0.005
        init_weight_name = 'initModel'
        self.last_best_upd = 0
        i = 0
        ### change re-start parameter here ####
        #######################################
        #
        if (isfile(pathBestModel+'maxAvgTotalReward.npy')):
            maxAvgTotalReward = np.load(pathBestModel+'maxAvgTotalReward.npy')
        
        print('Maximum Total Reward = %5.2f' % (maxAvgTotalReward))
        
        env_gym = gym.envs.registration.registry.copy()
        for env in env_gym:
            if 'forex-v0' in env:
                del gym.envs.registry['forex-v0']  
        register(id='forex-v0', entry_point='forexTrader.envs.forexEnv:ForexEnv')        

        env = gym.make(id='forex-v0', 
                       data_path=self.finalpath,
                       window_size=self.ts, 
                       batch_size=self.bs)
        trader = ForexDayTrader(env, 
                                self.finalpath, 
                                init_weight_name,
                                self.bs,
                                self.ts,
                                self.ff)
        result = []
        result_profit = []
        start_dt = '2005-02-01'
        end_dt = '2015-12-31'
        end_dt_fix = False
        while (end_dt < '2026-06-30'):
            #
            # äusserer Loop über alle Iterationen
            #
            i += 1
            self.setIterRates(trader, i)
            action_hold = 0
            action_buy = 0
            action_sell = 0
            if (i == 500000):
                #
                # Beginn des rollierenden Training - initialisieren eins rollierenden 3-Jahres-Fenster
                start_dt = '2013-01-02'
                end_dt = '2015-12-31'
            elif (i > 500000):
                if (i % 5000 == 0):
                    #
                    # alle 5000 iterationen wird das Zeitfenster um 1 Tag erhöht und der Folgetag bewertet
                    #
                    validDay = False
                    while not validDay:
                        start_dt = self.add_business_day(start_dt)
                        validDay = env.isValidDay(start_dt)

                    validDay = False
                    while not validDay:
                        end_dt = self.add_business_day(end_dt)
                        validDay = env.isValidDay(end_dt)
                    #
                    # die Daten für das neue end_dt sollen während des rollierenden Trainings immer in das Training Sample
                    # aufgenommen werden
                    end_dt_fix = True
                    #
                    print('Neues StartDate = {}, neues EndDate = {}'.format(start_dt, end_dt))
                    #
                    # das neue End-Datum mit dem aktuellen Modell bewerten
                    newDay = env.getRL_Data(end_dt)
                    nd_batch_id, nd_profit, num_action, num_trades = self.evalOneDay(trader, newDay)
                    result_profit.append([end_dt,
                                          nd_batch_id,
                                          nd_profit,
                                          num_action,
                                          num_trades])
                    df_profit = pd.DataFrame(result_profit, columns=['Datum',
                                                                     'Batch_id',
                                                                     'Profit',
                                                                     'Anzahl Action <> 0',
                                                                     'Anzahl Trades'])
                    df_profit.to_csv(self.finalpath + 'result/profit-results.csv', index=False)

            curr_state, info = env.reset(options={"start_dt": start_dt,
                                                  "end_dt": end_dt,
                                                  "end_dt_fix": end_dt_fix})

            curr_state = curr_state.reshape(1, self.ts, self.ff)

            for j in range(self.bs):
                #
                # innerer Loop über den Tag
                #
                action = trader.act(curr_state)
                new_state, reward, done, terminated, info = env.step(action)
                new_state = new_state.reshape(1,
                                              self.ts,
                                              self.ff)
                
                trader.remember(curr_state, action, reward)
        
                if (action == 0):
                    action_hold += 1
                elif (action == 1):
                    action_buy += 1
                elif (action == -1):
                    action_sell += 1
                    
                if done:
                    td_err = trader.dream(info["datetime"],
                                          info["batch_id"],
                                          info["total_reward"])
                else:
                    curr_state = new_state
                    
            result.append([i, 
                           info["datetime"][:10], 
                           info["batch_id"], 
                           info["total_reward"],
                           info["total_reward_long"],
                           info["total_reward_short"],
                           info["n_long_trades"],
                           info["n_short_trades"],
                           info["n_hold_trades"],
                           action_hold,
                           action_buy,
                           action_sell,
                           td_err[0],
                           td_err[1],
                           td_err[2],
                           td_err[3],
                           td_err[4],
                           td_err[5],
                           td_err[6],
                           self.dreaming_rate,
                           self.exploration_rate,
                           self.learning_rate])

            trader.replay(i)
            if (i % 50 == 0):
                # trainiere target_model alle xxx-Tage und sichere das Modell
                trader.train_target()
                trader.save_model(nameTmpModel)

            if (i % 500 == 0):
                df = pd.DataFrame(result, columns=['Iteration',
                                                   'DateTime',
                                                   'Batch-ID',
                                                   'Avg-Total-Reward',
                                                   'Long-Reward',
                                                   'Short_Reward',
                                                   '#Long_Trades',
                                                   '#Short_Trades',
                                                   '#Hold_Trades',
                                                   'Avg-Hold-Action',
                                                   'Avg-Buy-Action',
                                                   'Avg-Sell-Action',
                                                   'td_err Mean',
                                                   'td_err Abs Mean',
                                                   'td_err Median',
                                                   'td_err Min',
                                                   'td_err Max',
                                                   'td_err Pct10',
                                                   'td_err Pct90',
                                                   'Dreaming Rate',
                                                   'Exploration Rate',
                                                   'Learning Rate'])
                
                df.to_csv(self.finalpath + 'result/rl-results-' + str(i) + '.csv', 
                          index=False)

                avgTotalReward = np.mean(df["Avg-Total-Reward"])
                self.last_best_upd += 1
                print('Average Total Reward = %5.2f Max Total Reward = %5.2f' \
                      % (avgTotalReward, maxAvgTotalReward))
                print('exploration_rate = %5.4f dreaming_rate = %5.4f learning_rate = %5.4f' \
                      % (self.exploration_rate, 
                         self.dreaming_rate, 
                         self.learning_rate))
                    
                if (avgTotalReward > maxAvgTotalReward):
                # only save model with maximum Average Total Reward
                    print("saving model with avgTotalReward = %5.2f to %32s" \
                          % (avgTotalReward, nameBestModel))
                    trader.save_model(nameBestModel)
                    maxAvgTotalReward = avgTotalReward
                    np.save(pathBestModel + 'maxAvgTotalReward.npy',
                            maxAvgTotalReward)
                    self.last_best_upd = 0
                result = []

                                                               

periodStartDt = '2005-02-01 00:00:00'
periodEndDt = '2026-06-30 23:55:00'

eurusdData = forexRLDayTrading('EURUSD')

"""
eurusdData.processCandle()

eurusdData.addEcoFeatures()

eurusdData.formatFinalVersion(startDt = periodStartDt,
                              endDt = periodEndDt)

eurusdData.createABT()

eurusdData.fitInitialModel()
"""

eurusdData.fitForexDayTrader()





