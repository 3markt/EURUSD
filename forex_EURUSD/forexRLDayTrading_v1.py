#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:07:49 2023

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import datetime as dt
import random
import ta
import sys
from os import listdir, makedirs
from os.path import isfile, join, isdir
from sklearn.preprocessing import StandardScaler
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam, SGD
import tensorflow as tf
import gym
from gym.envs.registration import register
from forexTrader.envs import forexEnv
from forexTrader.agent import ForexDayTrader 



class forexRLDayTrading:
    
    headcols = ['Time', 'Close', 'Reward-Hold', 'Reward-Buy', 'Reward-Sell', 
                'Position']

    techindCols = ['zeitIndex', 'Close', 'scaledClose',  'round_lot', 'WL', 
                   'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi', 'macd_30Min', 
                   'macd_d_30Min', 'bb_h_30Min', 'bb_l_30Min', 'rsi_30Min', 
                   'macd_6H', 'macd_d_6H', 'bb_h_6H', 'bb_l_6H', 'rsi_6H']
    
    featureCols = ['zeitIndex', 'Close', 'scaledClose', 'round_lot', 'WL', 'macd', 
                   'macd_d', 'bb_h', 'bb_l', 'rsi', 'macd_30Min', 'macd_d_30Min', 
                   'bb_h_30Min', 'bb_l_30Min', 'rsi_30Min', 'macd_6H', 
                   'macd_d_6H', 'bb_h_6H', 'bb_l_6H', 'rsi_6H',
                   'CPI_US' , 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
                   'Zins_US', 'Zins_EU', 'Schock_Preis_US', 'Schock_Preis_EU', 
                   'Verbraucherstimmung_US', 'Verbraucherstimmung_EU', 'GfK', 
                   'Schock_Nachfrage_US', 'Schock_Nachfrage_EU', 
                   'PMI_Manufacturing_US', 'PMI_Service_US', 
                   'PMI_Manufacturing_EU', 'PMI_Service_EU', 
                   'Schock_Produktion_US', 'Schock_Produktion_DE', 
                   'JoblessInitial_US', 'UnemploymentRate_US', 
                   'UnemploymentRate_EU', 'Schock_Arbeitsmarkt_US', 
                   'Schock_Arbeitsmarkt_EU', 'GDP_US', 'GDP_EU', 
                   'TradeBalance_US', 'TradeBalance_EU', 
                   'Schock_Konjunktur_US', 'Schock_Konjunktur_EU']
        
    scaledCols = ['zeitIndex', 'scaledClose', 'round_lot', 'WL', 'macd', 'macd_d', 
                  'bb_h', 'bb_l', 'rsi', 'macd_30Min', 'macd_d_30Min', 
                  'bb_h_30Min', 'bb_l_30Min', 'rsi_30Min', 'macd_6H', 
                  'macd_d_6H', 'bb_h_6H', 'bb_l_6H', 'rsi_6H',
                  'CPI_US' , 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
                  'Zins_US', 'Zins_EU', 'Schock_Preis_US', 'Schock_Preis_EU', 
                  'Verbraucherstimmung_US', 'Verbraucherstimmung_EU', 'GfK', 
                  'Schock_Nachfrage_US', 'Schock_Nachfrage_EU', 
                  'PMI_Manufacturing_US', 'PMI_Service_US', 
                  'PMI_Manufacturing_EU', 'PMI_Service_EU', 
                  'Schock_Produktion_US', 'Schock_Produktion_DE', 
                  'JoblessInitial_US', 'UnemploymentRate_US', 
                  'UnemploymentRate_EU', 'Schock_Arbeitsmarkt_US', 
                  'Schock_Arbeitsmarkt_EU', 'GDP_US', 'GDP_EU', 
                  'TradeBalance_US', 'TradeBalance_EU', 
                  'Schock_Konjunktur_US', 'Schock_Konjunktur_EU']

    target = ['Reward-Hold', 'Reward-Buy', 'Reward-Sell']

    featureList = ['Position', 'zeitIndex', 'scaledClose', 'round_lot', 'WL', 
                   'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi', 'macd_30Min', 
                   'macd_d_30Min', 'bb_h_30Min', 'bb_l_30Min', 'rsi_30Min', 
                   'macd_6H', 'macd_d_6H', 'bb_h_6H', 'bb_l_6H', 'rsi_6H',
                   'CPI_US' , 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
                   'Zins_US', 'Zins_EU', 'Schock_Preis_US', 'Schock_Preis_EU', 
                   'Verbraucherstimmung_US', 'Verbraucherstimmung_EU', 'GfK', 
                   'Schock_Nachfrage_US', 'Schock_Nachfrage_EU', 
                   'PMI_Manufacturing_US', 'PMI_Service_US', 
                   'PMI_Manufacturing_EU', 'PMI_Service_EU', 
                   'Schock_Produktion_US', 'Schock_Produktion_DE', 
                   'JoblessInitial_US', 'UnemploymentRate_US', 
                   'UnemploymentRate_EU', 'Schock_Arbeitsmarkt_US', 
                   'Schock_Arbeitsmarkt_EU', 'GDP_US', 'GDP_EU', 
                   'TradeBalance_US', 'TradeBalance_EU', 
                   'Schock_Konjunktur_US', 'Schock_Konjunktur_EU']

    
    
    def __init__(self, currency):
        
        self.currency = currency
        #self.basepath = '/Users/uwe.muller/Hope/data/'
        self.basepath = '/home/chitlom/data/'
        self.rawpath = self.basepath + 'raw/'
        self.processedpath = self.basepath + 'processed/' + self.currency + '/'
        self.finalpath = self.basepath + 'final/' + self.currency + '/'
        self.schockStandardize = []
        self.bt = 288
        self.ts = 72
        self.bs = self.bt - self.ts
        self.units = 26
        self.tradeFee = 1.5
        self.pip = 10000
        self.gamma = 0.9
        self.learning_rate = 0.0005
        self.ongoing_model = self.finalpath + 'model/ongoingModel/ongoingModel'

        
        
        
    def normalDF(self, x, m=1., s=30.):
        
        f = np.exp(-0.5*((x-m)/s)**2)
        
        return f

        
        
    def convertCandle(self, df, freq):
        
        dfs = df.resample(freq, label='right', closed='right')
        df_new = dfs['Open'].first().to_frame()
        df_new['High'] = dfs['High'].max()
        df_new['Low'] = dfs['Low'].min()
        df_new['Close'] = dfs['Close'].last()
        df_new = df_new.dropna()
        
        return df_new
       


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
        row = row.replace(',', '')
        row = float(row.strip('%KMB'))
        
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
    
    
    
        
    def processCandle(self, startDt = '01.01.2007'):
        #
        # Erzeugen der Dateiliste
        #
        cpath = self.rawpath + 'forexCandle/'
        cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f)) and \
                    f[:6] == self.currency and f[27:38] >= startDt]
        cfiles.sort()
        
        df = pd.DataFrame()
        dateparse = lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f')
        for c in cfiles:
            dfi = pd.read_csv(cpath + c, parse_dates=['Gmt time'], 
                              date_parser=dateparse)
            dfi.rename({'Gmt time':'ttime'}, inplace=True, axis=1)
            dfi = dfi[['ttime', 'Open', 'High', 'Low', 'Close', ]]
            df = df.append(dfi)
            
        df.reset_index(drop=True, inplace=True)
        #
        # Addiere 5 Minuten zur Zeit, um das Label auf die rechte Seite zu bekommen
        df['ttime'] = df['ttime'] + pd.Timedelta(minutes=5)
        #
        # Verschieben der Eröffnungstransaktionen vom Sonntag auf den Freitag
        df.loc[df['ttime'].dt.dayofweek == 6, 'ttime'] = df['ttime'] \
                                                    - pd.to_timedelta(2, 'D')
        #
        # Beim Wechsel von Sommer- auf Winterzeit gibt es Überschneidungen,
        # die gelöscht werden müssen
        df.drop_duplicates(subset=['ttime'], 
                           keep='first', 
                           inplace=True, 
                           ignore_index=True)
        #
        # Sortieren und Datetimeindex setzen
        df = df.sort_values(by='ttime')
        df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
        df.drop('ttime', inplace=True, axis=1)
        #
        # Auffüllen von fehlenden Werte: aufgrund vom Wechsel von Winter-
        # auf Sommerzeit (es fehlt dann 1 Stunde), oder aufgrund von System-
        # ausfällen
        df = df.asfreq('5min', method='ffill')
        #
        # Löschen der Samstage und Sonntage
        df = df[(df.index.weekday != 5)]
        df = df[(df.index.weekday != 6)]

        path = self.processedpath + '/forexCandle/'
        
        #
        # Konvertierung auf andere Zeitskalen und Erzeugen der technischen Indikatoren
        #
        # 5 Minuten müssen nicht konvertiert werden
        min5Candle = self.createTechInd(df)
        min5Candle.to_csv(path + 'min5Candle.csv')
        
        # Konvertierung auf 30 Minuten und technischen Indikatoren
        min30Candle = self.convertCandle(min5Candle, '30min')
        min30Candle = self.createTechInd(min30Candle)
        min30Candle.to_csv(path + 'min30Candle.csv')
        
        #Konvertierung auf 6 Stunden und Erzeugen der technischen Indikatoren
        h6Candle = self.convertCandle(min5Candle, '6H')
        h6Candle = self.createTechInd(h6Candle)
        h6Candle.to_csv(path + 'h6Candle.csv')
  
    
        #
        # Zusammenführen der unterschiedlichen Zeitskalen
        #
        # zunächst 5 mit 30 Minuten
        df_all = pd.merge_ordered(min5Candle, min30Candle, 
                                  how='left', on='time', 
                                  suffixes=('', '_30Min'))
        
        # Fehlende Werte müssen nach Vorne aufgefüllt werden
        df_all = df_all.fillna(method='ffill')

        # und dann mit 6 Stunden
        df_all = pd.merge_ordered(df_all, h6Candle, 
                                  how='left', on='time', 
                                  suffixes=('', '_6H'))
        
        # Fehlende Werte müssen nach Vorne aufgefüllt werden
        df_all = df_all.fillna(method='ffill')
        
        # Fehlende Werte am Anfang werden gelöscht
        df_all = df_all.dropna()
        
        # und zum Schluss wird der Zeit-Index wieder erneuert
        df_all.rename({'time':'ttime'}, axis=1, inplace=True)
        df_all.set_index(pd.DatetimeIndex(df_all.ttime, name='time'), 
                         inplace=True)
        self.df_all = df_all[self.techindCols]



    def mergeEcoFeature(self, df_eco, name):
    
        df_eco.rename({'Actual':name}, axis=1, inplace=True)
        df_eco.drop('Forecast', axis=1, inplace=True)
        df = pd.merge_ordered(self.df_all, df_eco, 
                              how='left', on='time', 
                              suffixes=('', name))

        df.rename({'time':'ttime'}, axis=1, inplace=True)
        df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
        df.drop('ttime', axis=1, inplace=True)
        df = df.fillna(method='ffill')
        #
        # Verschmutzen der EcoFeatures
        std = df[name].std()
        df[name] = df[name].apply(lambda x: x + np.random.normal(loc=0.0, 
                                                                 scale=std*0.05))
        #df = df.dropna()
        
        return df
    
    

    def mergeEcoSchock(self, df_schock, name, fn):
        
        # calculate Mean and STD for standardization
        s_mean = (df_schock['Actual'] - df_schock['Forecast']).mean()
        s_std = (df_schock['Actual'] - df_schock['Forecast']).std()
        self.schockStandardize.append([fn, s_mean, s_std])
        # merge Schock mit df_all
        df = pd.merge_ordered(self.df_all, df_schock, 
                              how='left', on='time', 
                              suffixes=('', name))
        df.rename({'time':'ttime'}, axis=1, inplace=True)
        df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
        #
        # Berechnen der Schocks
        #
        # Länderspezifische Anpassung für Europa
        if ('_DE' in name):
            df['schock'] = 0.9*((df['Actual'] - df['Forecast'] - s_mean)/s_std) \
                            .round(decimals=2)
        elif ('_F' in name):
            df['schock'] = 0.75*((df['Actual'] - df['Forecast'] - s_mean)/s_std) \
                            .round(decimals=2)
        elif ('_I' in name or '_E' in name):
            df['schock'] = 0.5*((df['Actual'] - df['Forecast'] - s_mean)/s_std) \
                            .round(decimals=2)         
        else:
            df['schock'] = ((df['Actual'] - df['Forecast'] - s_mean)/s_std) \
                            .round(decimals=2)
        #
        # zeitlich abfallende Wirkung
        for i in range(1, 73):
            adjust = self.normalDF(i)
            df.loc[df['schock'].shift(i).notna(), 'schock'] = \
                (adjust*df['schock'].shift(i)).round(decimals=2)
        #
        # addiere die Schocks zum Schock-Feature
        df.loc[df['schock'].notna(), name] = df[name] + df['schock']
        #
        # Verschmutzen der EcoFeatures - damit wir nicht nur konstante Werte
        # in einem Batch bekommen (dies führt zu loss)
        std = df[name].std()
        df[name] = df[name].apply(lambda x: x + np.random.normal(loc=0.0, 
                                                                 scale=std*0.05))
        # und zum Schluß Löschen der Hilfsspalten
        df = df.drop(['ttime', 'schock',  'Actual', 'Forecast'], axis=1)
      
        return df
    


    def prepareEcoData(self, path, fn):
        #
        # Einlesen der Daten
        df = pd.read_csv(path + fn,
                         dtype={'Actual': str, 
                                'Forecast': str,
                                'Previous': str})
        df.rename({'Release Date':'Date'}, inplace=True, axis=1)
        #
        # Bereinigen der Daten
        df['Date'] = df['Date'].apply(self.cleanEcoDate)
        df = df.apply(self.missingEcoData, axis = 1)
        df['Actual'] = df['Actual'].apply(self.cleanEcoPct)
        df['Forecast'] = df['Forecast'].apply(self.cleanEcoPct)
        #
        # Zusammenfassen von Date und Time
        df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        #
        # Umrechung von lokaler auf GMT-Zeiten
        df['time'] = df['time'] + pd.to_timedelta(5, unit='h')
        #
        # Zeit-Index Festlegen
        df.set_index(pd.DatetimeIndex(df.time), inplace=True)
        #
        # Löschen der ueberfluessigen Spalten
        df.drop(['Date', 'Time', 'Previous', 'time'], axis=1, inplace=True)
        #
        # zur Sicherheit nochmals sortieren
        df = df.sort_values(by='time')
        
        return df
       
        
       
    def addEcoData(self):
        
        epath = self.processedpath + 'kalender/'
        
        df_cal = pd.read_csv(epath + 'Kalender_Features.csv')
        #
        # Feature Name und Dateiname in eine Liste
        features = df_cal[['FeatureName', 'FileName']].values.tolist()
        features = [[f[0].strip(), f[1].strip()] for f in features]
        #
        # Schleife über der Feature-Liste
        for feature in features:
            print('processing Feature', feature)
            if (feature[0][:7] != 'Schock_'):
                #
                # Verarbeitung der normalen Feature
                df_feature = self.prepareEcoData(epath, feature[1])
                self.df_all = self.mergeEcoFeature(df_feature, feature[0])
            else:
                #
                # Verarbeitung der Schocks
                fn_l = feature[1].split(',')
                self.df_all[feature[0]] = 0
                for fn in fn_l:
                    print('processing Schock', fn)
                    df_schock = self.prepareEcoData(epath, fn.strip())
                    self.df_all = self.mergeEcoSchock(df_schock, 
                                                      feature[0],
                                                      fn.strip())
                    
                print(feature[0], 
                      self.df_all[feature[0]].min(), 
                      self.df_all[feature[0]].max())



    def formatFinalVersion(self,
                           training,
                           startDt = '2008-01-01 00:00:00',
                           endDt = '2023-02-16 23:55:00'):
        #
        # einige Cleanups müssen noch gemacht werden
        cpath = self.processedpath + 'forexCandle/'

        print('Anzahl Beobachtungen vor Cleanup', len(self.df_all))
        sys.stdout.flush()

        
        self.df_all = self.df_all[self.featureCols]
        self.df_all = self.df_all.loc[self.df_all.index >= startDt]
        self.df_all = self.df_all.loc[self.df_all.index <= endDt]
        
        # Weihnachten, Sylvester und Neujahr müssen weg
        self.df_all = self.df_all[(self.df_all.index.month != 12) \
                                  | (self.df_all.index.day != 24)]
        self.df_all = self.df_all[(self.df_all.index.month != 12) \
                                  | (self.df_all.index.day != 25)]
        self.df_all = self.df_all[(self.df_all.index.month != 12) \
                                  | (self.df_all.index.day != 31)]
        self.df_all = self.df_all[(self.df_all.index.month != 1) \
                                  | (self.df_all.index.day != 1)]

        print('Anzahl Beobachtungen nach Cleanup', len(self.df_all))
        n_nan = self.df_all.isnull().sum().sum()
        print('Anzahl fehlender Werte =', n_nan)
        self.df_all[self.df_all.isnull()].to_csv(cpath + 'df_all_isnull.csv')
        self.df_all.to_csv(cpath + 'df_all.csv')
        
        if training:
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
        else:
            #
            # Mittelwert und Std müssen gelesen werden
            df_mean = pd.read_pickle(cpath + 'df_all_mean.pkl')
            df_std = pd.read_pickle(cpath + 'df_all_std.pkl')
 
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
        dff.to_csv(cpath + 'FullDayCheck.csv')
       
        
       
    def calcSingleReward(self, diffs, price, i, action):
        #
        # Berechnet den Reward an der Stelle i im Tag
        #
        r = 0
        if (action == 1):
            # reward for opening a Long position
            for j in range(self.bt-2, i, -1):
                r = self.gamma*(r + diffs[j])
            r += diffs[i] - self.tradeFee
        elif (action == 2):
            # reward for opening a short position
            for j in range(self.bt-2, i, -1):
                r = self.gamma*(r - diffs[j])
            r += -diffs[i] - self.tradeFee
                
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
        diffs = np.diff(price, append=price[-1])*self.pip
        #
        # Schleife über den Tag
        reward = np.full((self.bt, 3), 0.)
        for i in range(self.bt-1):
            reward[i, 1] += self.calcSingleReward(diffs, price, i, 1)
            reward[i, 2] += self.calcSingleReward(diffs, price, i, 2)
        
        position = np.array(self.bt*[0])
        for i in range(self.ts, self.bt):
            position [i] = random.choice([0.0, 0.5, 1.0])

        rw = np.c_[position, reward]

        df_day = pd.concat([df_day, 
                            pd.DataFrame(rw, 
                                         columns=['Position',
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
        dt_data = df['time'].values.astype(str)
        
        if (len(df) != self.bt):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht', self.bt)
            
        x = np.array([])
        y = np.array([])    
        dtd = np.array([])
        for i in range(ts+1, (bs+ts+1)):
            x = np.append(x, x_data[i-ts:i])
            y = np.append(y, y_data[i-1])
            dtd = np.append(dtd, dt_data[i-1])

        x = x.reshape(bs, ts, len(feature_list))
        y = y.reshape(bs, 3)
        dtd = dtd.reshape(bs, 1)
        
        return x, y, dtd
    


    def packRLXY(self, df, batch_id, bs, ts, feature_list, target):
        
        x_data = df[feature_list].values
        y_data = df[target].values
        dt_data = df['time'].values.astype(str)
        close_data = df['Close'].values
        
        if (len(df) != self.bt):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht', self.bt)
            
        x = np.array([])
        y = np.array([])    
        dtd = np.array([])
        close = np.array([])
        for i in range(ts+1, (bs+ts+1)):
            x = np.append(x, x_data[i-ts:i])
            y = np.append(y, y_data[i-1])
            dtd = np.append(dtd, dt_data[i-1])
            close = np.append(close, close_data[i-1])

        x = x.reshape(bs, ts, len(feature_list))
        y = y.reshape(bs, 3)
        dtd = dtd.reshape(bs, 1)
        close = close.reshape(bs, 1)
        all_data = np.array((batch_id, dtd, close, x, y), dtype=object)

        return all_data
    


    def createABT(self, training):
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
            
            if training:
                #
                # packe die Daten in eine Sequenz der Länge 72 für initiales Training
                x, y, dt = self.packInitXY(df_day, 
                                           self.bs, 
                                           self.ts, 
                                           self.featureList, 
                                           self.target)
                
                np.save(self.finalpath + 'X/x-' + day.strftime('%Y-%m-%d') + '.npy', x)
                np.save(self.finalpath + 'Y/y-' + day.strftime('%Y-%m-%d') + '.npy', y)
                np.save(self.finalpath + 'Dt/dt-' + day.strftime('%Y-%m-%d') + '.npy', dt)
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
            else:
                #
                # packe die Daten in eine Sequenz von 72 für die Vorhersage
                predict_data = self.packRLXY(df_day,
                                            self.days.index(day),
                                            self.bs, 
                                            self.ts, 
                                            self.featureList, 
                                            self.target)
                
                np.save(self.finalpath + 'exploit-data/day-' + \
                        day.strftime('%Y-%m-%d') + '.npy', predict_data)
                
            
            
    
    def fitInitialModel(self):
        #
        # trainiert das initial RL-Modell
        #
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
        model = Sequential()
                
        model.add(LSTM(units=self.units, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       recurrent_regularizer=l2(0.01),
                       kernel_initializer='truncated_normal',
                       batch_input_shape=(self.bs, 
                                          self.ts, 
                                          ff)))
                
        model.add(Dense(units=3, 
                        kernel_initializer='truncated_normal'))
            
        model.compile(loss=tf.keras.losses.Huber(), 
                      optimizer=SGD(),
                      metrics=['mae'])
        
        model.summary()
        K.set_value(model.optimizer.learning_rate, self.learning_rate)
        sys.stdout.flush()
        
        loss = 0.0
        mae = 0.0
        val_loss = 0.0
        val_mae = 0.0
        niter = 3875
        ntrain = 300
        ntest = 30
        nepoch = 1
        i = 1
        for j in range(ntrain, niter, 5):
            x_train = np.array([])
            y_train = np.array([])
            x_test = np.array([])
            y_test = np.array([])
            days = np.random.choice(range(j), size=ntrain, replace=False)
            for day in days:
                x_train = np.append(x_train, np.load(xpath + xfiles[day]))
                y_train = np.append(y_train, np.load(ypath + yfiles[day]))

            for i in range(j, j+ntest):
                x_test = np.append(x_test, np.load(xpath + xfiles[i]))
                y_test = np.append(y_test, np.load(ypath + yfiles[i]))

            x_train = x_train.reshape(ntrain*self.bs, self.ts, ff)
            y_train = y_train.reshape(ntrain*self.bs, 3)
            x_test = x_test.reshape(ntest*self.bs, self.ts, ff)
            y_test = y_test.reshape(ntest*self.bs, 3)
            fitted = model.fit(x_train, y_train, 
                               epochs=nepoch, 
                               batch_size=self.bs, 
                               #verbose=0,
                               shuffle=False,
                               validation_data=(x_test, y_test))
            
            loss = fitted.history['loss'][nepoch-1]
            mae = fitted.history['mae'][nepoch-1]
            val_loss = fitted.history['val_loss'][nepoch-1]
            val_mae = fitted.history['val_mae'][nepoch-1]
            
            model.save_weights(self.finalpath + 'model/init/initModel')
            
            print('Training Iteration %4i: loss=%2.3f, mae=%2.3f' % (i, 
                                                                     loss, 
                                                                     mae))
            print('Test Iteration %4i: loss=%2.3f, mae=%2.3f' % (i, 
                                                                 val_loss, 
                                                                 val_mae))
            sys.stdout.flush()
            i += 1
         

    
    def rate_scheduler(self, i, rate):
        
        return rate * np.exp(-(1/(100000 + i)))
       
        
       
    def setIterRates(self, trader, i):
        
        self.exploration_rate = self.rate_scheduler(i, self.exploration_rate)
        self.dr = self.rate_scheduler(i, self.dr)
        self.dreaming_rate = 0.84 - self.dr
        trader.set_rates(self.dreaming_rate, self.exploration_rate)
    
         
 
    def fitForexDayTrader(self):
                  
        name = 'ongoingModel'
        maxAvgTotalReward = -999.99
        training = True
        
        #######################################
        ### change re-start parameter here ####
        self.dr = 0.3
        self.exploration_rate = 0.99
        init_weight_name = 'initModel'
        start = 1
        end = 500000
        ### change re-start parameter here ####
        #######################################
        
        if (isfile(self.finalpath + 'model/ongoingModel/maxAvgTotalReward.npy')):
            maxAvgTotalReward = np.load(self.finalpath \
                                    + 'model/ongoingModel/maxAvgTotalReward.npy')
        
        print('Maximum Total Reward = %5.2f' % (maxAvgTotalReward))
        env_dict = gym.envs.registration.registry.env_specs.copy()
        for env in env_dict:
            if 'forex-v0' in env:
                del gym.envs.registration.registry.env_specs[env]
        
        register(id='forex-v0', entry_point='forexTrader.envs.forexEnv:ForexEnv')
        env = gym.make(id='forex-v0', 
                       data_path=self.finalpath + 'RL-data/', 
                       window_size=self.ts, 
                       batch_size=self.bs,
                       training=training)
        
        trader = ForexDayTrader(env, 
                                self.finalpath, 
                                init_weight_name,
                                self.units,
                                self.bs,
                                self.ts,
                                len(self.featureList))
        result = []
        for i in range(start, end):
            
            self.setIterRates(trader, i)    
            action_hold = 0
            action_buy = 0
            action_sell = 0
            curr_state = env.reset().reshape(1,
                                             self.ts,
                                             len(self.featureList))
            for j in range(self.bs):
                # batch loop
                action = trader.act(curr_state)
                new_state, reward, done, info = env.step(action)
                new_state = new_state.reshape(1,
                                              self.ts,
                                              len(self.featureList))
                
                trader.remember(curr_state, action, reward, new_state, done)
        
                if (action == 0):
                    action_hold += 1
                elif (action == 1):
                    action_buy += 1
                elif (action == 2):
                    action_sell += 1
                    
                if done:
                    trader.day_dream(info["datetime"], 
                                     info["batch_id"], 
                                     info["total_reward"])
                else:
                    curr_state = new_state
                    
            trader.dream()
            
            result.append([i, 
                           info["datetime"][:10], 
                           info["batch_id"], 
                           info["total_reward"],
                           info["total_reward_long"],
                           info["total_reward_short"],
                           info["n_long_trades"],
                           info["n_short_trades"],
                           action_hold,
                           action_buy,
                           action_sell])
        
            trader.replay(i)
            
            if (i % 1000 == 0):
                df = pd.DataFrame(result, columns=['Iteration',
                                                   'DateTime',
                                                   'Batch-ID',
                                                   'Avg-Total-Reward',
                                                   'Long-Reward',
                                                   'Short_Reward',
                                                   '#Long_Trades',
                                                   '#Short_Trades',
                                                   'Avg-Hold-Action',
                                                   'Avg-Buy-Action',
                                                   'Avg-Sell-Action'])
                
                df.to_csv(self.finalpath + 'result/rl-results-' + str(i) + '.csv', 
                          index=False)
                result = []
        
                avgTotalReward = np.mean(df["Avg-Total-Reward"])
                trader.train_target()
                print('Average Total Reward = %5.2f Max Total Reward = %5.2f' \
                      % (avgTotalReward, maxAvgTotalReward))
                print('exploration_rate = %5.4f dreaming_rate = %5.4f' \
                      % (self.exploration_rate, self.dreaming_rate))
                    
                if (avgTotalReward > maxAvgTotalReward):
                # only save model with maximum Average Total Reward
                    print("saving model with avgTotalReward = %5.2f to %32s" \
                          % (avgTotalReward, name))
                    trader.save_model(name)
                    maxAvgTotalReward = avgTotalReward
                    np.save(self.finalpath + 'model/ongoingModel/maxAvgTotalReward.npy',
                            maxAvgTotalReward)
                    
         
            
    def create_model(self, bs):

        model = Sequential()
                
        model.add(LSTM(units=self.units, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       recurrent_regularizer=l2(0.01),
                       kernel_initializer='truncated_normal',
                       batch_input_shape=(bs, 
                                          self.ts, 
                                          len(self.featureList))))
                
        model.add(Dense(units=3, 
                        kernel_initializer='truncated_normal'))
            
        model.compile(loss=tf.keras.losses.Huber(), 
                      optimizer=SGD(),
                      metrics=['mae'])
       
        model.load_weights(self.ongoing_model)
        
        print('weights loaded from ' + self.ongoing_model)
        
        model.summary()
        sys.stdout.flush()
        
        return model
    
    
    
    def exploitForexDayTrader(self):
        
        decision_model = self.create_model(1)
        training = False
        
        env_dict = gym.envs.registration.registry.env_specs.copy()
        for env in env_dict:
            if 'forex-v0' in env:
                del gym.envs.registration.registry.env_specs[env]
        
        register(id='forex-v0', entry_point='forexTrader.envs.forexEnv:ForexEnv')
        dpath = self.finalpath + 'exploit-data/'
        env = gym.make(id='forex-v0', 
                       data_path=dpath, 
                       window_size=self.ts, 
                       batch_size=self.bs,
                       training=training)
        
        
        dfiles = [f for f in listdir(dpath) if isfile(join(dpath, f)) \
                  and f[:3] == 'day']
        n_days = len(dfiles)
        
        result = []
        print(n_days)
        for day in range(n_days):
            print(day)
            #
            # neuer Tag, neues Glück
            state = env.reset().reshape(1,
                                        self.ts,
                                        len(self.featureList))
            
            for j in range(self.bs):
                
                # batch loop
                action = np.argmax(decision_model.predict(state, batch_size=1)[0])
                new_state, reward, done, info = env.step(action)
                new_state = new_state.reshape(1,
                                              self.ts,
                                              len(self.featureList))
                print(info)

                if done:
                    result.append([day, 
                                   info["datetime"][:10], 
                                   info["batch_id"], 
                                   info["total_reward"],
                                   info["total_reward_long"],
                                   info["total_reward_short"],
                                   info["n_long_trades"],
                                   info["n_short_trades"]])
                else:
                    state = new_state
            
            
     
        pd.DataFrame(result, columns=['Iteration',
                                      'DateTime',
                                      'Batch-ID',
                                      'Total-Reward',
                                      'Long-Reward',
                                      'Short_Reward',
                                      '#Long_Trades',
                                      '#Short_Trades']).to_csv(self.finalpath \
                                                               + 'result/exploitResults.csv', 
                                                               index=False)
        
                                                               
                                                               
fileStartDt = '01.01.2007'
periodStartDt = '2008-01-02 00:00:00'
periodEndDt = '2023-03-30 23:55:00'
training = True

eurusdData = forexRLDayTrading('EURUSD')
#eurusdData.processCandle(startDt = fileStartDt)
#eurusdData.addEcoData()
#eurusdData.formatFinalVersion(training,
#                              startDt = periodStartDt,
#                              endDt = periodEndDt)
#eurusdData.createABT(training)
#eurusdData.fitInitialModel()
eurusdData.fitForexDayTrader()

#eurusdData.exploitForexDayTrader()



