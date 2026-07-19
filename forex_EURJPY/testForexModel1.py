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
import gc
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



class forexModel:
    
    techindCols = ['zeitIndex', 'scaledClose',  'round_lot', 'WL', 
                   'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi', 
                   'macd_30Min', 'macd_d_30Min', 'bb_h_30Min', 'bb_l_30Min', 
                   'rsi_30Min', 'WL_6H', 'macd_6H', 'macd_d_6H', 'bb_h_6H', 
                   'bb_l_6H', 'rsi_6H']
    
    ecoCols = ['CPI_US', 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
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

        
    scaledCols = techindCols + ecoCols
    
    target = ['Reward-Buy', 'Reward-Sell']

    featureTest = scaledCols
    
    outName = 'lstm-ts24-ff52-u52-ecospot.csv'

    
    def __init__(self, currency):
        
        self.currency = currency
        #self.basepath = '/Users/uwe.muller/Hope/data/'
        self.basepath = '/home/chitlom/data/'
        self.rawpath = self.basepath + 'raw/'
        self.processedpath = self.basepath + 'processed/' + self.currency + '/'
        self.finalpath = self.basepath + 'final/' + self.currency + '/'
        self.schockStandardize = []
        self.bt = 288
        self.ts = 24
        self.bs = self.bt - self.ts
        self.units = 52
        self.tradeFee = 1.5
        self.pip = 10000
        self.gamma = 0.6
        self.learning_rate = 0.0005
        self.ntrain = 100

        
        
    def normalDF(self, x, m=0., s=5.):
        
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
            df = pd.concat([df, dfi])
            
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
        self.df_all = df_all[['Close'] + self.techindCols]



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
        # Berechnen des standardisierten Schocks
        df['schock'] = ((df['Actual'] - df['Forecast'] - s_mean)/s_std) \
                         .round(decimals=2)
        #
        # addiere die Schocks zum Schock-Feature
        df.loc[df['schock'].notna(), name] = df[name] + df['schock']
        #
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
        
        

    def calcInitReward(self, df_all, days):
        #
        # Methode berechnet die Rewards und fügt sie den Daten hinzu
        #
        df_tmp = pd.DataFrame()
        #
        # Schleife über alle Tage
        for day in days:
            df_day = df_all[df_all.index.date == day]
            df_day.reset_index(inplace=True)
            if (len(df_day) != self.bt):
                sys.exit('Anzahl Intervalle pro Tag entspricht nicht {}'.format(self.bt))
            #
            # Berechnen der Differenzen
            price = df_day['Close'].values
            diffs = np.diff(price, append=price[-1])*self.pip
            #
            # Schleife über den Tag
            reward = np.full((self.bt, 2), 0.)
            for i in range(self.bt-1):
                reward[i, 0] += self.calcSingleReward(diffs, price, i, 1)
                reward[i, 1] += self.calcSingleReward(diffs, price, i, 2)
                
            df_day = pd.concat([df_day, 
                                pd.DataFrame(reward, columns=['Reward-Buy',
                                                              'Reward-Sell'])], 
                               axis=1)
            df_tmp = pd.concat([df_tmp, df_day], axis=0)
            
        df_tmp.set_index(pd.DatetimeIndex(df_tmp.time), inplace=True, drop=True)
        
        return df_tmp
        




    def formatFinalVersion(self,
                           startDt = '2008-01-01 00:00:00',
                           endDt = '2023-02-16 23:55:00'):
        #
        # einige Cleanups müssen noch gemacht werden
        cpath = self.processedpath + 'forexCandle/'

        print('Anzahl Beobachtungen vor Cleanup', len(self.df_all))
        sys.stdout.flush()

        
        self.df_all = self.df_all[['Close'] + self.scaledCols]
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
        self.df_all.to_csv(cpath + 'df_all.csv', index=False)
        
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
        #
        # Berechnen der Anzahl und Liste der Tage
        dff = self.df_all.groupby(self.df_all.index.date).count()
        print('Anzahl Tage insgesamt:', len(dff))
        sys.stdout.flush()
        df_days = dff.index
        days = sorted(df_days.values.tolist())
        #
        # Erzeugen des initialen Rewards
        self.df_all = self.calcInitReward(self.df_all, days)
        self.df_all.to_csv(cpath + 'df_all_scaled.csv', index=False)
 
        #
        # und zum Schluß noch ein Check, ob jeder Tag 288 Einträge hat
        dff = dff[dff['Close'] < self.bt]
        dff.to_csv(cpath + 'FullDayCheck.csv')
               
       

    def packInitXY(self, df, bs, ts, feature_list, target):
        
        
        x_data = df[feature_list].values
        y_data = df[target].values
        
        if (len(df) != self.bt):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht', self.bt)
            
        x = np.array([])
        y = np.array([])    
        for i in range(ts+1, (bs+ts+1)):
            x = np.append(x, x_data[i-ts:i])
            y = np.append(y, y_data[i-1])

        x = x.reshape(bs, ts, len(feature_list))
        y = y.reshape(bs, 2)
        
        return x, y
    
    

    def createModel(self):
        
        ff = len(self.featureTest)
        print('Anzahl Features =', ff)
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
                
        model.add(Dense(units=2, 
                        kernel_initializer='truncated_normal'))
            
        model.compile(loss=tf.keras.losses.Huber(), 
                      optimizer=SGD(),
                      metrics=['mae'])
        
        model.summary()
        K.set_value(model.optimizer.learning_rate, self.learning_rate)
        sys.stdout.flush()
        
        return model
               


    def countEqSign(self, y_true, y_pred):
        #
        # zählt die Anzahl der gleichen Vorzeichen für y_true und y_pred
        #
        return np.sum(abs(np.sign(y_true) + np.sign(y_pred))/2)


    
    def fitModel(self, iteration, model, x_base, y_base):
        #
        # trainiert und testet das Model
        #

        #
        # Aufteilen der Daten in Train und Test
        ntrain = self.ntrain
        ntest = 30
        loss = 999.
        mae = 999.
        val_loss = 999.
        val_mae = 999.
        cntEqSign = -999
        
        if (iteration > 1000):
            #
            # zufällige Selektion der Trainingsdaten
            itrain = np.random.choice(range(iteration-ntest), 
                                      size=ntrain, 
                                      replace=False)
            #
            # lesen der x-y-Trainingsdaten
            x_train = np.array([])
            y_train = np.array([])
            for it in itrain:
                x_train = np.append(x_train, 
                                    np.load(self.finalpath + 'X/x-' \
                                            + self.days[it].strftime('%Y-%m-%d') \
                                            + '.npy'))
                y_train = np.append(y_train, 
                                    np.load(self.finalpath + 'Y/y-' \
                                            + self.days[it].strftime('%Y-%m-%d') \
                                            + '.npy'))
            # und noch ein reshape für train ...
            x_train = x_train.reshape(ntrain*self.bs, 
                                      self.ts, 
                                      len(self.featureTest))
            y_train = y_train.reshape(ntrain*self.bs, 2)
            #
            # lesen der x-y-Testdaten (immer am Ende der Iteration)
            itest = range(iteration-ntest, iteration)
            x_test = np.array([])
            y_test = np.array([])
            for it in itest:
                x_test = np.append(x_test, 
                                   np.load(self.finalpath + 'X/x-' \
                                           + self.days[it].strftime('%Y-%m-%d') \
                                        + '.npy'))
                y_test = np.append(y_test, 
                                   np.load(self.finalpath + 'Y/y-' \
                                           + self.days[it].strftime('%Y-%m-%d') \
                                           + '.npy'))
            # un d noch ein reshape für test
            x_test = x_test.reshape(ntest*self.bs,
                                    self.ts,
                                    len(self.featureTest))
            y_test = y_test.reshape(ntest*self.bs, 2)
            #
            # Fit Model
            fitted = model.fit(x_train, y_train, 
                               epochs=1, 
                               batch_size=self.bs, 
                               #verbose=0,
                               shuffle=False,
                               validation_data=(x_test, y_test))  
            #
            # Ausleden der Trainings-Metriken
            loss = fitted.history['loss'][0]
            mae = fitted.history['mae'][0]
            val_loss = fitted.history['val_loss'][0]
            val_mae = fitted.history['val_mae'][0]
            #
            # Erzeugen der Test-Metrik
            y_pred = model.predict(x_test, batch_size=ntest*self.bs)
            cntEqSign = self.countEqSign(y_test, y_pred)/(2*ntest*self.bs)
            print(iteration, y_pred.shape, loss, val_loss, mae, val_mae, cntEqSign)
            sys.stdout.flush()
            
            if (iteration % 10 == 0):
                del x_test
                del y_test
                del x_train
                del y_train
                gc.collect()

        return loss, mae, val_loss, val_mae, cntEqSign

        

    def readAll_Data(self):
        #
        # Erzeugen der gepakten XY-Daten und speichern auf Platte
        #
        cpath = self.processedpath + 'forexCandle/'
        c = 'df_all_scaled.csv'
        self.df_all= pd.read_csv(cpath + c, parse_dates=['time'], 
                                 date_format='%Y-%m-%d %H:%M:%S')
        
        self.df_all.set_index(pd.DatetimeIndex(self.df_all.time, name='time'), 
                              inplace=True, drop=True)
        #
        # Erzeugen der Tage
        dff = self.df_all.groupby(self.df_all.index.date).count()
        print('Anzahl Tage insgesamt:', len(dff))
        sys.stdout.flush()
        df_days = dff.index
        self.days = sorted(df_days.values.tolist())
        
        
    def createPackedData(self):
        #
        # packt die Daten und speichert sie auf Platte
        #
        for day in self.days:
            #
            # extrahiere den Tag
            df_day = self.df_all[self.df_all.index.date == day]            
            #
            # packe die Daten in eine Sequenz mit der Länge TS
            x, y = self.packInitXY(df_day, 
                                   self.bs, 
                                   self.ts, 
                                   self.featureTest, 
                                   self.target)
            
            np.save(self.finalpath + 'X/x-' + day.strftime('%Y-%m-%d') + '.npy', x)
            np.save(self.finalpath + 'Y/y-' + day.strftime('%Y-%m-%d') + '.npy', y)
            
       
        

    def testModel(self):
        #
        # Testen der unterschiedlichen Modellvariantionen
        #
        #
        # Freigabe nicht mehr benötigter Speicher
        del self.df_all
        gc.collect()
        
        # Initialisierungen
        model = self.createModel()
        out = []
        x_base = np.array([])
        y_base = np.array([])
        #
        # Schleife über alle Tage
        for day in self.days:
            #
            # extrahiere den Tag
            iteration = self.days.index(day) + 1
            #
            # trainiere das Modell und teste es
            loss, mae, val_loss, val_mae, cntEqSign = self.fitModel(iteration,
                                                                    model, 
                                                                    x_base,
                                                                    y_base)
            
            if (iteration > 1000):
                out.append([iteration, day, loss, mae, val_loss, val_mae, 
                            cntEqSign])
                out_cols = ['Iteration', 'Tag', 'loss', 'mae', 'val_loss', 
                            'val_mae', 'cntEqSign']
                
                if (iteration%500 == 0):
                    pd.DataFrame(out, columns=out_cols) \
                                 .to_csv(self.finalpath + 'testResult/' + \
                                         self.outName, index=False)
    
        pd.DataFrame(out, columns=out_cols).to_csv(self.finalpath + \
                                                   'testResult/' + \
                                                   self.outName,
                                                   index=False)
   
            

                                                               
fileStartDt = '01.01.2007'
periodStartDt = '2008-01-02 00:00:00'
periodEndDt = '2023-03-30 23:55:00'

eurusd = forexModel('EURUSD')
eurusd.processCandle(startDt = fileStartDt)
eurusd.addEcoData()
eurusd.formatFinalVersion(startDt = periodStartDt, endDt = periodEndDt)
eurusd.readAll_Data()
eurusd.createPackedData()
eurusd.testModel()


