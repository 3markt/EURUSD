#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:07:49 2023

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import datetime as dt
import ta
import sys
import psycopg2
import time
from os import listdir, makedirs
from os.path import isfile, join, isdir
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data




class myLSTM(nn.Module):
    
    def __init__(self, inputsize):
        
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=inputsize, 
                            hidden_size=6, 
                            num_layers=1, 
                            dropout=0.2,
                            batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=6, 
                            hidden_size=3, 
                            num_layers=1,                         
                            batch_first=True)
        
        self.linear = nn.Linear(3, 1)
        
        
        
    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        
        return x



class forexModel:
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'zeitIndex', 'round_lot', 
                    'WL', 'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi']
    target_col = ['target']
    
        
    
    
    def __init__(self, currency):
        
        self.currency = currency
        #self.basepath = '/Users/uwe.muller/Hope/data/'
        self.basepath = '/home/chitlom/data/'
        self.rawpath = self.basepath + 'raw/'
        self.processedpath = self.basepath + 'processed/' + self.currency + '/'
        self.finalpath = self.basepath + 'final/' + self.currency + '/'
        self.units = 52
        self.tradeFee = 1.5
        self.pip = 10000
        self.gamma = 0.6
        self.learning_rate = 0.0005



    def log(self, msg):
        
        real_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
        print(real_dt + ' ' + self.currency + ': ' + msg)
        sys.stdout.flush()
    
    
        
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
       
    
        
    def createTechInd(self, df):
        
        df['WL'] = 10000*(df['Close'] - df['Close'].shift(1))/df['Close']
        
        df['zeitIndex'] = df.index.hour + df.index.minute/60
        
        df['round_lot'] = 100*(df['Close'] - df['Close'].round(decimals=0))
        
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
        df['target'] = df['WL'].shift(-1)
                
        # Weihnachten, Sylvester und Neujahr müssen weg
        df = df[(df.index.month != 12) | (df.index.day != 24)]
        df = df[(df.index.month != 12) | (df.index.day != 25)]
        df = df[(df.index.month != 12) | (df.index.day != 31)]
        df = df[(df.index.month != 1) | (df.index.day != 1)]

        df = df.dropna()

        return df
    
    
         
    def processCandle(self, startDt = '01.01.2007', 
                      featureDT = '02.01.2008',
                      endDt = '16.01.2025 23:59:59'):
        #
        # Erzeugen der Dateiliste
        #
        cpath = self.rawpath + 'forexCandle/'
        cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f)) and \
                    f[:6] == self.currency and f[27:38] >= startDt]
        cfiles.sort()
        
        print(cfiles)
        
        df = pd.DataFrame()
        for c in cfiles:
            dfi = pd.read_csv(cpath + c, dayfirst=True, parse_dates=[0])
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
        df_5min = self.createTechInd(df)
        print(len(df_5min))
        df_5min = df_5min.loc[df_5min.index.weekday != 5]
        print(len(df_5min))

        df_5min.loc[(df_5min.index >= featureDT) \
                    & (df_5min.index <= endDt)].to_csv(path + 'min5Candle.csv')
        # Skalieren der Daten
        df_scaled_5min = (df_5min - df_5min.mean())/df_5min.std()
        df_scaled_5min.loc[(df_scaled_5min.index >= featureDT) \
                           & (df_scaled_5min.index <= endDt)].to_csv(path + 'min5Candle_scaled.csv')
        
        # Konvertierung auf 15 Minuten und technischen Indikatoren
        df_15min = self.convertCandle(df_5min, '15min')
        df_15min = self.createTechInd(df_15min)
        print(len(df_15min))
        df_15min = df_15min.loc[df_15min.index.weekday != 5]
        print(len(df_15min))
        df_15min.loc[(df_15min.index >= featureDT) \
                     & (df_15min.index <= endDt)].to_csv(path + 'min15Candle.csv')
        # Skalieren der Daten
        df_scaled_15min = (df_15min - df_15min.mean())/df_15min.std()
        df_scaled_15min.loc[(df_scaled_15min.index >= featureDT) \
                            & (df_scaled_15min.index <= endDt)].to_csv(path + 'min15Candle_scaled.csv')

        # Konvertierung auf 1 Stunde und technischen Indikatoren
        df_1h = self.convertCandle(df_5min, '1h')
        df_1h = self.createTechInd(df_1h)
        print(len(df_1h))
        df_1h = df_1h.loc[df_1h.index.weekday != 5]
        print(len(df_1h))
        df_1h.loc[(df_1h.index >= featureDT) \
                  & (df_1h.index <= endDt)].to_csv(path + 'h1Candle.csv')
        # Skalieren der Daten
        df_scaled_1h = (df_1h - df_1h.mean())/df_1h.std()
        df_scaled_1h.loc[(df_scaled_1h.index >= featureDT) \
                         & (df_scaled_1h.index <= endDt)].to_csv(path + 'h1Candle_scaled.csv')
        
        #Konvertierung auf 6 Stunden und Erzeugen der technischen Indikatoren
        df_6h = self.convertCandle(df_5min, '6h')
        df_6h = self.createTechInd(df_6h)
        print(len(df_6h))
        df_6h = df_6h.loc[df_6h.index.weekday != 5]
        print(len(df_6h))
        df_6h.loc[(df_6h.index >= featureDT) \
                  & (df_6h.index <= endDt)].to_csv(path + 'h6Candle.csv')
        # Skalieren der Daten
        df_scaled_6h = (df_6h - df_6h.mean())/df_6h.std()
        df_scaled_6h.loc[(df_scaled_6h.index >= featureDT) \
                         & (df_scaled_6h.index <= endDt)].to_csv(path + 'h6Candle_scaled.csv')
  
       

    def readData(self, timescale):
        #
        # Erzeugen der gepakten XY-Daten und speichern auf Platte
        #
        cpath = self.processedpath + 'forexCandle/'
        c = timescale + 'Candle_scaled.csv'
        df= pd.read_csv(cpath + c, dayfirst=True, parse_dates=[0])
        
        df.set_index(pd.DatetimeIndex(df.time, name='time'), inplace=True, 
                     drop=True)
        #
        # Erzeugen der Tage
        dff = df.groupby(df.index.date).count()
        print('Anzahl Tage insgesamt:', len(dff))
        sys.stdout.flush()
        df_days = dff.index
        days = sorted(df_days.values.tolist())
        
        return days, df
        
        
        
    def packInitXY(self, df, bt, ts):
        
        
        x_data = df[self.feature_cols].values
        y_data = df[self.target_col].values
        
        if (len(df) != bt):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht', bt)
            
        x = np.array([])
        y = np.array([])  
        bs = bt - ts
        for i in range(ts+1, (bs+ts+1)):
            x = np.append(x, x_data[i-ts:i])
            y = np.append(y, y_data[i-1])

        x = x.reshape(bs, ts, len(self.feature_cols))
        y = y.reshape(bs, 1)
        
        return x, y
    

    def createPackedData(self, timescale, df, days, bt, ts):
        #
        # packt die Daten und speichert sie auf Platte
        #
        xpath = self.finalpath + '/X/' + timescale + '/'
        ypath = self.finalpath + '/Y/' + timescale + '/'
            
        for day in days:
            #
            # extrahiere den Tag
            df_day = df[df.index.date == day]            
            #
            # packe die Daten in eine Sequenz mit der Länge TS
            x, y = self.packInitXY(df_day, bt, ts)
            np.save(xpath + 'x-' + day.strftime('%Y-%m-%d') + '.npy', x)
            np.save(ypath + 'y-' + day.strftime('%Y-%m-%d') + '.npy', y)
            
            
            
    def getXY(self, xpath, ypath, bt, ts, days):
        
        x = np.array([])
        y = np.array([])
        bs = bt - ts
        
        for day in days:
            x = np.append(x, np.load(xpath + 'x-' + day + '.npy'))
            y_raw = np.load(ypath + 'y-' + day + '.npy')
            yy = np.zeros((ts, 1), float)
            y1 = np.array([])
            for yi in y_raw:
                yy = np.append(yy[1:], yi)
                yy = yy.reshape(ts,1)
                y1 = np.append(y1, yy)
            y1 = y1.reshape(bs, ts, 1)
            y = np.append(y, y1)
             
        x = torch.tensor(x.reshape(len(days)*bs, 
                                   ts, 
                                   len(self.feature_cols))).type(torch.float)
        y = torch.tensor(y.reshape(len(days)*bs, 
                                   ts, 
                                   1)).type(torch.float)
           
        return x, y
                
    
       
    def sampleXY(self, n, split, timescale, bt, ts):
        #
        # creates a sample of XY data for training and test
        #
        xpath = self.finalpath + '/X/' + timescale + '/'
        ypath = self.finalpath + '/Y/' + timescale + '/'
        alldays = [f[2:12] for f in listdir(xpath) if isfile(join(xpath, f))]
        if (len(alldays) < n):
            sys.exit('Anzahl Intervalle pro Tag entspricht nicht', len(alldays), n)
            
        alldays.sort()
        
        isample = np.random.choice(range(len(alldays)), size=n, replace=False)
        isample.sort()
        itrain = isample[:int(n*split)]
        train_days = [alldays[i] for i in itrain]
        x_train, y_train = self.getXY(xpath, ypath, bt, ts, train_days)
        
        itest = isample[int(n*split):]
        test_days = [alldays[i] for i in itest]
        x_test, y_test = self.getXY(xpath, ypath, bt, ts, test_days)
        
        return x_train, y_train, train_days, x_test, y_test, test_days



    def update_db(self, result_values):
        
        insert_sql = """INSERT INTO simple_result 
                    (currency, timescale, tday, iteration, epoch, cnt_pos, 
                     cnt_neg, cnt_true_pos_pred, cnt_true_neg_pred,
                     cnt_false_pos_pred, cnt_false_neg_pred, wl_pos, wl_neg) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                    
        delete_sql = """DELETE FROM simple_result WHERE currency = %s and 
                        timescale = %s and tday = %s and iteration = %s
                        and epoch = %s"""      

        try:
            db_connect = psycopg2.connect(database = "dreimarktdb", 
                                          user = "chitlom", 
                                          password = "Pampe1muse", 
                                          host = "localhost", 
                                          port = "5432")
            
            db_cursor = db_connect.cursor()
            db_cursor.execute(delete_sql, (result_values[0], result_values[1], 
                                           result_values[2], result_values[3], 
                                           result_values[4], ))
            db_cursor.execute(insert_sql, result_values)
            db_connect.commit()
        
        except (Exception, psycopg2.Error) as error:
            self.log('Error updateing dreimarktDB Error: ' + str(error))
    



    def calcDayMetrics(self, iteration, epoch, day, y_true, y_pred):
        #
        # zählt die folgenden Anzahlen:
        #   - korrekte positive Prognosen
        #   - falsche positive Prognosen
        #   - Anzahl positiver Werte
        #
        
        pos_true = np.copy(y_true)
        pos_true[pos_true <= 0] = 0
        pos_true[pos_true > 0] = 1
        #
        # Anzahl positiver Werte
        cntPos = np.sum(pos_true)
        #
        # positive Prognosen       
        pos_pred = np.copy(y_pred)
        pos_pred[pos_pred <= 0] = 0
        pos_pred[pos_pred > 0] = 1
        #
        # Anzahl korrekt vorhergesagter positiver Werte              
        truePosPred = pos_true + pos_pred
        # korrekt prognostizierte positive Werte haben jetzt den Wert 2
        truePosPred[truePosPred!=2] = 0
        truePosPred[truePosPred==2] = 1
        cntTruePosPred = np.sum(truePosPred)
        win_pos = np.sum(truePosPred*abs(y_true))
        #
        # Anzahl falsch vorhergesagter positiver Werte
        falsePosPred = pos_pred - pos_true
        # falsch prognostizierte positive Werte haben den Wert 1
        falsePosPred[falsePosPred!=1] = 0
        cntFalsePosPred = np.sum(falsePosPred)
        loss_pos = np.sum(falsePosPred*abs(y_true))
        wl_Pos = win_pos - loss_pos
        #
        #  negativer Werte
        neg_true = np.copy(y_true)
        neg_true[neg_true >= 0] = 0
        neg_true[neg_true < 0] = 1
        #
        # Anzahl negativer Werte
        cntNeg = np.sum(neg_true)
        #
        # Anzahl korrekt vorhergesagter negativer Werte        
        neg_pred = np.copy(y_pred)
        neg_pred[neg_pred >= 0] = 0
        neg_pred[neg_pred < 0] = 1
        trueNegPred = neg_true + neg_pred
        trueNegPred[trueNegPred!=2] = 0
        trueNegPred[trueNegPred==2] = 1
        cntTrueNegPred = np.sum(trueNegPred)
        win_neg = np.sum(trueNegPred*abs(y_true))
        #
        # Anzahl falsch vorhergesagter negativer Werte
        falseNegPred = neg_pred - neg_true
        falseNegPred[falseNegPred!=1] = 0
        cntFalseNegPred = np.sum(falseNegPred)
        loss_neg = np.sum(falseNegPred*abs(y_true))
        wl_Neg = win_neg - loss_neg
        
        self.update_db((self.currency, self.timescale, day, 
                        iteration, epoch, 
                        cntPos.item(), cntNeg.item(), 
                        cntTruePosPred.item(), cntTrueNegPred.item(), 
                        cntFalsePosPred.item(), cntFalseNegPred.item(), 
                        wl_Pos.item(), wl_Neg.item()))
        
               
               
    def myMetrics(self, iteration, epoch, ypred, ytrue, days, bs):
        
        for day in days:
            i = days.index(day) 
            self.calcDayMetrics(iteration, 
                                epoch, 
                                day,
                                ypred[i*bs:(i+1)*bs], 
                                ytrue[i*bs:(i+1)*bs])
            
    
    
    def rate_scheduler(self, i, rate): 
        rate = rate*np.exp((-0.6/(10 + 1*i)))
        return rate


    
    
    def fitModel(self, timescale, bt, ts, n_fit, split, iternum):
        
        self.timescale = timescale

        ff = len(self.feature_cols)
        model = myLSTM(ff)
        lr = 0.05
        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss_fn = nn.MSELoss()
        init_epochs = 1
        
        for i in range(iternum):
            x_train, y_train, train_days, x_test, y_test, test_days \
                = self.sampleXY(n_days, split, timescale, bt, ts)
            
            
            train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), 
                                           shuffle=False, 
                                           batch_size=23,
                                           drop_last=True)
            
            epochs = init_epochs + int(i/100)

            for ii in range(epochs):
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
                    
                lr = self.rate_scheduler(i+ii, lr)
                optimizer.param_groups[0]['lr'] = lr
                
                self.myMetrics(i, ii, 
                               y_test[:,-1,:].detach().numpy().reshape((bt-ts)*(n_days - int(n_days*split)), 1, 1), 
                               y_pred[:,-1,:].detach().numpy().reshape((bt-ts)*(n_days - int(n_days*split)), 1, 1),
                               test_days, bt-ts)
 
                print("Iteration %d: lr %.6f train RMSE %.4f, test RMSE %.4f" \
                      % (i, lr, rmse, val_rmse))

                sys.stdout.flush()


                                                               
fileStartDt = '01.01.2007'
periodStartDt = '2008-01-02 00:00:00'
endDt = '2025-01-16 23:59:59'

eurusd = forexModel('EURUSD')

eurusd.processCandle(startDt = fileStartDt, 
                     featureDT = periodStartDt,
                     endDt = endDt)

days_5min, df_5min = eurusd.readData('min5')
days_15min, df_15min = eurusd.readData('min15')
days_1h, df_1h = eurusd.readData('h1')
days_6h, df_6h = eurusd.readData('h6')
eurusd.createPackedData('min5', df_5min, days_5min, 288, 12)
eurusd.createPackedData('min15', df_15min, days_15min, 96, 6)
eurusd.createPackedData('h1', df_1h, days_1h, 24, 3)
eurusd.createPackedData('h6', df_6h, days_6h, 4, 1)

iter_num = 500
n_days = 2000
split = 0.8

eurusd.fitModel('min5', 288, 12, n_days, split, iter_num)
eurusd.fitModel('min15', 96, 6, n_days, split, iter_num)
eurusd.fitModel('h1', 24, 3, n_days, split, iter_num)
eurusd.fitModel('h6', 4, 1, n_days, split, iter_num)

