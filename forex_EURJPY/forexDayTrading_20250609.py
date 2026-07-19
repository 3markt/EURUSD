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
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from os import listdir, makedirs
from os.path import isfile, join, isdir
from gym.envs.registration import register
from forexTrader.envs import forexEnv
from forexTrader.agent import ForexDayTrader 



class myLSTM(nn.Module):
    
    def __init__(self, inputsize):
        
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=inputsize, 
                            hidden_size=12, 
                            num_layers=2, 
                            dropout=0.2,
                            batch_first=True)
        
        self.linear = nn.Linear(12, 3)
        
        
        
    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x = self.linear(x)
        
        return x



class forexDayTrading:
    
    techCols = ['zeitIndex', 'Open', 'High', 'Low', 'scaledClose', 'round_lot', 
                'WL', 'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi']
    
    target = ['Reward-Hold', 'Reward-Buy', 'Reward-Sell']
    
    scaledCols = techCols + target

    featureList = ['Position', 'currProfit'] + techCols
    
    allCols = ['Close'] + featureList + target
    
    
    def __init__(self, currency):
        
        self.currency = currency
        #self.basepath = '/Users/uwe.muller/Hope/data/'
        self.basepath = '/home/chitlom/data/'
        self.rawpath = self.basepath + 'raw/'
        self.processedpath = self.basepath + 'processed/' + self.currency + '/'
        self.finalpath = self.basepath + 'final/' + self.currency + '/'
        self.bt = 288
        self.ts = 12
        self.bs = self.bt - self.ts
        self.ff = len(self.featureList)
        self.tradeFee = 1.0
        self.pip = 100
        self.gamma = 0.75
        self.learning_rate = 0.1
        self.best_model = self.finalpath + 'model/bestModel/bestModel'

    
        
    def createTechInd(self, df):

        df['scaledClose'] = df['Close']
        
        df['WL'] = 10000*(df['Close'] - df['Close'].shift(1))/df['Close']
        
        df['zeitIndex'] = df.index.hour + df.index.minute/60
        
        df['round_lot'] = 100*(df['Close'] - df['Close'].round(decimals=2))
        
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
    
    
        
    def processCandle(self, startDt = '01.01.2001'):
        #
        # Erzeugen der Dateiliste
        #
        cpath = self.rawpath + 'forexCandle/'
        cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f)) and \
                    f[:6] == self.currency and f[27:38] >= startDt]
        cfiles.sort()
        for fl in cfiles:
            print(fl)
        
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
        
        # Erzwugen der technischen Indikatoren
        df_all = self.createTechInd(df)

        # Fehlende Werte müssen nach Vorne aufgefüllt werden
        df_all = df_all.ffill()
        # Fehlende Werte am Anfang werden gelöscht
        df_all = df_all.dropna()
        # speichern als objekt und hinzufügen der original Close-Spalte, die
        # für den RL-Algorithmus benötigt wird
        self.df_all = df_all[['Close'] + self.techCols]

        
          
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
        
        position = np.array(self.bt*[0.]).astype(float)
        currProfit = np.array(self.bt*[0.]).astype(float)
        
        rw = np.c_[position, currProfit, reward]

        df_day = pd.concat([df_day, 
                            pd.DataFrame(rw, 
                                         columns=['Position',
                                                  'currProfit',
                                                  'Reward-Hold', 
                                                  'Reward-Buy',
                                                  'Reward-Sell'])], 
                           axis=1)
        
        return df_day
        


    def formatFinalVersion(self,
                           training,
                           startDt = '2004-01-02 00:00:00',
                           endDt = '2024-12-31 23:55:00'):
        #
        # einige Cleanups müssen noch gemacht werden
        cpath = self.processedpath + 'forexCandle/'
        #
        print('Anzahl Beobachtungen vor Cleanup', len(self.df_all))
        #
        self.df_all = self.df_all.loc[self.df_all.index >= startDt]
        self.df_all = self.df_all.loc[self.df_all.index <= endDt]
        #        
        print('Anzahl Beobachtungen nach Cleanup', len(self.df_all))
        #
        # 
        n_nan = self.df_all.isnull().sum().sum()
        n_nan1 = self.df_all.isnull().sum()
        print('Anzahl fehlender Werte =', n_nan, n_nan1)
        print(len(self.df_all[self.df_all.isnull()]))
        print(self.df_all.dtypes)
        self.df_all[self.df_all.isnull()].to_csv(cpath + 'df_all_isnull.csv')
        #
        # und zum Schluß noch ein Check, ob jeder Tag 288 Einträge hat
        dff = self.df_all.groupby(self.df_all.index.date).count()
        print('Anzahl Tage insgesamt:', len(dff))
        sys.stdout.flush()
        df_days = dff.index
        self.days = df_days.values.tolist()
        dff = dff[dff['Close'] < self.bt]
        dff.to_csv(cpath + 'FullDayCheck.csv')
       
        #
        # erzeugen der initial rewards
        df = pd.DataFrame([])
        for day in self.days:
            #
            # extrahiere den Tag
            df_day = self.df_all[self.df_all.index.date == day]
            #
            # Berechne die initiale Belohnung
            df = pd.concat([df, self.calcInitReward(df_day)])

        df = df.sort_values(by='time')
        self.df_all = df.set_index(pd.DatetimeIndex(df['time']))
        self.df_all[self.allCols].to_csv(cpath + 'df_all.csv')
        
        #
        # Skalieren der Daten
        if training:
            #
            # Berechne Mittelwert und Std für Skalierung und Speichere diese
            # als pickle-files
            #
            # Mean zunächst für die Features ...
            df_mean = self.df_all[self.techCols].mean()
            #
            # ... dann für die Initial Rewards - diese werden aus Reward-Buy
            # genommen und auf alle 3 Rewards angewendet
            mean_reward = self.df_all['Reward-Buy'].mean().item()
                
            df_mean.loc['Reward-Hold'] = mean_reward
            df_mean.loc['Reward-Buy'] = mean_reward
            df_mean.loc['Reward-Sell'] = mean_reward
            
            df_mean.to_pickle(cpath + 'df_all_mean.pkl')
            df_mean.to_csv(cpath + 'df_all_mean.csv')
            #
            # Standardabweichung zunächst für die Features ...
            df_std = self.df_all[self.techCols].std()
            #
            # ... dann für die Initial Rewards - diese werden über Reward-Buy
            # und Reward-Sell gemittelt und auf alle 3 Rewards angewendet
            std_reward = self.df_all['Reward-Buy'].std()
            df_std.loc['Reward-Hold'] = std_reward
            df_std.loc['Reward-Buy'] = std_reward
            df_std.loc['Reward-Sell'] = std_reward
            
            df_std.to_pickle(cpath + 'df_all_std.pkl')
            df_std.to_csv(cpath + 'df_all_std.csv')
        else:
            #
            # Mittelwert und Std müssen gelesen werden
            df_mean = pd.read_pickle(cpath + 'df_all_mean.pkl')
            df_std = pd.read_pickle(cpath + 'df_all_std.pkl') 
        #
        # eigentliche Skalierung   
        print(df_mean)
        print(df_std)
        self.df_all[self.scaledCols] = (self.df_all[self.scaledCols] - df_mean)/df_std
        self.df_all = self.df_all[self.allCols]
        self.df_all.to_csv(cpath + 'df_all_scaled.csv')
 


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
            
            if training:
                #
                # packe die Daten für initiales Training
                x, y = self.packInitXY(df_day, 
                                       self.bs, 
                                       self.ts, 
                                       self.featureList, 
                                       self.target)
                
                np.save(self.finalpath + 'X/x-' + day.strftime('%Y-%m-%d') + '.npy', x)
                np.save(self.finalpath + 'Y/y-' + day.strftime('%Y-%m-%d') + '.npy', y)
                #
                # packe die Daten für RL Training
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
                # packe die Daten für die Vorhersage
                predict_data = self.packRLXY(df_day,
                                            self.days.index(day),
                                            self.bs, 
                                            self.ts, 
                                            self.featureList, 
                                            self.target)
                
                np.save(self.finalpath + 'exploit-data/day-' + \
                        day.strftime('%Y-%m-%d') + '.npy', predict_data)
                


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

        niter = 5200
        ntrain = 4000
        ntest = 200
        nepoch = 1
        i = 1
        for i in range(ntrain, niter, 200):
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
            
         

    def rate_scheduler(self, i, rate):
        
        return rate * np.exp(-(1/(500000 + i)))

        
    def rate_LRscheduler(self, i, rate): 
        
       # rate = rate*np.exp((-0.525/(10 + 1*i)))
        rate = rate * np.exp(-(1/(50000 + i)))
        return rate
            
    
    def rate_DRscheduler(self, i, rate): 
        
        rate = rate * np.exp(-(1/(200000 + i)))
        return rate
    
    def rate_ERscheduler(self, i, rate): 
        
        if (rate > 0.1):
            rate = rate * np.exp(-(1/(500000 + i)))
        else:
            rate = 0.1
            
        return rate
    
            
             
    def setIterRates(self, trader, i):
        
        self.exploration_rate = self.rate_ERscheduler(i, self.exploration_rate)
        self.dr = self.rate_DRscheduler(i, self.dr)
        self.dreaming_rate = 0.9 - self.dr
        self.learning_rate = self.rate_LRscheduler(i, self.learning_rate)
            
        trader.set_rates(self.dreaming_rate, 
                         self.exploration_rate,
                         self.learning_rate)
    
         
 
    def fitForexDayTrader(self):
                  
        nameBestModel = 'bestModel'
        pathBestModel = self.finalpath + 'model/' + nameBestModel + '/'
        maxAvgTotalReward = -999.99
        training = True
        # sicherstellen, dass Model-Pfad existiert
        if (not isdir(pathBestModel)):
            makedirs(pathBestModel)
        #
        #######################################
        ### change re-start parameter here ####
        self.dr = 0.4
        self.exploration_rate = 0.4
        init_weight_name = 'bestModel'
        start = 1
        end = 2000000
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
                       data_path=self.finalpath + 'RL-data/', 
                       processed_path=self.processedpath + 'forexCandle/',
                       window_size=self.ts, 
                       batch_size=self.bs,
                       training=training)
        trader = ForexDayTrader(env, 
                                self.finalpath, 
                                init_weight_name,
                                self.bs,
                                self.ts,
                                self.ff)
        result = []
        for i in range(start, end):
            #
            # äusserer Loop über alle Iterationen
            #
            self.setIterRates(trader, i)    
            action_hold = 0
            action_buy = 0
            action_sell = 0
            curr_state, info = env.reset()
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
                elif (action == 2):
                    action_sell += 1
                    
                if done:
                    trader.dream(info["datetime"], 
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
                           action_hold,
                           action_buy,
                           action_sell])

            trader.replay(i)
            if (i % 500 == 0):
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

                avgTotalReward = np.mean(df["Avg-Total-Reward"])
                trader.train_target()
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
                result = []
                                        
        
        
    """            
    def create_model(self, bs):

        model = Sequential()
                
        model.add(LSTM(units=self.units1, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       recurrent_regularizer=l2(0.01),
                       kernel_initializer='truncated_normal',
                       return_sequences=True,
                       batch_input_shape=(self.bs, 
                                          self.ts, 
                                          len(self.featureList))))
    
        model.add(LSTM(units=self.units2, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       recurrent_regularizer=l2(0.01),
                       kernel_initializer='truncated_normal'))
                
        model.add(Dense(units=3, activation=activations.linear))
            
        model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                      optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['mae'])
       
        model.load_weights(self.best_model)
        
        print('weights loaded from ' + self.best_model)
        
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
            state = env.reset()
            state = state.reshape(1, self.ts, len(self.featureList))
            
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
"""        
                                                               
                                                               
fileStartDt = '01.01.2001'
periodStartDt = '2004-01-02 00:00:00'
periodEndDt = '2024-12-31 23:55:00'
training = True

eurjpy = forexDayTrading('EURJPY')

"""
eurjpy.processCandle(startDt = fileStartDt)
eurjpy.formatFinalVersion(training,
                              startDt = periodStartDt,
                              endDt = periodEndDt)

eurjpy.createABT(training)

eurjpy.fitInitialModel()
"""

eurjpy.fitForexDayTrader()

#eurjpy.exploitForexDayTrader()



