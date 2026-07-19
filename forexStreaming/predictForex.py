#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import psycopg2
import pandas as pd
import ta
import sys
import datetime as dt
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


class predictForex():
    #
    # 
    #
    techindCols = ['zeitIndex', 'scaledClose',  'round_lot', 'WL', 
                   'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi', 
                   'macd_30Min', 'macd_d_30Min', 'bb_h_30Min', 'bb_l_30Min', 
                   'rsi_30Min', 
                   'WL_6H', 'macd_6H', 'macd_d_6H', 'bb_h_6H', 
                   'bb_l_6H', 'rsi_6H',
                   'WL_1D', 'macd_1D', 'macd_d_1D', 'bb_h_1D', 
                   'bb_l_1D', 'rsi_1D']
    
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
    
    
    
    def __init__(self, currency, pip):
    
        self.currency = currency
        self.pip = pip
        #self.basepath = '/Users/uwe.muller/Hope/'
        self.basepath = '/home/chitlom/'
        self.processedpath = self.basepath + 'data/processed/' \
                            + self.currency + '/'
        self.finalpath = self.basepath + 'data/final/' + self.currency + '/'
        self.modelpath = self.basepath + 'model/' + self.currency + '/'
        self.bt = 288
        self.ts = 12
        self.bs = self.bt - self.ts
        self.units1 = 32
        self.units2 = 8
        self.learning_rate = 0.0005

        self.df_mean = pd.read_csv(self.processedpath \
                                   + 'forexCandle/df_all_mean.csv')
        self.df_mean.rename({'Unnamed: 0': 'name',
                             '0':'value'}, inplace=True, axis=1)
        self.df_std = pd.read_csv(self.processedpath \
                                   + 'forexCandle/df_all_std.csv')
        self.df_std.rename({'Unnamed: 0': 'name',
                             '0':'value'}, inplace=True, axis=1)
        self.df_schockMeanStd = pd.read_csv(self.processedpath \
                                   + 'forexCandle/schockStandardize.csv')
        



    def createModel(self, name):
        
        ff = len(self.scaledCols)
        print('Anzahl Features =', ff)
        model = Sequential()
                
        model.add(LSTM(units=self.units1, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       recurrent_regularizer=l2(0.01),
                       kernel_initializer='truncated_normal',
                       return_sequences=True,
                       batch_input_shape=(1, 
                                          self.ts, 
                                          ff)))
        model.add(LSTM(units=self.units2, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       recurrent_regularizer=l2(0.01),
                       kernel_initializer='truncated_normal'))
                 
        model.add(Dense(units=2, 
                        kernel_initializer='truncated_normal'))

        model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                      optimizer=Adam())
        
        model.summary()
        
        model.load_weights(self.modelpath + name + '/' + name)
        sys.stdout.flush()
        
        return model
               




eurusd = predictForex('EURUSD', 10000)
model = eurusd.createModel('lstm-ts12-ff58-u132-u28')

x = np.load(eurusd.finalpath + 'X/x-2008-01-21.npy')
py = model.predict(x)
print(py)

