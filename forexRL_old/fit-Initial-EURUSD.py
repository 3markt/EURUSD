#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
import datetime as dt
from os import listdir, makedirs
from os.path import isfile, join, isdir
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam
from sklearn.utils.class_weight import compute_class_weight
from math import log2
import tensorflow as tf
import ta



def lr_scheduler(epoch, lr):
    
    if epoch < 500:
        return lr
    else:
        return lr * np.exp(-10.0*lr)



def readABT(path):
    
    tsfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[:13] == 'eurusdRewards'
               and f[13] in ['1', '2', '3', '4', '5', '6', '7']]
    
    tsfiles.sort()
    df = pd.DataFrame()
    for f in tsfiles:
        dfi = pd.read_csv(path + f, parse_dates=['Time'], infer_datetime_format=True)
        print(f, len(dfi))
        df = df.append(dfi)
    
    sys.stdout.flush()
    df.sort_values(by='Time', inplace=True)
    df['minute'] = df['Time'].dt.minute.apply([lambda x: str(x).rjust(2, '0')])
    df['hour'] = df['Time'].dt.hour.apply([lambda x: str(x)])
    df['index'] = (df['hour'] + '.' + df['minute']).astype(float)
    df.reset_index(drop=True, inplace=True)
    
    return df


def scaleABT(df, headcols, features2scale):
    
    df_head = df[headcols]
    df['close_scaled'] = df['Close']
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features2scale]), columns=features2scale)
    
    return pd.concat([df_head, df_scaled], axis=1)



def packXY(df, bs, ts, feature_list, target):
    
    
    x_batchTrain = np.array([])
    y_batchTrain = np.array([])
    dt_batchTrain = np.array([])
    
    if (len(df) > bs + ts):
        x_data = df[feature_list].values
        y_data = df[target].values
        dt_data = df['Time'].values.astype(str)
        
        n = len(df)
        n_bs = n//(bs+ts)
        
        print(n, n/(bs+ts), n_bs)
        nbTrain = 0
        for b in range(n_bs):
            x = np.array([])
            y = np.array([])    
            dtd = np.array([])
            nbTrain += 1
            for i in range(ts+b*(bs+ts), (b+1)*(bs+ts)):
                x = np.append(x, x_data[i-ts:i])
                y = np.append(y, y_data[i-1])
                dtd = np.append(dtd, dt_data[i-1])
            x_batchTrain = np.append(x_batchTrain, x)
            y_batchTrain = np.append(y_batchTrain, y)
            dt_batchTrain = np.append(dt_batchTrain, dtd)
            
        x_batch = x_batchTrain.reshape(nbTrain*bs, ts, len(feature_list))
        y_batch = y_batchTrain.reshape(nbTrain*bs, 3)
        dt_batch = dt_batchTrain.reshape(nbTrain*bs, 1)
        
    return x_batch, y_batch, dt_batch
    



def unpackXY(x_batch, y_batch, dt_batch, i):
    global feature_list, target, ts, bs
    
    n = len(y_batch)
    if (i*bs > n):
        raise Exception('Sample data bigger than original')
#    i_train = int(i*(1-split))*bs
    i_train = (i-10)*bs
    i *= bs
    
    x_train = x_batch[:i_train]
    y_train = y_batch[:i_train]
    dt_train = dt_batch[:i_train]
    x_test = x_batch[i_train:i]
    y_test = y_batch[i_train:i]
    dt_test = dt_batch[i_train:i]

    return x_train, y_train, dt_train, x_test, y_test, dt_test
   
    
    


features2scale = ['index', 'close_scaled', 'WL', 
                  'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                  'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                  'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                  'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                  ]

headcols = ['Time', 'Close', 'Reward-Hold', 'Reward-Buy', 'Reward-Sell', 'Position']

feature_list = ['Position', 'index', 'close_scaled', 'WL', 
                'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                ]

target = ['Reward-Hold', 'Reward-Buy', 'Reward-Sell']

ts = 60
bs = 192
numLSTM1 = 120
nEpochs = 10
lr = 0.0005

#basepath = '/Users/uwe.muller/Hope/'
basepath = '/home/chitlom/'
currency = 'eurusd'

baseModel = 'Initial'
dataName = currency + '-Data'

weightName = currency + 'Weights-' + baseModel
modelName = currency + 'Model-' + baseModel

path = basepath + 'data/forex/'
apath = basepath + 'data/abt/'
mpath = basepath + 'model/' + currency + '-' + baseModel + '/'
rpath = basepath + 'results/'

# check model directory exists
if (not isdir(mpath)):
    makedirs(mpath)

if (isfile(apath + dataName + '-x-batch.npy')):
    x_batch = np.load(apath + dataName + '-x-batch.npy')
    y_batch = np.load(apath + dataName + '-y-batch.npy')
    dt_batch = np.load(apath + dataName + '-dt-batch.npy')
else:    
    tsABT = readABT(apath)
    tsABT['Position'] = tsABT['Position'] * 0.5
    tsABT = scaleABT(tsABT, headcols, features2scale)
    tsABT.to_csv(apath + dataName + '-ABT.csv', index=False)
    n = len(tsABT)
    x_batch, y_batch, dt_batch = packXY(tsABT, bs, ts, feature_list, target)
    np.save(apath + dataName + '-x-batch.npy', x_batch)
    np.save(apath + dataName + '-y-batch.npy', y_batch)
    np.save(apath + dataName + '-dt-batch.npy', dt_batch)

n_bs = y_batch.shape[0]//bs    
print()        
print('Sample-Shapes:', x_batch.shape, y_batch.shape)
print('Number of Batches:', n_bs)

# define LSTM-Model
model = Sequential()

model.add(LSTM(units=72, 
               recurrent_dropout=0.3,
               return_sequences=True,
               kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01),
               recurrent_regularizer=l2(0.01),
               kernel_initializer='truncated_normal',
               batch_input_shape=(bs, ts, len(feature_list))))
  
model.add(LSTM(units=48, 
               recurrent_dropout=0.2,
               return_sequences=True,
               kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01),
               recurrent_regularizer=l2(0.01),
               kernel_initializer='truncated_normal',
               batch_input_shape=(bs, ts, len(feature_list))))

model.add(LSTM(units=24, 
               recurrent_dropout=0.2,
               kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01),
               recurrent_regularizer=l2(0.01),
               kernel_initializer='truncated_normal',
               batch_input_shape=(bs, ts, len(feature_list))))

model.add(Dense(units=3, 
                kernel_initializer='truncated_normal'))

model.summary()
print(feature_list)

model.compile(loss='mean_squared_error',
              optimizer="Adam",
              metrics=['mae'])

# load excisting weights, if available
if (isfile(mpath + weightName + '.index')):
    print('-------> loading weights from file ' + weightName)
    model.load_weights(mpath + weightName)


if (np.isnan(x_batch).any()):
    print('feature-data conatins nans')
if (np.isnan(y_batch).any()):
    print('target-data conatins nans')


for i in range(n_bs-500, n_bs+1, 10):
#for i in range(240, 341, 10):
    
    print('-----------------------------------------------------------------------------------------------')
    x_train, y_train, dt_train, x_test, y_test, dt_test = unpackXY(x_batch, y_batch, dt_batch, i)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    lr = lr_scheduler(i, lr)
    print('Iteration %4i mit Learning Rate %1.6f' % (i, lr))
    print('Training Dates Start %10s - End %10s' % (dt_train[0], dt_train[-1]))
    print('Validation Dates Start %10s - End %10s' % (dt_test[0], dt_test[-1]))
    K.set_value(model.optimizer.learning_rate, lr)

    ffit = model.fit(x_train, y_train, 
                     epochs=nEpochs, 
                     batch_size=bs, 
                     verbose=0,
                     shuffle=False, 
                     validation_data=(x_test, y_test))
            
    loss = np.mean(ffit.history['loss'])
    val_loss = np.mean(ffit.history['val_loss'])
    mae = np.mean(ffit.history['mae'])
    val_mae = np.mean(ffit.history['val_mae'])
    print('--------------> loss=%1.4f val-loss=%1.4f mae=%1.4f val-mae=%1.4f <-----------------' % 
          (loss, val_loss, mae, val_mae))
    
    sys.stdout.flush()

    model.save_weights(mpath + weightName)
    model.save(mpath + modelName)

print('---------------------------------------------------------------------------------------------')
# final iterations - no validation data
ffit = model.fit(x_batch, y_batch, 
                 epochs=nEpochs, 
                 batch_size=bs, 
                 verbose=0,
                 shuffle=False)

loss = np.mean(ffit.history['loss'])
mae = np.mean(ffit.history['mae'])
print('--FINAL EPOCH------------> loss=%1.4f mae=%1.4f <---------------------' % 
      (loss, mae))

model.save_weights(mpath + weightName)
model.save(mpath + modelName)

# make predictions
y_hat = model.predict(x_batch, batch_size=bs)
pd.DataFrame(y_hat, columns=['Hold-Prediction', 
                             'Long-Prediction', 
                             'Short-Prediction']).to_csv(apath + currency + '-' + baseModel + '-predictions.csv')
        
