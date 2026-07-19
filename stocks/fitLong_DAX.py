#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam
from sklearn.utils.class_weight import compute_class_weight

from os import listdir
from os.path import isfile, join
import datetime
from dateutil.relativedelta import relativedelta
import sys
import math
import gc
import ta
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")




def getLearningRateIter(lr, patience):
    
    if (patience >= 10):
        lrate = lr - lr*0.05
        patience = 0
    else:
        lrate = lr
 
    return lrate, patience




def getLearningRate(ii):
    
    if (ii < 75):
        lrate = 0.005
    elif (ii < 150):
        lrate = 0.004
    elif (ii < 225):
        lrate = 0.003
    elif (ii < 300):
        lrate = 0.002
    elif (ii < 375):
        lrate = 0.001
    elif (ii < 450):
        lrate = 0.0009
    elif (ii < 525):
        lrate = 0.0008
    elif (ii < 600):
        lrate = 0.0007
    elif (ii < 675):
        lrate = 0.0006
    elif (ii < 750):
        lrate = 0.0005
    elif (ii < 825):
        lrate = 0.0004
    elif (ii < 900):
        lrate = 0.0003
    elif (ii < 975):
        lrate = 0.0002
    elif (ii < 1050):
        lrate = 0.0001
    elif (ii < 1125):
        lrate = 0.00009
    elif (ii < 1200):
        lrate = 0.00008
    elif (ii < 1275):
        lrate = 0.00007
    elif (ii < 1350):
        lrate = 0.00006
    elif (ii < 1425):
        lrate = 0.00005
    elif (ii < 1425):
        lrate = 0.00004
    elif (ii < 1500):
        lrate = 0.00003
    elif (ii < 1575):
        lrate = 0.00002
    else:
        lrate = 0.00001
 
    return lrate




def getWeights(ii, path, model, modelName):
    
    if (ii == 0):
        print(model.summary())
        print(features)   
    else:            
        if (isfile(path + 'results/' + modelName + '_bestMyAccuWeights.hdf')):
            print()
            print('------------------> loading bestMyAccuWeights <---------------------')
            print()
            model.load_weights(path + 'results/' + modelName + '_bestMyAccuWeights.hdf')
        elif (isfile(path + 'results/' + modelName + '_bestLossWeights.hdf')):
            print()
            print('------------------> loading bestLossWeights <---------------------')
            print()
            model.load_weights(path + 'results/' + modelName + '_bestLossWeights.hdf')
        else:
            print(model.summary())
            
            



def myEvaluation(y_hat, y_true):
    
    n = len(y_true)
    acc = 0
    acc_1 = 0
    n_1 = 0
    acc_0 = 0
    n_0 = 0
    tp = 0
    tpr = 0
    for i in range(n):                
        if (y_true[i] == 1):
            tp += 1
            if (y_hat[i] >= 0.5):
                tpr += 1

        if (y_hat[i] >= 0.5):
            n_1 += 1
            if (y_true[i] == 1):
                acc_1 += 1
                acc += 1
                
        if (y_hat[i] < 0.5):
            n_0 += 1
            if (y_true[i] == 0):
                acc_0 += 1
                acc += 1

    if (tp > 0):
        tpr = tpr/tp
    else:
        tpr = np.nan

    if (n > 0):                
        acc = acc/n
    else:
        acc = np.nan
        
    if (n_1 > 0):
        acc_1 = acc_1/n_1
    else:
        acc_1 = np.nan
    
    if (n_0 > 0):
        acc_0 = acc_0/n_0
    else:
        acc_0 = np.nan
 
    return acc, acc_0, n_0, acc_1, n_1, tpr
            





def prepData(path, symbls, features, bs, ts):
    
    x_all = []  
    y_all = []
    n_smpl = 0
    symbls_nb = []
    for symbl in symbls:
        df = pd.read_csv(path + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)

        if (len(df) >= 60):    
            x_data = df[features].values
            y_data = df['TargetLong'].values
            sc = MinMaxScaler(feature_range=(0,1))
            x_data = sc.fit_transform(x_data)
    
            n = len(df)
            n_bs = n//bs
            if (n_bs + ts > n):
                n_bs -= 1

            symbl_x = np.array([])
            symbl_y = np.array([])
            for b in range(n_bs):
                batch_x = np.array([])
                batch_y = np.array([])
                for i in range(b*bs+ts, (b+1)*bs+ts):
                    batch_x = np.append(batch_x, x_data[i-ts:i])
                    batch_y = np.append(batch_y, y_data[i-1])
                symbl_x = np.append(symbl_x, batch_x)
                symbl_y = np.append(symbl_y, batch_y)
            x_all.append(symbl_x.reshape(n_bs*bs, ts, len(features)))
            y_all.append(symbl_y.reshape(n_bs*bs, 1))
            n_smpl += n_bs*bs
            symbls_nb.append([symbl, n_bs])
    
    print('------> all samples: ', n_smpl)
            
    return x_all, y_all, symbls_nb




def unpackXY(all_x, all_y, symbls, bs, ts, ff, split):
    global features_check

    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    nbs_train = 0
    nbs_test = 0
    
    for symbl in symbls:
        i_symbl = symbls.index(symbl)
        symbl_x = all_x[i_symbl]
        symbl_y = all_y[i_symbl]
        for i in range(symbl[1]):

            if (random.random() < split):
                x_train = np.append(x_train, symbl_x[i*bs:(i+1)*bs])
                y_train = np.append(y_train, symbl_y[i*bs:(i+1)*bs])
                nbs_train += 1            
            else:
                x_test = np.append(x_test, symbl_x[i*bs:(i+1)*bs])
                y_test = np.append(y_test, symbl_y[i*bs:(i+1)*bs])
                nbs_test += 1

    x_train= x_train.reshape(nbs_train*bs, ts, ff)
    y_train= y_train.reshape(nbs_train*bs, 1)

    x_test= x_test.reshape(nbs_test*bs, ts, ff)
    y_test= y_test.reshape(nbs_test*bs, 1)
    
    return x_train, y_train, x_test, y_test



path = '/home/umueller/data/aktienkurse/'
name = 'daxTrain.csv'

df = pd.read_csv(path + name, sep=';')

symbls = df['Symbl'].values.tolist()


features = ['WL', 'WL5', 'HL', 'coGap', 'cMA5', 'WL2DAX', 'WL2DAX5', 'Vola', 
            'highCandle', 'lowCandle', 'ichimoku_conv', 'ichimoku_a', 
            'momentum_rsi', 'bbh', 'bbl'
            ]


###### START #############################################
############### Update this section - all ################
##########################################################

ii = int(sys.argv[1])
toDate = sys.argv[2]

modelName = 'DAXLong' + toDate
toDt = datetime.datetime.strptime(toDate, '%Y-%m-%d')

bs = 50
ts = 10

train_test_split = 0.9
n_epoch = 75

bestLoss = 99.99
bestLossIter = -999
bestAccu = -99.99
bestAccuIter = -999
bestMyAccu = -99.99
bestMyAccuIter = -999

patience = 0

##########################################################
############### Update this section - all ################
###### END ###############################################

model = Sequential()
model.add(LSTM(units=30, 
               recurrent_dropout=0.3,
               kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01),
               recurrent_regularizer=l2(0.01),
               kernel_initializer='truncated_normal',
               batch_input_shape=(bs, ts, len(features))))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='truncated_normal'))

getWeights(ii, path, model, modelName)
     
if (isfile(path + 'tmp/' + modelName + '_bestLoss.npy')):
    bestLoss = np.load(path + 'tmp/' + modelName + '_bestLoss.npy')

if (isfile(path + 'tmp/' + modelName + '_bestLossIter.npy')):
    bestLossIter = np.load(path + 'tmp/' + modelName + '_bestLossIter.npy')

if (isfile(path + 'tmp/' + modelName + '_bestAccu.npy')):
    bestAccu = np.load(path + 'tmp/' + modelName + '_bestAccu.npy')

if (isfile(path + 'tmp/' + modelName + '_bestAccuIter.npy')):
    bestAccuIter = np.load(path + 'tmp/' + modelName + '_bestAccuIter.npy')

if (isfile(path + 'tmp/' + modelName + '_bestMyAccu.npy')):
    bestMyAccu = np.load(path + 'tmp/' + modelName + '_bestMyAccu.npy')

if (isfile(path + 'tmp/' + modelName + '_bestMyAccuIter.npy')):
    bestMyAccuIter = np.load(path + 'tmp/' + modelName + '_bestMyAccuIter.npy')
 
   
lr = getLearningRate(ii)
opt = Adam(learning_rate=lr)
model.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy'])


if (isfile(path + 'tmp/' + modelName + '_x_train.npy')):
    x_train = np.load(path + 'tmp/' + modelName + '_x_train.npy')
    y_train = np.load(path + 'tmp/' + modelName + '_y_train.npy')
    x_test = np.load(path + 'tmp/' + modelName + '_x_test.npy')
    y_test = np.load(path + 'tmp/' + modelName + '_y_test.npy') 
else:
    x_all, y_all, symbls = prepData(path + 'abt/', symbls, features, bs, ts)
    x_train, y_train, x_test, y_test = unpackXY(x_all, y_all, symbls, bs, ts, len(features), train_test_split)   
    del x_all
    del y_all
    np.save(path + 'tmp/' + modelName + '_x_train.npy', x_train)
    np.save(path + 'tmp/' + modelName + '_y_train.npy', y_train)
    np.save(path + 'tmp/' + modelName + '_x_test.npy', x_test)
    np.save(path + 'tmp/' + modelName + '_y_test.npy', y_test)

print('----> Train Shape: ', x_train.shape, y_train.shape, 'Validate Shape: ', x_test.shape, y_test.shape)
print()

nTrain = len(y_train)
train_y1_sum = np.sum(y_train)
train_y1 = train_y1_sum/nTrain
train_y0 = (nTrain - train_y1_sum)/nTrain

nTest = len(y_test)
test_y1_sum = np.sum(y_test)
test_y1 = test_y1_sum/nTest
test_y0 = (nTest - test_y1_sum)/nTest

print('----> Train Target=0: %1.4f Target=1: %1.4f/%5i' % \
      (train_y0, train_y1, train_y1_sum))
print('----> TestTarget=0: %1.4f Target=1: %1.4f/%5i' % \
      (test_y0, test_y1, test_y1_sum))
print()

print()

out = []

#class_weights = {0:3.0, 1:1.5, 2:1.0}

for i in range(n_epoch):
    ffit = model.fit(x_train, y_train, epochs=1, batch_size=bs, verbose=0,
                        shuffle=False, validation_data=(x_test, y_test))
#                        class_weight=class_weights)
    
    y_hat = model.predict(x_train, batch_size=bs)            
    train_acc, train_acc_0, train_n_0, train_acc_1, train_n_1, trainTPR = myEvaluation(y_hat, y_train)

    y_hat = model.predict(x_test, batch_size=bs)            
    tst_acc, tst_acc_0, tst_n_0, tst_acc_1, tst_n_1, testTPR = myEvaluation(y_hat, y_test)

    if (ffit.history['loss'][0] < bestLoss):
        bestLoss = ffit.history['loss'][0]
        model.save(path + 'results/' + modelName + '_bestLossModel.hdf')        
        model.save_weights(path + 'results/' + modelName + '_bestLossWeights.hdf')
        np.save(path + 'tmp/' + modelName + '_bestLoss.npy', bestLoss)
        bestLossIter = i+ii
        np.save(path + 'tmp/' + modelName + '_bestLossIter.npy', bestLossIter)
                       
    if (ffit.history['accuracy'][0] > bestAccu):
        bestAccu = ffit.history['accuracy'][0]
        model.save(path + 'results/' + modelName + '_bestAccuModel.hdf')        
        model.save_weights(path + 'results/' + modelName + '_bestAccuWeights.hdf')
        np.save(path + 'tmp/' + modelName + '_bestAccu.npy', bestAccu)
        bestAccuIter = i+ii
        np.save(path + 'tmp/' + modelName + '_bestAccuIter.npy', bestAccuIter)
                       
    if (train_acc > bestMyAccu):
        bestMyAccu = train_acc
        model.save(path + 'results/' + modelName + '_bestMyAccuModel.hdf')        
        model.save_weights(path + 'results/' + modelName + '_bestMyAccuWeights.hdf')
        np.save(path + 'tmp/' + modelName + '_bestMyAccu.npy', bestMyAccu)
        bestMyAccuIter = i+ii
        np.save(path + 'tmp/' + modelName + '_bestMyAccuIter.npy', bestMyAccuIter)
        patience = 0
    else:
        patience += 1
        

    
    print()
    print('----------LONG---------- Iteration = %3i Learning Rate = %1.8f ------------ %8s -----------------' % \
          (i+ii, lr, toDate))
    print('----------- Train Loss = %1.4f   Accuracy = %1.4f ---------------------------------------------------------' % \
          (ffit.history['loss'][0], ffit.history['accuracy'][0]))
    print('------ Validation Loss = %1.4f   Accuracy = %1.4f ---------------------------------------------------------' % \
          (ffit.history['val_loss'][0], ffit.history['val_accuracy'][0]))
    print('------ myEvaluation Train Accu=%1.4f 0-Accu=%1.4f/%5i 1-Accu=%1.4f/%5i' % \
          (train_acc, train_acc_0, train_n_0, train_acc_1, train_n_1))
    print('------ myEvaluation Test Accu=%1.4f 0-Accu=%1.4f/%5i 1-Accu=%1.4f/%5i' % \
          (tst_acc, tst_acc_0, tst_n_0, tst_acc_1, tst_n_1))
    print('------ bestLoss=%1.4f at %4i bestAccu=%1.4f at %4i bestMyAccu=%1.4f at %4i' % \
         (bestLoss, bestLossIter, bestAccu, bestAccuIter, bestMyAccu, bestMyAccuIter))
    print('---------LONG------------------------ %8s ------------------------------------------------------------' %
          (toDate))
    sys.stdout.flush()
    
    out.append([i+ii,
                ffit.history['loss'][0], 
                ffit.history['accuracy'][0],
                ffit.history['val_loss'][0], 
                ffit.history['val_accuracy'][0],
                train_acc,
                train_acc_0, 
                train_n_0, 
                train_acc_1, 
                train_n_1, 
                tst_acc_0, 
                tst_n_0, 
                tst_acc_1, 
                tst_n_1, 
                bestAccu,
                bestMyAccu,
                lr,
                bestAccuIter,
                bestMyAccuIter
                ])
    
    lr, patience = getLearningRateIter(lr, patience)        
    K.set_value(model.optimizer.learning_rate, lr)
    gc.collect()

outDF = pd.DataFrame(out, columns=['Iteration', 
                                   'Train-Loss',
                                   'Train-Accu', 
                                   'Test-Loss', 
                                   'Test-Accu',
                                   'myTrain-Accu',
                                   'Train-0-Accu',
                                   '#Train-0',
                                   'Train-1-Accu',
                                   '#Train-1',
                                   'Test-0-Accu',
                                   '#Test-0',
                                   'Test-1-Accu',
                                   '#Test-1',
                                   'Best-Accu',
                                   'Best-My-Accu',
                                   'Learning-Rate',
                                   'Best-Accu-Iter',
                                   'Best-My-Accu-Iter'
                                   ])

if (isfile(path + 'results/' + modelName + '_result.csv')):
    xOutDF = pd.read_csv(path + 'results/' + modelName + '_result.csv')
    outDF = xOutDF.append(outDF)

outDF.to_csv(path + 'results/' + modelName + '_result.csv', index=False)
K.clear_session()
    


