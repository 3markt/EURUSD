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

from os import listdir
from os.path import isfile, join
import datetime
import sys
import math
import gc
import ta
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")



def unpackData(df, features, bs, ts):
    
    rc = 1
    sc = MinMaxScaler(feature_range=(0,1))
    xData = df[features].values
    xData = sc.fit_transform(xData)
    n = len(df)
    n_bs = n//bs
    if (n_bs + ts > n):
        n_bs -= 1
    x_data = np.array([])
    if (n_bs > 0):
        rc = 0                
        for b in range(n_bs):
            batch_x = np.array([])
            for i in range(b*bs+ts, (b+1)*bs+ts):
                batch_x = np.append(batch_x, xData[i-ts:i])
            x_data = np.append(x_data, batch_x)
    
        x_data = x_data.reshape(n_bs*bs, ts, len(features))

    print(n_bs, n_bs*bs, len(xData), n, x_data.shape)

    return rc, x_data




def packData(df, pHat, name, bs, ts):
    
    dfAdd = df.loc[ts-1:,'date']
    dfAdd = dfAdd.reset_index(drop=True)
    print(len(dfAdd), len(pHat))
    dfHat = pd.DataFrame(pHat, columns=[name+'-p'])
#    dfHat[name+'-TargetHat'] = dfHat.idxmax(axis=1).map({name+'-Null':0, name+'-Trade':1})
    dfHat[name+'-TargetHat'] = 0
    dfHat.loc[(dfHat[name+'-p'] > 0.5), name+'-TargetHat'] = 1
    dfAdd = pd.concat([dfAdd, dfHat], axis=1)    
    df = df.merge(dfAdd, how='left', on='date')

    return df


###### START #############################################
############### Update this section - all ################
##########################################################

basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
indexName = 'daxTrain.csv'

df = pd.read_csv(path + indexName, sep=';')

symbls = df['Symbl'].values.tolist()

numB = 1

bs = 50
ts = 10

nameModel = 'DAX2'


##########################################################
############### Update this section - all ################
###### END ###############################################


features = ['WL', 'WL5', 'HL', 'coGap', 'cMA5', 'WL2DAX', 'WL2DAX5', 'Vola', 
            'highCandle', 'lowCandle', 'ichimoku_conv', 'ichimoku_a', 
            'momentum_rsi', 'bbh', 'bbl'
            ]

modelLongTPR = load_model(basePath + 'model/' + nameModel + '/DAXLongTPR2020-12-31_bestTPRModel.hdf')
modelShort = load_model(basePath + 'model/' + nameModel + '/DAXShort2020-12-31_bestMyAccuModel.hdf')

inpath = basePath + 'Data/Aktienkurse/abt/'
outpath = basePath + 'Data/Aktienkurse/predicted/'

for symbl in symbls:
    
    df = pd.read_csv(inpath + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    rc, x_data = unpackData(df, features, bs, ts)    
    if (rc == 0):    
        pHat = modelLongTPR.predict(x_data, batch_size=bs)
        df = packData(df, pHat, 'longTPR', bs, ts)
        pHat = modelShort.predict(x_data, batch_size=bs)
        df = packData(df, pHat, 'short', bs, ts)
        df.to_csv(outpath + symbl + '.csv', index=False)
        
