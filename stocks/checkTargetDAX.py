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




def calcTarget(df, shift, limit, tarName):
        
    df[tarName] = 0
    df['sumWin'] = 0
    df['sumLoss'] = 0

    for i in range(1, shift+1):
        df['Win'+str(i)] = 100*(df['High'].shift(-i) - df['Open'].shift(-1))/df['Open'].shift(-1)
        df['Loss'+str(i)] = 100*(df['Low'].shift(-i) - df['Open'].shift(-1))/df['Open'].shift(-1)
        df['sumWin'] += df['Win'+str(i)].abs()
        df['sumLoss'] += df['Loss'+str(i)].abs()

        df.loc[(df['Win'+str(i)] > limit) & 
               (df['Win'+str(i)].abs() > df['Loss'+str(i)].abs()) &
               (df[tarName] == 0), tarName] = 1

        df.loc[(df['Loss'+str(i)] < (-1)*limit) & 
               (df['Loss'+str(i)].abs() > df['Win'+str(i)].abs()) &
               (df[tarName] == 0), tarName] = 2
            
    return df    




def calcTargetMedium(df, shift, tarName):
    
    df[tarName] = 100*(df['Close'].shift(-shift) - df['Open'].shift(-1))/df['Open'].shift(-1)
    
    return df  




def prepData(path, symbls, nDays, limit):
    
    
    nAll = 0
    nLongAll = 0
    nShortAll = 0
    n0All = 0
    
    for symbl in symbls:
        df = pd.read_csv(path + 'aktuell/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
        df = calcTarget(df, nDays, limit, 'Target')
        df = calcTargetMedium(df, nDays, 'TargetMedium')
        
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        n = len(df)
        n0 = len(df.loc[df['Target'] == 0])
        nL = len(df.loc[df['Target'] == 1])
        nS = len(df.loc[df['Target'] == 2])
        print('%10s: #=%5i #Long=%6.2f #Short=%6.2f' % (symbl, n, 100*nL/n, 100*nS/n))
        
        nAll += n
        nLongAll += nL
        nShortAll += nS
        n0All += n0
        df.to_csv(path + 'aktuell/test/Target' + symbl + '.csv', index=False)
    
    print('---> All: #=%6i #0=%6.2f #Long=%6.2f #Short=%6.2f' % (nAll, 100*n0All/nAll, 100*nLongAll/nAll, 100*nShortAll/nAll))
    



###### START #############################################
############### Update this section - all ################
##########################################################

basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
indexName = 'dax.csv'

df = pd.read_csv(path + indexName, sep=';')

symbls = df['Symbl'].values.tolist() 


limit = 2.5

nDays = 5

nameModel = 'DAX'

##########################################################
############### Update this section - all ################
###### END ###############################################

    
prepData(path, symbls, nDays, limit)
    
            


