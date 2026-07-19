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



def calcRand(df, name, TP, SL):
    
    p = abs(SL)/(abs(SL)+abs(TP))
    df[name] = 0
    df[name] = df.applymap(lambda x: 1 if random.random() < p else 0)
    df['p-' + name] = p
#    df.loc[df['Close'].shift(-1) < df['Open'].shift(-1), 'Random'] = 0
    
    return df


def calcPCT(df):
    
    df['pctLow'] = 100*(df['Low'].shift(-1) - df['Open'].shift(-1))/df['Open'].shift(-1)
    df['pctHigh'] = 100*(df['High'].shift(-1) - df['Open'].shift(-1))/df['Open'].shift(-1)
    df['pctClose'] = 100*(df['Close'].shift(-1) - df['Open'].shift(-1))/df['Open'].shift(-1)

    return df


def calcStd(df):
    
    df['Std'] = df[['Open', 'High', 'Low', 'Close']].std(axis=1)
    df['Std10'] = df['Std'].rolling(50).mean().pow(1./2)
    
    return df




def calcLongWL(df, TP, SL):
        
    df['LongTP'] = 0
    df.loc[df['pctHigh'] >= TP, 'LongTP'] = 1
    
    df['LongSL'] = 0
    df.loc[df['pctLow'] <= SL, 'LongSL'] = 1
    
    df.loc[(df['LongTP'] == 1) & (df['LongSL'] == 1) & (df['randomLong'] == 1), 'LongSL'] = 0
    df.loc[(df['LongTP'] == 1) & (df['LongSL'] == 1) & (df['randomShort'] == 0), 'LongTP'] = 0
    
    df['LongWL'] = 0
    df.loc[(df['LongTP'] == 1), 'LongWL'] = TP
    df.loc[(df['LongSL'] == 1), 'LongWL'] = SL
    df.loc[(df['LongTP'] == 0) & (df['LongSL'] == 0), 'LongWL'] = df['pctClose']
   
    
    return df
    



def calcShortWL(df, TP, SL):
        
    df['ShortTP'] = 0
    df.loc[df['pctLow'] <= TP, 'ShortTP'] = 1
    
    df['ShortSL'] = 0
    df.loc[df['pctHigh'] >= SL, 'ShortSL'] = 1
    
    df.loc[(df['ShortTP'] == 1) & (df['ShortSL'] == 1) & (df['randomShort'] == 1), 'ShortSL'] = 0
    df.loc[(df['ShortTP'] == 1) & (df['ShortSL'] == 1) & (df['randomShort'] == 0), 'ShortTP'] = 0

    df['ShortWL'] = 0
    df.loc[(df['ShortTP'] == 1), 'ShortWL'] = -TP
    df.loc[(df['ShortSL'] == 1), 'ShortWL'] = -SL
    df.loc[(df['ShortTP'] == 0) & (df['ShortSL'] == 0), 'ShortWL'] = -df['pctClose']  
    
    return df
    




def evalLong(df, name):
    
    nLongHat = len(df.loc[(df[name+'-TargetHat'] == 0)])
    trueLong = len(df.loc[(df['Target'] == 0) & (df[name+'-TargetHat'] == 0)])
    falseLong0 = len(df.loc[(df['Target'] == 1) & (df[name+'-TargetHat'] == 0)])
    falseLongS = len(df.loc[(df['Target'] == 2) & (df[name+'-TargetHat'] == 0)])
    if (nLongHat > 0):
        corrLong = trueLong/nLongHat
        errLong0 = falseLong0/nLongHat
        errLongS = falseLongS/nLongHat
        
    longWL = df.loc[df[name+'-TargetHat'] == 0, 'LongWL'].sum()
        
    return nLongHat, longWL, corrLong, errLong0, errLongS
    


def evalShort(df, name):
            
    nShortHat = len(df.loc[(df[name+'-TargetHat'] == 2)])
    trueShort = len(df.loc[(df['Target'] == 2) & (df[name+'-TargetHat'] == 2)])
    falseShort0 = len(df.loc[(df['Target'] == 1) & (df[name+'-TargetHat'] == 2)])
    falseShortL = len(df.loc[(df['Target'] == 0) & (df[name+'-TargetHat'] == 2)])
    if (nShortHat > 0):
        corrShort = trueShort/nShortHat
        errShort0 = falseShort0/nShortHat
        errShortL = falseShortL/nShortHat

    shortWL = df.loc[df[name+'-TargetHat'] == 2, 'ShortWL'].sum()
        
    return nShortHat, shortWL, corrShort, errShort0, errShortL
    



###### START #############################################
############### Update this section - all ################
##########################################################

basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
indexName = 'daxTrain.csv'

df = pd.read_csv(path + indexName, sep=';')

symbls = df['Symbl'].values.tolist()

##########################################################
############### Update this section - all ################
###### END ###############################################

inpath = basePath + 'Data/Aktienkurse/predicted/'
outpath = basePath + 'Data/Aktienkurse/work/'

longWLAll = 0
nLongAll = 0
shortWLAll = 0
nShortAll = 0

longTP = 1.0
longSL = -1.0
shortTP = -5.0
shortSL = 1.0

for symbl in symbls:
    
    df = pd.read_csv(inpath + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)

    df = calcPCT(df)
    df = calcRand(df, 'randomLong', longTP, longSL)
    df = calcRand(df, 'randomShort', shortTP, shortSL)
    df = calcLongWL(df, longTP, longSL)
    df = calcShortWL(df, shortTP, shortSL)
    df.to_csv(outpath + symbl + '.csv', index=False)
    
    nLong, longWL, corrLong, errLong0, errLongS = evalLong(df, 'p1')

    nShort, shortWL, corrShort, errShort0, errShortL = evalShort(df, 'p1')
        
    print('%8s LONG: WL=%6.2f corr=%4.2f err0=%4.2f errS=%4.2f tradeR=%4i' % \
          (symbl, longWL, corrLong, errLong0, errLongS, nLong))

    print('        SHORT: WL=%6.2f corr=%4.2f err0=%4.2f errL=%4.2f tradeR=%4i' % \
          (shortWL, corrShort, errShort0, errShortL, nShort))

    longWLAll += longWL
    nLongAll += nLong
    shortWLAll += shortWL
    nShortAll += nShort

print()    
print('Overall nLong=%5i LongWL=%6.2f longWLPCT=%4.2f nShort=%5i ShortWL=%6.2f shortWLPCT=%4.2f' % \
      (nLongAll, longWLAll, longWLAll/nLongAll, nShortAll, shortWLAll, shortWLAll/nShortAll))
    
        
