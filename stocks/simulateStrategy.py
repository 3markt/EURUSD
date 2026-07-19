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







def calcTarget(df, limit):
    
    df['Today'] = (df['Low']+df['High']+df['Close']+df['Open'])/4
    df['Tomorrow'] = (df['Low'].shift(-1) + df['High'].shift(-1) + df['Close'].shift(-1) + df['Open'].shift(-1))/4
    
    df['WLTarget'] = 100*(df['Tomorrow'] - df['Today'])/df['Today'] 
    
    df['Target'] = 0
    df.loc[df['WLTarget'] > limit, 'Target'] = 1
    df.loc[df['WLTarget'] < (-1)*limit, 'Target'] = 2
            
    return df



def calcRand(df):
    
    
    df['Random'] = 0
    df['Random'] = df.applymap(lambda x: random.randint(0,1))
    
    return df




def calcLongRule(df, TP, SL):
    
    df['LongTP'] = 0
    df.loc[df['pctHigh'] >= TP, 'LongTP'] = 1
    
    df['LongSL'] = 0
    df.loc[df['pctLow'] <= SL, 'LongSL'] = 1
    
    df.loc[(df['LongTP'] == 1) & (df['LongSL'] == 1) & (df['Random'] == 1), 'LongSL'] = 0
    df.loc[(df['LongTP'] == 1) & (df['LongSL'] == 1) & (df['Random'] == 0), 'LongTP'] = 0
    
    df['LongGood'] = 0
    df.loc[(df['Target'] == 1) & (df['LongTP'] == 1), 'LongGood'] = 1
    
    df['LongBad'] = 0
    df.loc[(df['Target'] == 1) & (df['LongSL'] == 1), 'LongBad'] = 1
    
    df['LongMed'] = 0
    df.loc[(df['Target'] == 1) & (df['LongTP'] == 0) & (df['LongSL'] == 0), 'LongMed'] = df['pctClose']
   
    
    return df
    

def calcShortRule(df, TP, SL):
    
    df['ShortTP'] = 0
    df.loc[df['pctLow'] <= TP, 'ShortTP'] = 1
    
    df['ShortSL'] = 0
    df.loc[df['pctHigh'] >= SL, 'ShortSL'] = 1
    
    df.loc[(df['ShortTP'] == 1) & (df['ShortSL'] == 1) & (df['Random'] == 0), 'ShortSL'] = 0
    df.loc[(df['ShortTP'] == 1) & (df['ShortSL'] == 1) & (df['Random'] == 1), 'ShortTP'] = 0

    df['ShortGood'] = 0
    df.loc[(df['Target'] == 2) & (df['ShortTP'] == 1), 'ShortGood'] = 1
    
    df['ShortBad'] = 0
    df.loc[(df['Target'] == 2) & (df['ShortSL'] == 1), 'ShortBad'] = 1
    
    df['ShortMed'] = 0
    df.loc[(df['Target'] == 2) & (df['ShortTP'] == 0) & (df['ShortSL'] == 0), 'ShortMed'] = -df['pctClose']    
    
    return df
    


def prepData(path, symbls, startDate, endDate):
    
    ltargetAll = 0    
    lgoodAll = 0
    lbadAll = 0
    lmedAll = 0
    stargetAll = 0
    sgoodAll = 0
    sbadAll = 0
    smedAll = 0
    
    longTP = 1.0
    longSL = -1.0
    shortTP = -1.0
    shortSL = 1.0
    
    for symbl in symbls:
        df = pd.read_csv(path + 'aktuell/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
        df = df.loc[(df['date'] > startDate[symbls.index(symbl)]) & (df['date'] < endDate[symbls.index(symbl)])]
        df = df.reset_index(drop=True)

        if (len(df) >= 200):
            df = df.sort_values('date')
            df = df.reset_index(drop=True)
            del df['Split']
            del df['Volume']
            df['Std'] = df.std(axis=1)
            df['Std10'] = df['Std'].rolling(10).mean()
            df = calcTarget(df, 1.0)
            del df['Today']
            del df['Tomorrow']
            del df['WLTarget']
            df['pctLow'] = 100*(df['Low'].shift(-1) - df['Close'])/df['Close']
            df['pctHigh'] = 100*(df['High'].shift(-1) - df['Close'])/df['Close']
            df['pctClose'] = 100*(df['Close'].shift(-1) - df['Close'])/df['Close']
            df = calcRand(df)
            df = calcLongRule(df, longTP, longSL)
            df = calcShortRule(df, shortTP, shortSL)
            df.to_csv(path + 'analysis/' + symbl + '.csv', index=False)
            
            longTarget = len(df.loc[df['Target'] == 1])
            longGood = longTP*df['LongGood'].sum()
            longBad = longSL*df['LongBad'].sum()
            longMed = df['LongMed'].sum()
            shortTarget = len(df.loc[df['Target'] == 2])
            shortGood = (-1)*shortTP*df['ShortGood'].sum()
            shortBad = (-1)*shortSL*df['ShortBad'].sum()
            shortMed = df['ShortMed'].sum()
            
            print(' %8s LONG: Target=%5i Good=%6.2f Bad=%6.2f Med=%6.2f All=%6.2f' % \
                 (symbl, longTarget, longGood/longTarget, longBad/longTarget, longMed/longTarget, 
                 (longGood+longBad+longMed)/longTarget))
            print(' %8s SHORT: Target=%5i Good=%6.2f Bad=%6.2f Med=%6.2f All=%6.2f' % \
                 (symbl, shortTarget, shortGood/shortTarget, shortBad/shortTarget, shortMed/shortTarget, 
                 (shortGood+shortBad+shortMed)/shortTarget))
            
            ltargetAll += longTarget
            lgoodAll += longGood
            lbadAll += longBad
            lmedAll += longMed
            stargetAll += shortTarget
            sgoodAll += shortGood
            sbadAll += shortBad
            smedAll += shortMed

    print('All LONG: Target=%5i Good=%6.2f Bad=%6.2f Med=%6.2f' % \
          (ltargetAll, lgoodAll/ltargetAll, lbadAll/ltargetAll, lmedAll/ltargetAll))
    print('All SHORT: Target=%5i Good=%6.2f Bad=%6.2f Med=%6.2f' % \
          (stargetAll, sgoodAll/stargetAll, sbadAll/stargetAll, smedAll/stargetAll))


path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'
name = 'daxTrain.csv'

df = pd.read_csv(path + name, sep=';')

symbls = df['Symbl'].values.tolist()
startDate = df['StartDate'].values.tolist() 
endDate = df['EndDate'].values.tolist() 
    
startDate = [datetime.datetime.strptime(x, '%Y-%m-%d')+relativedelta(months=-6) for x in startDate]
endDate = [datetime.datetime.strptime(x, '%Y-%m-%d')+relativedelta(months=-3) for x in endDate]
     
prepData(path, symbls, startDate, endDate)



