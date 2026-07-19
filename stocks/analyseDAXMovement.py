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




def calcWL(df):
    
    df['WL5'] = 100*(df['Close'].shift(-5) - df['Close'])/df['Close']
    df['WL10'] = 100*(df['Close'].shift(-10) - df['Close'])/df['Close']
    df['WL20'] = 100*(df['Close'].shift(-20) - df['Close'])/df['Close']
    
    df['WL2DAX5'] = df['WL5'] - df['DAXWL5']
    df['WL2DAX10'] = df['WL10'] - df['DAXWL10']
    df['WL2DAX20'] = df['WL20'] - df['DAXWL20']
    
    
    return df



def prepData(path, symbls):
    

    for symbl in symbls:
        df = pd.read_csv(path + 'aktuell/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
        dfIndex = pd.read_csv(path + 'analysis/' + 'DAX_WL.csv', parse_dates=['date'], infer_datetime_format=True)
        df = df.merge(dfIndex, on='date')
        
        df = calcWL(df)
        print(symbl, len(df))
        df.dropna(inplace=True)
        print(symbl, len(df))
        
        df.to_csv(path + 'analysis/' + symbl + '.csv', index=False)




###### START #############################################
############### Update this section - all ################
##########################################################

path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'
name = 'daxTrain.csv'

df = pd.read_csv(path + name, sep=';')

symbls = df['Symbl'].values.tolist() 


##########################################################
############### Update this section - all ################
###### END ###############################################
prepData(path, symbls)



