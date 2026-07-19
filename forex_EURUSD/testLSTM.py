#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam



bs = 215
ts = 72
ff = 52
lr = 0.001

model = Sequential()
        
model.add(LSTM(units=78, 
               recurrent_dropout=0.2,
               kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01),
               recurrent_regularizer=l2(0.01),
               kernel_initializer='truncated_normal',
               batch_input_shape=(bs, 
                                  ts, 
                                  ff)))
        
model.add(Dense(units=3, 
                kernel_initializer='truncated_normal'))
    
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=lr),
              metrics=['mae'])

model.summary()



