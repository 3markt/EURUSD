#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
import ta
from os import listdir
from os.path import isfile, join
from datetime import datetime
import alpha_vantage
import requests
import sys
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



basePath = '/Users/uwe.muller/Hope/'

name = 'DAX_History.csv'

df = pd.read_csv(basePath + 'data/aktienkurse/aktuell/' + name, 
                 sep=';',
                 thousands='.',
                 decimal=',', 
                 parse_dates=['Datum'], 
                 infer_datetime_format=True
                 )

df.rename(columns={'Datum':'date'}, inplace=True)
df = df.sort_values('date')
df = df.reset_index(drop=True)


df.dropna(inplace=True)

df.to_csv(basePath + 'data/aktienkurse/aktuell/DAX_WL.csv', index=False)         
