#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random

feature = 'Schock_Preis_US'

cols = ['WL', feature]

path = '/Users/uwe.muller/Hope/data/processed/EURUSD/forexCandle/'

df = pd.read_csv(path + 'df_all.csv', parse_dates=['time'])

df = df[cols]
df = df[df[feature] != 0.0].copy()
print(len(df))
print(df.corr())

print(df.dtypes)

