#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import datetime as dt
import ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


path = '/Users/uwe.muller/Hope/data/processed/EURUSD/forexCandle/'

df_abt = pd.read_csv(path + 'abtEcoDataRF_all.csv')

print(df_abt)

