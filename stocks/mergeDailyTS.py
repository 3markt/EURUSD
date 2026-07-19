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



path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'


dax = pd.read_csv(path + 'dax.csv', sep=';')

symbls = dax['Symbl'].values.tolist()

for symbl in symbls:
    neu = pd.read_csv(path + 'endofday/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    try:
        alt = pd.read_csv(path + 'aktuell/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
        alt.set_index('date', drop=False, inplace=True)
        neu.set_index('date', drop=False, inplace=True)
        l_alt = len(alt)
        l_neu = len(neu)
        new = pd.concat([alt, neu], axis=0).drop_duplicates(subset=['date']).reset_index(drop=True)
        print(symbl, l_alt, l_neu)
    except:
        new = neu
        print('---------> No old file available <-------------------')

    l_new = len(new)
    print(symbl, l_new)
    new = new.sort_values(by=['date'])
    new.to_csv(path + 'aktuell/' + symbl + '.csv', index=False)
