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


path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'

dax = 'dax.csv'
dj = 'dowjones.csv'
nasdaq = 'nasdaq100.csv'
tecdax = 'tecdax.csv'


df = pd.read_csv(path + dax, sep=';')
symbls = df['Symbl'].values.tolist()
df = pd.read_csv(path + dj, sep=';')
symbls = symbls + df['Symbl'].values.tolist()
df = pd.read_csv(path + nasdaq, sep=';')
symbls = symbls + df['Symbl'].values.tolist()
df = pd.read_csv(path + tecdax, sep=';')
symbls = symbls + df['Symbl'].values.tolist()

for symbl in symbls:
    df = pd.read_csv(path + 'aktuell/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
    print(symbl + ' len bevore=', len(df))
    df = df.drop_duplicates(subset=['date'])
    df.rename({"Unnamed: 0":"AA"}, axis="columns", inplace=True)
    del df["AA"]
    print(symbl + ' len after=', len(df))
    df.to_csv(path + 'alt/' + symbl + '.csv', index=False)
