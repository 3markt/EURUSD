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


symbl = 'SAP.DEX'
path = '/Users/uwe.muller/Hope/Data/Aktienkurse/'

df = pd.read_csv(path + 'aktuell/' + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
df = ta.add_trend_ta(df, "High", "Low", "Close", "Volume")

df.to_csv(path + 'aktuell/explore/SAP.csv')
