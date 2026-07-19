#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
from enum import Enum
import datetime as dt


path = '/Users/uwe.muller/Hope/data/abt/'
f = 'eurusdMaxRewards6.csv'

df = pd.read_csv(path + f, parse_dates=['Time'], infer_datetime_format=True)
df.set_index(pd.DatetimeIndex(df.Time), inplace=True)

print(df.dtypes)

