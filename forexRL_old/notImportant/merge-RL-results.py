#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
from collections import deque
import datetime
import time
import ta
from enum import Enum
from os import listdir
from os.path import isfile, join
import sys



basepath = '/Users/uwe.muller/Hope/'
path = basepath + 'data/rl/'


df = pd.read_csv(path + 'aLL-rl-results.csv')
dfi = pd.read_csv(path + 'rl-results.csv')
df = df.append(dfi)
print(len(df))

df.to_csv(path + 'aLL-rl-results-new.csv', index=False)


