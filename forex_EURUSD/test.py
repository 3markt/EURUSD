#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
from os import listdir, makedirs
from os.path import isfile, join, isdir
from datetime import datetime, timedelta


a = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff']

a = [f for f in a if f != 'ff']
print(a)
s = np.random.choice(a, size=3, replace=False)
s = np.append(s, ['ff'])
print(s)