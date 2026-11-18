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


a = np.array([1, 2, 3, 4, 5, 6])
#r = np.roll(a, -1)/a
r[:-1] = a[1:]/a[:-1]
r[-1] = 0.0
print(r)