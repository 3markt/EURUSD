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


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#r = np.zeros_like(a, dtype=float)
#r[:-1] = a[1:]/a[:-1]
p = np.ones_like(a, dtype=float)
n=len(p)
n1=int(n/3)
n2=int(2*n/3)
print(n, n1, n2)
p[n1:n2] += 1
p[n2:] += 2
p[-1] += 2
print(p)
p = p/sum(p)
print(p)
s = np.random.choice(a, size=5, replace=False, p=p)
print(s)