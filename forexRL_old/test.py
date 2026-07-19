#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
import joblib
from os import listdir, makedirs
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


a = [[1, 2], [1, 2], [3, 3], [4, 4], [4, 4], [5, 6]]

#aa = []
#[aa.append(s) for s in a if s not in aa]

aa = list(set(a))

print(aa)
