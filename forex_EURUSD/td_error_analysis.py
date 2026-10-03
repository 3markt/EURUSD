#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8)) # Schön groß machen!

td_error = pd.read_csv('/home/uwe/Hope/data/final/EURUSD/td_error/td_error_217000.csv',
                       index_col=0, header=0)
td_err_mean = td_error.mean(axis=0)
atd_error = td_error.abs()
td_err_amv = atd_error.mean(axis=1)
x = pd.DataFrame(list(range(1000)))
y = td_err_amv
print(y)
print(len(x), len(y))

plt.scatter(x, y, linewidth=0.1)
plt.show()