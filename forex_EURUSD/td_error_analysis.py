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

row = -11
td_error = pd.read_csv('/home/uwe/Hope/data/final/EURUSD/td_error/td_error_1000.csv')
x = pd.DataFrame(list(range(252)))
y = td_error.iloc[row, 1:]
print(len(x), len(y))
atd_error = td_error.abs()
print(atd_error.iloc[row,1:].mean())
print(td_error.iloc[row,1:].min())
print(td_error.iloc[row,1:].quantile(q=0.01))
print(td_error.iloc[row,1:].quantile(q=0.05))
print(td_error.iloc[row,1:].quantile(q=0.25))
print(td_error.iloc[row,1:].quantile(q=0.75))
print(td_error.iloc[row,1:].quantile(q=0.95))
print(td_error.iloc[row,1:].quantile(q=0.99))
print(td_error.iloc[row,1:].max())

plt.scatter(x, y, linewidth=0.1)
plt.show()