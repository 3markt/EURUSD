#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.io as pio

pio.renderers.default='browser'

file = '/Users/uwe.muller/Hope/data/processed/forexCandle/EURUSD/df_all.csv'
df = pd.read_csv(file, parse_dates=['time'])

fig = px.line(df, x='time', y="Close")
fig.show()

