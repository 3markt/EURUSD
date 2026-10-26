#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""
import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, isdir

cpath = '/home/uwe/Hope/data/raw/EURUSD/'
ppath = '/home/uwe/Hope/data/processed/EURUSD/'

cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f))]
cfiles.sort()
print(cfiles)
for c in cfiles[:1]:
    df = pd.read_csv(cpath + c)
    df.rename({'Gmt time': 'ttime'}, inplace=True, axis=1)
    df = df[['ttime', 'Close', ]]
    df['ttime'] = pd.to_datetime(df['ttime'], utc=True, format='mixed', dayfirst=True)
    df = df.sort_values(by='ttime').reset_index(drop=True)
    df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
    df.drop('ttime', inplace=True, axis=1)
    df_tag = df.resample("D").size().to_frame(name="anzahl_eintraege")
    df_tag["wochentag"] = df_tag.index.day_name(locale="de_DE.utf8")
    df_tag.to_csv(ppath + 'df_tage_' + c)