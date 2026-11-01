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



def add_business_day(date_string, date_format="%Y-%m-%d"):
    # 1. String in ein Datumsobjekt umwandeln
    date_obj = datetime.strptime(date_string, date_format)

    # 2. Immer mindestens 1 Tag addieren
    date_obj += timedelta(days=1)

    # 3. Wenn Samstag (5), addiere 2 Tage extra -> Montag
    if date_obj.weekday() == 5:
        date_obj += timedelta(days=2)
    # Wenn Sonntag (6), addiere 1 Tag extra -> Montag
    elif date_obj.weekday() == 6:
        date_obj += timedelta(days=1)

    # 4. Zurück in einen String konvertieren
    return date_obj.strftime(date_format)


"""
data_path = '/home/uwe/Hope/data/final/EURUSD/RL-data/'

dfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f[:3] == 'day']
dfiles.sort()
print(len(dfiles))
print(dfiles)
start_dt = '2018-01-29'
end_dt = '2018-02-15'
batch_files = [f for f in dfiles if f[4:14] >= start_dt and f[4:14] <= end_dt]
print('Start-Datum = {}, End-Datum = {}'.format(start_dt, end_dt))
"""
start_dt = '2018-01-29'
end_dt = '2018-02-15'
for i in range(10):
    start_dt = add_business_day(start_dt)
    end_dt = add_business_day(end_dt)
    print(start_dt, end_dt)