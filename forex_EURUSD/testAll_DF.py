#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""
import numpy as np
import pandas as pd
import datetime as dt
from os import listdir, makedirs
from os.path import isfile, join, isdir

path = '/home/uwe/Hope/data/processed/EURUSD/'

df_all = pd.read_csv(path + 'df_all.csv')
df_all["time"] = pd.to_datetime(df_all["time"])
df_all = df_all.set_index("time")

#print(len(df_all))
#print(df_all.columns)
dt_format = '%d.%m.%Y'
start_datum = dt.date(2005, 2, 1)
end_datum = dt.date(2026, 6, 30)
while True:
    test_datum = input("Test-Datum (dd.mm.yyyy): ")
    try:
        dt.datetime.strptime(test_datum, dt_format)
        print('Test-Datum =', {test_datum})
        test_dt = dt.datetime.strptime(test_datum, dt_format)
        if (start_datum <= test_dt.date() <= end_datum):
            df_test = df_all[df_all.index.date == test_dt.date()]
            df_test.to_csv(path + 'df_all_' + test_datum + '.csv')
        else:
            print('Testdatum liegt ausserhalb des Bereichs 01.02.2005 - 30.06.2026')

    except ValueError:
        print(test_datum, 'ist kein valides Datum')


