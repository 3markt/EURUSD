#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import requests as rq
import time
import sys


def log(msg):
    global currency

    real_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(real_dt + ' ' + currency + ': ' + msg)
    sys.stdout.flush()



def getTxt(s, currency, url):
    global buy, sell, low, high, txt, volume

    try:
        f = s.get(url + currency)
        txt = f.text
        try:
            sell = float(txt[7:14])
            buy = float(txt[14:21])
            high = float(txt[21:28])
            low = float(txt[28:35])
            volume = float(txt[35:])
        except:
            pass

    except:
        # if session closed get a new one
        log('Session closed - re-open it')
        s = rq.Session()



time.tzset()
url = 'http://webrates.truefx.com/rates/connect.html?c='
currency = 'EUR/USD'
session = rq.Session()
while True:
    getTxt(session, currency, url)
    print(txt)
    print(buy, sell, low, high, volume)