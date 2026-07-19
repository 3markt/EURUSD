#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import yfinance as yf
import pandas as pd




basePath = '/Users/uwe.muller/Hope/'
path = basePath + 'Data/Aktienkurse/'
name = 'dax.csv'

df = pd.read_csv(path + name, sep=';')
symbls = df['Symbl'].values.tolist()

ysymbls = []
for symbl in symbls:
    ysymbl = symbl.split('.')[0] + '.DE'
    ysymbls.append(ysymbl)


for symbl in ysymbls:
    ticker = yf.Ticker(symbl)
    
    Open = ticker.info['regularMarketOpen']
    High = ticker.info['regularMarketDayHigh']
    Low = ticker.info['regularMarketDayLow']
    Close = (ticker.info['bid'] + ticker.info['ask'])/2    
    
    print(symbl, Open, High, Low, Close)
    

