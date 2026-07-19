#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import psycopg2
import pandas as pd
import os

eco = {'aa':[]}
eco.update({'bb':[]})
eco.update({'cc':[]})
print(eco)
vl = []
for i in range(5):
    vl.append(i)
eco.update({'c':vl})

print(eco)
    
