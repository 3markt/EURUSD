#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import psycopg2
import pandas as pd
import os

eco = {}
eco.update({'aa':'zz'})
eco.update({'bb':'yy'})
eco.update({'cc':'xx'})
print(eco)
print(eco['bb'])

for e in eco:
    print(e, eco[e])
