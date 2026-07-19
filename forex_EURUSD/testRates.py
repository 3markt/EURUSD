#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
import requests 
import json 

   
def setIterRates(rate, i):
    
    rate = rate * np.exp(-(1/(950000 + i)))
    
    return rate


lr = 0.005

for i in range(100):
    lr = setIterRates(lr, i)
    print(lr, np.exp(-(1/(950000 + i))))
    print(('%5i -te') % (i))

