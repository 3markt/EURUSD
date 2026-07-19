#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
import ta
from os import listdir
from os.path import isfile, join
from datetime import datetime
import alpha_vantage
import requests



API_URL = "https://www.alphavantage.co/query"
data = {
    "function": "CASH_FLOW",
    "symbol": "AAPL",
 "outputsize": "compact",
    "apikey": "JW22M5RPKKUT9ZX3"
    }

response = requests.get(API_URL, params=data)
print(response.json())