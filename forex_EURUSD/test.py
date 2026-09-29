#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
pip = 10000
a = [1.13001, 1.13003, 1.3000, 1.3008 , 1.3004, 1.3007]
d = pip*np.diff(a, append=a[-1])/a
print(d)