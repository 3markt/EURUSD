#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd

a = []
a.append([1.13001, 1.13003, 1.3000, 1.3008 , 1.3004, 1.3007])
a.append([1.13010, 1.13013, 1.3010, 1.3018 , 1.3014, 1.3017])
a.append([1.13010, 1.13013, 1.3010, 1.3018 , 1.3014, 1.3017])
print(len(a), pd.DataFrame(a))
