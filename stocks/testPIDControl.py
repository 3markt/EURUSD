#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import sys
import time

a = sys.argv[1]

print('Python Process %3s started ....' % (a))
time.sleep(10)
print('Python Process %3s ended ...' % (a))

