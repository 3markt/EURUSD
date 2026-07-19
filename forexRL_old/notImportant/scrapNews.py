#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:02:16 2021

@author: uwe.mueller
"""

import requests
import bs4

URL = 'https://spiegel.de/schlagzeilen' 
website = requests.get(URL)

results = bs4.BeautifulSoup(website.content, 'html.parser')
l = results.find_all('h2')
print(l)
