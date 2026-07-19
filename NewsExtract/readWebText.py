#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import numpy as np
import pandas as pd
import datetime
import re
import requests
from bs4 import BeautifulSoup

link = "https://www.finanzen.net/index/dax/30-werte"
f = requests.get(link)
html_doc = f.text
soup = BeautifulSoup(html_doc, 'lxml')
print('==================================================')
data = soup.findAll(text=True)

names = []
for d in data:
    names.append(d.parent.name)

print(list(set(names))) 

"""
def visible(element):
    if element.parent.name in ['source', 'span', 'li', 'h1', 'section', 'div',
                               'time', 'ol', 'h4', 'footer',
                               'ul', 'nav', 'noscript', 'figcaption', 'figure', 
                               'style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    
    return True

"""


def visible(element):
    if element.parent.name in ['p', 'title']:
        return True
    else:
        return False
 
result = filter(visible, data)
for r in result:
#    if (r != ' '):
    print(r)

