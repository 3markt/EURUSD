#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:59:52 2021

@author: uwe.mueller
"""

import pandas as pd
import requests
import newspaper as np
import spacy

def groupStockNames(stockNames):
   
    doNotCheck = ['AG', 'SE', '&', 'Co.', 'S.A.', 'KGaA', 'Co', 'N.V.', 'GmbH',
                  '&', 'und', '-', 
                  'Partner', 'Holding', 'Group', 'Deutsche', 'Stiftung', 'Bank',
                  'Technologies', 'Technology', 'Aero', 'Engines', 'Laser', 'Electronics',
                  'Medical', 'Care', 'Meditec', 'Optical', 'Networking', 'Network', 'Solar',
                  'Carl', 'Capital', 'Beteiligungs', 'Strahlen', 'Medizintechnik',
                  'Baumarkt', 'Office', 'Industries', 'Immobilien', 'Licht', 'Media',
                  'Energy', 'Healthineers', 'Deutschland'
                  ]
    
    completeCheck = ['Delivery Hero',
                     'Deutsche Bank',
                     'Deutsche Börse',
                     'Deutsche Wohnen',
                     'Münchener Rück',
                     'Eckert & Ziegler',
                     'Eckert und Ziegler'
                     'New Work',
                     'Pfeiffer Vacuum',
                     'United Internet',
                     'Adler Group',
                     'Amadeus FiRe',
                     'Borussia Dortmund',
                     'Deutsche Beteiligungs',
                     'Deutsche Pfandbriefbank',
                     'DIC Asset',
                     'Global Fashion',
                     'Hamburger Hafen und Logistik',
                     'Indus Holding',
                     'Instone Real Estate',
                     'Jost Werke',
                     'Koenig & Bauer',
                     'Koenig und Bauer',
                     'KWS SAAT',
                     'Wacker Neuson',
                     'Grand City Properties',
                     'Hannover Rückversicherung',
                     'Hugo Boss',
                     'Shop Apotheke',
                     'Wacker Chemie',
                     'Fuchs Petrolub',
                     'RTL Group',
                     'Metro AG'
                     ]
    
    for s in stockNames:
        for e in completeCheck:
            if (e in s):
                stockNames.remove(s)
    
    shortNames = []
    for s in stockNames:
        fullName = s.split(''' ''')
        for d in doNotCheck:
            if d in fullName:
                fullName.remove(d)
        shortNames.append(' '.join(fullName))
        shortNames = list(set(shortNames))
        
    return shortNames[1:] + completeCheck

            
    
ipath = '/Users/uwe.muller/Hope/Data/AktienIndicies/'
dfNames = pd.read_csv(ipath + 'dax.csv')
stockNames = dfNames['Name'].tolist()
dfNames = pd.read_csv(ipath + 'techdax.csv')
stockNames += dfNames['Name'].tolist()
dfNames = pd.read_csv(ipath + 'mdax.csv', sep=';')
stockNames += dfNames['Name'].tolist()
dfNames = pd.read_csv(ipath + 'sdax.csv', sep=';')
stockNames += dfNames['Name'].tolist()
stockNames = list(set(stockNames))

stockNames = groupStockNames(stockNames)
print(stockNames)