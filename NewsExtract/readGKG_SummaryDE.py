#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:20:54 2018

@author: uwe.mueller
"""

import requests
import pandas as pd
import numpy as np
import zipfile
import io
import os
from datetime import datetime
import newspaper
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
                     'Deutsche Post',
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
                     'Metro AG',
                     'Software AG'
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
        
    return shortNames[1:] + completeCheck + ['DAX', 'MDAX', 'TECHDAX', 'SDAX',
                                             'dax', 'mdax', 'techdax', 'sdax',
                                             'EUR/USD', 'eur/usd']

            

def findStockNames(entities, stockNames):
    
    foundStockNames = ''
    for s in stockNames:
        for entity in entities:
            ee = entity.split(' ')
            for e in ee:
                if s == e:
                    foundStockNames += s + ';'
            
    return foundStockNames



def isNaN(string):
    return string != string


def countOrgs(row):
    
    orgs = []
    cntOrgs = []
    if not isNaN(row):
        rawOrgs = row.split(';')
        for s in rawOrgs:
            orgs.append(s.split(',')[0])
                    
        corgs = list(set(orgs))
        for c in corgs:
            cntOrgs.append([c, orgs.count(c)])
    return cntOrgs
        

def wordCNT(row):
    
    if not isNaN(row):
        cnt = row.split(',')[0].split(':')[1]
    
    try:
        return int(cnt)
    except:
        return 0
        

                
def selectDeuGKG(gkg):
    
    # extract German entries
    gkg = gkg[gkg['TranslationInfo'].str.contains('srclc:deu;') == True]
            
    return gkg    




#
# set some configuration items
#
index_url = 'http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt'
gpath = '/Users/uwe.muller/Hope/Data/GDELT/selectedNews/'
ipath = '/Users/uwe.muller/Hope/Data/AktienIndicies/'
gkg_ext = '.translation.gkg.csv'

gkg_cols = ['GKGRECORDID',
            'DATE',
            'SourceCollectionIdentifier',
            'SourceCommonName',
            'DocumentIdentifier',
            'Counts',
            'V2Counts',
            'Themes',
            'V2Themes',
            'Locations',
            'V2Locations',
            'Persons',
            'V2Persons',
            'Organizations',
            'V2Organizations',
            'V2Tone',
            'Dates',
            'GCAM',
            'SharingImage',
            'RelatedImages',
            'SocialImageEmbeds',
            'SocialVideoEmbeds',
            'Quotations',
            'AllNames',
            'Amounts',
            'TranslationInfo',
            'Extras']

fgkg_cols = ['GKGRECORDID',
            'DATE',
            'SourceCollectionIdentifier',
            'SourceCommonName',
            'DocumentIdentifier',
            'V2Tone',
            'Dates',
            'TitleEntities',
            'TextEntities',
            'TitleOrgs',
            'TextOrgs',
            'wordCNT',
            'lenTextEntities'
            ]

# load spaCy model
deutsch = spacy.load('de_core_news_lg')

# load stock names
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

#
# lesen aller GDELT GKG Files bis zurück nach 2015
#
index = requests.get(index_url)
sindex = index.text.split('\n')
gkgLines = []
for s in sindex:
    if (len(s) > 1):
        sp = s.split('http://')[1]
    if (gkg_ext in sp):
    # extract translated gkg-lines only
        gkgLines.append(sp)

print(gkgLines[-4:-1])
print(len(gkgLines))
pd.DataFrame(gkgLines).to_csv(gpath + 'gkg-liste.csv')
for gkg in gkgLines[-4:-1]:
    dt = gkg[30:44]
    r = requests.get('http://' + gkg)

    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path = gpath)
    dfGKG = pd.read_csv(gpath + dt + gkg_ext, 
                        sep='\t', 
                        encoding = 'latin1', 
                        names = gkg_cols, 
                        header=None)

    dfGKG = selectDeuGKG(dfGKG)
    dfGKG = dfGKG.reset_index(drop=True)
    
    allNews = dfGKG['DocumentIdentifier'].to_list()

    
    dfGKG['TitleEntities'] = ''
    dfGKG['TextEntities'] = ''
    i = 0
    print('-------------------------%10s---------------------------------------' % (dt))
    for news in allNews:
        try:
            article = newspaper.Article(news, language='de')
            article.download()
            article.parse()
            doctitle = deutsch(article.title)
            print('--%4i-->%s' % (i, doctitle.text))
            titleentities = []
            for e in doctitle.ents:
                titleentities.append(e.text.strip().replace('-', ' '))            
            doc = deutsch(article.text)
            docentities = []
            n_e = 0
            for e in doc.ents:
                n_e += 1
                docentities.append(e.text.strip().replace('-', ' '))            
            dfGKG.loc[i, 'TitleEntities'] = ','.join(titleentities)
            dfGKG.loc[i, 'TextEntities'] = ','.join(docentities)
            dfGKG.loc[i, 'lenTextEntities'] = n_e
            dfGKG.loc[i, 'TitleOrgs'] = findStockNames(titleentities, stockNames)
            dfGKG.loc[i, 'TextOrgs'] = findStockNames(docentities, stockNames)
            dfGKG['wordCNT'] = 0
            dfGKG['wordCNT'] = dfGKG['GCAM'].apply(wordCNT)
                        
        except:
            print('#################################Fehler beim Lesen von ' + news)
        i += 1
    
    dfGKG[fgkg_cols].to_csv(gpath + dt + gkg_ext)
    
    
    