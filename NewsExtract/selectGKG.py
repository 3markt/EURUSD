# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:24:49 2018

This Code creates a .csv-file including all GDELT-Data
listet in csv_Liste.csv.

@author: gerrit.meinhardt
"""
import pandas as pd
import requests
import zipfile
import io
from os import listdir
from os.path import isfile, join

    
def selectGKGs(row, org):
    
    if (type(row['V2Organizations']) == str):
        if (org in row['V2Organizations'].lower() and
            'srclc:deu' in row['TranslationInfo'].lower() and
            row['SourceCollectionIdentifier'] == 1.0):
            row['Relevant'] = True
    
    return row


path = '/Users/uwe.muller/Hope/Data/GDELT/selectedNews/'

csv_files = [path + f for f in listdir(path) if isfile(join(path, f))]
  

columns = ['GKGRECORDID',
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

Data = [pd.read_csv(str(x), sep='\t', encoding = 'latin1', names = columns, header=None) for x in csv_files]
Data = pd.concat(Data)
Data['Relevant'] = False

Data = Data.apply(selectGKGs, args=['qiagen'], axis = 1)
Data = Data[Data['Relevant']]
print(Data[['V2Organizations', 'Themes']])


