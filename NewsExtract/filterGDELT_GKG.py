#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:20:54 2018

@author: uwe.mueller
"""

import requests
import pandas as pd
import zipfile
import io
import os
from datetime import datetime


class XIndex():
    
    def __init__(self, DAXC):
        
        self.startDate = DAXC['Datum'].min() 
        self.initMembers = DAXC['Addition'][DAXC['Datum'] == self.startDate].sort_values().values.tolist()
        self.addMembers = DAXC[DAXC['Datum'] != self.startDate]

    def searchMembers(self, datum):
        
        self.currMembers = self.initMembers.copy()
        if (datum < self.startDate):
            raise Exception('Datum kleiner ala Index-Anfangsdatum')
        else:
            deladds = self.addMembers[['Deletion', 'Addition']][self.addMembers['Datum'] <= datum].values.tolist()
            while (len(deladds) > 0):
                update = deladds.pop(0)
                if (update[0] != ''):
                    if (update[0] in self.currMembers):
                        self.currMembers.remove(update[0])
                    else:
                        raise Exception(update[0] + ' not in current DAX list')
                if (update[1] != ''):
                    self.currMembers.append(update[1])
           
            self.currMembers.sort()
                



def selectGKGs(row, orgs):
    
    if (type(row['V2Organizations']) == str and type(row['TranslationInfo']) == str):
        if ('srclc:deu' in row['TranslationInfo'].lower() and
            row['SourceCollectionIdentifier'] == 1.0):
            gkgOrgs = row['V2Organizations'].lower().split(';')
            for gorg in [x.split(',')[0].lower() for x in gkgOrgs]:
                if (gorg in orgs):
                    row['Relevant'] = True
    
    return row
                
                

#
# set some configuration items
#
from_dt = '20200920110000'
index_url = 'http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt'
gpath = '/Users/uwe.muller/Hope/Data/GDELT/selectedNews/'
gkg_ext = '.translation.gkg.csv'
ofile_ext = 'filteredGKG.csv'

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
            'Quotations',
            'AllNames',
            'Amounts',
            'TranslationInfo',
            'Extras']
#      
# lesen der xDAX Mitgliederänderungen
#
path = '/Users/uwe.muller/Hope/Data/AktienIndicies/'

daxc = pd.read_csv(path + 'DAX_Composition.csv', 
                   delimiter=';', 
                   infer_datetime_format=True, 
                   parse_dates=['Datum'],
                   dayfirst = True,
                   na_values = '')

daxc['Deletion'] = daxc['Deletion'].map(lambda x: x.lower() if isinstance(x,str) else x)
daxc['Addition'] = daxc['Addition'].map(lambda x: x.lower() if isinstance(x,str) else x)
daxComposite = daxc.sort_values(by='Datum')
daxComposite = daxComposite.fillna('')
dax = XIndex(daxComposite)

mdaxc = pd.read_csv(path + 'MDAX_Composition.csv', 
                   delimiter=';', 
                   infer_datetime_format=True, 
                   parse_dates=['Datum'],
                   dayfirst = True,
                   na_values = '')

mdaxc['Deletion'] = mdaxc['Deletion'].map(lambda x: x.lower() if isinstance(x,str) else x)
mdaxc['Addition'] = mdaxc['Addition'].map(lambda x: x.lower() if isinstance(x,str) else x)
mdaxComposite = mdaxc.sort_values(by='Datum')
mdaxComposite = mdaxComposite.fillna('')
mdax = XIndex(mdaxComposite)

sdaxc = pd.read_csv(path + 'SDAX_Composition.csv', 
                    delimiter=';', 
                    infer_datetime_format=True, 
                    parse_dates=['Datum'],
                    dayfirst = True,
                    na_values = '')

sdaxc['Deletion'] = sdaxc['Deletion'].map(lambda x: x.lower() if isinstance(x,str) else x)
sdaxc['Addition'] = sdaxc['Addition'].map(lambda x: x.lower() if isinstance(x,str) else x)
sdaxComposite = sdaxc.sort_values(by='Datum')
sdaxComposite = sdaxComposite.fillna('')
sdax = XIndex(sdaxComposite)

#
# lesen aller GDELT GKG Files bis zurück nach 2015
#
index = requests.get(index_url)
sindex = index.text.split('\n')
effectiveDate = '0'
monat = '202009'
filteredGKG = pd.DataFrame()

#
# loop über alle gelesenen GDELT GKG Files und filtern von Nachrichten entsprechend der Organisationen
#
for s in sindex[::-1]:
    sp = s.split('http://')
    if (len(sp) > 1 and '.gkg.csv.zip' in sp[1]):
        ind = sp[1]
        dt = ind[30:44]
        if (dt >= '20200920000000'):
#
# erzeugen der xDAX Member Liste
            if (effectiveDate != dt[:8]):
#
# neuer Tag: erzeuge neue xDAX Mitgliederliste   
                if (monat != dt[:6] and len(filteredGKG) > 0):
#
# neuer Monat: speichere gefilterte Liste vom Vormonat
                    filteredGKG[fgkg_cols].to_csv(gpath + monat + ofile_ext)
                    filteredGKG = pd.DataFrame()
                    monat = dt[:6]
                    
                effectiveDate = dt[:8]
                print('Processing Date:', effectiveDate)
                dax.searchMembers(datetime.strptime(effectiveDate, '%Y%m%d'))
                mdax.searchMembers(datetime.strptime(effectiveDate, '%Y%m%d'))
                sdax.searchMembers(datetime.strptime(effectiveDate, '%Y%m%d'))
                searchList = dax.currMembers + mdax.currMembers + sdax.currMembers
                
            r = requests.get('http://' + ind)
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(path = gpath)
                gkg = pd.read_csv(gpath + dt + gkg_ext, 
                                  sep='\t', 
                                  encoding = 'latin1', 
                                  names = gkg_cols, 
                                  header=None)
                
                gkg['Relevant'] = False
                gkg = gkg.apply(selectGKGs, args=[searchList], axis = 1)
                filteredGKG = filteredGKG.append(gkg[gkg['Relevant']]) 
                os.remove(gpath + dt + gkg_ext)
            except:
                print('ERROR: GKG File ' + ind + ' cprocessing error. Skipping file ...')
        
filteredGKG[fgkg_cols].to_csv(gpath + monat + ofile_ext)

            


