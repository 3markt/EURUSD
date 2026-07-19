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



def calcTarget(df, shift, limit):
        
    df['Target'] = 0
    for i in range(1, shift+1):
        df['CH'+str(i)] = 100*(df['High'].shift(-i) - df['Open'].shift(-1))/df['Open'].shift(-1)
        df['CL'+str(i)] = 100*(df['Low'].shift(-i) - df['Open'].shift(-1))/df['Open'].shift(-1)
#        df['CH'+str(i)] = 100*(df['High'].shift(-i) - df['Close'])/df['Close']
#        df['CL'+str(i)] = 100*(df['Low'].shift(-i) - df['Close'])/df['Close']
        df.loc[(df['CH'+str(i)] > (-1)*df['CL'+str(i)]) 
               & (df['CH'+str(i)] > limit)
               & (df['Target'] == 0), 'Target'] = 2
        df.loc[(df['CH'+str(i)] < (-1)*df['CL'+str(i)]) 
               & (df['CL'+str(i)] < (-1)*limit)
               & (df['Target'] == 0), 'Target'] = 1
        
    df.loc[(df['Target'] == 2), 'Target'] = 0  
        
    return df    




def splitSymbls(symbls, path):
    y = np.array([])
    for symbl in symbls:
        df = pd.read_csv(path + symbl + '.csv', parse_dates=['date'], infer_datetime_format=True)
        df = df.sort_values('date')
        l = len(df)
#        if (l > 2000):
#            df = df[l-2000:]
        df = df.reset_index(drop=True)        
        df = calcTarget(df, 2, 0.5)
        yi = np.sum(df['Target'])/len(df)
        y = np.append(y, yi)
    
    y50 = np.percentile(y, 50)
    y75 = np.percentile(y, 75)
    print(y50, y75)
    
    npsymbls = np.array(symbls)
    symbls = list(npsymbls[[list(y).index(x) for x in y if x > y50]])
    
    return symbls


path = '/Users/uwe.muller/Hope/Data/Aktienkurse/aktuell/'

dax = ['ADS.DEX',      # Adidas DAX
       'ALV.DEX',      # Allianz DAX
       'BAS.DEX',      # BASF DAX
       'BAYN.DEX',     # Bayer DAX
       'BEI.DEX',      # Beiersdorf DAX
       'BMW.DEX',      # BMW DAX
       'CON.DEX',      # Continental DAX
       '1COV.DEX',     # Covestro
       'DAI.DEX',      # Daimler
       'DBK.DEX',      # Dt. Bank
       'DB1.DEX',      # Dt. Boerse
       'DPW.DEX',      # Dt. Post
       'DTE.DEX',      # Dt. Telekom
       'EOAN.DEX',     # EO.N
       'FME.DEX',      # Fresenius Medical Care
       'FRE.DEX',      # Fresenius
       'HEI.DEX',      # HeidelbergCement
       'HEN3.DEX',     # Henkel Vz.
       'IFX.DEX',      # Infineon
       'LIN.DEX',      # Linde PLC
       'LHA.DEX',      # Lufthansa
       'MRK.DEX',      # Merck KGaA
       'MTX.DEX',      # MTU Aero Engines
       'MUV2.DEX',     # Munich RE
       'RWE.DEX',      # RWE
       'SAP.DEX',      # SAP
       'SIE.DEX',      # Siemens
       'VOW.DEX',      # VW Vz.
       'VNA.DEX',      # Vonovia
       'WDI.DEX'      # Wirecard  
       ]

dj = ['AAPL',         # Apple DJ
      'AMZN',         # Amazon DJ
      'AXP',          # American Express DJ
      'BA',           # Boeing DJ
      'CAT',          # Caterpillar
      'CSCO',         # Cisco Systems
      'CVX',          # Chevron
      'DIS',          # Walt Disney
      'DOW',          # Dow Inc.
      'GS',           # Goldman Sachs
      'IBM',          # IBM
      'INTC',         # Intel Corporation
      'JNJ',          # Johnson & Johnson
      'JPM',          # JP Morgan Chase
      'KO',           # Coca Cola
      'MCD',          # McDonal's Corporation
      'MMM',          # 3M Company
      'MRK',          # Merck & Co
      'MSFT',         # Microsoft
      'NKE',          # Nike Inc
      'PFE',          # Pfizer
      'PG',           # Procter & Gamble
      'RTX',          # Raytheon Technologie
      'TRV',          # Travelers Companies
      'UNH',          # United Health Group
      'V',            # VISA Inc.
      'VZ',           # Verizon Communications
      'WBA',          # Walgreens Boots Alliance
      'WMT',          # Wal-Mart
      'XOM'          # Exxon Mobile
      ]



mdax = ['AFX.DEX',      # Carl Zeiss Meditech MDAX
        'AIR.DEX',      # Airbus MDAX
        'AOX.DEX',      # alstria office REIT-AG MDAX
        'AT1.DEX',      # Aroundtown SA MDAX
        'ARL.DEX',      # Aareal Bank MDAX
        'B4B.DEX',      # Metro AG MDAX
        'BC8.DEX',      # Bechtle MDAX
        'BNR.DEX',      # Brenntag MDAX
        'BOSS.DEX',     # Hugo Boss MDAX
        'CBK.DEX',      # Commerzbank MDAX
        'COK.DEX',      # Cancom MDAX
        'COP.DEX',      # CompuGroup Medical MDAX
        'DHER.DEX',     # Delivery Hero MDAX
        'DUE.DEX',      # Dürr AG MDAX
        'DWNI.DEX',     # Deutsche Wohnen MDAX             
        'EVD.DEX',      # CTS Eventim MDAX
        'EVK.DEX',      # Evonik Industrie MDAX
        'EVT.DEX',      # Evotec MDAX
        'FNTN.DEX',     # Freent MDAX
        'FPE.DEX',      # Fuchs Petrolub MDAX
        'FRA.DEX',      # Fraport MDAX
        'G1A.DEX',      # GEA Group MDAX
        'G24.DEX',      # Scout24 MDAX
        'GLJ.DEX',      # Grenke AK MDAX
        'GXI.DEX',      # Gerresheimer MDAX
        'GYC.DEX',      # Grand City Properties MDAX
        'HFG.DEX',      # HelloFresh MDAX
        'HLE.DEX',      # Hella MDAX
        'HNR1.DEX',     # Hannover Rueck MDAX
        'HOT.DEX',      # Hochtief MDAX
        'KBX.DEX',      # Knorr-Bremse MDAX
        'KGX.DEX',      # Kion Group MDAX
        'LEG.DEX',      # LEG Immobilien MDAX
        'LXS.DEX',      # Lanxess MDAX
        'MOR.DEX',      # Morphosys MDAX
        'NDA.DEX',      # Aurubis MDAX
        'NEM.DEX',      # Nemetschek MDAX
        'O2D.DEX',      # Telefonica Deutschland MDAX
        'OSR.DEX',      # Osram Licht AG MDAX
        'PBB.DEX',      # Deutsche Pfandbrief MDAX
        'PSM.DEX',      # ProSiebenSat.1 Media MDAX
        'PUM.DEX',      # Puma MDAX
        'QIA.DEX',      # Qiagen MDAX
        'RAA.DEX',      # Rational AG MDAX
        'RHM.DEX',      # Rheinmetall MDAX
        'RKET.DEX',     # Rocket Internet MDAX
        'RRTL.DEX',     # RTL Group MDAX
        'SDF.DEX',      # K+S (Kali + Salz) MDAX
        'SHL.DEX',      # Siemens Healthineers MDAX
        'SOW.DEX',      # Software AG MDAX
        'SRT.DEX',      # Sartorius MDAX
        'SY1.DEX',      # Symrise AG MDAX
        'TEG.DEX',      # TAG Immobilien MDAX
        'TKA.DEX',      # thyssenkrupp AG MDAX
        'TMV.DEX',      # TemViewer AG MDAX
        'UN01.DEX',     # Uniper MDAX
        'UTDI.DEX',     # United Internbet AG MDAX
        'VAR1.DEX',     # Varta AG MDAX
        'WAF.DEX',      # Siltronic AG MDAX
        'ZAL.DEX'       # Zalando MDAX          
                ]
symbls = dj + dax + mdax

symblsShort = splitSymbls(symbls, path)

print(symblsShort)


