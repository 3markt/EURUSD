#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import spacy
import requests
from bs4 import BeautifulSoup



#link = "https://www.spiegel.de/wirtschaft/us-finanzkrise-lehman-kollaps-reisst-dax-in-die-tiefe-a-578368.html"
#link = 'https://www.derwesten.de/panorama/vermischtes/kaufland-werbung-angebot-song-lied-kunden-supermarkt-id231322758.html'
#link = 'https://www.rundschau-online.de/region/bonn/wachtberg/wer-zahlt-fuer-den-schrott--gemeinde-wachtberg-hat-stehengelassenes-auto-sichergestellt-37925336'
link = 'https://www.handelsblatt.com/politik/konjunktur/nachrichten/aussenhandel-chinas-exporte-legen-im-corona-jahr-weiter-zu/26797714.html'

f = requests.get(link)
html_doc = f.content
soup = BeautifulSoup(html_doc, 'html.parser')

print(soup)
myFile = open('crash.txt', 'w')
myFile.write(str(html_doc))
myFile.close()

print('######################################################################################')
text = soup.find_all(text=True)

names = []
i = 0
for t in text:
    i += 1
    names.append(t.parent.name)

nn = list(set(names))

news = ''
for t in text:
	if t.parent.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'time']:
		news += '{} '.format(t)
    
#print(news)
