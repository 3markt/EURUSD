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
import PyPDF2

npath = '/Users/uwe.muller/Hope/Data/news/fed-minutes/'

pdf_file = open(npath + 'fomcminutes20190501.pdf', 'rb')
read_pdf = PyPDF2.PdfFileReader(pdf_file)
nump = read_pdf.getNumPages()
info = read_pdf.getDocumentInfo()
print(nump, info)

txt = ''
for i in range(nump):
    page = read_pdf.getPage(i)
    txt += page.extractText()
