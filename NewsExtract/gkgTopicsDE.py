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
from collections import Counter
from pprint import pprint
import newspaper
import spacy
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary

def cleanTXT(txt):
    
    txt_list = []
    for token in txt:
        if (not token.is_stop 
            and not token.is_punct 
            and not token.is_digit
            and '\n' not in token.text):
            txt_list.append(token.lemma_.lower())
        
    return txt_list

    

def txtPipeline(article, lang):
    
    title = lang(article.title)
    title = cleanTXT(title)

    body = lang(article.text)    
    doc_list = cleanTXT(body)
       
    doc_list = [doc_list]
    
    bigram = Phrases(doc_list, min_count=1, threshold=1)
    
    for idx in range(len(doc_list)):
        for token in bigram[doc_list[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                doc_list[idx].append(token)
    
    doc = title + title + doc_list[0]
    
    return doc


                
def selectDeuGKG(gkg):
    
    # extract German entries
    gkg = gkg[gkg['TranslationInfo'].str.contains('srclc:deu;') == True]
            
    return gkg    



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

#
# set some configuration items
#
gpath = '/Users/uwe.muller/Hope/Data/GDELT/selectedNews/'
gkg_ext = '.translation.gkg.csv'


# load spaCy model
deutsch = spacy.load('de_core_news_lg')

#
# lesen aller GDELT GKG Files bis zurück nach 2015
#
df = pd.read_csv(gpath + 'gkg-liste.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
gkgLines = df.values.tolist()
gkgLines = [g[0] for g in gkgLines][-30:-1]

docs = []

for gkg in gkgLines:
    print(gkg)
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

    i = 0
    print('-------------------------%10s---------------------------------------' % (dt))
    
    for news in allNews:
        try:
            article = newspaper.Article(news, language='de')
            article.download()
            article.parse()
            print(article.title, len(article.text))
            docs.append(txtPipeline(article, deutsch))
            
        except:
            print('#################################Fehler beim Lesen von ' + news)
        i += 1

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=2, no_above=0.5)
print(len(dictionary))
corpus = [dictionary.doc2bow(doc) for doc in docs]
print(len(corpus))

num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None

temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus)
pprint(top_topics)


    