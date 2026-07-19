#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import newspaper as np
import spacy
from collections import Counter
from gensim.models import Phrases
import PyPDF2
import re
from os import listdir
from os.path import isfile, join
from pprint import pprint
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary




def cleanTXT(txt):
    
    txt_list = []
    for token in txt:
        if (not token.is_stop 
            and not token.is_punct 
            and not token.is_digit
            and not token.is_space
            and not token.is_oov
            and not token.is_bracket
            and not token.is_quote
            and not token.is_currency
            and '\n' not in token.text):
            txt_list.append(token.lemma_.lower())
        
    return txt_list

    

def txtPipeline(txt, lang):
    
    body = lang(txt)    
    doc_list = cleanTXT(body)
       
    doc_list = [doc_list]
    
    bigram = Phrases(doc_list, min_count=4, threshold=1)
    
    for idx in range(len(doc_list)):
        for token in bigram[doc_list[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                doc_list[idx].append(token)
                
    return doc_list[0]



# load spaCy model
en = spacy.load('en_core_web_lg')

npath = '/Users/uwe.muller/Hope/Data/news/fed-minutes/'


tsfiles = [f for f in listdir(npath) if isfile(join(npath, f)) and f[:11] == 'fomcminutes'
           and f != 'fomcminutes20190501.pdf']

docs = []
for f in tsfiles:
    pdf_file = open(npath + f, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    nump = read_pdf.getNumPages()
    print(f, nump)
    
    txt = ''
    for i in range(nump):
        page = read_pdf.getPage(i)
        txt += page.extractText()
    
    # removing new lines
    txt = re.sub('\n', '', txt)
    
    # removing one letter words
    txt = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', ' ', txt)
    
    # removing TM-sign
    txt = re.sub("\\u00AE|\\u00a9|\\u2122", "", txt)
        
    # merge of words split to next line
    txt = re.sub('-', '', txt)
    
    doc = txtPipeline(txt, en)
    docs.append(doc)

    
dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=2, no_above=0.5)
print(len(dictionary))
corpus = [dictionary.doc2bow(doc) for doc in docs]
print(len(corpus))

num_topics = 5
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
    


