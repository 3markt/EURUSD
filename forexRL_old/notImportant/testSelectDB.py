#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import sys
import datetime
import time
import psycopg2
import pandas as pd



def read_db(currency, sql):
    
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        df = pd.read_sql_query(sql, con=db_connect, params=[currency])
        db_connect = None    
    except (Exception, psycopg2.Error) as error:
        print('Error reading dreimarktDB Error: ' + str(error))
        
    return df


    
sql_one = """select date_time, open, low, high, close from forex_raw 
            where currency = %s 
            order by date_time desc limit 1;"""

sql_all = """select date_time, open, low, high, close from forex_raw 
            where currency = %s 
            order by date_time;"""
            



currency = 'EUR/USD'
df = read_db(currency, sql_all)
print(df)

df = read_db(currency, sql_one)
print(df)
