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


def log(msg):
    
    dt_raw = datetime.datetime.now()
    real_dt = dt_raw.strftime('%Y-%m-%d %H:%M:%S')
    
    print(real_dt + ' ' + ': ' + msg)
    sys.stdout.flush()



def update_db(forex_values):
    
    insert_sql = """INSERT INTO forex_raw (currency, date_time, open, low, high, close, cnt_diff_vals) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(insert_sql, forex_values)
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))




dt = datetime.datetime.now()
curr_open = 1.22000
curr_high = 1.23999
curr_low = 1.21345
curr_close = 1.22500
cnt_diff_kurs = 200

update_db(('EUR/USD', dt.strftime('%Y-%m-%d %H:%M'), curr_open, curr_high, curr_low, curr_close, cnt_diff_kurs))

