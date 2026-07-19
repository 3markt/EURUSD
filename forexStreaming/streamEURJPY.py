#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:46:46 2018

@author: uwe.mueller
"""


import requests as rq
import time
import sys
import psycopg2


def log(msg):
    global currency
    
    real_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
    print(real_dt + ' ' + currency + ': ' + msg)
    sys.stdout.flush()




def getLatestForexRaw():
    #
    # selects the latest Forex_Raw Record from DB
    global currency
    select_sql = "select open, close, low, high from forex_raw \
        where currency = '{0}' and date_time in (select max(date_time) \
        from forex_raw where currency = '{0}')".format(currency.replace('/', ''))

    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(select_sql)
        vals = db_cursor.fetchone()
        return vals
           
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))




def getTxt(s, currency, url):
    global buy, sell, kurs, txt
    
    try:
        f = s.get(url+currency)
        txt = f.text
        try:
            sell = float(txt[7:14])
            buy = float(txt[14:21])
            kurs = round((sell+buy)/2, 5)
        except:
            pass

    except:
# if session closed get a new one
        log('Session closed - re-open it')
        s = rq.Session()


def update_db(forex_values):
    
    insert_sql = """INSERT INTO forex_raw 
                (currency, date_time, open, low, high, close, cnt_diff_vals) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)"""
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



def marketIsOpen(dt):

    wd = dt.tm_wday
    hr = dt.tm_hour
    se = dt.tm_sec
    
    # set day-saving-time
    dtt = time.mktime(dt)
    dst = time.localtime(dtt).tm_isdst

    if (wd == 5):
        time.sleep(3600)
        return False
    elif (wd == 4 and dst and hr >= 21 and se > 1):
        time.sleep(3600)
        return False
    elif (wd == 4 and not dst and hr >= 22 and se > 1):
        time.sleep(3600)
        return False
    elif (wd == 6 and dst and hr < 21):
        time.sleep(5)
        return False
    elif (wd == 6 and not dst and hr < 22):
        time.sleep(5)
        return False
    else:
        time.sleep(1)
        return True
    
    


time.tzset()
url = 'http://webrates.truefx.com/rates/connect.html?c='
#currency = sys.argv[1]
currency = 'EUR/JPY'
if (currency not in ['EUR/USD', 'USD/JPY', 'EUR/GBP', 'USD/CHF', 'EUR/JPY', 'EUR/CHF', 'AUD/USD', 'USD/CAD']):
    exit('Wrong Currency: ' + currency)

session = rq.Session()
buy, sell, kurs = 0.0, 0.0, 0.0
getTxt(session, currency, url)
prev_vals = getLatestForexRaw()

if (prev_vals is None):
    curr_open = kurs
    curr_close = kurs
    curr_low = kurs
    curr_high = kurs
    prev_kurs = kurs
else:
    curr_open = prev_vals[0]
    curr_close = prev_vals[1]
    curr_low = prev_vals[2]
    curr_high = prev_vals[3]
    prev_kurs = prev_vals[1]
   
curr = currency.replace('/', '')

print('previous values:', curr_open, curr_close, curr_low, curr_high)

new_candle = True
cnt_diff_kurs = 0

while True:
    dt = time.gmtime()    
    if marketIsOpen(dt):
        if (dt.tm_min % 5 == 0):
            if new_candle:
                new_candle = False
                curr_close = prev_kurs
                # save candle to db
                if (cnt_diff_kurs < 5):
                    log('WARNING: Count of different Values less than 5: ' + str(cnt_diff_kurs))
                    
                update_db((curr, 
                           time.strftime('%Y-%m-%d %H:%M', dt), 
                           curr_open, 
                           curr_low,
                           curr_high, 
                           curr_close, 
                           cnt_diff_kurs))
                
                curr_high = kurs
                curr_low = kurs
                curr_open = kurs
                curr_close = kurs
                cnt_diff_kurs = 0
        else:
            if not new_candle: 
                new_candle = True
        
        getTxt(session, currency, url)
        if (kurs != prev_kurs):
            prev_kurs = kurs
            if (curr_low > kurs):
                curr_low = kurs
            if (curr_high < kurs):
                curr_high = kurs
            cnt_diff_kurs += 1
