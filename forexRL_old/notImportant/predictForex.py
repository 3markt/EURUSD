#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:46:46 2018

@author: uwe.mueller
"""


import datetime
import time
import sys
import psycopg2
import pandas as pd
import numpy as np
import ta
from pickle import load
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.models import Model, load_model, Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam




def log(msg):
    global dayopen_processing, dayend_processing, prev_dt, df, currency, scaler, basepath, position
    
    real_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(real_dt + ' ' + currency + ': ' + msg)
    sys.stdout.flush()



def convert_str2dt(row):
    
    try:
        return datetime.datetime.strptime(row, '%Y-%m-%d %H:%M')
    except:
        log('Error row =' + row)



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
        log('Error reading dreimarktDB Error: ' + str(error))
        
    return df



def calc_features(in_df):
    
    df = in_df.copy()
    df['WL'] = 10000*(df['Close'] - df['Close'].shift(1))/df['Close']
    
    BB = ta.volatility.BollingerBands(df['Close'], 
                                      window = 10, 
                                      window_dev = 2)
    df['bb_h'] = (BB.bollinger_hband() - df['Close'])/df['Close']
    df['bb_l'] = (BB.bollinger_lband() - df['Close'])/df['Close']
    df['bb_m'] = (BB.bollinger_mavg() - df['Close'])/df['Close']
    
    MACD = ta.trend.MACD(df['Close'],
                         window_slow = 26, 
                         window_fast = 12, 
                         window_sign = 9)
    df['macd'] = MACD.macd()
    df['macd_d'] = MACD.macd_diff()

    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        
    return df



def prep_data(df):
    
    ts5Min = df.copy()
    ts5Min.set_index(pd.DatetimeIndex(ts5Min.Time), inplace=True)
    ts5Min = calc_features(ts5Min)
    ts5Min.rename(columns={'Time':'time'}, inplace=True)   
    
    ts15Min = ts5Min.resample('15T', label='right').agg({'Open': 'first',
                                                         'Close': 'last',
                                                         'Low': 'min',
                                                         'High': 'max'})
    ts15Min = calc_features(ts15Min)
    ts15Min['time'] = ts15Min.index        
    abt = pd.merge_ordered(ts5Min, ts15Min, 
                           how='left', on='time',
                           suffixes=('', '_15Min'))
    
    ts1H = ts5Min.resample('60T', label='right').agg({'Open': 'first',
                                                      'Close': 'last',
                                                      'Low': 'min',
                                                      'High': 'max'})
    ts1H = calc_features(ts1H)
    ts1H['time'] = ts1H.index    
    abt = pd.merge_ordered(abt, ts1H, 
                           how='left', on='time',
                           suffixes=('', '_1H'))
    
    ts4H = ts5Min.resample('240T', label='right').agg({'Open': 'first',
                                                       'Close': 'last',
                                                       'Low': 'min',
                                                       'High': 'max'})
    ts4H = calc_features(ts4H)
    ts4H['time'] = ts4H.index    
    abt = pd.merge_ordered(abt, ts4H, 
                           how='left', on='time',
                           suffixes=('', '_4H'))
    
    abt = abt.fillna(method='ffill')
    abt = abt.dropna()
    abt['close_scaled'] = abt['Close']
    abt = abt[-60:]
    
    return abt



def initialize_data():
    global dayopen_processing, dayend_processing, prev_dt, df, currency, scaler, basepath, position
    
    sql = """select date_time, open, low, high, close from forex_raw 
                where currency = %s 
                order by date_time desc
                limit 2400;"""
    
    df = read_db(currency, sql)
    df.rename(columns={'date_time': 'Time',
                       'open': 'Open',
                       'low': 'Low',
                       'high': 'High',
                       'close': 'Close'}, 
              inplace=True)

    df = df.sort_values('Time')
    prev_dt = df['Time'].iat[-1]
    df['Time'] = df['Time'].apply(convert_str2dt)
    df['minute'] = df['Time'].dt.minute.apply([lambda x: str(x).rjust(2, '0')])
    df['hour'] = df['Time'].dt.hour.apply([lambda x: str(x)])
    df['index'] = (df['hour'] + '.' + df['minute']).astype(float)
    df['Position'] = 0.0
    df.reset_index(drop=True, inplace=True)



def update_data():
    global dayopen_processing, dayend_processing, prev_dt, df, currency, scaler, basepath, position
    
    features2scale = ['index', 'close_scaled', 'WL', 
                      'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                      'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                      'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                      'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                      ]
        
    feature_list = ['Position', 'index', 'close_scaled', 'WL', 
                    'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi',
                    'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min',
                    'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H', 
                    'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H'        
                    ]
    
    data_updated = False
    prev_dt = df['Time'].iat[-1]
        
    sql_new = """select date_time, open, low, high, close from forex_raw 
                    where currency = %s 
                    order by date_time desc limit 1;"""
    
    df_new = read_db(currency, sql_new)
    
    df_new.rename(columns={'date_time': 'Time',
                           'open': 'Open',
                           'low': 'Low',
                           'high': 'High',
                           'close': 'Close'}, 
                  inplace=True)

    if (df_new['Time'].iat[-1] != prev_dt):
        data_updated = True
        prev_dt = df_new['Time'].iat[-1]
        df_new['Time'] = df_new['Time'].apply(convert_str2dt)
        df_new['minute'] = df_new['Time'].dt.minute.apply([lambda x: str(x).rjust(2, '0')])
        df_new['hour'] = df_new['Time'].dt.hour.apply([lambda x: str(x)])
        df_new['index'] = (df_new['hour'] + '.' + df_new['minute']).astype(float)
        df_new['Position'] = position
        df = df.append(df_new)
        df.reset_index(drop=True, inplace=True)
        
    new_close = df['Close'].iat[-1]
    abt = prep_data(df)
    abt.index = pd.RangeIndex(len(abt.index))
    abt_scaled = pd.DataFrame(scaler.transform(abt[features2scale]), columns=features2scale)
    abt = pd.concat([abt['Position'], abt_scaled], axis=1)
    path = basepath + 'data/forex/duka/'
    df.to_csv(path + 'df.csv')
    abt.to_csv(path + 'abt.csv')

    return abt[feature_list].values, new_close, data_updated
    



def update_trade_log(trade_values):
    
    insert_sql = """INSERT INTO trade_log (currency, 
                                           date_time, 
                                           position, 
                                           close, 
                                           profit, 
                                           trade_cnt) 
                    VALUES (%s, %s, %s, %s, %s, %s)"""
                    
    try:
        db_connect = psycopg2.connect(database = "dreimarktdb", 
                                      user = "chitlom", 
                                      password = "Pampe1muse", 
                                      host = "localhost", 
                                      port = "5432")
        
        db_cursor = db_connect.cursor()
        db_cursor.execute(insert_sql, trade_values)
        db_connect.commit()
    
    except (Exception, psycopg2.Error) as error:
        log('Error updateing dreimarktDB Error: ' + str(error))



def init_model(path):
    
    # define LSTM-Model
    model = Sequential()
    model.add(LSTM(units=120, 
                   recurrent_dropout=0.3,
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01),
                   kernel_initializer='truncated_normal',
                   batch_input_shape=(1, 60, 24)))
    
    model.add(Dense(units=3, 
                    kernel_initializer='truncated_normal'))    
    model.compile(loss='mean_squared_error',
                  optimizer="Adam",
                  metrics=['mae'])
    
    model.load_weights(path)
    
    return model



def isTradingTime(dt):
    global dayopen_processing, dayend_processing, prev_dt, df, currency
    global scaler, basepath, position, trade_close, trade_cnt, end_of_day
    
    weekdays = [0, 1, 2, 3, 4]
    wd = dt.tm_wday
    hr = dt.tm_hour
    mt = dt.tm_min

    if ((wd in weekdays) and (hr >= 5) and (hr < 21)):
        if (not dayopen_processing):
            log('executing dayopen-processing')
            initialize_data()
            position = 0.0
            end_of_day = False
            position = 0.0
            trade_close = 0.0
            trade_cnt = 0
            dayopen_processing = True
            dayend_processing = False
        if (hr == 20 and mt >= 55):
            end_of_day = True
        time.sleep(1)
        return True
    elif ((wd in weekdays) and (hr >= 21)):
        if (not dayend_processing):
            log('executing dayend-processing')
            dayopen_processing = False
            dayend_processing = True
        time.sleep(3600)
        return False
    else:
        time.sleep(3600)
        return False



time.tzset()
basepath = '/home/chitlom/'
pip = 10000
trade_fee = 1.0

currency = sys.argv[1]
#currency = 'EUR/USD'
if (currency not in ['EUR/USD', 'USD/JPY', 'EUR/GBP', 'USD/CHF', 'EUR/JPY', 'EUR/CHF', 'AUD/USD', 'USD/CAD']):
    exit('Wrong Currency: ' + currency)

scaler = load(open(basepath + 'scaler/scaler-eurusd.pkl', 'rb'))

model = init_model(basepath + 'model/eurusd-Inititial/eurusdWeights-Inititial')
initialize_data()
dayopen_processing = False
dayend_processing = False
end_of_day = False

new_candle = False
position = 0.0
trade_close = 0.0
trade_cnt = 0

log('Previous Date = ' + str(prev_dt))
while True:
    dt = time.localtime()
    if isTradingTime(dt):
        if (dt.tm_min % 5 == 0):            
            if new_candle:
                time.sleep(1)
                abt, new_close, data_updated = update_data()
                if data_updated:
                    new_candle = False
                    predicted_reward = model.predict(abt.reshape(1, 60, 24))
                    action = np.argmax(predicted_reward)
                    log('Predicted Reward (Hold, Buy, Sell)' + str(predicted_reward))
                    log('Predicted Action = ' + str(action) + ' - current position = ' + str(position))
                    if end_of_day:
                        action = 0
                    if (action in [1, 2] and position == 0.0):
                        # open new trade - remember the close value
                        trade_close = new_close
                        trade_cnt += 1
                    elif (action in [0, 2] and position == 0.5):
                        # close a long position
                        profit = pip*(new_close - trade_close) - trade_fee
                        update_trade_log((currency, 
                                          prev_dt, 
                                          position, 
                                          new_close, 
                                          profit, 
                                          trade_cnt))
                        trade_close = new_close
                        if (action == 2):
                            trade_cnt = 1
                        else:
                            trade_cnt = 0
                    elif (action in [0, 1] and position == 1.0):
                        # close a short position
                        profit = pip*(trade_close - new_close) - trade_fee
                        update_trade_log((currency, 
                                          prev_dt, 
                                          position, 
                                          new_close, 
                                          profit, 
                                          trade_cnt))
                        trade_close = new_close
                        if (action == 1):
                            trade_cnt = 1
                        else:
                            trade_cnt = 0
                    elif (position == action * 0.5 and action in [1, 2]):
                        # keep long or short position
                        trade_cnt += 1                
                        
                    position = action * 0.5
                else:
                    log('No new data yet')
        else:
            if not new_candle:
                new_candle = True


    
