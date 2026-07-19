#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import psycopg2
import pandas as pd
import os
import ta
import datetime as dt
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir
import plotly.express as px
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default='browser'


featureList = ['Position', 'zeitIndex', 'scaledClose', 'round_lot', 'WL', 
               'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi', 'macd_30Min', 
               'macd_d_30Min', 'bb_h_30Min', 'bb_l_30Min', 'rsi_30Min', 
               'macd_6H', 'macd_d_6H', 'bb_h_6H', 'bb_l_6H', 'rsi_6H',
               'CPI_US' , 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
               'Zins_US', 'Zins_EU', 'Schock_Preis_US', 'Schock_Preis_EU', 
               'Verbraucherstimmung_US', 'Verbraucherstimmung_EU', 'GfK', 
               'Schock_Nachfrage_US', 'Schock_Nachfrage_EU', 
               'PMI_Manufacturing_US', 'PMI_Service_US', 
               'PMI_Manufacturing_EU', 'PMI_Service_EU', 
               'Schock_Produktion_US', 'Schock_Produktion_DE', 
               'JoblessInitial_US', 'UnemploymentRate_US', 
               'UnemploymentRate_EU', 'Schock_Arbeitsmarkt_US', 
               'Schock_Arbeitsmarkt_EU', 'GDP_US', 'GDP_EU', 
               'TradeBalance_US', 'TradeBalance_EU', 
               'Schock_Konjunktur_US', 'Schock_Konjunktur_EU']

    

basepath = '/Users/uwe.muller/Hope/data/'
path = basepath + 'processed/EURUSD/forexCandle/'
dt_select = '2010-05-07'
"""
df = pd.read_csv(path + 'df_all.csv', parse_dates=['time'])
print(len(df))
df = df[(df['time'] >= dt_select + ' 00:00:00') \
        & (df['time'] <= dt_select + ' 23:55:00')].copy()
df.to_csv(path + 'test.csv', index=False)
"""
df = pd.read_csv(path + 'test.csv', parse_dates=['time'])
feat1 = 'bb_l'
feat2 = 'bb_h'
#Create a figure and add three traces to it
fig = go.Figure()

fig.add_trace(                               #Add a bar graph to the figure
        go.Scatter(
        x=df['time'],
        y=df['Close'],
        name="Close",
        mode='lines',
        marker_color='#d99b16',              #Specify the color of the bars    
        hoverinfo='none',                    #Hide the hoverinfo
        ),
        )   

fig.add_trace(                               #Add the second trace (line graph) to the figure
    go.Scatter(                          
    x=df['time'],
    y=df[feat1],
    name=feat1,
    mode='lines',            
    hoverinfo='none',                        #Pass the 'text' column to the hoverinfo parameter to customize the tooltip
    line = dict(color='#f70f13', width=3),   #Specify the color of the line
    yaxis="y2" ),                            #By specifying yaxis='y2' we tell plotly that this line chart uses a secondary y-axis
    )                   

fig.add_trace(                               #Add the third trace (line graph) to the figure
    go.Scatter(
    x=df['time'],
    y=df[feat2],
    name=feat2,
    mode='lines',
    hoverinfo='none',                        #Pass the 'text' column to the hoverinfo parameter to customize the tooltip
    line = dict(color='#0000ff', width=3),   #Specify the color of the line
    yaxis="y2" ),                            #By specifying yaxis='y3' we tell plotly that this line graph uses the third y-axis
    )  

# Create axis objects
fig.update_layout(
      xaxis=dict(
          domain=[0, 1]                 #Sets the domain of this axis (in plot fraction). Play with the numbers to see how it affects the plot
      ),
    yaxis=dict(                         
        title="Close",               #Add an axis title for the primary axis
        titlefont=dict(
            color="#d99b16"               #Make the color of the y-axis the same as the bar color
        ),
        tickfont=dict(
            color="#d99b16"
        )
    ),
    yaxis2=dict(
        title='macd',                      #Add axis title for the second y-axis
        titlefont=dict(
            color="#0000ff"              #Make the color of the third y-axis the same as its corresponding line color
        ),
        tickfont=dict(
            color="#0000ff"
        ),
        anchor="x",                      #If set to "x", this axis is bound to the corresponding opposite-letter axis - in this case, y-axis. And the 'side' parameter specifies which side this y-axis is placed
        overlaying="y",
        side="right"
    ),
    )

                                 

fig.show()
