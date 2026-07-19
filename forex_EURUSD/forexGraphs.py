#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default='browser'

feature = 'val_mae'

file = '/Users/uwe.muller/Hope/Data/final/EURUSD/testResult/lstm-ts36-ff21-u20.csv'
df1 = pd.read_csv(file)
df1 = df1[df1['Iteration'] > 1000]

file = '/Users/uwe.muller/Hope/Data/final/EURUSD/testResult/lstm-ts72-ff21-u20.csv'
df2 = pd.read_csv(file)
df2 = df2[df2['Iteration'] > 1000]

var = 'cntEqSign'
ext1 = '_ts36'
ext2 = '_ts72'

df = pd.merge_ordered(df1, df2, 
                      how='left', on='Iteration', 
                      suffixes=(ext1, ext2))
var1 = var + ext1
var2 = var + ext2


#Create a figure and add traces to it
fig = go.Figure()

fig.add_trace(                               #Add a bar graph to the figure
        go.Scatter(
        x=df['Iteration'],
        y=df[var1],
        name=var1,
        mode='lines',
        marker_color='#d99b16',              #Specify the color of the bars    
        hoverinfo='none',                    #Hide the hoverinfo
        ),
        )                                    

fig.add_trace(                               #Add the second trace (line graph) to the figure
    go.Scatter(                          
    x=df['Iteration'],
    y=df[var2],
    name=var2,
    mode='lines',            
    hoverinfo='none',                        #Pass the 'text' column to the hoverinfo parameter to customize the tooltip
    line = dict(color='#f70f13', width=1),   #Specify the color of the line
    yaxis="y2" ),                            #By specifying yaxis='y2' we tell plotly that this line chart uses a secondary y-axis
    )                   

"""
fig.add_trace(                               #Add the third trace (line graph) to the figure
    go.Scatter(
    x=df['time'],
    y=df[schock],
    name=schock,
    mode='lines',
    hoverinfo='none',                        #Pass the 'text' column to the hoverinfo parameter to customize the tooltip
    line = dict(color='#0000ff', width=3),   #Specify the color of the line
    yaxis="y2" ),                            #By specifying yaxis='y3' we tell plotly that this line graph uses the third y-axis
    )  
"""
"""
# Create axis objects
fig.update_layout(
      xaxis=dict(
          domain=[0, 1]                 #Sets the domain of this axis (in plot fraction). Play with the numbers to see how it affects the plot
      ),
    yaxis=dict(                         
        title=var1,               #Add an axis title for the primary axis
        titlefont=dict(
            color="#d99b16"               #Make the color of the y-axis the same as the bar color
        ),
        tickfont=dict(
            color="#d99b16"
        )
    ),
    yaxis2=dict(
        title=var2,                      #Add axis title for the second y-axis
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
"""
fig.show()    

