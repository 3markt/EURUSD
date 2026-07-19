#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import math
from os import listdir, makedirs
from os.path import isfile, join, isdir


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=12):
        
        super(PositionalEncoding, self).__init__() 
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        return x + self.pe[:x.size(0), :]
       


class TransAm(nn.Module):
    
    def __init__(self,feature_size=60,num_layers=9,dropout=0.15):
        
        super(TransAm, self).__init__()
        
        self.model_type = 'Transformer'
        
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, 
                                                        nhead=6, 
                                                        dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                         num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, 2)
        self.init_weights()


    def init_weights(self):
        
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self,src):
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        
        return output



def get_XY(path, trainDays, testDays):
    
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    
    for day in trainDays:
        x_raw = np.load(path + 'X/x-' + day + '.npy')
        x_train = np.append(x_train, x_raw[:,:,:-3])
        y_raw = np.load(path + 'Y/y-' + day + '.npy')
        yy = np.zeros((12, 2), float)
        y = np.array([])
        for yi in y_raw:
            yy = np.append(yy[1:], yi)
            yy = yy.reshape(12,2)
            y = np.append(y, yy)
        y = y.reshape(276, 12, 2)
        y_train = np.append(y_train, y)
    x_train = torch.tensor(x_train.reshape(len(trainDays)*276, 12, 60)).type(torch.float)
    y_train = torch.tensor(y_train.reshape(len(trainDays)*276, 12, 2)).type(torch.float)

    for day in testDays:
        x_raw = np.load(path + 'X/x-' + day + '.npy')
        x_test = np.append(x_test, x_raw[:,:,:-3])
        yy = np.zeros((12, 2), float)
        y = np.array([])
        for yi in y_raw:
            yy = np.append(yy[1:], yi)
            yy = yy.reshape(12,2)
            y = np.append(y, yy)
        y = y.reshape(276, 12, 2)
        y_test = np.append(y_test, y)                        
    x_test = torch.tensor(x_test.reshape(len(testDays)*276, 12, 60)).type(torch.float)
    y_test = torch.tensor(y_test.reshape(len(testDays)*276, 12, 2)).type(torch.float)
    
    return x_train, y_train, x_test, y_test




def myMetrics(y_true, y_pred):
    #
    # zählt die folgenden Anzahlen:
    #   - korrekte positive Prognosen
    #   - falsche positive Prognosen
    #   - Anzahl positiver Werte
    #
    
    pos_true = np.copy(y_true)
    pos_true[pos_true <= 0] = 0
    pos_true[pos_true > 0] = 1
    #
    # Anzahl positiver Werte
    cntPos = np.sum(pos_true)
    #
    # Anzahl korrekt vorhergesagter positiver Werte        
    pos_pred = np.copy(y_pred)
    pos_pred[pos_pred <= 0] = 0
    pos_pred[pos_pred > 0] = -1
    truePred = pos_true - pos_pred
    truePred[truePred!=2] = 0
    truePred[truePred==2] = 1
    cntTruePred = np.sum(truePred)
    win = np.sum(truePred*y_true)
    #
    # Anzahl falsch vorhergesagter positiver Werte
    pos_pred[pos_pred < 0] = 1
    falsePred = pos_pred - pos_true
    falsePred[falsePred!=1] = 0
    cntFalsePred = np.sum(falsePred)
    loss = np.sum(falsePred*y_true)
    profit = win + loss

    return cntPos, cntTruePred, cntFalsePred, profit


techCols = ['zeitIndex', 'scaledClose',  'round_lot', 
            'WL', 'macd', 'macd_d', 'bb_h', 'bb_l', 'rsi', 
            'macd_15Min', 'macd_d_15Min', 'bb_h_15Min', 'bb_l_15Min', 'rsi_15Min', 
            'macd_1H', 'macd_d_1H', 'bb_h_1H', 'bb_l_1H', 'rsi_1H',
            'WL_4H', 'macd_4H', 'macd_d_4H', 'bb_h_4H', 'bb_l_4H', 'rsi_4H',
            'WL_1D', 'macd_1D', 'macd_d_1D', 'bb_h_1D', 'bb_l_1D', 'rsi_1D']

ecoCols = ['CPI_US' , 'CPI_EU', 'CPI_DE', 'PPI_US', 'PPI_EU', 'PPI_DE', 
           'Zins_US', 'Zins_EU', 'Schock_Preis_US', 'Schock_Preis_EU', 
           'Verbraucherstimmung_US', 'Verbraucherstimmung_EU', 
           'Schock_Nachfrage_US', 'Schock_Nachfrage_EU', 
           'PMI_Manufacturing_US', 'PMI_Service_US', 
           'PMI_Manufacturing_EU', 'PMI_Service_EU', 
           'Schock_Produktion_US', 'Schock_Produktion_DE', 
           'JoblessInitial_US', 'UnemploymentRate_US', 
           'UnemploymentRate_EU', 'Schock_Arbeitsmarkt_US', 
           'Schock_Arbeitsmarkt_EU', 'GDP_US', 'GDP_EU', 
           'TradeBalance_US', 'TradeBalance_EU', 
           'Schock_Konjunktur_US', 'Schock_Konjunktur_EU']
    
scaledCols = techCols + ecoCols

target = ['Reward-Buy', 'Reward-Sell']

featureList = ['Position', 'currProfit'] + techCols + ecoCols
ff = len(featureList)
print(ff)


path = '/Users/uwe.muller/Hope/data/final/EURUSD/'

model = TransAm(feature_size=60, num_layers=8, dropout=0.15)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

days = [f[2:12] for f in listdir(path + 'X/') if isfile(join(path + 'X/', f))]
days.sort()
maxDays = len(days)
print(maxDays)
start = 1000
ntrain = 100
ntest = 1

loss = 999.
mae = 999.
val_loss = 999.
val_mae = 999.
cntPos = 0
truePred = 0
falsePred = 0
profit = 0.0

iteration = 0
out_dat = []

for i in range(start, maxDays-ntest, ntest):
#for i in range(start, 1001):
    
    # get x-y-Data for training and testing
    itrain = np.random.choice(range(i), 
                              size=ntrain, 
                              replace=False)
    trainDays = [days[ii] for ii in itrain]    
    itest = range(i-ntest, i)
    testDays = [days[ii] for ii in itest] 
    x_train, y_train, x_test, y_test = get_XY(path, trainDays, testDays)
    
    train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), 
                                   shuffle=False, 
                                   batch_size=12,
                                   drop_last=True)
    test_loader = data.DataLoader(data.TensorDataset(x_test, y_test), 
                                  shuffle=False, 
                                  batch_size=1,
                                  drop_last=True)


    epochs = 1
    train_rmse = 0.0
    test_rmse = 0.0
    
    for ii in range(epochs):
        model.train()
        for X, Y in train_loader:
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            train_rmse += np.sqrt(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_rmse = train_rmse/2300

        y_hat = np.array([])
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                y_pred = model(x)
                y_hat = np.append(y_hat, y_pred[:,-1,:].numpy())
                test_rmse += np.sqrt(loss_fn(y_pred, y).detach().item())
                print(y_pred[:,-1,:].numpy())
                
        test_rmse = test_rmse/276
        
    print("Iteration %d: train RMSE %.4f, test RMSE %.4f" \
          % (i, train_rmse/epochs, test_rmse/epochs))
    
    cntPos, cntTruePred, cntFalsePred, profit = \
        myMetrics(y_test[:,-1,:].numpy().reshape(276, 2),  y_hat.reshape(276, 2))
    print("Iteration %d: cntPos %3i, cntTruePred %3i cntFalsePred %3i profit %4.1f" \
          % (i, cntPos, cntTruePred, cntFalsePred, profit))
    print(72*'#')

