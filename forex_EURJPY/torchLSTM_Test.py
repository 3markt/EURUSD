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
from os import listdir, makedirs
from os.path import isfile, join, isdir

class myLSTM(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=63, 
                            hidden_size=35, 
                            num_layers=1, 
                            dropout=0.2,
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=35, 
                            hidden_size=7, 
                            num_layers=1,                         
                            batch_first=True)
        self.linear = nn.Linear(7, 2)
        
        
        
    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        
        return x



def get_XY(path, trainDays, testDays):
    
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    
    for day in trainDays:
        x_train = np.append(x_train, np.load(path + 'X/x-' + day + '.npy'))
        y_raw = np.load(path + 'Y/y-' + day + '.npy')
        yy = np.zeros((12, 2), float)
        y = np.array([])
        for yi in y_raw:
            yy = np.append(yy[1:], yi)
            yy = yy.reshape(12,2)
            y = np.append(y, yy)
        y = y.reshape(276, 12, 2)
        y_train = np.append(y_train, y)
    x_train = torch.tensor(x_train.reshape(len(trainDays)*276, 12, 63)).type(torch.float)
    y_train = torch.tensor(y_train.reshape(len(trainDays)*276, 12, 2)).type(torch.float)

    for day in testDays:
        x_test = np.append(x_test, np.load(path + 'X/x-' + day + '.npy'))
        y_raw = np.load(path + 'Y/y-' + day + '.npy')
        yy = np.zeros((12, 2), float)
        y = np.array([])
        for yi in y_raw:
            yy = np.append(yy[1:], yi)
            yy = yy.reshape(12,2)
            y = np.append(y, yy)
        y = y.reshape(276, 12, 2)
        y_test = np.append(y_test, y)                        
    x_test = torch.tensor(x_test.reshape(len(testDays)*276, 12, 63)).type(torch.float)
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





path = '/Users/uwe.muller/Hope/data/final/EURUSD/'
model = myLSTM()
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
                                   batch_size=23,
                                   drop_last=True)
    epochs = 1
    train_rmse = 0.0
    test_rmse = 0.0
    for ii in range(epochs):
        model.train()
        for X, Y in train_loader:
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            Y_pred = model(x_train)
            train_rmse += np.sqrt(loss_fn(Y_pred, y_train).detach().item())        
            y_pred = model(x_test)
            test_rmse += np.sqrt(loss_fn(y_pred, y_test).detach().item())
        
    cntPos, cntTruePred, cntFalsePred, profit = \
        myMetrics(y_test[:,-1,:].detach().numpy().reshape(276, 1, 2), 
                  y_pred[:,-1,:].detach().numpy().reshape(276, 1, 2))

    print("Iteration %d: train RMSE %.4f, test RMSE %.4f" \
          % (i, train_rmse/epochs, test_rmse/epochs))
    print("Iteration %d: cntPos %3i, cntTruePred %3i cntFalsePred %3i profit %4.1f" \
          % (i, cntPos, cntTruePred, cntFalsePred, profit))
    print('###########################################################')
    
    out_dat.append([i, days[i], 
                    round(train_rmse/epochs,4), round(test_rmse/epochs, 4), 
                    cntPos, cntTruePred, cntFalsePred, profit])
    if (i%10 == 0):
        pd.DataFrame(out_dat, columns=['Iteration', 
                                       'Tag',
                                       'RMSE-Train', 
                                       'RMSE-Test',
                                       'cntPos',
                                       'truePred',
                                       'falsePred',
                                       'profit']).to_csv(path + 'testResult/torch_no_shuffle_35_7.csv',
                                                         index=False)
    
pd.DataFrame(out_dat, columns=['Iteration', 
                               'Tag',
                               'RMSE-Train', 
                               'RMSE-Test',
                               'cntPos',
                               'truePred',
                               'falsePred',
                               'profit']).to_csv(path + 'testResult/torch_no_shuffle_35_7.csv',
                                                 index=False)
