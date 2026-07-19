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


class myLSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=54, hidden_size=30, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=54, hidden_size=10, num_layers=1, batch_first=True)
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        return x
        
        


model = myLSTM()
for p in model.parameters():
    print(len(p))

xx = torch.randn(276, 12, 54)

