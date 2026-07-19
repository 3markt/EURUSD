#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import gym
from collections import deque
import datetime
import time
import ta
from enum import Enum


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Nothing = 0.0
    Long = 0.5
    Short = 1.0


class Test():
    
    def __init__(self, position, last_trade, current_tick):
        
        self.bs = 10
        self.pip = 10000
        self.position = position
        self.prices = [1.22344, 1.22144, 1.22044, 1.22344, 1.22544, 1.22844, 1.23344, 1.22944, 1.22544, 1.22044]
        self.diffs = np.diff(self.prices, append=self.prices[-1])*self.pip
        self.trade_fee = 1.5
        self.gamma = 0.90
        self.current_tick = current_tick
        self.last_trade_tick = last_trade


        
    def calculate_reward(self, action):

        r = 0.0
        if (self.position == Positions.Nothing and action == Actions.Buy.value):
        # reward for opening a Long position
            for i in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r + self.diffs[i])
            r += self.diffs[self.current_tick] - self.trade_fee                
        elif (self.position == Positions.Nothing and action == Actions.Sell.value):
        # reward for opening a short position
            for i in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r - self.diffs[i])
            r += -self.diffs[self.current_tick] - self.trade_fee
        elif (self.position == Positions.Long and action == Actions.Hold.value):
        # reward for closing a long position
            r = self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
        elif (self.position == Positions.Long and action == Actions.Buy.value):
        # reward for holding a long position
            # discounted future rewards
            for i in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r + self.diffs[i])
            # reward from the past
            r += self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            # current reward minus trade fee
            r += self.diffs[self.current_tick] - self.trade_fee
        elif (self.position == Positions.Long and action == Actions.Sell.value):
        # reward for closing a long position and opening a short one
            # rewards for the long position
            rL = self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            # discounted future rewards
            for i in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r - self.diffs[i])
            # current reward
            r += - self.diffs[self.current_tick] 
            # add all together and subtract trading fee
            r += rL - self.trade_fee
        elif (self.position == Positions.Short and action == Actions.Hold.value):
        # reward for closing a short position
            r = -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
        elif (self.position == Positions.Short and action == Actions.Buy.value):
        # reward for closing a short position and opening a long one
            # reward for the short position
            rS = -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            # discounted future rewards
            for i in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r + self.diffs[i])
            # current reward
            r += self.diffs[self.current_tick]                       
            # add all together and subtract trading fee
            r += rS - self.trade_fee
        elif (self.position == Positions.Short and action == Actions.Sell.value):
        # reward for holding a short position
            # discounted future rewards
            for i in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r - self.diffs[i])
            # reward from the past
            r += -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            # current reward minus trade fee
            r += -self.diffs[self.current_tick] - self.trade_fee
            
        return r

pos = Positions.Short
action = 1
last_trade = 0
current_tick = 2
t = Test(pos, last_trade, current_tick)
print(t.diffs)
print(round(t.calculate_reward((action)), 5))
