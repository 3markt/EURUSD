#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))

class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = -1



"""
path = '/home/uwe/Hope/data/raw/'

df = pd.read_csv(path + 'EURUSD_Candlestick_5_m_BID_09.02.2025-08.10.2026.csv')

prices = np.array(df['close'].iloc[:288])

logp_rewards = 10000*np.log(np.roll(prices, -1)/prices)
logp_rewards[-1] = 10000*np.log(prices[-1]/prices[-2])
logp_rewards = np.clip(logp_rewards, -5, 5)
logp_transcost = 10000*np.log(1 + ((1/10000)/prices))

print(prices,
      logp_rewards,
      logp_transcost)

start = 288
end = 576
x = range(end - start)
y = df_logp.iloc[start:end]
#y = np.clip(df_logp, a_min=-5, a_max=5)[start:end]

plt.scatter(x, y)
plt.grid()
plt.show()
"""

