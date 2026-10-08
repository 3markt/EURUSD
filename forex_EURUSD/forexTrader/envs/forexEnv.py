import gym
# import gymnasium as gym
import gc
import sys
import pandas as pd
from gym import spaces
# from gymnasium import spaces
import numpy as np
from enum import Enum
import random
from os import listdir
from os.path import isfile, join


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Nothing = 0
    Long = 1
    Short = -1


class ForexEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, window_size, batch_size, training):

        self.data_path = data_path
        self.ts = window_size
        self.bs = batch_size
        self.training = training
        self.n_batch = 500
        self.pip = 10000
        self.position = Positions.Nothing
        self.start_tick = 0
        self.end_tick = self.bs - 1
        self.trade_fee = 1.25

        # episode
        self.done = False
        self.current_tick = self.start_tick
        self.last_trade_tick = None
        self.total_reward = 0.
        self.total_reward_long = 0.
        self.total_reward_short = 0.
        self.n_long = 0
        self.n_short = 0
        self.n_hold = 0
        self.batch_duration = 500
        self.iter = 0
        
        # training files
        self.dfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) \
                       and f[:3] == 'day']
        self.n_days = len(self.dfiles)
        
        if self.training:
            self.days = np.random.choice(range(self.n_days), 
                                         size=self.n_batch, 
                                         replace=False)
        else:
            self.days = range(self.n_days)
            
        self.all_data = np.array((), dtype=object)
        for day in self.days:
            self.all_data = np.append(self.all_data, 
                                      np.load(self.data_path + self.dfiles[day],
                                              allow_pickle=True))
        
        if self.training:
            self.all_data = self.all_data.reshape(self.n_batch, 5)
            #
            # wähle aus den geladenen Batches einen zufällig aus
            self.batch_id = random.randint(0, self.n_batch - 1)
        else:
             self.all_data = self.all_data.reshape(self.n_days, 5)
             self.batch_id = -1
                  
        self.mean_cp = 0.0
        self.std_cp = 1.0
        
        self.cnt_calculate_current_profit = 0
        self.sum_current_profit = 0.0
        self.sum2_current_profit = 0.0
        
        # spaces
        self.ff = self.all_data[0][3].shape[2]
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(self.ts*self.ff,), 
                                            dtype=np.float32)



    def reset(self):

        self.iter += 1
        if self.training:
            if (self.iter%self.batch_duration == 0):
                #
                # lade zufällig ausgewählte neue Batches
                del self.all_data
                gc.collect()
                self.days = np.random.choice(range(self.n_days), 
                                             size=self.n_batch, 
                                             replace=False)
                
                self.all_data = np.array((), dtype=object)
                for day in self.days:
                    self.all_data = np.append(self.all_data, 
                                              np.load(self.data_path + self.dfiles[day],
                                                      allow_pickle=True))    
                self.all_data = self.all_data.reshape(self.n_batch, 5)
                print('ForexEnv: neue Daten geladen bei Iteration', self.iter)
            #
            # wähle zufällig eine batch_id 
            self.batch_id = random.randint(0, self.n_batch - 1)
        else:
            self.batch_id += 1
            if (self.batch_id >= self.n_days):
                print('Alle Daten verarbeitet. Ende.')
                exit()
                
        self.position = Positions.Nothing
        self.prices, self.rewards, self.dttimes, self.feature_values = self.prepare_data()
        self.done = False
        self.end_tick = self.bs - 1
        self.current_tick = self.start_tick
        self.last_trade_tick = None
        self.total_reward = 0.
        self.total_reward_long = 0.
        self.total_reward_short = 0.
        self.n_long = 0
        self.n_short = 0
        self.n_hold = 0
        self.state = self.feature_values[self.start_tick]
        self.state = self.get_state()
        
        info = dict(
            datetime = self.dttimes[self.current_tick][0][:16],
            batch_id = self.all_data[self.batch_id][0],
            i = self.current_tick,
            last_trade = self.last_trade_tick,
            materialized_reward = 0.0,
            total_reward = self.total_reward,
            total_reward_long = self.total_reward_long,
            total_reward_short = self.total_reward_short,
            n_long_trades = self.n_long,
            n_short_trades = self.n_short,
            n_hold_trades = self.n_hold,
            position = self.position,
            action = Actions.Hold
        )

        return (self.state, info)


    def step(self, action): 
        assert action in [0, 1, 2]

        if (self.current_tick == self.end_tick):
            self.done = True
            action = Actions.Hold.value
            
        reward = self.calculate_reward(action)
        
        mat_reward, mr_long, mr_short, long, short, hold = self.materialized_reward(action)
        self.total_reward += mat_reward
        self.total_reward_long += mr_long
        self.total_reward_short += mr_short
        self.n_long += long
        self.n_short += short
        self.n_hold += hold
        info = dict(
            datetime = self.dttimes[self.current_tick][0][:16],
            batch_id = self.all_data[self.batch_id][0],
            i = self.current_tick,
            last_trade = self.last_trade_tick,
            materialized_reward = mat_reward,
            total_reward = self.total_reward,
            total_reward_long = self.total_reward_long,
            total_reward_short = self.total_reward_short,
            n_long_trades = self.n_long,
            n_short_trades = self.n_short,
            n_hold_trades = self.n_hold,
            position = self.position,
            action = action
        )

        if (action == Actions.Buy.value and self.position in [Positions.Nothing, Positions.Short]):
            self.position = Positions.Long
            self.last_trade_tick = self.current_tick
        elif (action == Actions.Sell.value and self.position in [Positions.Nothing, Positions.Long]):
            self.position = Positions.Short
            self.last_trade_tick = self.current_tick
        elif (action == Actions.Hold.value and self.position in [Positions.Long, Positions.Short]):
            self.position = Positions.Nothing
            self.last_trade_tick = None
            
        if not self.done:
            self.current_tick += 1
            
        new_state = self.get_state()
        
        return new_state, reward, self.done, False, info



    def get_state(self):
        #
        # zwischenspeichern von simulierter Position und Current Profit
        self.state = self.state.reshape(self.ts, self.ff)
        position_currProfit = self.state[1:, :2]
        self.state = self.feature_values[self.current_tick]
        #
        # überschreiben der simulierten Position und Current Profit
        self.state[:-1, :2] = position_currProfit
        self.state[-1, 0] = self.position.value
        self.state[-1, 1] = self.calculate_current_profit()

        return np.float32(self.state).reshape(self.ts*self.ff)



    def render(self, mode='human'):

        pass


    def prepare_data(self):      
        
        dttimes = self.all_data[self.batch_id][1]
        prices = self.all_data[self.batch_id][2].reshape(self.bs)
        rewards = self.pip*np.diff(prices, append=prices[-1])/prices
        feature_values = self.all_data[self.batch_id][3]

        return prices, rewards, dttimes, feature_values



    def calculate_reward(self, action):
        assert action in [0, 1, 2]

        r = 0.0
        if (self.position == Positions.Nothing and action == Actions.Buy.value):
        # reward for opening a Long position
            r = self.rewards[self.current_tick] - self.trade_fee
        elif (self.position == Positions.Nothing and action == Actions.Sell.value):
        # reward for opening a short position
            r = -self.rewards[self.current_tick] - self.trade_fee
        elif (self.position == Positions.Long and action == Actions.Hold.value):
        # reward for closing a long position
            r = 0.0
        elif (self.position == Positions.Short and action == Actions.Hold.value):
        # reward for closing a short position
            r = 0.0
        elif (self.position == Positions.Long and action == Actions.Buy.value):
        # reward for holding a long position - no trading fees
            r = self.rewards[self.current_tick]
        elif (self.position == Positions.Short and action == Actions.Sell.value):
       # reward for holding a short position
            r = -self.rewards[self.current_tick]
        elif (self.position == Positions.Long and action == Actions.Sell.value):
        # reward for closing a long position and opening a short one
            r = -self.rewards[self.current_tick] - self.trade_fee
        elif (self.position == Positions.Short and action == Actions.Buy.value):
        # reward for closing a short position and opening a long one
            r = self.rewards[self.current_tick] - self.trade_fee
        return r
    
    
    
    def calculate_current_profit(self):
        #
        # calculates the profit until the current state
        #
        if (self.position == Positions.Long):
            currProfit = self.pip * (self.prices[self.current_tick] \
                                          - self.prices[self.last_trade_tick]) \
                                     / self.prices[self.last_trade_tick] - self.trade_fee
        elif (self.position == Positions.Short):
            currProfit = -self.pip * (self.prices[self.current_tick] \
                                          - self.prices[self.last_trade_tick]) \
                                    / self.prices[self.last_trade_tick] - self.trade_fee
        else:
            currProfit = 0.0

        return currProfit
        
        
    
    def materialized_reward(self, action):
        assert action in [0, 1, 2]

        r = 0.
        r_long = 0.
        r_short = 0.
        long = 0
        short = 0
        hold = 0
        if (self.position == Positions.Long and action in (Actions.Hold.value, Actions.Sell.value)):
            r = self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
            r_long = r
            long = 1
        elif (self.position == Positions.Short and action in (Actions.Hold.value, Actions.Buy.value)):
            r = -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
            r_short = r
            short = 1
        elif (self.position == Positions.Nothing and action in (Actions.Buy.value, Actions.Sell.value)):
            r = 0.
            hold = 1
        return r, r_long, r_short, long, short, hold
    

