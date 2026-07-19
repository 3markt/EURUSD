#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:10:14 2021

@author: uwe.mueller
"""

import gym
import numpy as np
import random
import time
import sys
from os import listdir, makedirs
from os.path import isfile, join, isdir
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l1, l2
from keras.optimizers import Adam, RMSprop, Nadam



class ForexDayTrader:
    
    def __init__(self, env, model_path, model_weights):
        
        #basepath = '/Users/uwe.muller/Hope/'
        basepath = '/home/chitlom/'
        self.env     = env
        self.model_init_path = basepath + 'model/eurusd-ongoing/Professional/'
        self.model_path = basepath + 'model/eurusd-ongoing/'
        self.model_weights = 'Professional'
        self.data_path = basepath + 'data/rl/'
        self.gamma = 0.9
        self.epsilon = 0.9
        self.min_epsilon = 0.05
        self.learning_rate = 0.0005
        self.min_lr = 0.0002
        self.dreaming_rate = 0.75
        self.min_dr = 0.1

        self.bs = 192
        self.ts = 60
        self.ff = 24
        self.model = self.create_model(self.bs)
        self.decision_model = self.create_model(1)
        self.target_model = self.create_model(self.bs)
        
        self.train_size = 100
        self.training_cnt = 0
        self.max_memory_size = 5000
        
        # initialize structures required for memory
        self.memory = []
        self.replay_memory = []
        self.x = np.array([])
        self.y = np.array([])
        self.datetime = ''
        self.batch_id = 0
        self.total_reward = 0.0
        self.daily_state = np.array([])
        self.daily_action = np.array([]).astype(int)
        self.daily_reward = np.array([])



    def create_model(self, bs):

        model = Sequential()
                
        model.add(LSTM(units=120, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       recurrent_regularizer=l2(0.01),
                       kernel_initializer='truncated_normal',
                       batch_input_shape=(bs, 
                                          self.ts, 
                                          self.ff)))
                
        model.add(Dense(units=3, 
                        kernel_initializer='truncated_normal'))
            
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['mae'])
        
        model.load_weights(self.model_init_path + self.model_weights)
        
        print('weights loaded from ' + self.model_init_path + self.model_weights)
        
        model.summary()
        sys.stdout.flush()
        
        return model


    def act(self, state):

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.decision_model.predict(state, batch_size=1)[0])


    def remember(self, state, action, reward, new_state, done):

        self.daily_state = np.append(self.daily_state, state)
        self.daily_action = np.append(self.daily_action, action)
        self.daily_reward = np.append(self.daily_reward, reward)


    def day_dream(self, dt, batch_id, total_reward):
        
        self.replay_memory.append([dt,
                                   batch_id,
                                   total_reward,
                                   self.daily_action,
                                   self.daily_state,
                                   self.daily_reward])
        
        self.daily_state = np.array([])
        self.daily_action = np.array([]).astype(int)
        self.daily_reward = np.array([])


    def dream(self, take_best):
        
        # take_best must be smaller or equal available replay-memory
        assert take_best <= len(self.replay_memory)
        assert take_best > 0
        
        # sort replay-memory according to best total reward
        self.replay_memory.sort(key=(lambda x: x[2]), reverse=True)
                
        for element in self.replay_memory[:take_best]:
            # loop over the take_best best elements - only dream about these
            dt, batch_id, total_reward, action, state, reward = element
            state = state.reshape(self.bs, self.ts, self.ff)
            
            # calculate new target values according to Bellmann-equation
            target = self.target_model.predict(state, batch_size=self.bs)
            idx_maxTarget = np.argmax(target, axis=1)
            # take the max rewards as future values
            Q_future = target[range(target.shape[0]), idx_maxTarget]
            # future reward is discounted and shifted by one
            Q_future = self.gamma*Q_future[1:]
            # get old values
            old_target = target[range(target.shape[0]), action]
            # calculate modification part
            target_mod = reward[:-1] + Q_future - old_target[:-1]
            # for the done item we just use the reward
            done_mod = reward[-1] - old_target[-1]
            # add the done item at the end
            target_mod = np.append(target_mod, done_mod)
            # calculate new target values
            # dreaming rate == learning rate of Bellmann equation
            target[range(target.shape[0]), action] = old_target + \
                self.dreaming_rate * target_mod
    
            # remember everything
            self.memory.append([dt, 
                                batch_id, 
                                total_reward,
                                action,
                                state, 
                                target.reshape(self.bs, 3)])
        
        # clean daily replay memory
        self.replay_memory = []


    def replay(self):

        if len(self.memory) < self.train_size: 
            return

        n = len(self.memory)
        samples = random.sample(self.memory, self.train_size)
        xx = np.array([])
        yy = np.array([])
        for sample in samples:
            dt, batch_id, total_reward, action, x, y  = sample
            xx = np.append(xx, x)
            yy = np.append(yy, y)

        ffit = self.model.fit(x, y, 
                              epochs=self.train_size, 
                              batch_size=self.bs, 
                              verbose=0,
                              shuffle=False)
        self.training_cnt += self.train_size
                        
        if (self.training_cnt % 1000 == 0):    
            loss = np.mean(ffit.history['loss'])
            mae = np.mean(ffit.history['mae'])
            print('Fit: count=%5i loss=%2.3f mae=%2.3f' %
                  (self.training_cnt, loss, mae))
            print('RL: learn=%1.5f dream=%1.2f explore=%1.3f' %
                  (self.learning_rate, self.dreaming_rate, self.epsilon))
            sys.stdout.flush()
        
        # pass-on re-trained weights to decision-model
        weights = self.model.get_weights()
        self.decision_model.set_weights(weights)
        
        if (n >= self.max_memory_size):
            self.memory = self.memory[int(self.max_memory_size/2):]


    def train_target(self):
        
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)


    def save_model(self, name):
        
        # check model directory exists
        if (not isdir(self.model_path + name + '/')):
            makedirs(self.model_path + name + '/')
        
        self.model.save_weights(self.model_path + name + '/' + name)
        
        
    def set_rates(self, learning_rate, dreaming_rate, exploration_rate):
        
        self.learning_rate = learning_rate
        self.dreaming_rate = dreaming_rate
        self.epsilon = exploration_rate
        
        
        
