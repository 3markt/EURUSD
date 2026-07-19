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
from keras.optimizers import Adam, RMSprop, Nadam, SGD
import tensorflow as tf



class ForexDayTrader:
    
    def __init__(self, env, basepath, weight_name, units, bs, ts, ff):
        
        self.env     = env
        self.init_model = basepath + 'model/' + weight_name + '/' + weight_name
        self.model_path = basepath + 'model/'
        self.data_path = basepath + 'RL-data/'
        self.longterm_path = basepath + 'longterm_memory/'
        self.longterm_index = 0
        self.longterm_index_file = self.longterm_path + 'index.npy'
        self.ltmem = []
        self.longterm_memory = []
        if (isfile(self.longterm_index_file)):
            self.longterm_index = np.load(self.longterm_index_file)

        self.gamma = 0.9
        self.epsilon = 0.9
        self.min_epsilon = 0.05
        self.learning_rate = 0.0005
        self.min_lr = 0.0002
        self.dreaming_rate = 0.75
        self.min_dr = 0.1

        self.bs = bs
        self.ts = ts
        self.ff = ff
        self.units = units
        self.model = self.create_model(self.bs)
        self.decision_model = self.create_model(1)
        self.target_model = self.create_model(self.bs)
        
        self.train_size = 10
        self.training_cnt = 0
        self.max_memory_size = 1000
        self.min_memory_size = 800
        self.max_longterm_mem = 1000
        self.sample_size = 50
        
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
                
        model.add(LSTM(units=self.units, 
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
            
        model.compile(loss=tf.keras.losses.Huber(), 
                      optimizer=SGD(),
                      metrics=['mae'])
       
        model.load_weights(self.init_model)
        
        print('weights loaded from ' + self.init_model)
        
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



    def dream(self):
        
        for element in self.replay_memory:
            # loop over the replay_memory
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
        
            self.longterm_memory.append([dt, 
                                         batch_id, 
                                         total_reward,
                                         action,
                                         state, 
                                         target.reshape(self.bs, 3)])
        # clean daily replay memory
        self.replay_memory = []



    def replay(self, iteration):

        if len(self.memory) < self.min_memory_size: 
            return

        loss = 0.0
        mae = 0.0
        samples = random.sample(self.memory + self.ltmem, self.train_size)
        for sample in samples:
            self.training_cnt += 1
            dt, batch_id, total_reward, action, x, y  = sample
            ffit = self.model.fit(x, y, 
                                  epochs=1, 
                                  batch_size=self.bs, 
                                  verbose=0,
                                  shuffle=False)
            
            loss += ffit.history['loss'][0]
            mae += ffit.history['mae'][0]
            
        if (self.training_cnt % 1000 == 0):    
            print('Fit: iteration=%6i count=%5i loss=%2.3f mae=%2.3f' %
                  (iteration+1, self.training_cnt, loss/self.train_size, mae/self.train_size))
            sys.stdout.flush()
        
        # pass-on re-trained weights to decision-model
        weights = self.model.get_weights()
        self.decision_model.set_weights(weights)

        n = len(self.memory)
        nl = len(self.longterm_memory)
        if (iteration <= 100000):
            num_lt_sample_files = 0
        elif (iteration <= 200000):
            num_lt_sample_files = 1
        elif (iteration <= 300000):
            num_lt_sample_files = 2
        elif (iteration <= 400000):
            num_lt_sample_files = 3
        else:
            num_lt_sample_files = 4
            
        if (n >= self.max_memory_size):
            if (self.longterm_index >= num_lt_sample_files):
                ltind = random.sample(range(self.longterm_index), num_lt_sample_files)
                self.ltmem = []
                for l in ltind:
                    self.ltmem += np.load(self.longterm_path + 'ltmem' + str(l) + '.npy',
                                          allow_pickle=True).tolist()
            self.memory = self.memory[n-self.min_memory_size:]
            
        if (nl >= self.max_longterm_mem):
             np.save(self.longterm_path + 'ltmem' + str(self.longterm_index) + '.npy', 
                    np.array(random.sample(self.longterm_memory, self.sample_size), dtype=object),
                    allow_pickle=True)
             self.longterm_memory = []
             self.longterm_index += 1
             np.save(self.longterm_index_file, self.longterm_index)
           


    def train_target(self):
        
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)



    def save_model(self, name):
        
        # check model directory exists
        if (not isdir(self.model_path + name + '/')):
            makedirs(self.model_path + name + '/')
        
        self.model.save_weights(self.model_path + name + '/' + name)
     
        
        
    def set_rates(self, dreaming_rate, exploration_rate):
        
        self.dreaming_rate = dreaming_rate
        self.epsilon = exploration_rate
        
        
        
