#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:10:14 2021

@author: uwe.mueller
"""

import gym
# import gymnasium as gym
import numpy as np
import random
import time
import sys
import gc
from os import listdir, makedirs
from os.path import isfile, join, isdir

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data



class myLSTM(nn.Module):
    
    def __init__(self, inputsize):
        
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=inputsize, 
                            hidden_size=48, 
                            num_layers=1, 
                            batch_first=True)
        
        self.linear = nn.Linear(48, 3)
        #self.double()
        torch.set_default_dtype(torch.float32)
        
        
        
    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x = self.linear(x)
        
        return x



class ForexDayTrader:
    
    def __init__(self, env, basepath, weight_name, bs, ts, ff):
        
        self.env     = env
        self.model_path = basepath + 'model/'
        self.init_model = self.model_path + weight_name + '/' + weight_name \
                            + '.pt'
        self.data_path = basepath + 'RL-data/'
        self.longterm_path = basepath + 'longterm_memory/'
        self.td_error_path = basepath + 'td_error/'
        self.longterm_index = 0
        self.longterm_index_file = self.longterm_path + 'index.npy'
        self.ltmem = []
        self.longterm_memory = []
        if (isfile(self.longterm_index_file)):
            self.longterm_index = np.load(self.longterm_index_file)

        self.gamma = 0.975
        self.epsilon = 0.9
        self.learning_rate = 0.1
        self.dreaming_rate = 0.5

        self.bs = bs
        self.ts = ts
        self.ff = ff
        # instance of the model which will be continously trained
        model, opt, loss = self.create_model(self.init_model)
        self.model = model
        self.opt_m = opt
        self.loss_m = loss
        # instance of the decision model
        model, opt, loss = self.create_model(self.init_model)
        self.decision_model = model
        self.opt_dm = opt
        self.loss_dm = loss
        # instance of the target model
        model, opt, loss = self.create_model(self.init_model)
        self.target_model = model
        self.opt_tar = opt
        self.loss_tar = loss
        
        self.train_size = 10
        self.training_cnt = 0
        self.max_memory_size = 500
        self.min_memory_size = 300
        self.max_longterm_mem = 1000
        self.sample_size = 20
        
        # initialize structures required for memory
        self.memory = []
        self.x = np.array([])
        self.y = np.array([])
        self.datetime = ''
        self.batch_id = 0
        self.total_reward = 0.0
        self.daily_state = np.array([])
        self.daily_action = np.array([]).astype(int)
        self.daily_reward = np.array([])

        # memory for td-error analysis
        self.td_err_save_flag = True
        self.td_error_size = 1000
        self.td_error = []
        self.td_err_save_cnt = 0



    def create_model(self, name):

        model = myLSTM(self.ff)
        lr = self.learning_rate
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        #model.load_state_dict(torch.load(name, weights_only=True))
        model.load_state_dict(torch.load(name))
        print('weights loaded from ' + name) 
        sys.stdout.flush()
        
        return model, optimizer, loss_fn
  
 
   

    def act(self, state):

        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state_tt = torch.from_numpy(state)
            self.decision_model.eval()
            action_tt = self.decision_model(state_tt)
            action = np.argmax(action_tt[-1, -1, :].detach().numpy().reshape(1, 1, 3))
            
        return action




    def remember(self, state, action, reward):

        self.daily_state = np.append(self.daily_state, state)
        self.daily_action = np.append(self.daily_action, action)
        self.daily_reward = np.append(self.daily_reward, reward)



    def dream(self, dt, batch_id, total_reward):
        
        state = self.daily_state
        self.daily_state = np.array([])
        action = self.daily_action
        self.daily_action = np.array([]).astype(int)
        reward = self.daily_reward
        self.daily_reward = np.array([])
        
        state = np.float32(state.reshape(self.bs, self.ts, self.ff))
        state_tt = torch.from_numpy(state)
        
        # calculate new target values according to Bellmann-equation
        self.target_model.eval()
        target_tt = self.target_model(state_tt)
        target = target_tt[:,-1,:].detach().numpy().reshape(self.bs, 3)

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

        # td_error wird für die error analyse benötigt. Später werden wir in abhängigkeit der td_errors to
        # dreaming_rate festlegen
        if self.td_err_save_flag:
            self.td_error.append(target_mod)
            self.td_err_save_cnt += 1
            if (len(self.td_error) >= self.td_error_size):
                pd.DataFrame(self.td_error).to_csv(self.td_error_path \
                                    + 'td_error_' + str(self.td_err_save_cnt) + '.csv')
                self.td_error = []
                self.td_err_save_cnt = 0



    def y_reshape(self, yy):
        #
        # Methode bringt die y-Werte von der Form (bs, 3) auf die Form (bs, ts, 3)
        # mit den vergangenen Werten in der Zeitreihe
        #
        y = np.zeros((self.bs,self.ts,3))

        y[:,-1,:] = yy
        for i in range(1, self.ts):
            y[i:,-i-1,:] = yy[:-i]
            
        return y



    def replay(self, iteration):

        if len(self.memory) < self.min_memory_size: 
            return

        self.opt_m.param_groups[0]['lr'] = self.learning_rate

        samples = random.sample(self.memory + self.ltmem, self.train_size)
        running_loss = 0
        for sample in samples:
            self.training_cnt += 1
            dt, batch_id, total_reward, action, x, yy  = sample
            
            y = self.y_reshape(yy)
            
            x_tt = torch.tensor(np.float32(x)).reshape(self.bs, 
                                                       self.ts, 
                                                       self.ff)
            
            y_tt = torch.tensor(np.float32(y)).reshape(self.bs, 
                                                       self.ts, 
                                                       3)
            
            train_loader = data.DataLoader(data.TensorDataset(x_tt, y_tt), 
                                           shuffle=False, 
                                           batch_size=self.bs,
                                           drop_last=True)
            
            self.model.train()
            batch_loss = 0
            for X, Y in train_loader:
                Y_pred = self.model(X)
                loss = self.loss_m(Y_pred, Y)
                self.opt_m.zero_grad()
                loss.backward()
                self.opt_m.step()
                batch_loss += loss.item()
                
            running_loss += batch_loss
                
            
        if (self.training_cnt % 1000 == 0):    
            print('Fit: iteration=%6i count=%5i loss=%2.3f' %
                  (iteration+1, 
                   self.training_cnt, 
                   running_loss/self.train_size))
            sys.stdout.flush()
            running_loss = 0
        
        # pass-on re-trained weights to decision-model
        self.decision_model.load_state_dict(self.model.state_dict())

        n = len(self.memory)
        nl = len(self.longterm_memory)
        if (iteration <= 500000):
            num_lt_sample_files = 0
        else:
            num_lt_sample_files = 1
        
        #
        # Memory-management: keep historical sample memory and re-use it in 
        #                    current training in order to avoid the NN-forget-problem
        #
        if (n >= self.max_memory_size):
            if (self.longterm_index >= num_lt_sample_files):
                # load historical sample memory from long-term memory
                ltind = random.sample(range(self.longterm_index), num_lt_sample_files)
                del self.ltmem
                gc.collect()
                self.ltmem = []
                for l in ltind:
                    self.ltmem += np.load(self.longterm_path + 'ltmem' + str(l) + '.npy',
                                          allow_pickle=True).tolist()
            self.memory = self.memory[n-self.min_memory_size:]
            
        if (nl >= self.max_longterm_mem):
            # save historical sample memory as long-term memory
             np.save(self.longterm_path + 'ltmem' + str(self.longterm_index) + '.npy', 
                    np.array(random.sample(self.longterm_memory, 
                                           self.sample_size), 
                             dtype=object),
                    allow_pickle=True)
             del self.longterm_memory
             gc.collect()
             self.longterm_memory = []
             self.longterm_index += 1
             np.save(self.longterm_index_file, self.longterm_index)
           


    def train_target(self):
        
        self.target_model.load_state_dict(self.model.state_dict())



    def save_model(self, name):
        #
        # check model directory exists
        if (not isdir(self.model_path + name + '/')):
            makedirs(self.model_path + name + '/')
        
        npath = self.model_path + name + '/' + name + '.pt'
        torch.save(self.model.state_dict(), npath)
     
        

    def set_rates(self, dreaming_rate, exploration_rate, learning_rate):
        
        self.dreaming_rate = dreaming_rate
        self.epsilon = exploration_rate
        self.learning_rate = learning_rate
        
       
        
