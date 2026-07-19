import gym
from gym import spaces
import numpy as np
from enum import Enum


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Nothing = 0.0
    Long = 0.5
    Short = 1.0


class ForexEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, data, feature_list, window_size, batch_size, n_batch):

        self.all_data = data
        self.feature_list = feature_list
        self.ts = window_size
        self.bs = batch_size
        self.n_batch = n_batch
        self.pip = 10000
        self.position = Positions.Nothing
        self.batch_id = 0
        self.start_tick = 0
        self.end_tick = self.bs - 1
        self.prices, self.diffs, self.dttimes, self.feature_values = self.prepare_data()
        self.shape = (self.ts, self.feature_values.shape[1])
        self.trade_fee = 1.0
        self.gamma = 0.95

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self.done = False
        self.current_tick = self.start_tick
        self.last_trade_tick = None
        self.history = None
        self.total_reward = 0.


    def reset(self):

        if (self.batch_id < self.n_batch):
            self.batch_id += 1
        else:
            self.batch_id = 0
            
        self.position = Positions.Nothing
        self.prices, self.diffs, self.dttimes, self.feature_values = self.prepare_data()
        self.done = False
        self.end_tick = self.bs - 1
        self.current_tick = self.start_tick
        self.last_trade_tick = None
        self.total_reward = 0.
        self.history = {}

        return self.get_state()


    def step(self, action):

        self.current_tick += 1
        
        if (self.current_tick == self.end_tick):
            self.done = True
            action = Actions.Hold.value
            
        reward = self.calculate_reward(action)
        
        mat_reward = self.materialized_reward(action)
        self.total_reward += mat_reward
        info = dict(
            datetime = self.dttimes[self.current_tick][0][:16],
            batch_id = self.batch_id,
            i = self.current_tick,
            last_trade = self.last_trade_tick,
            materialized_reward = mat_reward,
            total_reward = self.total_reward,
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
            
        new_state = self.get_state()

        return new_state, reward, self.done, info


    def get_state(self):

        state = self.feature_values[self.current_tick]
        state[-1, 0] = self.position.value

        return state


    def render(self, mode='human'):

        pass


    def prepare_data(self):
        
        assert self.batch_id == self.all_data[self.batch_id][0]
        
        dttimes = self.all_data[self.batch_id][1]
        prices = self.all_data[self.batch_id][2].reshape(self.bs)
        diffs = np.diff(prices, prepend=prices[0])*self.pip
        feature_values = self.all_data[self.batch_id][3]

        return prices, diffs, dttimes, feature_values


    def calculate_reward(self, action):

        r = 0
        if (self.position == Positions.Nothing and action == Actions.Buy.value):
        # reward for opening a Long position
            for j in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma * (r + self.diffs[j])
            r += self.diffs[self.current_tick] - self.trade_fee                
        elif (self.position == Positions.Nothing and action == Actions.Sell.value):
        # reward for opening a short position
            for j in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma * (r - self.diffs[j])
            r -= self.diffs[self.current_tick] + self.trade_fee
        elif (self.position == Positions.Long and action == Actions.Hold.value):
        # reward for closing a long position
            r = self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
        elif (self.position == Positions.Long and action == Actions.Buy.value):
        # reward for holding a long position
            # discounted future reward
            for j in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r + self.diffs[j])
            # reward from the past
            r += self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            # current reward minus trade fee
            r += self.diffs[self.current_tick] - self.trade_fee
        elif (self.position == Positions.Long and action == Actions.Sell.value):
        # reward for closing a long position and opening a short one
            r1 = self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            for j in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r - self.diffs[j])
            r -= self.diffs[self.current_tick]
            r = r + r1 - self.trade_fee
        elif (self.position == Positions.Short and action == Actions.Hold.value):
        # reward for closing a short position
            r = -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
        elif (self.position == Positions.Short and action == Actions.Buy.value):
        # reward for closing a short position and opening a long one
            r1 = -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            for j in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r + self.diffs[j])
            r += self.diffs[self.current_tick]
            r = r + r1 - self.trade_fee
        elif (self.position == Positions.Short and action == Actions.Sell.value):
        # reward for holding a short position
            # discounted future rewards
            for j in range(self.bs - 1, self.current_tick, -1):
                r = self.gamma*(r - self.diffs[j])
            # reward from the past
            r -= self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick])
            # current reward minus trade fee
            r -= self.diffs[self.current_tick] + self.trade_fee
            
        """
        print('Index=%03i LTT=%03i Reward=%08.3f Kurs=%1.5f Diff=%07.3f SumDiff=%08.3f Pos=%1.1f Action=%1i' %\
              (self.current_tick, 
               self.last_trade_tick,
               r, 
               self.prices[self.current_tick], 
               self.diffs[self.current_tick], 
               sum(self.diffs[self.current_tick:]),
               self.position.value, 
               action))
        """
        return r
    
    
    def materialized_reward(self, action):
        
        r = 0
        if (self.position == Positions.Long and action in (Actions.Hold.value, Actions.Sell.value)):
            r = self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
        elif (self.position == Positions.Short and action in (Actions.Hold.value, Actions.Buy.value)):
            r = -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
        
        return r
    

