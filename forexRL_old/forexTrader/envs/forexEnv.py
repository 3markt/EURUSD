import gym
from gym import spaces
import numpy as np
from enum import Enum
import random


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

    def __init__(self, data, window_size, batch_size, n_batch):

        self.all_data = data
        self.ts = window_size
        self.bs = batch_size
        self.n_batch = n_batch
        self.pip = 10000
        self.position = Positions.Nothing
        self.batch_id = random.randint(0, self.n_batch -1)
        self.start_tick = 0
        self.end_tick = self.bs - 1
        self.trade_fee = 1.5
        self.gamma = 0.90

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.ts, self.all_data[0,3].shape[2]), dtype=np.float32)

        # episode
        self.done = False
        self.current_tick = self.start_tick
        self.last_trade_tick = None
        self.total_reward = 0.
        self.total_reward_long = 0.
        self.total_reward_short = 0.
        self.n_long = 0
        self.n_short = 0


    def reset(self):

        self.batch_id = random.randint(0, self.n_batch -1)
        self.position = Positions.Nothing
        self.prices, self.diffs, self.dttimes, self.feature_values = self.prepare_data()
        self.done = False
        self.end_tick = self.bs - 1
        self.current_tick = self.start_tick
        self.last_trade_tick = None
        self.total_reward = 0.
        self.total_reward_long = 0.
        self.total_reward_short = 0.
        self.n_long = 0
        self.n_short = 0

        return self.get_state()


    def step(self, action): 
        assert action in [0, 1, 2]

        if (self.current_tick == self.end_tick):
            self.done = True
            action = Actions.Hold.value
            
        reward = self.calculate_reward(action)
        
        mat_reward, mr_long, mr_short, long, short = self.materialized_reward(action)
        self.total_reward += mat_reward
        self.total_reward_long += mr_long
        self.total_reward_short += mr_short
        self.n_long += long
        self.n_short += short
        info = dict(
            datetime = self.dttimes[self.current_tick][0][:16],
            batch_id = self.batch_id,
            i = self.current_tick,
            last_trade = self.last_trade_tick,
            materialized_reward = mat_reward,
            total_reward = self.total_reward,
            total_reward_long = self.total_reward_long,
            total_reward_short = self.total_reward_short,
            n_long_trades = self.n_long,
            n_short_trades = self.n_short,
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
        self.current_tick += 1
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
        diffs = np.diff(prices, append=prices[-1])*self.pip
        feature_values = self.all_data[self.batch_id][3]

        return prices, diffs, dttimes, feature_values


    def calculate_reward(self, action):
        assert action in [0, 1, 2]

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
    
    
    def materialized_reward(self, action):
        assert action in [0, 1, 2]

        r = 0.
        r_long = 0.
        r_short = 0.
        long = 0
        short = 0
        if (self.position == Positions.Long and action in (Actions.Hold.value, Actions.Sell.value)):
            r = self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
            r_long = r
            long = 1
        elif (self.position == Positions.Short and action in (Actions.Hold.value, Actions.Buy.value)):
            r = -self.pip * (self.prices[self.current_tick] - self.prices[self.last_trade_tick]) - self.trade_fee
            r_short = r
            short = 1
        return r, r_long, r_short, long, short
    

