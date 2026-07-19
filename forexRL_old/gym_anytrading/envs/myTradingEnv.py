import gym
from gym import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2
    Close = 3


class Positions(Enum):
    Nothing = 0
    Long = 1
    Short = 2


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound):

        self.seed()
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self._position = None
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self.trade_fee = 0.00001

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._current_trade_tick = None
        self._last_trade_tick = None
        self._position_history = None
        self._total_reward = None
        self._first_rendering = None
        self.history = None


    def reset(self, frame_bound):

        self.frame_bound = frame_bound
        self._position = Positions.Nothing
        self.prices, self.signal_features = self._process_data()
        self._done = False
        self._current_tick = self._start_tick
        self._current_trade_tick = 0
        self._last_trade_tick = self._current_tick - 1
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._first_rendering = True
        self.history = {}
        return self._get_observation()


    def step(self, action):
        self._done = False
        self._current_tick += 1
        
        if (self._current_tick == self._end_tick):
            self._done = True
            
        if (self._position in [Positions.Long, Positions.Short]):
            self._current_trade_tick += 1
                                
        if (action == Actions.Close.value):
            self._current_trade_tick = 0

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        if (action == Actions.Buy.value and self._position == Positions.Nothing):
            self._position = Positions.Long
            self._last_trade_tick = self._current_tick
        elif (action == Actions.Sell.value and self._position == Positions.Nothing):
            self._position = Positions.Short
            self._last_trade_tick = self._current_tick
        elif (action == Actions.Close.value and self._position in [Positions.Long, Positions.Short]):
            self._position = Positions.Nothing
            
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info



    def _get_observation(self):

        observation = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
        observation[:,0] = np.full((self.window_size), self._position.value)
        return observation


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward)

        plt.pause(0.01)



    def _process_data(self):
        
        prices = self.df.loc[:, 'Close'].to_numpy()
        diff = self.df.loc[:, 'WL'].to_numpy()
        macd = self.df.loc[:, 'macd'].to_numpy()
        macd_d = self.df.loc[:, 'macd_d'].to_numpy()
        macd_H = self.df.loc[:, 'macd_1H'].to_numpy()
        macd_d_H = self.df.loc[:, 'macd_d_1H'].to_numpy()
        macd_D = self.df.loc[:, 'macd_1D'].to_numpy()
        macd_d_D = self.df.loc[:, 'macd_d_1D'].to_numpy()
        rsi_D = self.df.loc[:, 'rsi_1D'].to_numpy()

        ts = self.window_size
        bs = self.frame_bound[1]
        iteration = self.frame_bound[0]
        iStart = iteration*bs
        iEnd = (iteration+1)*bs + ts
        prices = prices[iStart:iEnd]
        position = np.full((bs+ts), self._position)
        diff = diff[iStart:iEnd]
        macd = macd[iStart:iEnd]
        macd_d = macd_d[iStart:iEnd]
        macd_H = macd_H[iStart:iEnd]
        macd_d_H = macd_d_H[iStart:iEnd]
        macd_D = macd_D[iStart:iEnd]
        macd_d_D = macd_d_D[iStart:iEnd]
        rsi_D = rsi_D[iStart:iEnd]

        state = np.column_stack((position, diff, macd, macd_d, macd_H, macd_d_H, macd_D, macd_d_D, rsi_D))
        return prices, state


    def _calculate_reward(self, action):

        step_reward = 0  # pip

        close_position = False
        if ((action == Actions.Close.value or self._done) and (self._position in [Positions.Long, Positions.Short])):
            close_position = True

        if close_position:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Short:
                price_diff = current_price - last_trade_price + self.trade_fee
                step_reward += -price_diff * 10000
            elif self._position == Positions.Long:
                price_diff = current_price - last_trade_price - self.trade_fee
                step_reward += price_diff * 10000

        return step_reward


