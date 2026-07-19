import gym
from gym import spaces
from gym.utils import seeding
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

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._current_trade_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._first_rendering = None
        self.history = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._current_trade_tick = 0
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Nothing
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
            
        if (self._done and self._position in [Positions.Long, Positions.Short]):
            action = Actions.Close.value

        if (self._position in [Positions.Long, Positions.Short]):
            self._current_trade_tick += 1
                    
        if (self._position == Positions.Nothing and action == Actions.Close.value):
            action = Actions.Hold.value
        
        if ((self._position in [Positions.Long, Positions.Short]) and (action in [Actions.Buy.value, Actions.Sell.value])):
            action = Actions.Hold.value
            
        if (action == Actions.Close.value):
            self._current_trade_tick = 0

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        if (action == Actions.Buy.value):
            self._position = Positions.Long
            self._last_trade_tick = self._current_tick
        elif (action == Actions.Sell.value):
            self._position = Positions.Short
            self._last_trade_tick = self._current_tick
        elif (action == Actions.Close.value):
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
        return self.signal_features[(self._current_tick-self.window_size):self._current_tick]


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


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'r.')
        plt.plot(long_ticks, self.prices[long_ticks], 'g.')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward)
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError

