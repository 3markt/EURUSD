import numpy as np

from .trading_env import TradingEnv, Actions, Positions


class ForexEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.trade_fee = 0.00001  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        diff = self.df.loc[:, 'WL'].to_numpy()
        macd = self.df.loc[:, 'macd'].to_numpy()
        macd_d = self.df.loc[:, 'macd_d'].to_numpy()

        ts = self.window_size
        bs = self.frame_bound[1]
        iteration = self.frame_bound[0]
        iStart = iteration*bs
        iEnd = (iteration+1)*bs + ts
        prices = prices[iStart:iEnd]
        diff = diff[iStart:iEnd]
        macd = macd[iStart:iEnd]
        macd_d = macd_d[iStart:iEnd]

        signal_features = np.column_stack((diff, macd, macd_d))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0  # pip

        close_position = False
        if ((action == Actions.Close.value) and (self._position in [Positions.Long, Positions.Short])):
            close_position = True

        if close_position:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff * 10000
            elif self._position == Positions.Long:
                step_reward += price_diff * 10000

        return step_reward

