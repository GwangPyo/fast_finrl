import gymnasium as gym
import matplotlib.pyplot as plt
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pandas as pd
from copy import deepcopy
from typing import Literal, Sequence, Tuple
import numpy as np
from collections import defaultdict



class StockTradingEnvExtension(StockTradingEnv):
    def __init__(
            self,
            df: pd.DataFrame,
            hmax: int = 15,
            initial_amount: int = 30_000,
            num_stock_shares: Sequence[int] | None = None,
            buy_cost_pct: Sequence[float] | float = 0.01,
            sell_cost_pct: Sequence[float] | float = 0.01,
            reward_scaling: float = 1e-3,
            tech_indicator_list: list[str] | Literal['auto'] = 'auto',
            turbulence_threshold=None,
            risk_indicator_col="turbulence",
            make_plots: bool = False,
            print_verbosity=10,
            day=0,
            initial=True,
            previous_state=(),
            model_name="",
            mode="",
            iteration="",
            randomize_day: bool = True,
            bidding: Literal['default', 'uniform', 'adv_uniform'] = 'default',
            stop_loss_tolerance: float = 0.8,  # 0.8 means short selling when the loss is over 20% => 80% price reach
            stop_loss_calculation: Literal['close', 'low'] = 'close',
            seed: int = 42
    ):
        df['day'] = df['date'].rank(method='dense', ascending=True).astype(int) - 1
        df.set_index(keys='day', inplace=True)

        self.np_rng = np.random.default_rng(seed)
        self.stop_loss_tolerance = stop_loss_tolerance
        self.tic_names = df.tic.unique()
        self.ticker_to_idx = { ticker: idx for idx, ticker in enumerate(self.tic_names) }

        tomorrow_open = df.loc[1:]['open'].values
        # pad
        last_day = (df.index.max())
        if len(df.tic.unique()) > 1:
            pad = df.loc[last_day]['close'].values
        else:
            print([df.loc[last_day]])
            pad = np.asarray([df.loc[last_day]['close']])

        df["start"] = np.concatenate([tomorrow_open, pad], axis=-1)

        df_columns = deepcopy(list(df.columns))

        # df_columns.remove('day')
        df_columns.remove('date')
        df_columns.remove('tic')
        df_columns.remove('close')
        if previous_state == ():
            previous_state = []

        if tech_indicator_list == 'auto':
            # dataframe other than close, high, low, volume
            # they are not available in real-world scenario when the trading is happening

            tech_indicator_list = ['macd',  # ONE DAY BEFORE
                                   'rsi_7', 'rsi_14',  # ONE DAY BEFORE
                                   'cci_7', 'cci_14',  # ONE DAY BEFORE
                                   'close_10_ema', 'close_5_ema',  # ONE DAY BEFORE
                                   'close_20_ema', 'close_50_ema', 'close_100_ema',  # ONE DAY BEFORE
                                   "boll", "boll_ub", "boll_lb",
                                   "atr", "atr_14",
                                   "adx", "adxr",
                                   "cci_7", "cci_14",
                                   "mfi",
                                   "pdi", "ndi",
                                   "close_10_roc",

                                   'log_prev_close',
                                   'log_prev_volume',
                                   'log_prev_high',
                                   'log_prev_low',
                                   ]

        # calculate sotck dimension, and set state and action space automatically
        stock_dim = len(df['tic'].unique())
        state_space = 1 + 3 * stock_dim + len(tech_indicator_list) * stock_dim


        action_space = stock_dim
        # casting cost
        if isinstance(buy_cost_pct, float):
            buy_cost_pct = [buy_cost_pct] * stock_dim
        else:
            buy_cost_pct = list(buy_cost_pct)
        if isinstance(sell_cost_pct, float):
            sell_cost_pct = [sell_cost_pct] * stock_dim
        else:
            sell_cost_pct = list(sell_cost_pct)
        if num_stock_shares is None:
            num_stock_shares = [0] * stock_dim
        self.randomize_day = randomize_day
        self.max_day = len(df.index.unique())
        self.bidding = bidding
        self.stop_loss_calculation = stop_loss_calculation
        super().__init__(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            initial_amount=initial_amount,
            num_stock_shares=num_stock_shares,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            state_space=state_space,
            action_space=action_space,
            tech_indicator_list=tech_indicator_list,
            turbulence_threshold=turbulence_threshold,
            risk_indicator_col=risk_indicator_col,
            make_plots=make_plots,
            print_verbosity=print_verbosity,
            day=day,
            initial=initial,
            previous_state=previous_state,
            model_name=model_name,
            mode=mode,
            iteration=iteration
        )


    def open_price_today(self):
        today = self.df.loc[self.day, :]
        open_price = today['open']
        return np.asarray(open_price.values).copy()

    def process_obs(self):
        state = np.asarray(self.state).copy()
        tech = state[2 * self.stock_dim + 1:]

        obs = np.concatenate(
            [[np.log1p(state[0] / self.initial_amount)],  # cash
             np.log(self.open_price_today()),
             self.share / 100 - 3,
             np.log1p(self.avg_buy_price.copy() / self.open_price_today()),
             tech, ], axis=0)  # NOTE: also technical indicators are from the previous day.
        return obs

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.initial_amount]
                        + self.data.close.values.tolist()
                        + self.num_stock_shares
                        + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                        [self.initial_amount]
                        + [self.data.close]
                        + [0] * self.stock_dim
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.previous_state[0]]
                        + self.data.close.values.tolist()
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
                )
            else:
                # for single stock
                state = (
                        [self.previous_state[0]]
                        + [self.data.close]
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return np.asarray(state, dtype=np.float64)

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ):
        self.num_stop_loss = 0
        self.avg_buy_price = np.zeros(self.stock_dim)

        state, info = super().reset(seed=seed, options=options)
        if self.randomize_day:
            self.day = self.np_rng.integers(0, int(self.max_day * 0.8))
            self.data = self.df.loc[self.day, :]
            self.state = self._initiate_state()
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1: 1 + self.stock_dim])
                )
            ]
            self.date_memory = [self._get_date()]

        return np.asarray(self.process_obs()).copy(), info

    @property
    def share(self) -> np.ndarray:  # amount of asset except cash
        return np.copy(np.asarray(self.state[(1 + self.stock_dim):(1 + 2 * self.stock_dim)]))

    @property
    def price(self) -> np.ndarray:  # current price of the assets
        return np.array(self.state[1: (self.stock_dim + 1)])

    @property
    def vec_asset(self) -> np.ndarray:  # cash + current value of each asset
        return np.concatenate([self.price * self.share, [self.state[0]]], axis=0).copy()

    @property
    def asset(self) -> float:  # total value of the current asset
        return self.vec_asset.sum().item()

    def low(self, tic):
        price_low = self.df.loc[self.day, :]
        price_low = price_low[price_low.tic == tic]['low']
        return np.asarray(price_low.values)

    def high(self, tic):
        price_high = self.df.loc[self.day, :]
        price_high = price_high[price_high.tic == tic]['high']
        return np.asarray(price_high.values)

    def open(self, tic):

        price_open = self.df.loc[self.day, :]
        price_open = price_open[price_open.tic == tic]['open']
        return np.asarray(price_open.values)

    def close(self, tic):
        price_close = self.df.loc[self.day, :]
        price_close = price_close[price_close.tic == tic]['close']
        return np.asarray(price_close.values)

    def olhc(self, tic):
        price = self.df.loc[self.day, :]
        tic_price = price[price.tic == tic]
        return (tic_price['open'].values, tic_price['low'].values,
                tic_price['high'].values, tic_price['close'].values)

    def randomized_price(self, index, randomize_option: Literal['default', 'uniform',
    'low_uniform', 'high_uniform']):
        tic = self.tic_names[index]
        open, low, high, close = self.olhc(tic)

        if randomize_option == 'default':
            return self.state[index + 1]

        elif randomize_option == 'uniform':

            return self.np_rng.uniform(low, high).item()
        elif randomize_option == 'low_uniform':

            maximum = np.min([open, close]).item()

            return self.np_rng.uniform(low, maximum).item()
        else:
            minimum = np.max([open, close]).item()

            return self.np_rng.uniform(minimum, high).item()

    def _sell_stock(self, index, action):
        options = { 'default': 'default', 'uniform': 'uniform', 'adv_uniform': 'low_uniform' }
        bids = options[self.bidding]
        def _do_sell_normal():
            if (
                    self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )

                    sell_price = self.randomized_price(index, bids)

                    sell_amount = (
                            sell_price
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                            sell_price
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]

                        sell_amount = (
                                self.state[index + 1]
                                * sell_num_shares
                                * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                                self.state[index + 1]
                                * sell_num_shares
                                * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:

            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        options = { 'default': 'default', 'uniform': 'uniform', 'adv_uniform': 'high_uniform' }
        bids = options[self.bidding]
        def _do_buy():
            if (
                    self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                buy_price = self.randomized_price(index, bids)

                available_amount = self.state[0] // (
                        buy_price * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = int(min(available_amount, action))
                buy_amount = (
                        buy_price
                        * buy_num_shares
                        * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                        buy_price * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def step(self, actions):

        self.terminal = (self.day + 1 >= len(self.df.index.unique()) - 1)
        n_trades = 0

        actions = actions * self.hmax  # actions initially is scaled between 0 to 1
        actions = actions.astype(
            int
        )  # convert into integer because we can't by fraction of shares
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)

        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        # print("begin_total_asset:{}".format(begin_total_asset))

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
        # share_before_selling = self.share

        for index in sell_index:
            # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
            # print(f'take sell action before : {actions[index]}')
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
            n_trades += np.abs(actions[index])
            # print(f'take sell action after : {actions[index]}')
            # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

        share_after_selling = self.share
        # reset all avg buy price if there is no more share
        self.avg_buy_price = np.where(share_after_selling == 0, 0, self.avg_buy_price)

        for index in buy_index:
            # previous
            cash_prev = self.state[0]
            prev_total_i = share_after_selling[index] * self.avg_buy_price[index]
            # actual buy amount
            actions[index] = self._buy_stock(index, actions[index])
            n_trades += np.abs(actions[index])

            # buy amount * price
            cash_now = self.state[0]
            spent = abs(cash_now - cash_prev)
            total_spent = prev_total_i + spent
            new_shares = actions[index] + share_after_selling[index]
            #  (action[index] == 0 & share_after_selling ==0) can be True. In that case, we should not change avg buy price
            #  otherwise, do moving average
            self.avg_buy_price[index] = np.where(actions[index] + share_after_selling[index] > 0,
                                                 total_spent / np.clip(new_shares, 1e-6, np.inf),
                                                 0
                                                 )
        self.avg_buy_price = np.clip(self.avg_buy_price, 0, np.inf)

        # do stop_loss
        now = self.share
        if self.stop_loss_calculation == 'close':
            price = self.price
        else:
            price = self.df.loc[self.day]['low']

        stop_loss_index = np.where((now > 0) & (self.avg_buy_price * self.stop_loss_tolerance > price))[0]
        before_loss_cut = self.state[0]

        for index in stop_loss_index:
            num_sell_stocks = self._sell_stock(index, now[index])
            # no trading rewards for the stop loss. this is passive

            self.avg_buy_price[index] = 0
            self.num_stop_loss += 1

        after_loss_cut = self.state[0]
        loss_cut_amount = abs(before_loss_cut - after_loss_cut)

        self.avg_buy_price = np.where(self.share <= 0, np.zeros_like(self.avg_buy_price),
                                      self.avg_buy_price)

        self.actions_memory.append(actions)

        # state: s -> s+1
        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif len(self.df.tic.unique()) > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        self.state = self._update_state()

        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())
        self.reward = np.log(end_total_asset / begin_total_asset)
        self.rewards_memory.append(end_total_asset - begin_total_asset)
        # self.reward = self.reward * self.reward_scaling
        self.state_memory.append(
            self.state
        )  # add current state in state_recorder for each step

        if self.asset <= 25000:
            done = True
        else:
            done = False
        if done or self.terminal:
            print("Asset", self.asset)
        # next_obs, reward, done, timeout, info
        return np.asarray(self.process_obs()).copy(), self.reward, done, self.terminal, {
            "loss_cut_amount": loss_cut_amount,
            "n_trades": n_trades
            }

    def dict_obs(self):
        data = self.df.query(f'index == {self.day}')

        tech = [data[tech].tolist()
                for tech in self.tech_indicator_list]

        # Get unique tickers
        tickers = self.df['tic'].unique()
        obs = { "Cash": self.state[0], "date": self.df.query(f'index == {self.day}')['date'].values[0] }

        for i, tic in enumerate(tickers):
            obs[tic] = {
                "price": self.price[i].item(),
                "share": self.share[i].item(),
                "avg_price": self.avg_buy_price[i].item(),
                "tech": [(self.tech_indicator_list[j], tech_values[i]) for j, tech_values in enumerate(tech)]
            }

        return obs


class StockTradingMOEnv(StockTradingEnvExtension):

    def __init__(self,
                 df: pd.DataFrame,
                 hmax: int = 500,
                 initial_amount: int = 100_000,
                 num_stock_shares: Sequence[int] | None = None,
                 buy_cost_pct: Sequence[float] | float = 0.001,
                 sell_cost_pct: Sequence[float] | float = 0.001,
                 reward_scaling: float = 1e-4,
                 cash_coef: float = 1.,
                 portfolio_value_coef: float = 100.,
                 tech_indicator_list: list[str] | Literal['auto'] = 'auto',
                 turbulence_threshold=None,
                 risk_indicator_col="turbulence",
                 make_plots: bool = False,
                 print_verbosity=10,
                 day=0,
                 initial=True,
                 previous_state=(),
                 model_name="",
                 mode="",
                 iteration="",
                 randomize_day: bool = True,
                 bidding: Literal['default', 'uniform', 'adv_uniform'] = 'default',
                 stop_loss_tolerance: float = 0.8,
                 # 0.8 means short selling when the loss is over 20% => 80% price reach
                 stop_loss_calculation: Literal['close', 'low'] = 'close',
                 seed: int = 42
                 ):
        self.cash_coef = cash_coef
        self.portfolio_value_coef = portfolio_value_coef
        self.asset_class = { 'Commodities': ['XLE', 'GLD', 'SLV'],
                             'Bonds': ['TLT', 'TIP', 'JNK'],
                             'Equities': ['SPY', 'QQQ', 'SOXX']
                             }

        self.asset_keys = ["Commodities", "Bonds", "Equities", "Cash"]
        self.reward_dim = 3

        super().__init__(df=df, hmax=hmax, initial_amount=initial_amount, num_stock_shares=num_stock_shares,
                         buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct, reward_scaling=reward_scaling,
                         tech_indicator_list=tech_indicator_list, turbulence_threshold=turbulence_threshold,
                         risk_indicator_col=risk_indicator_col, make_plots=make_plots, print_verbosity=print_verbosity,
                         day=day, initial=initial, previous_state=previous_state, model_name=model_name, mode=mode,
                         iteration=iteration, stop_loss_tolerance=stop_loss_tolerance,
                         randomize_day=randomize_day, bidding=bidding, stop_loss_calculation=stop_loss_calculation,
                         seed=seed,
                         )

        # state_space = self.observation_space.shape[0] + self.stock_dim

        # self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(state_space,))

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        info['loss_cut_amount'] = 0
        info['n_trades'] = 0
        _, asset_dict = self.classified_asset()
        info.update(**asset_dict)
        info['day'] = self.day

        return self.process_obs(), info

    def open_price_today(self):

        today = self.df.loc[self.day, :]
        open_price = today['open']
        return np.asarray(open_price.values).copy()

    def classified_asset(self) -> Tuple[dict, dict]:
        vec_asset = self.vec_asset
        ret = defaultdict(list)
        asset_dict = defaultdict(list)
        for class_name, tickers in self.asset_class.items():
            for ticker in tickers:
                if ticker in self.ticker_to_idx:
                    ret[class_name].append(vec_asset[self.ticker_to_idx[ticker]].item())
                    asset_dict[ticker] = vec_asset[self.ticker_to_idx[ticker]].item()
        ret['Cash'] = [vec_asset[-1].item()]
        asset_dict['Cash'] = vec_asset[-1].item()

        return ret, asset_dict

    def classified_asset_value(self):
        assets, asset_dict = self.classified_asset()
        return np.asarray([sum(assets[k]) for k in self.asset_keys]), asset_dict

    def step(self, actions):
        prev, _ = self.classified_asset_value()
        prev_portfolio_value = prev.sum().item()
        obs, reward, done, timeout, info = super().step(actions)
        after, asset_dict = self.classified_asset_value()

        p = after / after.sum()
        log_p = np.log(np.clip(p, 1e-6, 1))

        # how uniform the asset distribution is?
        entropy = (-(p * log_p).sum()) - 0.5
        after_portfolio_value = after.sum().item()
        # cash_ratio = (after[-1] /after_portfolio_value)
        # too low cash is penalty
        # distribution_reward = 0.5 * entropy + np.where(cash_ratio < 0.3, np.where(after[-1]< 1e-6, -10, np.log(cash_ratio)), 0)

        portfolio_value_rate = after_portfolio_value / prev_portfolio_value
        log_delta = np.where(after_portfolio_value < 1e-6, -10, np.log(portfolio_value_rate))
        # this controls MDD, sortino ratio, ...

        # num trades for boosting exploration.
        vec_reward = [np.clip(after_portfolio_value - prev_portfolio_value, -np.inf, 0) * self.reward_scaling,
                      # lower log volatility
                      entropy * 0.1,
                      log_delta * self.portfolio_value_coef,  # log return

                      # linear_delta * self.reward_scaling
                      ]
        info.update(**asset_dict)
        info['day'] = self.day

        return self.process_obs(), np.asarray(vec_reward), done, timeout, info

    def gold_norm_close(self):
        """
        This seems like data leakage, but due to the data processing (shift), this is the previous day's closing price.
        Therefore, there is no data leakage here.
        :return: PREVIOUS DAY of gold normalized closing price
        """
        gold_norm_close = self.df.loc[self.day, :]
        gold_norm_close = gold_norm_close['gold_norm_close']
        return np.asarray(gold_norm_close.values).copy()

    def gold_norm_open(self):

        today = self.df.loc[self.day, :]
        gold_norm_open = today['gold_norm_open']
        return np.asarray(gold_norm_open.values).copy()

    def process_obs(self):
        state = np.asarray(self.state).copy()
        tech = state[2 * self.stock_dim + 1:]

        obs = np.concatenate(
            [[np.log1p(state[0] / self.initial_amount)],  # cash
             np.log(self.open_price_today()),
             self.share / 100 - 3,
             np.log1p(self.avg_buy_price.copy() / self.open_price_today()),
             tech, ], axis=0)  # NOTE: also technical indicators are from the previous day.
        return obs

