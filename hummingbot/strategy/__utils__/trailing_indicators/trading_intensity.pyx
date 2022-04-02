import logging
import math

import time
from decimal import Decimal
# from math import floor, ceil
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from typing import (
    Tuple,
)
import warnings

from sklearn import linear_model

intensity_logger = None

from hummingbot.core.event.events import TradeType

cdef class TradingIntensityIndicator():
    @classmethod
    def logger(cls):
        global intensity_logger
        if intensity_logger is None:
            intensity_logger = logging.getLogger(__name__)
        return intensity_logger

    def __init__(self, order_refresh_time = 1, sampling_length: int = 30):
        self._alpha = 0
        self._kappa = 0
        self._old_kappa = 0
        self._old_alpha = 0
        self._trades = []
        self._bids_df = None
        self._asks_df = None
        self._sampling_length = sampling_length
        self._samples_length = 0
        self._last_inserted_trade_time = time.time()
        self._last_inserted_trade_price = 0
        self._last_inserted_trade_amount = 0
        self._last_inserted_trade_type = 0
        self._last_price = 0
        self._last_price_time = 0
        self._last_price_type = 0
        self._last_price_amount = 0
        self._average_sold_qty = 0
        self._average_bought_qty = 0
        self._nb_sells = 0
        self._nb_buys = 0
        # self.lambda_2 = 0
        # self.lambda_spread = {}
        # self.spread_levels = []
        self._order_refresh_time = order_refresh_time
        self.initial_time = time.time()
        # self.mids = []

        warnings.simplefilter("ignore", OptimizeWarning)

    def _simulate_execution(self, bids_df, asks_df):
        self.c_simulate_execution(bids_df, asks_df)

    cdef c_simulate_execution(self, new_bids_df, new_asks_df):
        cdef:
            object _bids_df = self._bids_df
            object _asks_df = self._asks_df
            object bids_df = new_bids_df
            object asks_df = new_asks_df
            int _sampling_length = self._sampling_length
            object bid
            object ask
            object price
            object bid_prev
            object ask_prev
            object price_prev
            list trades
            delta_spread = 0.001

        if self._last_price_time is np.nan:
            self.logger().info("_last_price_time is nan")
            return

        # if len(self.mids) < self._order_refresh_time:
        #     return

        if (self._last_inserted_trade_time == self._last_price_time) and \
                (self._last_inserted_trade_price == self._last_price) and\
                (self._last_price_amount == self._last_inserted_trade_amount) and\
                (self._last_price_type == self._last_inserted_trade_type):
            # self.logger().info("no trade to register")
            return

        # if self._last_inserted_trade_time == 0:
        #     self._last_inserted_trade_time = self._last_price_time

        bid = bids_df["price"].iloc[0]
        ask = asks_df["price"].iloc[0]
        price = (bid + ask) / 2

        bid_prev = _bids_df["price"].iloc[0]
        ask_prev = _asks_df["price"].iloc[0]
        price_prev = (bid_prev + ask_prev) / 2

        # divide by tick_size ?
        # maybe we should use the last_mid_price instead, the order has probably already eaten the lob
        spread = self._last_price - price_prev
        # rounding
        spread = round(abs(int(round(spread  / delta_spread, 0)) * delta_spread),4)
        # self.logger().info(f"spread:{spread}")

        # real_trades = []
        if self._last_price_type == TradeType.BUY:
            # self.logger().info(f"last_price_time: {self._last_price_time} ")
            # self.logger().info(f"_last_inserted_trade_time: {self._last_inserted_trade_time} ")
            # self.lambda_2 += (self._last_price_time - self._last_inserted_trade_time) / self._order_refresh_time
            # self.logger().info(f"lambda 2: {self.lambda_2} ")
            self._nb_buys += 1
            if self._nb_buys == 1:
                self._average_bought_qty = self._last_price_amount
            else:
                self._average_bought_qty = self._average_bought_qty - (1 / (self._nb_buys + 1)) * (self._average_bought_qty - self._last_price_amount)

            if spread == 0:
                # do nothing
                return
            elif self._last_price <= bid_prev:
                # limit order, do nothing
                return
            elif (self._last_price > bid_prev) and (self._last_price < ask_prev):
                # aggressive limit order, do nothing
                return
            elif self._last_price >= ask_prev:
                # we don't differentiate the buy from the sell side unfortunately
                # if spread not in self.spread_levels:
                #     self.spread_levels += [spread]
                #     self.lambda_spread[spread] = 1
                # else:
                #     self.lambda_spread[spread] += 1
                self._trades += [{'price_level': spread, 'amount': self._last_price_amount, 'type': 'BUY',
                                 'time': self._last_price_time}]
                self._last_inserted_trade_time = self._last_price_time
                self._last_inserted_trade_price = self._last_price
                self._last_inserted_trade_type = self._last_price_type
                self._last_inserted_trade_amount = self._last_price_amount
        elif self._last_price_type == TradeType.SELL:
            # self.logger().info(f"last_price_time: {self._last_price_time} ")
            # self.logger().info(f"_last_inserted_trade_time: {self._last_inserted_trade_time} ")
            # self.lambda_2 += (self._last_price_time - self._last_inserted_trade_time) / self._order_refresh_time
            # self.logger().info(f"lambda 2: {self.lambda_2}")
            # average sell quantity:
            self._nb_sells += 1
            if self._nb_sells == 1:
                self._average_sold_qty = self._last_price_amount
            else:
                self._average_sold_qty = self._average_sold_qty - (1 / (self._nb_sells + 1)) * (self._average_sold_qty - self._last_price_amount)

            if spread == 0:
                # do nothing
                return
            elif self._last_price >= ask_prev:
                # limit order, do nothing
                return
            elif (self._last_price > bid_prev) and (self._last_price < ask_prev):
                # aggressive limit order, do nothing... for now !
                return
            elif self._last_price <= bid_prev:
                # market order
                # if spread not in self.spread_levels:
                #     self.spread_levels += [spread]
                #     self.lambda_spread[spread] = 1
                # else:
                #     self.lambda_spread[spread] += 1
                self._trades += [{'price_level': spread, 'amount': self._last_price_amount, 'type': 'SELL', 'time': self._last_price_time}]

                self._last_inserted_trade_time = self._last_price_time
                self._last_inserted_trade_price = self._last_price
                self._last_inserted_trade_type = self._last_price_type
                self._last_inserted_trade_amount = self._last_price_amount

        self.logger().info(f"average buy volume: {self._average_bought_qty}"
                           f"average_sell volume: {self._average_sold_qty}")
        self.logger().info(f"nb buys: {self._nb_buys}"
                           f"nb sells: {self._nb_sells}")

        self.logger().info(f"len trade: {len(self._trades)}")
        if len(self._trades) > self._sampling_length:
            initial_time = self._trades[0]['time']
            self._trades = self._trades[1:]
            self.logger().info("set initial time")
            self.logger().info(f"t0={initial_time}")

    def _estimate_intensity(self):
        self.c_estimate_intensity()

    cdef c_estimate_intensity(self):
        cdef:
            dict trades_consolidated
            list lambdas
            list price_levels

        # Fit the probability density function; reuse previously calculated parameters as initial values
        try:
            self.logger().info("c_estimate_intensity....")
            last_trade_time = self.initial_time
            avg_volume = 0
            i = 0
            lambda_2 = 0
            spread_levels = []
            lambda_spread = {}
            for trade in self._trades:
                lambda_2 += (trade['time'] - last_trade_time) / self._order_refresh_time
                if trade['price_level'] not in spread_levels:
                    spread_levels += [trade['price_level']]
                    lambda_spread[trade['price_level']] = 1
                else:
                    lambda_spread[trade['price_level']] += 1

                if avg_volume == 0:
                    avg_volume = trade['amount']
                else:
                    avg_volume = avg_volume - (1 / (i + 1)) * (avg_volume - trade['amount'])

                i += 1
                last_trade_time = trade['time']
                #self.logger().info(f"trade['time']: {trade['time']}\n "
                #               f"trade['price_level']: {trade['price_level']}  ")

            self.logger().info(f"avg_volume: {avg_volume}\n "
                               f"lambda_2: {lambda_2}  ")

            spread_levels.sort()
            lambda_emp = []

            real_lambda_spread = lambda_spread.copy()
            for spread_j in lambda_spread:
                real_lambda_spread[spread_j]=0
                for spread_k in lambda_spread:
                    if spread_j <= spread_k:
                        real_lambda_spread[spread_j] += lambda_spread[spread_k]

            lambda_emp = [np.log(real_lambda_spread[spread] / lambda_2) for
                               spread in spread_levels]
            lambda_emp = pd.DataFrame(lambda_emp)
            spread_levels = pd.DataFrame(spread_levels)
            spread_levels = spread_levels.values.reshape(len(spread_levels), 1)
            lambda_emp = (lambda_emp.values.reshape(len(lambda_emp), 1)).squeeze()
            #lin_reg_model = linear_model.LinearRegression()
            lin_reg_model = linear_model.TheilSenRegressor()
            lin_reg_model.fit(spread_levels, lambda_emp)
            r_2=lin_reg_model.score(spread_levels, lambda_emp)
            kappa = -lin_reg_model.coef_.item()
            intensity_a = math.exp(lin_reg_model.intercept_)
            # average_volume = (self._average_bought_qty + self._average_sold_qty ) / 2

            self._alpha = Decimal(avg_volume * intensity_a)
            self._kappa = Decimal(kappa)

            self.logger().info(f"alpha: {self._alpha}")
            self.logger().info(f"kappa: {self._kappa}")
            self.logger().info(f"R_2: {r_2}")

            #########
            #self.logger().info(f"spread: {spread_levels}")
            #self.logger().info(f"lambda_spread: {lambda_spread}")
            #self.logger().info(f"real_lambda_spread: {real_lambda_spread}")
            #self.logger().info(f"lambda_2: {lambda_2}")
            #self.logger().info(f"Intensité: {lambda_emp}")
            #########

        except (RuntimeError, ValueError) as e:
            pass

    def add_sample(self, value: Tuple[pd.DataFrame, pd.DataFrame], last_trade: Tuple[double, double, TradeType, double]):
        bids_df = value[0]
        asks_df = value[1]

        self._last_price_time = last_trade[0]
        self._last_price = last_trade[1]
        self._last_price_type = last_trade[2]
        self._last_price_amount = last_trade[3]

        if bids_df.empty or asks_df.empty:
            return

        if self._bids_df is not None and self._asks_df is not None:
            # Retrieve previous order book, evaluate execution
            self.c_simulate_execution(bids_df, asks_df)
            if self.is_sampling_buffer_full:
                # Estimate alpha and kappa
                self.c_estimate_intensity()

        # Store the orderbook
        self._bids_df = bids_df
        self._asks_df = asks_df

        # price = (bids_df["price"].iloc[0] + asks_df["price"].iloc[0]) / 2
        # self.mids.append(price)



    @property
    def current_value(self) -> Tuple[float, float]:
        return self._alpha, self._kappa

    @property
    def is_sampling_buffer_full(self) -> bool:
        return len(self._trades) == self._sampling_length

    @property
    def is_sampling_buffer_changed(self) -> bool:
        is_changed = self._samples_length != len(self._trades)
        self._samples_length = len(self._trades)
        return is_changed

    @property
    def sampling_length(self) -> int:
        return self._sampling_length

    @property
    def average_bought_qty(self) -> float:
        return self._average_bought_qty

    @property
    def average_sold_qty(self) -> float:
        return self._average_sold_qty



