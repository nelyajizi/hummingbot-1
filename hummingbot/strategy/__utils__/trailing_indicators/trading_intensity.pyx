import logging
import math

import time
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from typing import Tuple
from sklearn import linear_model
from hummingbot.core.event.events import TradeType

import warnings

intensity_logger = None


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
        self._abs_price_changes = []
        self._price_changes = []
        self._net_volume = []
        self._median_price_impact = np.nan
        self._avg_impact = np.nan
        self._bid_ask_spread = []
        self._lambda_coef = np.nan
        self._lambda_intercept = np.nan
        self._order_imbalance = np.nan

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
            return


        bid = bids_df["price"].iloc[0]
        ask = asks_df["price"].iloc[0]
        price = (bid + ask) / 2

        self._bid_ask_spread += [ask - bid]

        bid_prev = _bids_df["price"].iloc[0]
        ask_prev = _asks_df["price"].iloc[0]
        price_prev = (bid_prev + ask_prev) / 2

        diff = price - price_prev
        if diff != 0:
            prev_bid_vol = _bids_df["amount"].iloc[0]
            prev_ask_vol = _asks_df["amount"].iloc[0]
            if self._last_price_type == TradeType.BUY:
                epsilon = 1
            else:
                epsilon = -1
            self._abs_price_changes += [np.abs(diff)]
            self._price_changes += [diff]
            self._net_volume += [prev_ask_vol - prev_bid_vol]

        # divide by tick_size ?
        spread = self._last_price - price_prev
        # rounding
        spread = round(abs(int(round(spread  / delta_spread, 0)) * delta_spread),4)

        if self._last_price_type == TradeType.BUY:
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
                self._trades += [{'price_level': spread, 'amount': self._last_price_amount, 'type': 'BUY',
                                 'time': self._last_price_time}]
                self._last_inserted_trade_time = self._last_price_time
                self._last_inserted_trade_price = self._last_price
                self._last_inserted_trade_type = self._last_price_type
                self._last_inserted_trade_amount = self._last_price_amount
        elif self._last_price_type == TradeType.SELL:
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
                self._trades += [{'price_level': spread, 'amount': self._last_price_amount, 'type': 'SELL', 'time': self._last_price_time}]
                self._last_inserted_trade_time = self._last_price_time
                self._last_inserted_trade_price = self._last_price
                self._last_inserted_trade_type = self._last_price_type
                self._last_inserted_trade_amount = self._last_price_amount

        self.logger().info(f"average buy volume: {self._average_bought_qty}"
                           f"average_sell volume: {self._average_sold_qty}")
        self.logger().info(f"nb buys: {self._nb_buys}"
                           f"nb sells: {self._nb_sells}")

        if len(self._trades) > self._sampling_length:
            initial_time = self._trades[0]['time']
            self._trades = self._trades[1:]

    def _estimate_intensity(self):
        self.c_estimate_intensity()

    cdef c_estimate_intensity(self):
        cdef:
            dict trades_consolidated
            list lambdas
            list price_levels

        try:
            self.logger().info("c_estimate_intensity....")
            #last_trade_time = self.initial_time
            #avg_volume = 0
            #avg_volume_v2 = 0
            #median_volume = 0
            #i = 0
            #lambda_2 = 0
            #spread_levels = []
            #trade_amount = []
            #lambda_spread = {}
            #for trade in self._trades:
            #    lambda_2 += (trade['time'] - last_trade_time) #/ self._order_refresh_time
            #    if trade['price_level'] not in spread_levels:
            #        spread_levels += [trade['price_level']]
            #        lambda_spread[trade['price_level']] = 1
            #    else:
            #        lambda_spread[trade['price_level']] += 1

            #    trade_amount += [trade['amount']]
            #    last_trade_time = trade['time']
            #    #self.logger().info(f"trade['time']: {trade['time']}\n "
            #    #               f"trade['price_level']: {trade['price_level']}  ")

            #avg_volume= np.mean(trade_amount)
            #median_volume= np.median(trade_amount)
            #self.logger().info(f"median_volume: {median_volume}\n "
            #                   f"avg_volume: {avg_volume}\n "
            #                   f"lambda_2: {lambda_2}  ")

            #spread_levels.sort()
            #lambda_emp = []

            #real_lambda_spread = lambda_spread.copy()
            #for spread_j in lambda_spread:
            #    real_lambda_spread[spread_j]=0
            #    for spread_k in lambda_spread:
            #        if spread_j <= spread_k:
            #            real_lambda_spread[spread_j] += lambda_spread[spread_k]#

            #lambda_emp = [np.log(real_lambda_spread[spread] / lambda_2) for
            #                   spread in spread_levels]
            #lambda_emp = pd.DataFrame(lambda_emp)
            #spread_levels = pd.DataFrame(spread_levels)
            #spread_levels = spread_levels.values.reshape(len(spread_levels), 1)
            #lambda_emp = (lambda_emp.values.reshape(len(lambda_emp), 1)).squeeze()
            ##lin_reg_model = linear_model.LinearRegression()
            #lin_reg_model = linear_model.TheilSenRegressor()
            #lin_reg_model.fit(spread_levels, lambda_emp)
            #r_2=lin_reg_model.score(spread_levels, lambda_emp)
            #kappa = -lin_reg_model.coef_.item()
            #intensity_a = math.exp(lin_reg_model.intercept_)
            ## average_volume = (self._average_bought_qty + self._average_sold_qty ) / 2

            ##################### NOUVEAU CALCUL D'INTENSITE
            avg_volume = 0
            median_volume = 0
            spread_levels = []
            trade_amount = []
            count_spread = {}
            count_emp = []
            duree=0
            nombre_de_periode_dt=0
            i=0
            
            premier_trade_du_buffer = list(self._trades)[0]
            dernier_trade_du_buffer = list(self._trades)[-1]
            nombre_de_periode_dt=int((dernier_trade_du_buffer['time']-premier_trade_du_buffer['time'])/self._order_refresh_time)
            # self.logger().info(f"nombre de delta_t: {nombre_de_periode_dt}")
            if nombre_de_periode_dt==0:
                self.logger().info(f"Attention : Le trading_buffer n'est pas suffisamment grand au regard de l'order_refresh")
                nombre_de_periode_dt=1

            order_imb = 0
            for trade in self._trades:
                duree = (trade['time'] - premier_trade_du_buffer['time'])
                i = int(duree/self._order_refresh_time)
                if i<=nombre_de_periode_dt:
                    if trade['price_level'] not in spread_levels:
                        spread_levels += [trade['price_level']]
                        count_spread[trade['price_level']] = 1
                    else:
                        count_spread[trade['price_level']] += 1
                trade_amount += [trade['amount']]
                trade_sign = 1 if trade['type'] == 'BUY' else -1
                order_imb += trade_sign * trade['amount']

            self._order_imbalance = order_imb
            avg_volume= np.mean(trade_amount)
            median_volume= np.median(trade_amount)
            self.logger().info(f"median_volume: {median_volume}\n "
                               f"avg_volume: {avg_volume}")

            spread_levels.sort()
            real_count_spread = count_spread.copy()
            for spread_j in count_spread:
                real_count_spread[spread_j]=0
                for spread_k in count_spread:
                    if spread_j <= spread_k:
                        real_count_spread[spread_j] += count_spread[spread_k]

            count_emp = [np.log((real_count_spread[spread] / nombre_de_periode_dt) / self._order_refresh_time) for
                         spread in spread_levels]
            count_emp = pd.DataFrame(count_emp)
            spread_levels = pd.DataFrame(spread_levels)
            spread_levels = spread_levels.values.reshape(len(spread_levels), 1)
            count_emp = (count_emp.values.reshape(len(count_emp), 1)).squeeze()
            #lin_reg_model = linear_model.LinearRegression()
            lin_reg_model = linear_model.TheilSenRegressor()
            lin_reg_model.fit(spread_levels, count_emp)
            r_2=lin_reg_model.score(spread_levels, count_emp)
            kappa = -lin_reg_model.coef_.item()
            intensity_a = math.exp(lin_reg_model.intercept_)
            ############ FIN DU NOUVEAU CALCUL D'INTENSITE

            self._alpha = Decimal(median_volume * intensity_a)
            self._kappa = Decimal(kappa)

            self._median_price_impact = np.median(self._abs_price_changes)
            self._avg_impact = np.mean(self._abs_price_changes)

            self.logger().info(f"calculating Kyle's lambda:")
            if len(self.price_changes) > 3:
                price_changes = pd.DataFrame(self._price_changes)
                order_imb = pd.DataFrame(self._net_volume)
                order_imb = order_imb.values.reshape(len(order_imb), 1)
                price_changes = price_changes.values.reshape(len(price_changes), 1)
                reg_model = linear_model.LinearRegression()
                reg_model.fit(order_imb, price_changes)
                self._lambda_coef = lin_reg_model.coef_.item()
                self._lambda_intercept = lin_reg_model.intercept_
                self.logger().info(f"coef: {self._lambda_coef}")
                self.logger().info(f"intercept: {self._lambda_intercept}")

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

    @property
    def net_volume(self) -> list:
        return self._net_volume

    @property
    def price_changes(self) -> list:
        return self._price_changes

    @property
    def median_price_impact(self) -> float:
        return self._median_price_impact

    @property
    def avg_impact(self):
        return self._avg_impact

    @property
    def bid_ask_spread(self):
        return self._bid_ask_spread

    @property
    def avg_bid_ask_spread(self):
        return np.average(self._bid_ask_spread)

    @property
    def lambda_coef(self):
        return self._lambda_coef

    @property
    def lambda_intercept(self):
        return self._lambda_intercept

    @property
    def order_imbalance(self):
        return self._order_imbalance

