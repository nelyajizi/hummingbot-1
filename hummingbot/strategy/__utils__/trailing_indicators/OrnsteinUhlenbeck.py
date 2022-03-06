from decimal import Decimal
from typing import Tuple
from sklearn import linear_model
import numpy as np
import pandas as pd
from .base_trailing_indicator import BaseTrailingIndicator
from ..ring_buffer import RingBuffer


class OUModelIndicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 200, processing_length: int = 15):
        super().__init__(sampling_length, processing_length)
        self.mean_rev_speed_buffer = RingBuffer(processing_length)
        self.stdev = RingBuffer(processing_length)

    def _indicator_calculation(self) -> Tuple[Decimal, Decimal, Decimal]:
        data = self._sampling_buffer.get_as_numpy_array()
        if data.size < self._sampling_length:
            return np.nan, np.nan, np.nan
        data = pd.DataFrame(data)
        lag = data.shift(1)[1:]
        lag = lag.values.reshape(len(lag), 1)
        lin_reg_model = linear_model.LinearRegression()
        close = data[1:]
        lin_reg_model.fit(lag, close)
        coef = lin_reg_model.coef_.item()
        intercept = lin_reg_model.intercept_[0]
        if coef != 0:
            mean_rev_speed = -np.log(coef)
            mean_rev = intercept / (1 - coef)

            close = close.values.reshape(len(close), 1)
            prediction = lin_reg_model.predict(lag)
            residual = close - pd.DataFrame(prediction.flatten())
            realized_vol = self.realized_vol(residual)[0]
            realized_vol /= np.sqrt(2 * mean_rev_speed / (1 - np.exp(-2 * mean_rev_speed)))
            return mean_rev, mean_rev_speed, realized_vol
        else:
            delta = data[1:] - data.shift(1)[1:]
            delta = delta.values.reshape(len(delta), 1)
            lin_reg_model = linear_model.LinearRegression()
            lin_reg_model.fit(lag, delta)
            intercept = lin_reg_model.intercept_
            coef = lin_reg_model.coef_.item()
            mean_rev_speed = -coef
            if mean_rev_speed != 0:
                mean_rev = intercept / mean_rev_speed
                prediction = lin_reg_model.predict(lag)
                residual = delta - pd.DataFrame(prediction.flatten())
                realized_vol = self.realized_vol(residual)[0]
                return mean_rev, mean_rev_speed, realized_vol
            else:
                return self.current_value

    def _processing_calculation(self) -> float:
        return self._processing_buffer.get_last_value()

    @property
    def current_value(self) -> Tuple[Decimal, Decimal, Decimal]:
        return self._processing_buffer.get_last_value(),\
            self.mean_rev_speed_buffer.get_last_value(),\
            self.stdev.get_last_value()

    @staticmethod
    def realized_vol(sample):
        delta = sample[1:] - sample.shift(1)[1:]
        abs_log_return = delta.apply(np.abs)
        vol = abs_log_return * abs_log_return
        vol = vol.sum() / len(sample)
        vol = np.sqrt(vol)
        return vol

    def add_sample(self, value: float):
        self._sampling_buffer.add_value(value)
        mean_rev, mean_rev_speed, realized_vol = self._indicator_calculation()
        self._processing_buffer.add_value(mean_rev)
        self.mean_rev_speed_buffer.add_value(mean_rev_speed)
        self.stdev.add_value(realized_vol)

    def get_mean_vol(self, length):
        mean_rev, mean_rev_speed, realized_vol = self.current_value
        if np.isnan(mean_rev):
            return np.nan, np.nan
        mid = self._sampling_buffer.get_last_value()
        mean = mid * np.exp(-mean_rev_speed * length)
        mean += mean_rev * (1 - np.exp(-mean_rev_speed * length))
        variance = realized_vol * realized_vol * (1 - np.exp(-2 * mean_rev_speed * length)) / 2 * mean_rev_speed
        mean += 0.5 * variance
        vol_model = np.sqrt(variance)
        return mean, vol_model
