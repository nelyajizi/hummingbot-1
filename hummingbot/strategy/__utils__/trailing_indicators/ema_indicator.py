from datetime import datetime

import numpy as np
import pandas as pd
from .base_trailing_indicator import BaseTrailingIndicator

underlyings = ["price", "volatility", "diff_price"]


class EMAIndicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 30, processing_length: int = 1, underlying_type="price"):
        if processing_length != 1:
            raise Exception("Exponential moving average processing_length should be 1")
        super().__init__(sampling_length, processing_length)
        self.half_life = sampling_length
        self.prev_time = datetime.now()
        if underlying_type in underlyings:
            self.underlying_type = underlying_type
        else:
            raise Exception("unknown underlyings")

    def _indicator_calculation(self) -> float:
        ema_last = self._processing_buffer.get_last_value()
        np.isnan(ema_last)
        data = self._sampling_buffer.get_as_numpy_array()
        if data.size < self._sampling_length:
            return np.nan
        ema = 0.0
        if np.isnan(ema_last):
            if self.underlying_type == "price":
                ema = np.average(data)
            elif self.underlying_type == "diff_price":
                delta = np.diff(data)
                ema = np.average(delta)
            elif self.underlying_type == "volatility":
                ema = self.realized_vol(pd.Series(data))
            return ema

        now = datetime.now()
        time_decay = (now - self.prev_time).microseconds
        time_decay = float(time_decay * 1e-6)

        alpha = 1 - np.exp(np.log(0.5) * time_decay / self.half_life)
        if self.underlying_type == "price":
            ema = alpha * data[-1] + (1 - alpha) * ema_last
        elif self.underlying_type == "diff_price":
            ema = alpha * (data[-1] - data[-2]) + (1 - alpha) * ema_last
        elif self.underlying_type == "volatility":
            square_return = (data[-1] - data[-2])**2
            ema = alpha * square_return + (1 - alpha) * ema_last
        self.prev_time = now
        return ema

    def _processing_calculation(self) -> float:
        return self._processing_buffer.get_last_value()

    def update_half_life(self, factor):
        if not np.isnan(factor) and factor != 0:
            self.half_life = factor

    @staticmethod
    def realized_vol(sample):
        delta = sample[1:] - sample.shift(1)[1:]
        abs_log_return = delta.apply(np.abs)
        vol = abs_log_return * abs_log_return
        vol = vol.sum() / len(sample)
        vol = np.sqrt(vol)
        return vol
