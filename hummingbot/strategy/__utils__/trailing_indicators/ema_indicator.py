from datetime import datetime

import numpy as np
import pandas as pd
from .base_trailing_indicator import BaseTrailingIndicator


class EMAIndicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 30, processing_length: int = 1):
        if processing_length != 1:
            raise Exception("Exponential moving average processing_length should be 1")
        super().__init__(sampling_length, processing_length)
        self.half_life = sampling_length
        self.prev_time = datetime.now()

    def _indicator_calculation(self) -> float:
        ema_last = self._processing_buffer.get_last_value()
        np.isnan(ema_last)
        data = self._sampling_buffer.get_as_numpy_array()
        if data.size < self._sampling_length:
            return np.nan

        if np.isnan(ema_last):
            ema = pd.Series(data).tail(self._sampling_length).mean()
            return ema

        now = datetime.now()
        time_decay = (now - self.prev_time).microseconds
        time_decay = float(time_decay * 1e-6)

        alpha = 1 - np.exp(np.log(0.5) * time_decay / self.half_life)
        ema = alpha * data[-1] + (1 - alpha) * ema_last
        self.prev_time = now

        return ema

    def _processing_calculation(self) -> float:
        return self._processing_buffer.get_last_value()

    def update_half_life(self, factor):
        if not np.isnan(factor) and factor != 0:
            self.half_life = factor
