from decimal import Decimal
from typing import Tuple
import numpy as np
import pandas as pd
from .base_trailing_indicator import BaseTrailingIndicator
from ..ring_buffer import RingBuffer



class BollingerBandsIndicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 200, processing_length: int = 15):
        super().__init__(sampling_length, processing_length)
        self.bb_up_buffer = RingBuffer(processing_length)
        self.bb_down_buffer = RingBuffer(processing_length)

    def _indicator_calculation(self) -> Tuple[Decimal, Decimal, Decimal]:
        data = self._sampling_buffer.get_as_numpy_array()
        if data.size < self._sampling_length:
            return np.nan, np.nan, np.nan

        sample = pd.DataFrame(self._sampling_buffer.get_as_numpy_array())
        mean = sample.mean()
        std_dev = sample.std()
        bb_up = mean + 2 * std_dev
        bb_down = mean - 2 * std_dev
        return mean, bb_up, bb_down

    def _processing_calculation(self) -> float:
        return self._processing_buffer.get_last_value()

    @property
    def current_value(self) -> Tuple[Decimal, Decimal, Decimal]:
        return self._processing_buffer.get_last_value(), self.bb_up_buffer.get_last_value(), \
               self.bb_down_buffer.get_last_value()

    def add_sample(self, value: float):
        self._sampling_buffer.add_value(value)
        bb_mean, bb_up, bb_down = self._indicator_calculation()
        self._processing_buffer.add_value(bb_mean)
        self.bb_up_buffer.add_value(bb_up)
        self.bb_down_buffer.add_value(bb_down)
