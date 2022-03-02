import time
from datetime import datetime

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator as rsi_indicator

from .base_trailing_indicator import BaseTrailingIndicator


class RSIIndicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 14, processing_length: int = 1):
        if processing_length != 1:
            raise Exception("RSI processing_length should be 1")
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        data = self._sampling_buffer.get_as_numpy_array()
        if data.size < self._sampling_length:
            return np.nan
        data = pd.Series(data)
        rsi = rsi_indicator(close=data, window=self._sampling_length,
                            fillna=True).rsi()
        # diff = data.diff(1).dropna()
        # up = diff.copy()
        # down = diff.copy()
        # up[up < 0] = 0
        # down[down > 0] = 0
        # avg_gain = float(up.mean())
        # avg_loss = abs(float(down.mean()))
        # relative_strength = 0
        # if avg_loss != 0:
        #     relative_strength = avg_gain / avg_loss
        # rsi = 100.0 - (100.0 / (1.0 + relative_strength))
        # return rsi
        return rsi.iloc[-1]

    def _processing_calculation(self) -> float:
        return self._processing_buffer.get_last_value()

