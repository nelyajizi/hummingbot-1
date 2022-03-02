import time
from datetime import datetime

import numpy as np
import pandas as pd
from .base_trailing_indicator import BaseTrailingIndicator


class HurstIndicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 300, processing_length: int = 1):
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self):
        data = self._sampling_buffer.get_as_numpy_array()
        if data.size < self._sampling_length:
            return np.nan

        lags = range(2, 100)
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
    # Return the Hurst exponent from the polyfit output
        if np.isnan(poly[0]):
            return self._processing_buffer.get_last_value()
        return poly[0] * 2.0

    def _processing_calculation(self) -> float:
        return self._processing_buffer.get_last_value()
