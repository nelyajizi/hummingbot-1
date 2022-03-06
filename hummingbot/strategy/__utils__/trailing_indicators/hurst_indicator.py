import math
import sys

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
        hurst, c, data = self.compute_hurst(data, simplified=False)
        if hurst != 0:
            return hurst
        hurst, c, data = self.compute_hurst(data, simplified=True)
        return hurst
    #     lags = range(2, 100)
    #     # Calculate the array of the variances of the lagged differences
    #     tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
    #     # Use a linear fit to estimate the Hurst Exponent
    #     poly = np.polyfit(np.log(lags), np.log(tau), 1)
    # # Return the Hurst exponent from the polyfit output
    #     if np.isnan(poly[0]):
    #         return self._processing_buffer.get_last_value()
    #     return poly[0] * 2.0

    def _processing_calculation(self) -> float:
        return self._processing_buffer.get_last_value()

    def __get_RS(self, series):
        incs = self.__to_pct(series)
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)
        if R == 0 or S == 0:
            return 0  # return 0 to skip this interval due undefined R/S

        return R / S

    def __get_simplified_RS(self, series):
        pcts = self.__to_pct(series)
        R = max(series) / min(series) - 1.  # range in percent
        S = np.std(pcts, ddof=1)
        if R == 0 or S == 0:
            return 0  # return 0 to skip this interval due the undefined R/S ratio
        return R / S

    @staticmethod
    def __to_pct(x):
        pcts = x[1:] / x[:-1] - 1.
        return pcts

    def compute_hurst(self, series, min_window=10, max_window=None, simplified=True):
        if len(series) < 100:
            raise ValueError("Series length must be greater or equal to 100")

        ndarray_likes = [np.ndarray]
        if "pandas.core.series" in sys.modules.keys():
            ndarray_likes.append(pd.core.series.Series)

        # convert series to numpy array if series is not numpy array or pandas Series
        if type(series) not in ndarray_likes:
            series = np.array(series)

        if "pandas.core.series" in sys.modules.keys() and type(series) == pd.core.series.Series:
            if series.isnull().values.any():
                raise ValueError("Series contains NaNs")
            series = series.values  # convert pandas Series to numpy array
        elif np.isnan(np.min(series)):
            raise ValueError("Series contains NaNs")

        if simplified:
            RS_func = self.__get_simplified_RS
        else:
            RS_func = self.__get_RS

        err = np.geterr()
        np.seterr(all='raise')

        max_window = max_window or len(series) - 1
        window_sizes = list(map(
            lambda x: int(10 ** x),
            np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
        window_sizes.append(len(series))
        window = []
        RS = []
        for w in window_sizes:
            rs = []
            for start in range(0, len(series), w):
                if (start + w) > len(series):
                    break
                _ = RS_func(series[start:start + w])
                if _ != 0:
                    rs.append(_)
            if len(rs) != 0:
                window.append(w)
                RS.append(np.mean(rs))

        A = np.vstack([np.log10(window), np.ones(len(RS))]).T
        H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
        np.seterr(**err)

        c = 10 ** c
        return H, c, [window, RS]
