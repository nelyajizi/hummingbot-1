from email.headerregistry import DateHeader
from .base_trailing_indicator import BaseTrailingIndicator
import numpy as np

#AB : ArithmeticBrownian -   dS / S = drift x dt + vol x dW

class VolatilityAB_Indicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 30, processing_length: int = 15):
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        np_sampling_buffer = np.log(self._sampling_buffer.get_as_numpy_array())
        if np_sampling_buffer.size == 1:
            vol=0
        else :
            mu_delta = np.sum(np.diff(np_sampling_buffer)) / (np.diff(np_sampling_buffer)).size
            var_delta = np.sum(np.square(np.diff(np_sampling_buffer)-mu_delta)) / ((np.diff(np_sampling_buffer)).size-1)
            vol = np.sqrt(var_delta)
        return vol

    def _processing_calculation(self) -> float:
        # Only the last calculated indicator, not an average of multiple past indicator
        return self._processing_buffer.get_last_value()


class DriftAB_Indicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 30, processing_length: int = 15):
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        np_sampling_buffer = np.log(self._sampling_buffer.get_as_numpy_array())
        if np_sampling_buffer.size == 1:
            drift=0
        else :
            mu_delta = np.sum(np.diff(np_sampling_buffer)) / np_sampling_buffer.size
            var_delta = np.sum(np.square(np.diff(np_sampling_buffer)-mu_delta)) / ((np.diff(np_sampling_buffer)).size-1)
            drift = mu_delta + var_delta/2
        return drift

    def _processing_calculation(self) -> float:
        # Only the last calculated indicator, not an average of multiple past indicator
        return self._processing_buffer.get_last_value()