from email.headerregistry import DateHeader
from .base_trailing_indicator import BaseTrailingIndicator
from sklearn import linear_model
import pandas as pd
import numpy as np


class InstantDriftIndicator(BaseTrailingIndicator):
    def __init__(self, sampling_length: int = 30, processing_length: int = 15):
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        data = self._sampling_buffer.get_as_numpy_array()
        data_t= np.arange(0,data.size,1)
        data = pd.DataFrame(data)
        data_t = pd.DataFrame(data_t)
        
        if data.size < self._sampling_length:
            return 0
     

        time_duration= data_t[1:]
        close = data[1:]
        lin_reg_model = linear_model.LinearRegression()
        lin_reg_model.fit(time_duration, close)
        drift = lin_reg_model.coef_.item()

        self.logger().info(f"drift: {drift} ")
        #self.logger().info(f"data: {close} ")
        #self.logger().info(f"time_date: {time_duration} ")

        return drift

    def _processing_calculation(self) -> float:
        # Only the last calculated 
        return self._processing_buffer.get_last_value()

