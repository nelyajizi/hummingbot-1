from libc.stdint cimport int64_t


cdef class TradingIntensityIndicator():
    cdef:
        double _alpha
        double _kappa
        list _trades
        object _bids_df
        object _asks_df
        int _sampling_length
        int _samples_length
        double _last_inserted_trade_time
        double _last_inserted_trade_price
        double _last_price
        double _last_price_time
        object _last_price_type

    cdef c_simulate_execution(self, bids_df, asks_df)
    cdef c_estimate_intensity(self)
