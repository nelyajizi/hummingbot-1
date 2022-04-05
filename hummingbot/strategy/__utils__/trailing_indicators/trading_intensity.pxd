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
        double _last_inserted_trade_amount
        object _last_inserted_trade_type
        double _last_price
        double _last_price_time
        object _last_price_type
        double _last_price_amount
        double _average_sold_qty
        double _average_bought_qty
        double _nb_sells
        double _nb_buys
        # double lambda_2
        # object lambda_spread
        object count_spread
        list spread_levels
        double _old_kappa
        double _old_alpha
        int _order_refresh_time
        double initial_time
        list _price_changes
        list _order_sizes
        double _median_price_impact
        double _avg_impact
        list _bid_ask_spread
        # list mids


    cdef c_simulate_execution(self, bids_df, asks_df)
    cdef c_estimate_intensity(self)
