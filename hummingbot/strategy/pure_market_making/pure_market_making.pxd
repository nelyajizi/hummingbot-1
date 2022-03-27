# distutils: language=c++

from libc.stdint cimport int64_t

from hummingbot.strategy.strategy_base cimport StrategyBase


cdef class PureMarketMakingStrategy(StrategyBase):
    cdef:
        object _market_info

        object _bid_spread
        object _ask_spread
        object _minimum_spread
        object _order_amount
        int _order_levels
        int _buy_levels
        int _sell_levels
        object _order_level_spread
        object _order_level_amount
        double _order_refresh_time
        double _max_order_age
        object _order_refresh_tolerance_pct
        double _filled_order_delay
        bint _inventory_skew_enabled
        object _inventory_target_base_pct
        object _inventory_range_multiplier
        bint _hanging_orders_enabled
        object _hanging_orders_tracker
        bint _order_optimization_enabled
        object _ask_order_optimization_depth
        object _bid_order_optimization_depth
        bint _add_transaction_costs_to_orders
        object _asset_price_delegate
        object _inventory_cost_price_delegate
        object _price_type
        bint _take_if_crossed
        object _price_ceiling
        object _price_floor
        bint _ping_pong_enabled
        list _ping_pong_warning_lines
        bint _hb_app_notification
        object _order_override

        double _cancel_timestamp
        double _create_timestamp
        object _limit_order_type
        bint _all_markets_ready
        int _filled_buys_balance
        int _filled_sells_balance
        double _last_timestamp
        double _status_report_interval
        int64_t _logging_options
        object _last_own_trade_price
        bint _should_wait_order_cancel_confirmation
        object _avg_vol
        object _lt_avg_vol
        object _ema
        object _ema_lt
        object _ouprocess
        double _last_sampling_timestamp
        object _latest_parameter_calculation_vol
        str _debug_csv_path
        object _rsi
        object _auto_correl
        object _bb
        double _last_price
        double _last_rsi
        object _hurst
        double _last_ema
        double _last_ema_lt
        bint _is_debug
        double _max_spread
        int _lob_depth
        int sell_signal
        int buy_signal
        double _gamma

    cdef object c_get_mid_price(self)
    cdef object c_create_base_proposal(self)
    cdef tuple c_get_adjusted_available_balance(self, list orders)
    cdef c_apply_order_levels_modifiers(self, object proposal)
    cdef c_apply_price_band(self, object proposal)
    cdef c_apply_ping_pong(self, object proposal)
    cdef c_apply_order_price_modifiers(self, object proposal)
    cdef c_apply_order_size_modifiers(self, object proposal)
    cdef c_apply_inventory_skew(self, object proposal)
    cdef c_apply_budget_constraint(self, object proposal)

    cdef c_filter_out_takers(self, object proposal)
    cdef c_apply_order_optimization(self, object proposal)
    cdef c_apply_add_transaction_costs(self, object proposal)
    cdef bint c_is_within_tolerance(self, list current_prices, list proposal_prices)
    cdef c_cancel_active_orders(self, object proposal)
    cdef c_cancel_orders_below_min_spread(self)
    cdef c_cancel_active_orders_on_max_age_limit(self)
    cdef bint c_to_create_orders(self, object proposal)
    cdef c_execute_orders_proposal(self, object proposal)
    cdef set_timers(self)

    cdef bint c_is_algorithm_ready(self)
    cdef c_collect_market_variables(self, double timestamp)
    cdef object c_get_order_book_snapshot(self)
    cdef object c_calculate_target_inventory(self)

