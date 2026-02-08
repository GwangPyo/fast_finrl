FastFinRL
=========

Single-environment trading simulator.

.. py:class:: fast_finrl_py.FastFinRL(csv_path, initial_amount=30000.0, failure_threshold=25000.0, hmax=15, buy_cost_pct=0.01, sell_cost_pct=0.01, stop_loss_tolerance=0.8, bidding='default', stop_loss_calculation='close', tech_indicator_list=[], macro_tickers=[], return_format='json')

   Create a single trading environment.

   :param str csv_path: Path to data file (CSV or Parquet, auto-detected by extension)
   :param float initial_amount: Starting cash (default: 30000.0)
   :param float failure_threshold: Episode terminates if total_asset <= this (default: 25000.0)
   :param int hmax: Maximum shares per trade (default: 15)
   :param float buy_cost_pct: Buy transaction cost percentage (default: 0.01)
   :param float sell_cost_pct: Sell transaction cost percentage (default: 0.01)
   :param float stop_loss_tolerance: Stop loss threshold (default: 0.8)
   :param str bidding: Price execution mode ('default', 'uniform', 'adv_uniform')
   :param str stop_loss_calculation: Stop loss price source ('close', 'open')
   :param list tech_indicator_list: Technical indicators to include
   :param list macro_tickers: Macro reference tickers
   :param str return_format: Output format ('json' or 'vec')

   .. py:method:: reset(ticker_list, seed=0, shifted_start=0)

      Reset environment to initial state.

      :param list ticker_list: List of ticker symbols to trade
      :param int seed: Random seed for reproducibility
      :param int shifted_start: Day offset from start
      :return: Initial state dict

   .. py:method:: step(actions)

      Execute one trading step.

      :param list actions: Action values [-1, 1] for each ticker. Negative=sell, Positive=buy
      :return: Next state dict with 'reward', 'done', 'terminal'

   .. py:method:: get_all_tickers()

      Get all available tickers in the dataset.

      :return: List of ticker symbols

   .. py:method:: get_indicator_names()

      Get names of technical indicators.

      :return: Set of indicator names

   .. py:method:: get_max_day()

      Get the maximum day index in the dataset.

      :return: Maximum day number

   .. py:method:: get_market_window_numpy(tickers, day, h, future=0)

      Get market data window as numpy arrays.

      :param list tickers: Ticker symbols
      :param int day: Center day
      :param int h: History length
      :param int future: Future length (default: 0)
      :return: Dict with 'ohlcv', 'indicators', 'mask' arrays

   .. py:method:: get_raw_value(ticker, day, column)

      Get raw OHLCV value for a specific ticker/day.

      :param str ticker: Ticker symbol
      :param int day: Day index
      :param str column: Column name ('open', 'high', 'low', 'close', 'volume')
      :return: Value as float

State Format (vec)
------------------

When ``return_format='vec'``, state dict contains:

- **day** (int): Current day index
- **cash** (float): Available cash
- **shares** (ndarray): Share holdings per ticker
- **avg_buy_price** (ndarray): Average buy price per ticker
- **tickers** (list): Ticker symbols
- **open** (ndarray): Current open prices
- **indicators** (ndarray): Technical indicators [n_tickers, n_indicators]
- **reward** (float): Step reward (after step only)
- **done** (bool): Episode done flag
- **terminal** (bool): Terminal state flag

State Format (json)
-------------------

When ``return_format='json'``, state dict contains nested structure:

- **day**: Current day index
- **portfolio**: Dict with 'cash', 'holdings', 'total_asset'
- **market**: Dict with per-ticker OHLCV and indicators
