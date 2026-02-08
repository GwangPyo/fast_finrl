VecFastFinRL
============

Vectorized multi-environment trading simulator for parallel training.

.. py:class:: fast_finrl_py.VecFastFinRL(csv_path, n_envs, initial_amount=100000, hmax=100, buy_cost_pct=0.001, sell_cost_pct=0.001, stop_loss_tolerance=0.9, bidding='default', stop_loss_calculation='close', initial_seed=0, tech_indicator_list=[], macro_tickers=[], auto_reset=True, return_format='json', num_tickers=0, shuffle_tickers=False)

   Create vectorized trading environments.

   :param str csv_path: Path to data file (CSV or Parquet, auto-detected by extension)
   :param int n_envs: Number of parallel environments
   :param float initial_amount: Starting cash per env (default: 100000)
   :param int hmax: Maximum shares per trade (default: 100)
   :param float buy_cost_pct: Buy transaction cost percentage (default: 0.001)
   :param float sell_cost_pct: Sell transaction cost percentage (default: 0.001)
   :param float stop_loss_tolerance: Stop loss threshold (default: 0.9)
   :param str bidding: Price execution mode ('default', 'uniform', 'adv_uniform')
   :param str stop_loss_calculation: Stop loss price source ('close', 'open')
   :param int initial_seed: Base random seed
   :param list tech_indicator_list: Technical indicators to include
   :param list macro_tickers: Macro reference tickers
   :param bool auto_reset: Auto-reset done environments (default: True)
   :param str return_format: Output format ('json' or 'vec')
   :param int num_tickers: Number of tickers per env (0=all)
   :param bool shuffle_tickers: Shuffle ticker selection per env

   .. py:method:: reset(seed=-1, tickers=None)

      Reset all environments.

      :param int seed: Random seed (-1 for random)
      :param list tickers: Optional ticker list override
      :return: Batched state dict

   .. py:method:: step(actions)

      Execute one step across all environments.

      :param ndarray actions: Actions array [n_envs, n_tickers]
      :return: Batched next state dict

   .. py:method:: num_envs()

      Get number of environments.

      :return: Number of parallel environments

   .. py:method:: n_tickers()

      Get number of tickers per environment.

      :return: Ticker count

   .. py:method:: n_indicators()

      Get number of technical indicators.

      :return: Indicator count

   .. py:method:: get_tickers()

      Get ticker lists for all environments.

      :return: List of ticker lists [n_envs][n_tickers]

   .. py:method:: get_all_tickers()

      Get all available tickers in dataset.

      :return: List of all ticker symbols

   .. py:method:: get_indicator_names()

      Get names of technical indicators.

      :return: Set of indicator names

Batched State Format (vec)
--------------------------

When ``return_format='vec'``, state dict contains batched arrays:

- **day** (ndarray[n_envs]): Current day per env
- **cash** (ndarray[n_envs]): Available cash per env
- **shares** (ndarray[n_envs, n_tickers]): Share holdings
- **avg_buy_price** (ndarray[n_envs, n_tickers]): Average buy prices
- **tickers** (list[n_envs]): Ticker lists per env
- **open** (ndarray[n_envs, n_tickers]): Current open prices
- **indicators** (ndarray[n_envs, n_tickers, n_indicators]): Technical indicators
- **reward** (ndarray[n_envs]): Step rewards
- **done** (ndarray[n_envs]): Done flags
- **terminal** (ndarray[n_envs]): Terminal flags

Ticker Configuration
--------------------

Control which tickers each environment trades:

.. code-block:: python

   # All tickers, same for all envs
   env = VecFastFinRL(path, n_envs=4, num_tickers=0)

   # Fixed 3 tickers (alphabetically first), same for all envs
   env = VecFastFinRL(path, n_envs=4, num_tickers=3, shuffle_tickers=False)

   # Random 3 tickers, different per env
   env = VecFastFinRL(path, n_envs=4, num_tickers=3, shuffle_tickers=True)
