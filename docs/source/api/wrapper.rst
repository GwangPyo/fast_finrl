VecFinRLWrapper
===============

Gymnasium-compatible wrapper for VecFastFinRL with integrated replay buffer.

.. py:class:: VecFinRLWrapper(path, buffer_capacity, history_length=20, auto_add=True, batch_size=256, n_envs=4, initial_amount=30000.0, failure_threshold=25000.0, hmax=15, buy_cost_pct=0.01, sell_cost_pct=0.01, stop_loss_tolerance=0.8, bidding='adv_uniform', stop_loss_calculation='close', initial_seed=0, tech_indicator_list=[], macro_tickers=[], auto_reset=True, num_tickers=0, shuffle_tickers=True)

   Create a Gymnasium-compatible wrapper for vectorized trading environment.

   :param Path path: Path to data file (CSV or Parquet)
   :param int buffer_capacity: Replay buffer capacity
   :param int history_length: Default history window length (default: 20)
   :param bool auto_add: Auto-add transitions to buffer on step (default: True)
   :param int batch_size: Default buffer sample batch size (default: 256)
   :param int n_envs: Number of parallel environments (default: 4)
   :param float initial_amount: Initial cash amount (default: 30000.0)
   :param float failure_threshold: Episode terminates if total_asset <= this (default: 25000.0)
   :param int hmax: Maximum shares per trade (default: 15)
   :param float buy_cost_pct: Buy transaction cost percentage (default: 0.01)
   :param float sell_cost_pct: Sell transaction cost percentage (default: 0.01)
   :param float stop_loss_tolerance: Stop loss tolerance (default: 0.8)
   :param str bidding: Price execution mode (default: 'adv_uniform')
   :param str stop_loss_calculation: Stop loss calculation method (default: 'close')
   :param int initial_seed: Random seed (default: 0)
   :param list tech_indicator_list: Technical indicators to use
   :param list macro_tickers: Macro/index tickers for context (required)
   :param bool auto_reset: Auto-reset on done (default: True)
   :param int num_tickers: Number of tickers per env (0 = all)
   :param bool shuffle_tickers: Shuffle ticker assignment (default: True)

   .. py:method:: reset(seed=None, options=None)

      Reset all environments.

      :param int seed: Random seed
      :param dict options: Additional reset options
      :return: Tuple (obs, info)

   .. py:method:: step(action)

      Step all environments.

      :param ndarray action: Actions [n_envs, num_tickers, 2]
      :return: Tuple (obs, reward, done, truncated, info)

   .. py:method:: sample_actions()

      Sample random actions.

      :return: ndarray [n_envs, num_tickers, 2]

   .. py:method:: sample_buffer(batch_size=None, history_length=None, future_length=None)

      Sample from replay buffer with structured output.

      :param int batch_size: Override default batch size
      :param int history_length: Override default history length
      :param int future_length: Future window length (default: 0)
      :return: Tuple (obs, action, reward, next_obs, done, mask, mask_next)

Observation Structure
---------------------

**From reset() and step():**

.. code-block:: python

   obs['portfolio']['cash']       # [n_envs, 1] - Cash amounts
   obs['portfolio']['shares']     # [n_envs, n_tickers] - Share holdings

   obs['market']['open']          # [n_envs, n_tickers] - Current open prices
   obs['market']['indicators']    # [n_envs, n_tickers, n_ind] - Current indicators

   obs['macro']['open']           # [n_envs, n_macro] - Macro open prices
   obs['macro']['indicators']     # [n_envs, n_macro, n_ind] - Macro indicators

   obs['tics']                    # [n_envs, n_tickers] - Tokenized ticker IDs
   obs['macro_tics']              # [n_envs, n_macro] - Tokenized macro ticker IDs

   # Market history
   obs['hist']['market']['ohlcvs']     # [n_envs, n_tickers, H, 5] - History OHLCV
   obs['hist']['market']['indicators'] # [n_envs, n_tickers, H, n_ind] - History indicators
   obs['hist']['market']['masks']      # [n_envs, n_tickers, H] - History validity masks
   obs['hist']['market']['tickers']    # [n_envs, n_tickers] - Tokenized tickers

   # Macro history
   obs['hist']['macro']['ohlcvs']      # [n_envs, n_macro, H, 5] - Macro history OHLCV
   obs['hist']['macro']['indicators']  # [n_envs, n_macro, H, n_ind] - Macro history indicators
   obs['hist']['macro']['masks']       # [n_envs, n_macro, H] - Macro history validity masks
   obs['hist']['macro']['tickers']     # [n_envs, n_macro] - Tokenized macro tickers

sample_buffer() Return Format
-----------------------------

.. code-block:: python

   obs, action, reward, next_obs, done, mask, mask_next = wrapper.sample_buffer(
       batch_size=256, history_length=20, future_length=5
   )

**Observation dict (obs, next_obs):**

.. code-block:: python

   obs['portfolio']['cash']       # [B, 1]
   obs['portfolio']['shares']     # [B, n_tickers]

   obs['market']['open']          # [B, n_tickers]
   obs['market']['indicators']    # [B, n_tickers, n_ind]

   obs['macro']['open']           # [B, n_macro]
   obs['macro']['indicators']     # [B, n_macro, n_ind]

   obs['tics']                    # [B, n_tickers] - Tokenized tickers
   obs['macro_tics']              # [B, n_macro] - Tokenized macro tickers

   # Market history
   obs['hist']['market']['ohlcvs']     # [B, n_tickers, H, 5]
   obs['hist']['market']['indicators'] # [B, n_tickers, H, n_ind]
   obs['hist']['market']['masks']      # [B, n_tickers, H]
   obs['hist']['market']['tickers']    # [B, n_tickers]

   # Macro history
   obs['hist']['macro']['ohlcvs']      # [B, n_macro, H, 5]
   obs['hist']['macro']['indicators']  # [B, n_macro, H, n_ind]
   obs['hist']['macro']['masks']       # [B, n_macro, H]
   obs['hist']['macro']['tickers']     # [B, n_macro]

   # Future (only if future_length > 0)
   obs['future']['market']['ohlcvs']     # [B, n_tickers, F, 5]
   obs['future']['market']['indicators'] # [B, n_tickers, F, n_ind]
   obs['future']['market']['masks']      # [B, n_tickers, F]
   obs['future']['market']['tickers']    # [B, n_tickers]

   obs['future']['macro']['ohlcvs']      # [B, n_macro, F, 5]
   obs['future']['macro']['indicators']  # [B, n_macro, F, n_ind]
   obs['future']['macro']['masks']       # [B, n_macro, F]
   obs['future']['macro']['tickers']     # [B, n_macro]

**Mask dict (mask, mask_next):**

.. code-block:: python

   mask['market']          # [B, n_tickers, H] - Market history validity
   mask['macro']           # [B, n_macro, H] - Macro history validity

   # Future (only if future_length > 0)
   mask['future']['market'] # [B, n_tickers, F]
   mask['future']['macro']  # [B, n_macro, F]

**Action, Reward, Done:**

.. code-block:: python

   action  # [B, n_tickers, 2]
   reward  # [B]
   done    # [B]

TickerTokenizer
---------------

.. py:class:: TickerTokenizer

   Converts ticker symbols to integer IDs and back.

   .. py:method:: encode(tic)

      Encode single ticker to ID.

      :param str tic: Ticker symbol
      :return: Integer ID (1-indexed, 0 reserved for <CASH>)

   .. py:method:: encode_batch(tics)

      Encode batch of tickers.

      :param tics: List of tickers or list of lists
      :return: ndarray of IDs

   .. py:method:: decode(token_id)

      Decode ID to ticker symbol.

      :param int token_id: Token ID
      :return: Ticker symbol (or "<CASH>" for 0)

   .. py:method:: decode_batch(token_ids)

      Decode batch of IDs.

      :param ndarray token_ids: Token IDs
      :return: List of ticker symbols

   .. py:attribute:: vocab_size
      :type: int

      Current vocabulary size

   .. py:method:: save(path)

      Save tokenizer to JSON file.

   .. py:method:: load(path)

      Load tokenizer from JSON file.

Usage Example
-------------

.. code-block:: python

   from vec_finrl_wrapper import VecFinRLWrapper
   from pathlib import Path

   # Create wrapper with macro tickers
   wrapper = VecFinRLWrapper(
       path=Path('data.csv'),
       buffer_capacity=100000,
       history_length=20,
       n_envs=4,
       num_tickers=10,
       macro_tickers=['SPY', 'QQQ', 'VIX'],
   )

   # Standard Gymnasium interface
   obs, info = wrapper.reset(seed=42)

   for _ in range(1000):
       action = wrapper.sample_actions()  # [4, 10, 2]
       obs, reward, done, truncated, info = wrapper.step(action)

   # Sample from buffer with future prediction data
   obs, action, reward, next_obs, done, mask, mask_next = wrapper.sample_buffer(
       batch_size=256,
       history_length=20,
       future_length=5
   )

   # Access structured data
   market_history = obs['hist']['past']['ohlcvs']      # [256, 10, 20, 5]
   market_future = obs['future']['market']['ohlcvs']   # [256, 10, 5, 5]
   macro_future = obs['future']['macro']['ohlcvs']     # [256, 3, 5, 5]

   # Use tokenized tickers for embedding
   ticker_ids = obs['tics']  # [256, 10]
