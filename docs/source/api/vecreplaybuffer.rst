VecReplayBuffer
===============

Experience replay buffer for vectorized multi-environment training.

.. py:class:: fast_finrl_py.VecReplayBuffer(env, capacity=1000000, batch_size=256, seed=42, action_shape=None)

   Create a replay buffer linked to a VecFastFinRL environment.

   :param VecFastFinRL env: Vectorized environment instance
   :param int capacity: Maximum buffer size (default: 1000000)
   :param int batch_size: Default sample batch size (default: 256)
   :param int seed: Random seed for sampling (-1 for random_device)
   :param tuple action_shape: Custom action shape (default: (n_tickers,))

   .. py:method:: add(states, actions, rewards, next_states, dones)

      Add batch of transitions to the buffer.

      Accepts two formats (auto-detected):

      - **Vec format**: Batched dicts from VecFastFinRL with ``return_format='vec'``
      - **Object list**: List of state objects with attributes

      :param states: Current states (dict or list)
      :param ndarray actions: Actions [n_envs, ...]
      :param rewards: Rewards (ndarray or list)
      :param next_states: Next states (dict or list)
      :param dones: Done flags (ndarray or list)

   .. py:method:: sample(batch_size=None, history_length=0, future_length=0)

      Sample a batch of transitions with market history and future data.

      :param int batch_size: Override default batch size
      :param int history_length: History window length (default: 0)
      :param int future_length: Future window length (default: 0)
      :return: Tuple (s, action, reward, s_next, done, mask, mask_next)

   .. py:method:: sample_indices(batch_size)

      Sample random indices from buffer.

      :param int batch_size: Number of indices to sample
      :return: List of indices

   .. py:method:: get(index)

      Get transition by index.

      :param int index: Buffer index
      :return: VecStoredTransition object

   .. py:method:: get_market_data(index, h, future, next_state=False)

      Get market data for a specific transition.

      :param int index: Buffer index
      :param int h: History window length
      :param int future: Future window length
      :param bool next_state: Use next_state day instead of state day
      :return: Dict with per-ticker market data arrays

   .. py:method:: size()

      Get current buffer size.

      :return: Number of stored transitions

   .. py:method:: capacity()

      Get buffer capacity.

      :return: Maximum buffer size

   .. py:method:: clear()

      Clear all transitions from buffer.

   .. py:method:: save(path)

      Save buffer to file.

      :param str path: File path

   .. py:method:: load(path)

      Load buffer from file.

      :param str path: File path

sample() Return Format
----------------------

.. code-block:: python

   s, action, reward, s_next, done, mask, mask_next = buffer.sample(
       batch_size=256, history_length=20, future_length=5
   )

**State dict (s, s_next):**

.. code-block:: python

   # Per-ticker market data
   s['AAPL']['ohlcv']              # [B, H, 5] - OHLCV history
   s['AAPL']['indicators']         # [B, H, n_ind] - Indicator history
   s['AAPL']['future_ohlcv']       # [B, F, 5] - OHLCV future (if future_length > 0)
   s['AAPL']['future_indicators']  # [B, F, n_ind] - Indicator future
   s['AAPL']['future_mask']        # [B, F] - Future validity mask

   # Macro ticker data (same structure)
   s['macro']['SPY']['ohlcv']           # [B, H, 5]
   s['macro']['SPY']['indicators']      # [B, H, n_ind]
   s['macro']['SPY']['future_ohlcv']    # [B, F, 5]
   s['macro']['SPY']['future_indicators']# [B, F, n_ind]
   s['macro']['SPY']['future_mask']     # [B, F]

   # Portfolio state
   s['portfolio']['cash']          # [B] - Cash amounts
   s['portfolio']['shares']        # [B, n_tickers] - Share holdings
   s['portfolio']['avg_buy_price'] # [B, n_tickers] - Average buy prices

   # Metadata
   s['tickers']           # [B][n_tickers] - Ticker lists per sample
   s['unique_tickers']    # List of unique tickers in batch
   s['macro_tickers']     # List of macro tickers
   s['indicator_names']   # List of indicator names
   s['action_shape']      # Action shape tuple
   s['env_ids']           # [B] - Environment IDs

**Action, Reward, Done:**

.. code-block:: python

   action  # [B, *action_shape] - Reshaped by action_shape
   reward  # [B, n_objectives] - Multi-objective rewards
   done    # [B] - Done flags

**Mask dict (mask, mask_next):**

.. code-block:: python

   mask['AAPL']        # [B, H] - History validity mask per ticker
   mask['macro']['SPY'] # [B, H] - Macro ticker history mask

Usage Example
-------------

.. code-block:: python

   import fast_finrl_py as ff
   import numpy as np

   # Create environment with macro tickers
   env = ff.VecFastFinRL(
       'data.csv',
       n_envs=4,
       return_format='vec',
       macro_tickers=['SPY', 'QQQ']
   )

   # Create buffer with custom action shape
   buffer = ff.VecReplayBuffer(
       env,
       capacity=100000,
       batch_size=256,
       action_shape=(env.n_tickers(), 2)  # [n_tickers, 2] per action
   )

   # Collect experience
   state = env.reset(seed=42)
   for _ in range(1000):
       actions = np.random.uniform(-1, 1, (4, env.n_tickers(), 2))
       next_state = env.step(actions[..., 0])  # env uses first dim
       buffer.add(state, actions, next_state['reward'], next_state, next_state['done'])
       state = next_state

   # Sample with history and future
   s, action, reward, s_next, done, mask, mask_next = buffer.sample(
       batch_size=128,
       history_length=20,
       future_length=5
   )

   # Access market data
   print(s['AAPL']['ohlcv'].shape)        # (128, 20, 5)
   print(s['AAPL']['future_ohlcv'].shape) # (128, 5, 5)

   # Access macro data
   print(s['macro']['SPY']['ohlcv'].shape)        # (128, 20, 5)
   print(s['macro']['SPY']['future_ohlcv'].shape) # (128, 5, 5)

   # Action is reshaped by action_shape
   print(action.shape)  # (128, n_tickers, 2)

VecStoredTransition
-------------------

.. py:class:: fast_finrl_py.VecStoredTransition

   Vectorized transition data structure.

   .. py:attribute:: env_id
      :type: int

      Environment index

   .. py:attribute:: state_day
      :type: int

      State day index

   .. py:attribute:: next_state_day
      :type: int

      Next state day index

   .. py:attribute:: tickers
      :type: list

      Ticker symbols

   .. py:attribute:: state_cash
      :type: float

      State cash amount

   .. py:attribute:: next_state_cash
      :type: float

      Next state cash amount

   .. py:attribute:: state_shares
      :type: list

      State share holdings

   .. py:attribute:: next_state_shares
      :type: list

      Next state share holdings

   .. py:attribute:: action
      :type: list

      Action taken (flat)

   .. py:attribute:: rewards
      :type: list

      Rewards (supports multi-objective)

   .. py:attribute:: done
      :type: bool

      Done flag

   .. py:attribute:: terminal
      :type: bool

      Terminal flag
