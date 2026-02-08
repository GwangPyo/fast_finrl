ReplayBuffer
============

Experience replay buffer for single-environment training.

.. py:class:: fast_finrl_py.ReplayBuffer(env, capacity=1000000, batch_size=256, seed=42)

   Create a replay buffer linked to a FastFinRL environment.

   :param FastFinRL env: Environment instance (for market data retrieval)
   :param int capacity: Maximum buffer size (default: 1000000)
   :param int batch_size: Default sample batch size (default: 256)
   :param int seed: Random seed for sampling

   .. py:method:: add(state, action, reward, next_state, done)

      Add a transition to the buffer.

      :param dict state: Current state (json or vec format, auto-detected)
      :param list action: Action taken
      :param float reward: Reward received
      :param dict next_state: Next state
      :param bool done: Episode done flag

   .. py:method:: sample(h=0, batch_size=None)

      Sample a batch of transitions with market history.

      :param int h: History window length (default: 0)
      :param int batch_size: Override default batch size
      :return: Tuple (s, a, r, s_next, dones, mask, mask_next)

   .. py:method:: sample_indices(batch_size)

      Sample random indices from buffer.

      :param int batch_size: Number of indices to sample
      :return: List of indices

   .. py:method:: get(index)

      Get transition by index.

      :param int index: Buffer index
      :return: StoredTransition object

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

Sample Return Format
--------------------

The ``sample()`` method returns a tuple:

.. code-block:: python

   s, a, r, s_next, dones, mask, mask_next = buffer.sample(h=10)

   # s, s_next: Dict[ticker] -> {'ohlcv': [B, h, 5], 'indicators': [B, h, n_ind]}
   # a: [B, n_tickers] actions
   # r: [B] rewards
   # dones: [B] done flags
   # mask, mask_next: Dict[ticker] -> [B, h] validity masks (None if h=0)

StoredTransition
----------------

.. py:class:: fast_finrl_py.StoredTransition

   Single transition data structure.

   .. py:attribute:: day
      :type: int

      Day index

   .. py:attribute:: tickers
      :type: list

      Ticker symbols

   .. py:attribute:: cash
      :type: float

      Cash amount

   .. py:attribute:: shares
      :type: list

      Share holdings

   .. py:attribute:: action
      :type: list

      Action taken

   .. py:attribute:: reward
      :type: float

      Reward received

   .. py:attribute:: done
      :type: bool

      Done flag
