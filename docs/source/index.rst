FastFinRL Documentation
=======================

FastFinRL is a high-performance reinforcement learning environment for financial trading,
implemented in C++ with Python bindings.

Features
--------

- **High Performance**: C++ core with zero-copy numpy arrays
- **Vectorized Environments**: Run multiple environments in parallel
- **Replay Buffers**: Efficient experience replay with market data retrieval
- **Flexible Bidding**: Multiple price execution modes (default, uniform, adv_uniform)
- **Multiple Data Formats**: CSV and Parquet support (auto-detected)

Quick Start
-----------

.. code-block:: python

   import fast_finrl_py as ff
   import numpy as np

   # Single environment
   env = ff.FastFinRL('data.csv', return_format='vec')
   state = env.reset(['AAPL', 'GOOGL'], seed=42)
   next_state = env.step([0.5, -0.3])

   # Vectorized environment
   vec_env = ff.VecFastFinRL('data.csv', n_envs=4, return_format='vec')
   states = vec_env.reset(seed=42)
   actions = np.random.uniform(-1, 1, (4, vec_env.n_tickers()))
   next_states = vec_env.step(actions)

   # Replay buffer
   buffer = ff.VecReplayBuffer(vec_env, capacity=100000)
   buffer.add(states, actions, next_states['reward'], next_states, next_states['done'])
   batch = buffer.sample(h=10)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/fastfinrl
   api/vecfastfinrl
   api/replaybuffer
   api/vecreplaybuffer
   api/wrapper

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
