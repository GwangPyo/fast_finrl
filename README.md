# FastFinRL

High-performance C++ implementation of FinRL StockTradingEnv.

## Installation

```bash
pip install fast-finrl
```

Build from source:
```bash
sudo apt install -y cmake g++ libtbb-dev  # Ubuntu
pip install .
```

---

## Usage

```python
from fast_finrl_py import FastFinRL

env = FastFinRL("data/stock.csv")
state = env.reset(["AAPL", "GOOGL"], seed=42)

while not state["done"]:
    actions = model.predict(state)  # range: [-1, 1]
    state = env.step(actions)
    reward = state["reward"]
```

---

## Constructor

```python
FastFinRL(
    csv_path,                      # CSV or Parquet file path
    initial_amount=30000.0,        # Initial capital
    hmax=15,                       # Max trade quantity (action * hmax)
    buy_cost_pct=0.01,             # Buy transaction fee
    sell_cost_pct=0.01,            # Sell transaction fee
    stop_loss_tolerance=0.8,       # Stop-loss threshold (0.8 = sell at 20% loss)
    bidding="default",             # Fill price policy
    tech_indicator_list=[],        # Indicators to use (empty = auto-detect from CSV columns)
    macro_tickers=[]               # Tickers always included in state["macro"] section
)
```

**Notes:**
- `tech_indicator_list`: Order determines indicator array order in `get_market_window_numpy()`. Empty list auto-detects all non-OHLC numeric columns from CSV.
- `macro_tickers`: Always included in state regardless of trading tickers. Useful for market indicators like VIX, TLT. Can overlap with trading tickers (appears in both `market` and `macro` sections).

**Bidding options:**
| Value | Buy Price | Sell Price |
|-------|-----------|------------|
| default | close | close |
| uniform | random(low, high) | random(low, high) |
| adv_uniform | random(max(open,close), high) | random(low, min(open,close)) |

---

## API

### reset(ticker_list, seed, shifted_start=0)

Initialize episode.

```python
state = env.reset(["AAPL", "GOOGL"], seed=42, shifted_start=100)
```

- `ticker_list`: Tickers to trade. **This order determines action order in step()**
- `seed`: Random seed (-1 = previous seed + 1)
- `shifted_start`: Delay start by N days

### step(actions)

Execute one step.

```python
# Order matches ticker_list from reset()
# If reset(["AAPL", "GOOGL"], ...), then:
state = env.step([0.5, -0.3])  # actions[0]=AAPL, actions[1]=GOOGL
```

- `actions`: List of values in [-1, 1], **same order as ticker_list in reset()**
- Positive=buy, Negative=sell, 0=hold

### get_market_window_numpy(ticker_list, day, h, future)

Get market data for ML models. **Returns zero-copy numpy arrays.**

```python
data = env.get_market_window_numpy(["AAPL"], day=500, h=100, future=20)

# Usage:
past_prices = data["AAPL"]["past_ohlc"]      # shape: (100, 4)
past_mask = data["AAPL"]["past_mask"]        # shape: (100,), 1=valid, 0=padding
future_prices = data["AAPL"]["future_ohlc"]  # shape: (20, 4)
```

### Utilities

```python
env.get_all_tickers()      # All available tickers
env.get_macro_tickers()    # Macro tickers (always in state)
env.get_max_day()          # Maximum day index
env.get_indicator_names()  # Technical indicator column names
```

---

## State Structure

### reset() return value
```python
{
    "day": 150,
    "date": "2023-06-15",
    "portfolio": {
        "cash": 100000.0,
        "holdings": {"AAPL": {"shares": 0, "avg_buy_price": 0.0}}
    },
    "market": {
        "AAPL": {"open": 150, "high": 152, "low": 149, "close": 151,
                 "indicators": {"macd": 0.5, "rsi_14": 55}}
    },
    "macro": {  # Only present if macro_tickers configured
        "VIX": {"open": 15, "high": 16, "low": 14, "close": 15,
                "indicators": {"macd": -0.1, "rsi_14": 40}}
    }
}
```

### step() return value
```python
{
    "day": 151,
    "reward": 0.0023,
    "done": False,
    "terminal": False,
    "portfolio": {"cash": 95000, "total_asset": 100230, ...},
    "market": {...},
    "info": {"n_trades": 5, "num_stop_loss": 0},
    "debug": {"AAPL": {"fill_price": 151, "cost": 15.1, "quantity": 10}}
}
```

---

## Data Format

Required CSV columns:

```csv
day,date,tic,open,high,low,close,volume,macd,rsi_14
0,2023-01-01,AAPL,150,152,149,151,1000000,0.5,55
0,2023-01-01,GOOGL,2800,2850,2790,2820,500000,-0.3,52
1,2023-01-02,AAPL,151,153,150,152,1100000,0.7,58
```

- `day`: Time index (starts from 0)
- `tic`: Ticker symbol
- Other numeric columns: Automatically detected as technical indicators

---

## ReplayBuffer

C++ implementation of experience replay buffer with on-demand market data fetching.

```python
from fast_finrl_py import FastFinRL, ReplayBuffer

env = FastFinRL("data/stock.csv")
buffer = ReplayBuffer(env)  # capacity=1M, batch_size=256
```

### Constructor

```python
ReplayBuffer(
    env,                    # FastFinRL instance
    capacity=1_000_000,     # 100K (small), 1M (default), 5M (large)
    batch_size=256          # Default batch size for sample()
)
```

### add(state, action, reward, next_state, done)

```python
state = env.reset(tickers, seed=42)
while not state["done"]:
    action = model.predict(state)
    next_state = env.step(action)

    # Explicit interface
    buffer.add(state, action, next_state["reward"], next_state, next_state["done"])

    state = next_state
```

Multi-objective reward:
```python
reward = [return_reward, sharpe_reward, risk_penalty]
buffer.add(state, action, reward, next_state, done)
```

### sample(h, batch_size=None)

```python
s, a, r, s_next, done, s_mask, s_next_mask = buffer.sample(h=20)
```

**Returns:**
| Variable | Shape | Description |
|----------|-------|-------------|
| `s[ticker]["ohlc"]` | (batch, h+1, 4) | OHLC prices |
| `s[ticker]["indicators"]` | (batch, h+1, n_ind) | Technical indicators |
| `s["macro"][ticker]["ohlc"]` | (batch, h+1, 4) | Macro ticker OHLC |
| `s["portfolio"]["cash"]` | (batch,) | Cash balance |
| `s["portfolio"]["shares"]` | (batch, n_tickers) | Share holdings |
| `a` | (batch, n_tickers) | Actions |
| `r` | (batch, reward_size) | Rewards (always 2D) |
| `done` | (batch, 1) | Done flags (always 2D) |
| `s_mask[ticker]` | (batch, h+1) or None | Mask (None if h=0) |
| `s_mask["macro"][ticker]` | (batch, h+1) or None | Macro mask |

### save / load

```python
buffer.save("buffer.json")
buffer.load("buffer.json")
```

---

## Example: Training Loop

```python
env = FastFinRL("train.csv", initial_amount=100000, hmax=30)
buffer = ReplayBuffer(env, capacity=1_000_000, batch_size=256)
tickers = list(env.get_all_tickers())

# Collect experience
for episode in range(100):
    state = env.reset(tickers, seed=episode)
    while not state["done"]:
        action = model.predict(state)
        next_state = env.step(action)
        buffer.add(state, action, next_state["reward"], next_state, next_state["done"])
        state = next_state

# Train
for step in range(10000):
    s, a, r, s_next, done, s_mask, _ = buffer.sample(h=50)
    loss = model.train(s, a, r, s_next, done)
```

---

## VecFastFinRL (Vectorized Environment)

N parallel environments with TBB parallelization.

```python
from fast_finrl_py import VecFastFinRL
import numpy as np

vec_env = VecFastFinRL("data/stock.csv")
```

### Constructor

```python
VecFastFinRL(
    csv_path,                      # CSV or Parquet file path
    initial_amount=30000.0,        # Initial capital per env
    hmax=15,                       # Max trade quantity (action * hmax)
    buy_cost_pct=0.01,             # Buy transaction fee
    sell_cost_pct=0.01,            # Sell transaction fee
    stop_loss_tolerance=0.8,       # Stop-loss threshold
    bidding="default",             # Fill price policy: "default", "uniform", "adv_uniform"
    stop_loss_calculation="close", # Stop-loss reference: "close" or "low"
    tech_indicator_list=[],        # Indicators (empty = auto-detect)
    macro_tickers=[],              # Macro tickers for state["macro_ohlc"]
    auto_reset=True                # Auto-reset done envs with seed+1
)
```

### reset(tickers_list, seeds)

Initialize N environments. Each environment can trade different tickers (count must match).

```python
tickers_list = [
    ["AAPL", "GOOGL"],  # env 0
    ["MSFT", "AMZN"],   # env 1
    ["AAPL", "MSFT"],   # env 2
]
seeds = np.arange(3, dtype=np.int64)
states = vec_env.reset(tickers_list, seeds)  # List[dict], len=3
```

- `tickers_list`: `List[List[str]]` - shape (N, n_tickers). All envs must have same n_tickers.
- `seeds`: `np.ndarray[int64]` - shape (N,). Random seed per env.
- **Returns**: `List[dict]` - state dict per env.

### step(actions)

Execute one step for all environments.

```python
actions = np.array([
    [0.5, -0.3],   # env 0: buy AAPL, sell GOOGL
    [0.0, 1.0],    # env 1: hold MSFT, buy AMZN
    [-1.0, 0.5],   # env 2: sell AAPL, buy MSFT
])
states = vec_env.step(actions)  # List[dict], len=3
```

- `actions`: `np.ndarray[float64]` - shape (N, n_tickers). Range [-1, 1].
- **Returns**: `List[dict]` - state dict per env with `reward`, `done`, `terminal`.

### reset_indices(indices, seeds)

Partial reset - reset only specified environment indices.

```python
# Reset only env 1 and 3
states = vec_env.reset_indices([1, 3], np.array([100, 200], dtype=np.int64))
```

- `indices`: `List[int]` - env indices to reset.
- `seeds`: `np.ndarray[int64]` - seeds for each index (same length as indices).
- **Returns**: `List[dict]` - full state list (all N envs). Only specified indices are modified.

### set_auto_reset(enabled) / auto_reset()

Control auto-reset behavior. When enabled, done environments automatically reset with seed+1.

```python
vec_env.set_auto_reset(False)  # Disable auto-reset
is_enabled = vec_env.auto_reset()  # Returns bool
```

### Accessor Methods

```python
vec_env.num_envs()         # int: Number of environments (N)
vec_env.n_tickers()        # int: Tickers per env
vec_env.n_indicators()     # int: Number of technical indicators
vec_env.n_macro()          # int: Number of macro tickers
vec_env.get_all_tickers()  # set[str]: All available tickers in CSV
vec_env.get_macro_tickers() # List[str]: Configured macro tickers
vec_env.get_tickers()      # List[List[str]]: Tickers per env (shape N x n_tickers)
```

### State Structure

Each state dict in the returned list:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `day` | int | - | Current day index |
| `cash` | float | - | Cash balance |
| `total_asset` | float | - | Cash + holdings value |
| `shares` | np.ndarray | (n_tickers,) | Shares held per ticker |
| `avg_buy_price` | np.ndarray | (n_tickers,) | Average buy price per ticker |
| `ohlc` | np.ndarray | (n_tickers, 4) | OHLC prices [open, high, low, close] |
| `indicators` | np.ndarray | (n_tickers, n_ind) | Technical indicators |
| `macro_ohlc` | np.ndarray | (n_macro, 4) | Macro ticker OHLC (if configured) |
| `macro_indicators` | np.ndarray | (n_macro, n_ind) | Macro indicators (if configured) |
| `tickers` | List[str] | (n_tickers,) | Ticker symbols for this env |
| `done` | bool | - | Episode finished |
| `terminal` | bool | - | Reached max day (vs early stop) |
| `reward` | float | - | Step reward (0.0 on reset) |

---

## VecReplayBuffer

Replay buffer for vectorized environments. Stores transitions with env_id for tracking.

```python
from fast_finrl_py import FastFinRL, VecReplayBuffer

env = FastFinRL("data/stock.csv")  # For market data lookup
buffer = VecReplayBuffer(env, capacity=1_000_000, batch_size=256)
```

### Constructor

```python
VecReplayBuffer(
    env,                    # FastFinRL instance (for market data lookup)
    capacity=1_000_000,     # Max transitions to store
    batch_size=256          # Default batch size for sample()
)
```

### add_batch(states, actions, rewards, next_states, dones)

Add N transitions from VecFastFinRL step.

```python
# states/next_states: List[dict] from VecFastFinRL (must have .day, .cash, .shares, .avg_buy_price, .tickers, .terminal)
# Use wrapper if needed:
class DictWrapper:
    def __init__(self, d): self._d = d
    def __getattr__(self, name): return self._d[name]

states_wrapped = [DictWrapper(s) for s in states]
next_states_wrapped = [DictWrapper(s) for s in next_states]

buffer.add_batch(
    states_wrapped,          # List[obj]: N state objects with .attr access
    actions,                 # np.ndarray: shape (N, n_tickers)
    rewards,                 # List[float] or List[List[float]]: rewards per env
    next_states_wrapped,     # List[obj]: N next_state objects
    dones                    # List[bool]: done flags per env
)
```

Multi-objective rewards supported:
```python
rewards = [[r1, r2, r3] for _ in range(N)]  # 3 objectives
buffer.add_batch(states, actions, rewards, next_states, dones)
```

### sample(h, batch_size=None)

Sample batch with market data windows.

```python
s, a, r, s_next, done, s_mask, s_next_mask = buffer.sample(h=10, batch_size=256)
```

**Returns tuple of 7 elements:**

| Variable | Type | Description |
|----------|------|-------------|
| `s` | dict | State market data + portfolio |
| `a` | np.ndarray (batch, n_tickers) | Actions |
| `r` | np.ndarray (batch, n_obj) | Rewards |
| `s_next` | dict | Next state market data + portfolio |
| `done` | np.ndarray (batch, 1) | Done flags |
| `s_mask` | dict or None | Masks (None if h=0) |
| `s_next_mask` | dict or None | Next state masks |

**State dict structure (`s` and `s_next`):**

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `s[ticker]["ohlc"]` | np.ndarray | (batch, h+1, 4) | OHLC window |
| `s[ticker]["indicators"]` | np.ndarray | (batch, h+1, n_ind) | Indicator window |
| `s["macro"][ticker]["ohlc"]` | np.ndarray | (batch, h+1, 4) | Macro OHLC |
| `s["macro"][ticker]["indicators"]` | np.ndarray | (batch, h+1, n_ind) | Macro indicators |
| `s["portfolio"]["cash"]` | np.ndarray | (batch,) | Cash balance |
| `s["portfolio"]["shares"]` | np.ndarray | (batch, n_tickers) | Share holdings |
| `s["portfolio"]["avg_buy_price"]` | np.ndarray | (batch, n_tickers) | Avg buy prices |
| `s["env_ids"]` | np.ndarray | (batch,) | Original env indices |
| `s["tickers"]` | List[str] | (n_tickers,) | Ticker names |
| `s["macro_tickers"]` | List[str] | (n_macro,) | Macro ticker names |
| `s["indicator_names"]` | List[str] | (n_ind,) | Indicator names |

**Mask dict structure (`s_mask`):**

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `s_mask[ticker]` | np.ndarray | (batch, h+1) | 1=valid, 0=padding |
| `s_mask["macro"][ticker]` | np.ndarray | (batch, h+1) | Macro mask |

### Utility Methods

```python
buffer.size()              # int: Current number of transitions
buffer.capacity()          # int: Maximum capacity
buffer.clear()             # Clear all transitions
buffer.save("path.json")   # Save buffer to JSON file
buffer.load("path.json")   # Load buffer from JSON file
```

---

## Example: VecFastFinRL Training Loop

```python
from fast_finrl_py import FastFinRL, VecFastFinRL, VecReplayBuffer
import numpy as np

# Setup
env = FastFinRL("train.csv")
vec_env = VecFastFinRL("train.csv", auto_reset=True)
buffer = VecReplayBuffer(env, capacity=1_000_000, batch_size=256)

N = 64
all_tickers = list(vec_env.get_all_tickers())
tickers_list = [all_tickers[:4] for _ in range(N)]

# Wrapper for add_batch
class W:
    def __init__(self, d): self._d = d
    def __getattr__(self, n): return self._d[n]

# Collect experience
states = vec_env.reset(tickers_list, np.arange(N, dtype=np.int64))
for step in range(10000):
    actions = model.predict(states)  # (N, 4)
    next_states = vec_env.step(actions)

    buffer.add_batch(
        [W(s) for s in states],
        actions,
        [s["reward"] for s in next_states],
        [W(s) for s in next_states],
        [s["done"] for s in next_states]
    )
    states = next_states

    # Train every 100 steps
    if step % 100 == 0 and buffer.size() > 1000:
        s, a, r, s_next, done, _, _ = buffer.sample(h=20)
        loss = model.train(s, a, r, s_next, done)
```

