# FastFinRL

High-performance C++ implementation of FinRL StockTradingEnv.

## Installation

```bash
pip install  git+https://github.com/GwangPyo/fast_finrl
```

Build from source:
```bash
# Ubuntu/Debian - install dependencies
sudo apt install -y \
    cmake g++ \
    nlohmann-json3-dev \
    pybind11-dev \
    libtbb-dev \
    libarrow-dev \
    libparquet-dev

pip install .
```

---

## Quick Start

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

# FastFinRL

Single-environment trading simulator.

## Constructor

```python
FastFinRL(
    csv_path: str,
    initial_amount: float = 30000.0,
    hmax: int = 15,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    stop_loss_tolerance: float = 0.8,
    bidding: str = "default",
    stop_loss_calculation: str = "close",
    initial_seed: int = 0,
    tech_indicator_list: List[str] = [],
    macro_tickers: List[str] = [],
    return_format: str = "json"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | str | required | Path to CSV or Parquet data file |
| `initial_amount` | float | 30000.0 | Starting capital |
| `hmax` | int | 15 | Max trade quantity per action (actual_qty = action * hmax) |
| `buy_cost_pct` | float | 0.01 | Buy transaction fee (1% = 0.01) |
| `sell_cost_pct` | float | 0.01 | Sell transaction fee |
| `stop_loss_tolerance` | float | 0.8 | Stop-loss threshold (0.8 = sell at 20% loss from avg_buy_price) |
| `bidding` | str | "default" | Fill price policy (see table below) |
| `stop_loss_calculation` | str | "close" | Price used for stop-loss check: "close" or "low" |
| `initial_seed` | int | 0 | Initial random seed |
| `tech_indicator_list` | List[str] | [] | Indicator columns to use. Empty = auto-detect all non-OHLC numeric columns |
| `macro_tickers` | List[str] | [] | Tickers always included in state["macro"] regardless of trading tickers |
| `return_format` | str | "json" | State format: "json" (nested dict) or "vec" (flat numpy arrays) |

**Bidding policies:**

| Value | Buy Fill Price | Sell Fill Price |
|-------|----------------|-----------------|
| `"default"` | close | close |
| `"uniform"` | random(low, high) | random(low, high) |
| `"adv_uniform"` | random(max(open,close), high) | random(low, min(open,close)) |

---

## Methods

### reset(ticker_list, seed, shifted_start=0) -> state

Initialize episode with specified tickers.

```python
state = env.reset(["AAPL", "GOOGL"], seed=42, shifted_start=100)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker_list` | List[str] | Tickers to trade. **Order determines action order in step()** |
| `seed` | int | Random seed. Use -1 for auto-increment (previous seed + 1) |
| `shifted_start` | int | Skip first N days before starting |

**Returns:** State dict (format depends on `return_format`)

---

### step(actions) -> state

Execute one trading step.

```python
state = env.step([0.5, -0.3])  # Buy AAPL, sell GOOGL
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `actions` | List[float] | Actions in [-1, 1]. **Order must match ticker_list from reset()** |

- Positive = buy (action * hmax shares)
- Negative = sell (abs(action) * hmax shares)
- Zero = hold

**Returns:** State dict with `reward`, `done`, `terminal`, `debug` fields

---

### get_market_window_numpy(ticker_list, day, h, future) -> dict

Fetch historical + future market data for ML models. **Zero-copy numpy arrays.**

```python
data = env.get_market_window_numpy(["AAPL", "GOOGL"], day=500, h=100, future=20)

# Per-ticker data:
data["AAPL"]["past_ohlc"]           # shape: (h, 4) = (100, 4)
data["AAPL"]["past_indicators"]     # shape: (h, n_ind)
data["AAPL"]["past_mask"]           # shape: (h,), 1=valid, 0=padding
data["AAPL"]["past_days"]           # shape: (h,), day indices

data["AAPL"]["current_ohlc"]        # shape: (4,)
data["AAPL"]["current_indicators"]  # shape: (n_ind,)
data["AAPL"]["current_mask"]        # int, 1 or 0
data["AAPL"]["current_day"]         # int

data["AAPL"]["future_ohlc"]         # shape: (future, 4) = (20, 4)
data["AAPL"]["future_indicators"]   # shape: (future, n_ind)
data["AAPL"]["future_mask"]         # shape: (future,)
data["AAPL"]["future_days"]         # shape: (future,)

# Metadata:
data["indicator_names"]  # List[str]
data["h"]                # int
data["future"]           # int
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker_list` | List[str] | Tickers to fetch |
| `day` | int | Reference day index |
| `h` | int | Number of past days (before current) |
| `future` | int | Number of future days (after current) |

---

### set_return_format(format) / get_return_format() -> str

Change state serialization format at runtime.

```python
env.set_return_format("vec")   # Switch to numpy array format
env.set_return_format("json")  # Switch back to nested dict format
fmt = env.get_return_format()  # Returns "json" or "vec"
```

---

### Accessor Methods

```python
env.get_all_tickers()       # set[str]: All unique tickers in CSV
env.get_indicator_names()   # set[str]: Technical indicator column names
env.get_macro_tickers()     # List[str]: Configured macro tickers
env.get_max_day()           # int: Maximum day index in data
env.get_state()             # dict: Current state (json format)
env.get_raw_value(ticker, day, column)  # float: Raw DataFrame value
```

---

### Public Attributes (read-write)

```python
env.initial_amount          # float
env.hmax                    # int
env.buy_cost_pct            # float
env.sell_cost_pct           # float
env.stop_loss_tolerance     # float
env.bidding                 # str
env.stop_loss_calculation   # str
```

---

## State Structure

### return_format="json" (default)

**reset() returns:**
```python
{
    "day": 150,
    "date": "2023-06-15",
    "seed": 42,
    "done": False,
    "terminal": False,
    "portfolio": {
        "cash": 100000.0,
        "holdings": {
            "AAPL": {"shares": 0, "avg_buy_price": 0.0},
            "GOOGL": {"shares": 0, "avg_buy_price": 0.0}
        }
    },
    "market": {
        "AAPL": {
            "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0,
            "indicators": {"macd": 0.5, "rsi_14": 55.0, ...}
        },
        "GOOGL": {...}
    },
    "macro": {  # Only if macro_tickers configured
        "VIX": {"open": 15.0, ..., "indicators": {...}}
    }
}
```

**step() returns (additional fields):**
```python
{
    ...,
    "reward": 0.0023,
    "portfolio": {
        "cash": 95000.0,
        "total_asset": 100230.0,
        "holdings": {...}
    },
    "info": {
        "loss_cut_amount": 0.0,
        "n_trades": 5,
        "num_stop_loss": 0
    },
    "debug": {
        "AAPL": {"fill_price": 151.0, "cost": 15.1, "quantity": 10},
        "GOOGL": {"fill_price": 2820.0, "cost": 14.1, "quantity": -5}
    }
}
```

### return_format="vec"

Flat dict with numpy arrays. No nested structure.

```python
{
    "day": 150,                              # int
    "date": "2023-06-15",                    # str
    "seed": 42,                              # int (reset only)
    "done": False,                           # bool
    "terminal": False,                       # bool
    "reward": 0.0023,                        # float (step only)
    "cash": 100000.0,                        # float
    "total_asset": 100230.0,                 # float
    "shares": np.array([0, 0]),              # shape: (n_tickers,)
    "avg_buy_price": np.array([0., 0.]),     # shape: (n_tickers,)
    "ohlc": np.array([[150, 152, 149, 151],  # shape: (n_tickers, 4)
                      [2800, 2850, 2790, 2820]]),
    "indicators": np.array([[0.5, 55], ...]),# shape: (n_tickers, n_ind)
    "tickers": ["AAPL", "GOOGL"],            # List[str]
    "indicator_names": ["macd", "rsi_14"],   # List[str]
    "macro_ohlc": np.array([...]),           # shape: (n_macro, 4), if configured
    "macro_indicators": np.array([...]),     # shape: (n_macro, n_ind), if configured
    "macro_tickers": ["VIX"]                 # List[str], if configured
}
```

---

# ReplayBuffer

Experience replay buffer with on-demand market data fetching.

## Constructor

```python
from fast_finrl_py import FastFinRL, ReplayBuffer

env = FastFinRL("data/stock.csv")
buffer = ReplayBuffer(
    env: FastFinRL,
    capacity: int = 1_000_000,
    batch_size: int = 256
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | FastFinRL | required | Environment instance (used for market data lookup) |
| `capacity` | int | 1,000,000 | Maximum transitions to store |
| `batch_size` | int | 256 | Default batch size for sample() |

---

## Methods

### add(state, action, reward, next_state, done)

Add single transition. State must be json format (with "portfolio", "market" keys).

```python
state = env.reset(tickers, seed=42)
while not state["done"]:
    action = [0.5, -0.3]
    next_state = env.step(action)
    buffer.add(state, action, next_state["reward"], next_state, next_state["done"])
    state = next_state
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | dict | Current state (json format) |
| `action` | List[float] | Actions taken |
| `reward` | float or List[float] | Reward(s). List for multi-objective |
| `next_state` | dict | Next state (json format) |
| `done` | bool | Episode done flag |

---

### sample(h, batch_size=None) -> tuple

Sample batch with market data windows.

```python
s, a, r, s_next, done, s_mask, s_next_mask = buffer.sample(h=20)
s, a, r, s_next, done, s_mask, s_next_mask = buffer.sample(h=20, batch_size=512)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h` | int | required | History length (past days) |
| `batch_size` | int | None | Batch size. None = use constructor default |

**Returns tuple of 7:**

| Index | Variable | Type | Shape | Description |
|-------|----------|------|-------|-------------|
| 0 | `s` | dict | - | State with market data |
| 1 | `a` | np.ndarray | (batch, n_tickers) | Actions |
| 2 | `r` | np.ndarray | (batch, n_obj) | Rewards (always 2D) |
| 3 | `s_next` | dict | - | Next state with market data |
| 4 | `done` | np.ndarray | (batch, 1) | Done flags (always 2D) |
| 5 | `s_mask` | dict or None | - | Validity masks. None if h=0 |
| 6 | `s_next_mask` | dict or None | - | Next state masks |

**State dict (`s`, `s_next`) structure:**

```python
# Per-ticker market data
s["AAPL"]["ohlc"]              # shape: (batch, h+1, 4)
s["AAPL"]["indicators"]        # shape: (batch, h+1, n_ind)

# Macro tickers
s["macro"]["VIX"]["ohlc"]      # shape: (batch, h+1, 4)
s["macro"]["VIX"]["indicators"] # shape: (batch, h+1, n_ind)

# Portfolio
s["portfolio"]["cash"]          # shape: (batch,)
s["portfolio"]["shares"]        # shape: (batch, n_tickers)
s["portfolio"]["avg_buy_price"] # shape: (batch, n_tickers)

# Metadata
s["tickers"]                   # List[str]: ticker names
s["indicator_names"]           # List[str]: indicator names
s["macro_tickers"]             # List[str]: macro ticker names
```

**Mask dict (`s_mask`) structure:**

```python
s_mask["AAPL"]           # shape: (batch, h+1), 1=valid, 0=padding
s_mask["macro"]["VIX"]   # shape: (batch, h+1)
```

---

### get(index) -> StoredTransition

Get raw transition by buffer index.

```python
t = buffer.get(0)
t.state_day          # int
t.next_state_day     # int
t.tickers            # List[str]
t.state_cash         # float
t.next_state_cash    # float
t.state_shares       # List[int]
t.next_state_shares  # List[int]
t.state_avg_buy_price     # List[float]
t.next_state_avg_buy_price # List[float]
t.action             # List[float]
t.rewards            # List[float]
t.done               # bool
t.terminal           # bool
```

---

### get_market_data(index, h, future, next_state=False) -> dict

Get market window for specific transition. Same format as `env.get_market_window_numpy()`.

```python
data = buffer.get_market_data(index=0, h=10, future=5, next_state=False)
```

---

### Utility Methods

```python
buffer.size()               # int: Current number of transitions
buffer.capacity()           # int: Maximum capacity
buffer.clear()              # Clear all transitions
buffer.save("buffer.json")  # Save to JSON file
buffer.load("buffer.json")  # Load from JSON file
buffer.sample_indices(batch_size)  # List[int]: Sample random indices
```

---

# VecFastFinRL

N parallel environments with TBB parallelization. **Inherits all FastFinRL features** with the following differences:

## Constructor Differences

```python
VecFastFinRL(
    ...,                        # Same as FastFinRL
    auto_reset: bool = True,    # NEW: Auto-reset done envs with seed+1
    return_format: str = "json" # "json" returns List[dict], "vec" returns single dict
)
```

| New Parameter | Type | Default | Description |
|---------------|------|---------|-------------|
| `auto_reset` | bool | True | When env is done, automatically reset with seed+1 |

**Note:** `initial_seed` parameter is NOT available. Seeds are provided per-env in reset().

---

## Method Differences

### reset(tickers_list, seeds) -> states

```python
tickers_list = [["AAPL", "GOOGL"], ["MSFT", "AAPL"]]  # N=2 envs
seeds = np.arange(2, dtype=np.int64)
states = vec_env.reset(tickers_list, seeds)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tickers_list` | List[List[str]] | Shape (N, n_tickers). All envs must have same n_tickers |
| `seeds` | np.ndarray[int64] | Shape (N,). Per-env random seeds |

**Returns:**
- `return_format="json"`: `List[dict]` of length N
- `return_format="vec"`: Single `dict` with batched arrays

---

### step(actions) -> states

```python
actions = np.array([[0.5, -0.3], [0.1, 0.2]])  # shape: (N, n_tickers)
states = vec_env.step(actions)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `actions` | np.ndarray[float64] | Shape (N, n_tickers). Range [-1, 1] |

---

### reset_indices(indices, seeds) -> states (NEW)

Partial reset - reset only specified environment indices.

```python
states = vec_env.reset_indices([1, 3], np.array([100, 200], dtype=np.int64))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `indices` | List[int] | Env indices to reset |
| `seeds` | np.ndarray[int64] | Seeds (same length as indices) |

**Returns:** Full states (all N envs). Only specified indices are modified.

---

### set_auto_reset(enabled) / auto_reset() (NEW)

```python
vec_env.set_auto_reset(False)  # Disable
is_enabled = vec_env.auto_reset()
```

---

### Additional Accessor Methods

```python
vec_env.num_envs()      # int: Number of environments (N)
vec_env.n_tickers()     # int: Tickers per env
vec_env.n_indicators()  # int: Number of indicators
vec_env.n_macro()       # int: Number of macro tickers
vec_env.get_tickers()   # List[List[str]]: Per-env tickers (N, n_tickers)
```

---

## State Structure Differences

### return_format="json" (default)

Returns `List[dict]` where each dict is same as FastFinRL vec format:

```python
states[0] = {
    "day": 150,
    "cash": 100000.0,
    "total_asset": 100000.0,
    "done": False,
    "terminal": False,
    "reward": 0.0,
    "shares": np.array([0, 0]),            # (n_tickers,)
    "avg_buy_price": np.array([0., 0.]),   # (n_tickers,)
    "ohlc": np.array([[...], [...]]),      # (n_tickers, 4)
    "indicators": np.array([[...], [...]]),# (n_tickers, n_ind)
    "tickers": ["AAPL", "GOOGL"],
    "macro_ohlc": np.array([...]),         # (n_macro, 4) if configured
    "macro_indicators": np.array([...])    # (n_macro, n_ind) if configured
}
```

### return_format="vec"

Returns single `dict` with all envs batched:

```python
state = {
    "day": np.array([150, 151]),                # (N,)
    "cash": np.array([100000., 100000.]),       # (N,)
    "total_asset": np.array([100000., 100000.]),# (N,)
    "done": np.array([False, False]),           # (N,)
    "terminal": np.array([False, False]),       # (N,)
    "reward": np.array([0., 0.]),               # (N,)
    "shares": np.array([[0, 0], [0, 0]]),       # (N, n_tickers)
    "avg_buy_price": np.array([[0., 0.], ...]), # (N, n_tickers)
    "ohlc": np.array([[[...]], [[...]]]),       # (N, n_tickers, 4)
    "indicators": np.array([[[...]], [[...]]]), # (N, n_tickers, n_ind)
    "tickers": [["AAPL", "GOOGL"], ["MSFT", "AAPL"]],
    "macro_ohlc": np.array([...]),              # (N, n_macro, 4)
    "macro_indicators": np.array([...]),        # (N, n_macro, n_ind)
    "n_envs": 2,
    "n_tickers": 2,
    "n_indicators": 3,
    "n_macro": 1
}
```

---

## Not Implemented

| Feature | Reason |
|---------|--------|
| `get_state()` | Use reset/step return values |
| `get_raw_value()` | Not applicable for vec env |
| `get_market_window()` (json) | Use `get_market_window_numpy()` |
| Public attribute setters | Config is fixed after construction |

---

# VecReplayBuffer

Replay buffer for vectorized environments. **Inherits all ReplayBuffer features** with the following differences:

## Constructor Differences

```python
# From VecFastFinRL (recommended)
buffer = VecReplayBuffer(vec_env, capacity=1_000_000, batch_size=256)

# From FastFinRL (also works)
buffer = VecReplayBuffer(env, capacity=1_000_000, batch_size=256)
```

---

## Method Differences

### add(states, actions, rewards, next_states, dones)

Add N transitions at once. Same interface as ReplayBuffer.

```python
# States must have attribute access (.day, .cash, etc.)
# Wrap dicts if needed:
class W:
    def __init__(self, d): self._d = d
    def __getattr__(self, n): return self._d[n]

buffer.add(
    [W(s) for s in states],           # List[obj] with .day, .cash, .shares, .avg_buy_price, .tickers, .terminal
    actions,                           # np.ndarray (N, n_tickers)
    [s["reward"] for s in next_states],# List[float] or List[List[float]]
    [W(s) for s in next_states],
    [s["done"] for s in next_states]   # List[bool]
)
```

---

### sample() Differences

**Additional fields in state dict:**

```python
s["env_ids"]         # np.ndarray (batch,): Original env indices
s["tickers"]         # List[List[str]] (batch, n_tickers): Per-sample tickers
s["unique_tickers"]  # List[str]: Union of all tickers in batch
```

**Note:** `s["tickers"]` is `(batch, n_tickers)` since different samples may have different tickers (when env_id changes and tickers change).

---

# Data Format

Required CSV columns:

```csv
day,date,tic,open,high,low,close,volume,macd,rsi_14
0,2023-01-01,AAPL,150,152,149,151,1000000,0.5,55
0,2023-01-01,GOOGL,2800,2850,2790,2820,500000,-0.3,52
1,2023-01-02,AAPL,151,153,150,152,1100000,0.7,58
```

| Column | Required | Description |
|--------|----------|-------------|
| `day` | Yes | Time index (starts from 0) |
| `date` | Yes | Date string |
| `tic` | Yes | Ticker symbol |
| `open`, `high`, `low`, `close` | Yes | OHLC prices |
| `volume` | No | Trading volume (excluded from indicators) |
| Other numeric columns | No | Auto-detected as technical indicators |

---

# Examples

## FastFinRL Training Loop

```python
from fast_finrl_py import FastFinRL, ReplayBuffer

env = FastFinRL("train.csv", initial_amount=100000, hmax=30)
buffer = ReplayBuffer(env, capacity=1_000_000, batch_size=256)
tickers = ["AAPL", "GOOGL", "MSFT"]

for episode in range(100):
    state = env.reset(tickers, seed=episode)
    while not state["done"]:
        action = model.predict(state)
        next_state = env.step(action)
        buffer.add(state, action, next_state["reward"], next_state, next_state["done"])
        state = next_state

for step in range(10000):
    s, a, r, s_next, done, s_mask, _ = buffer.sample(h=50)
    loss = model.train(s, a, r, s_next, done)
```

## VecFastFinRL Training Loop

```python
from fast_finrl_py import VecFastFinRL, VecReplayBuffer
import numpy as np

vec_env = VecFastFinRL("train.csv", auto_reset=True, return_format="json")
buffer = VecReplayBuffer(vec_env, capacity=1_000_000, batch_size=256)

N = 64
tickers_list = [["AAPL", "GOOGL", "MSFT"] for _ in range(N)]

class W:
    def __init__(self, d): self._d = d
    def __getattr__(self, n): return self._d[n]

states = vec_env.reset(tickers_list, np.arange(N, dtype=np.int64))
for step in range(10000):
    actions = model.predict(states)  # (N, 3)
    next_states = vec_env.step(actions)

    buffer.add(
        [W(s) for s in states],
        actions,
        [s["reward"] for s in next_states],
        [W(s) for s in next_states],
        [s["done"] for s in next_states]
    )
    states = next_states

    if step % 100 == 0 and buffer.size() > 1000:
        s, a, r, s_next, done, _, _ = buffer.sample(h=20)
        loss = model.train(s, a, r, s_next, done)
```

## Using vec return_format

```python
# Batched numpy arrays - easier for neural networks
vec_env = VecFastFinRL("data.csv", return_format="vec")

state = vec_env.reset(tickers_list, seeds)
# state["ohlc"].shape = (N, n_tickers, 4)
# state["done"].shape = (N,)

# Or switch at runtime
vec_env.set_return_format("vec")
state = vec_env.step(actions)
```

