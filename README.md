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
    tech_indicator_list=[]         # Indicators to use (empty = auto-detect from CSV columns)
)
```

**Notes:**
- `tech_indicator_list`: Order determines indicator array order in `get_market_window_numpy()`. Empty list auto-detects all non-OHLC numeric columns from CSV.

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

## Example: Training Loop

```python
env = FastFinRL("train.csv", initial_amount=100000, hmax=30)
tickers = list(env.get_all_tickers())

for episode in range(1000):
    state = env.reset(tickers, seed=episode)

    while not state["done"] and not state["terminal"]:
        # Get market data
        window = env.get_market_window_numpy(tickers, state["day"], h=50, future=0)

        # Model inference
        actions = model(window)

        # Step
        state = env.step(actions)

        # Train
        model.train(state["reward"])
```
