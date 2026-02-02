# FastFinRL

High-performance stock trading environment for reinforcement learning. A drop-in replacement for [FinRL](https://github.com/AI4Finance-Foundation/FinRL)'s StockTradingEnv with **speedup with C++**.

## Installation

### Requirements
- Python 3.8+
- GCC 13+ (for C++23 `<format>` support)
- TBB (Threading Building Blocks)
- CMake 3.14+

### Install via pip
```bash
pip install .
```

Or for development:
```bash
pip install -e .
```

## Quick Start

```python
from fast_finrl_py import FastFinRL, FastFinRLConfig

# Create environment
env = FastFinRL("data/stock_data.csv")

# Reset with tickers and seed
state = env.reset(["AAPL", "MSFT", "GOOGL"], seed=42)

# Training loop
while not state["done"] and not state["terminal"]:
    # Actions: [-1, 1] range, positive=buy, negative=sell, 0=hold
    actions = [0.5, -0.3, 0.0]
    state = env.step(actions)

    reward = state["reward"]
    total_asset = state["portfolio"]["total_asset"]
```

## Configuration

```python
from fast_finrl_py import FastFinRL, FastFinRLConfig

config = FastFinRLConfig()
config.initial_amount = 30000.0      # Starting cash
config.hmax = 15                      # Max shares per trade
config.buy_cost_pct = 0.01           # 1% buy fee
config.sell_cost_pct = 0.01          # 1% sell fee
config.stop_loss_tolerance = 0.8     # Stop-loss at 20% loss
config.bidding = "deterministic"     # Price execution policy

env = FastFinRL("data/stock_data.csv", config)
```

### Bidding Policies

| Policy | Description |
|--------|-------------|
| `deterministic` | Execute at close price |
| `default` | Execute at close price |
| `uniform` | Random price between low and high |
| `adv_uniform` | Adversarial: worst price for the agent |

## API Reference

### `reset(ticker_list, seed) -> dict`

Reset environment with given tickers.

- `ticker_list`: List of ticker symbols to trade
- `seed`: Random seed (`-1` for auto-increment from previous seed)

Returns state dict:
```python
{
    "day": 42,
    "date": "2023-05-15",
    "seed": 42,
    "done": False,
    "terminal": False,
    "portfolio": {
        "cash": 30000.0,
        "holdings": {
            "AAPL": {"shares": 0, "avg_buy_price": 0.0}
        }
    },
    "market": {
        "AAPL": {
            "open": 150.0,
            "high": 152.0,
            "low": 149.0,
            "close": 151.0,
            "indicators": {"macd": 0.5, "rsi_14": 50.1, ...}
        }
    }
}
```

### `step(actions) -> dict`

Execute one trading step.

- `actions`: List of floats in `[-1, 1]` range (one per ticker)
  - Positive = buy, Negative = sell, Zero = hold
  - Scaled by `hmax` internally

Returns state dict with additional fields:
```python
{
    ...,
    "reward": 0.0023,  # log(end_asset / begin_asset)
    "portfolio": {
        ...,
        "total_asset": 30500.0
    },
    "debug": {
        "AAPL": {"fill_price": 150.5, "cost": 15.05, "quantity": 10}
    },
    "info": {
        "n_trades": 1,
        "num_stop_loss": 0,
        "loss_cut_amount": 0.0
    }
}
```

### Properties

```python
env.get_indicator_names()  # Set of technical indicator names
env.get_state()            # Current state dict
env.get_raw_value(ticker, day, column)  # Raw DataFrame value
```

## Data Format

CSV file with columns:

| Column | Type | Required |
|--------|------|----------|
| day | int | Yes |
| date | string | Yes |
| tic | string | Yes |
| open, high, low, close | float | Yes |
| volume | float | Yes |
| *(any other columns)* | float | Auto-detected as indicators |

Example:
```csv
day,date,tic,open,high,low,close,volume,macd,rsi_14
0,2023-01-01,AAPL,150.0,152.0,149.0,151.0,1000000,0.5,50.1
0,2023-01-01,MSFT,250.0,255.0,248.0,252.0,800000,0.3,48.5
```

## Performance

| Tickers | Step Time |
|---------|-----------|
| 4       | 0.03ms    |
| 9       | 0.07ms    |
| 37      | 0.19ms    |

## License

MIT
