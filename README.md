# FastFinRL

High-performance stock trading environment for reinforcement learning. A drop-in replacement for [FinRL](https://github.com/AI4Finance-Foundation/FinRL)'s StockTradingEnv with **~200x speedup**.

## Installation

```bash
pip install fast-finrl
```

## Quick Start

```python
from fast_finrl_py import FastFinRL

# Create environment
env = FastFinRL("data/stock_data.csv")

# Reset with tickers and seed
state = env.reset(["AAPL", "MSFT", "GOOGL"], seed=42)

# Training loop
while not state["done"] and not state["terminal"]:
    actions = [0.5, -0.3, 0.0]  # buy AAPL, sell MSFT, hold GOOGL
    state = env.step(actions)

    reward = state["reward"]
    total_asset = state["portfolio"]["total_asset"]
```

## Configuration

```python
from fast_finrl_py import FastFinRL

env = FastFinRL(
    csv_path="data/stock_data.csv",
    initial_amount=100000.0,
    hmax=30,
    buy_cost_pct=0.001,
    sell_cost_pct=0.001,
    stop_loss_tolerance=0.85,
    bidding="uniform",
    initial_seed=42,
    tech_indicator_list=["macd", "rsi_14", "cci_14"],
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_path` | (required) | Path to CSV file with stock data |
| `initial_amount` | 30000.0 | Starting cash amount |
| `hmax` | 15 | Max shares per trade (action * hmax) |
| `buy_cost_pct` | 0.01 | Buy fee (1% = 0.01) |
| `sell_cost_pct` | 0.01 | Sell fee (1% = 0.01) |
| `stop_loss_tolerance` | 0.8 | Stop-loss threshold (0.8 = sell at 20% loss) |
| `bidding` | "default" | Price execution policy |
| `stop_loss_calculation` | "close" | Stop-loss reference: "close" or "low" |
| `initial_seed` | 0 | Random seed |
| `tech_indicator_list` | [] | Indicator columns (empty = auto-detect) |

### Bidding Policies

| Policy | Buy Price | Sell Price |
|--------|-----------|------------|
| `default` | close | close |
| `uniform` | random(low, high) | random(low, high) |
| `adv_uniform` | random(max(open,close), high) | random(low, min(open,close)) |

## API

### `reset(ticker_list, seed) -> dict`

Reset environment with given tickers.

```python
state = env.reset(["AAPL", "MSFT"], seed=42)
# seed=-1: auto-increment from previous seed
```

Returns:
```python
{
    "day": 0,
    "date": "2023-01-01",
    "seed": 42,
    "done": False,
    "terminal": False,
    "portfolio": {
        "cash": 30000.0,
        "holdings": {"AAPL": {"shares": 0, "avg_buy_price": 0.0}}
    },
    "market": {
        "AAPL": {
            "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0,
            "indicators": {"macd": 0.5, "rsi_14": 50.1}
        }
    }
}
```

### `step(actions) -> dict`

Execute one trading step.

- `actions`: List of floats in [-1, 1] (one per ticker)
  - Positive = buy, Negative = sell, Zero = hold

Returns:
```python
{
    "day": 1,
    "reward": 0.0023,  # log(end_asset / begin_asset)
    "done": False,
    "terminal": False,
    "portfolio": {"cash": 25000.0, "total_asset": 30500.0, ...},
    "market": {...},
    "debug": {"AAPL": {"fill_price": 150.5, "cost": 15.05, "quantity": 10}},
    "info": {"n_trades": 1, "num_stop_loss": 0, "loss_cut_amount": 0.0}
}
```

## Data Format

CSV with columns:

| Column | Type | Required |
|--------|------|----------|
| day | int | Yes |
| date | string | Yes |
| tic | string | Yes |
| open, high, low, close | float | Yes |
| volume | float | Yes |
| *(other columns)* | float | Used as indicators |

```csv
day,date,tic,open,high,low,close,volume,macd,rsi_14
0,2023-01-01,AAPL,150.0,152.0,149.0,151.0,1000000,0.5,50.1
0,2023-01-01,MSFT,250.0,255.0,248.0,252.0,800000,0.3,48.5
```

## Performance

| Tickers | Step Time |
|---------|-----------|
| 4 | 0.03ms |
| 9 | 0.07ms |
| 37 | 0.19ms |

## Build from Source

### Requirements (Ubuntu/Debian)

```bash
sudo apt install -y libarrow-dev libparquet-dev libtbb-dev
```

### Build

```bash
git clone https://github.com/GwangPyo/fast_finrl.git
cd fast_finrl
pip install .
```
 
