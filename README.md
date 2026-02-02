# FastFinRL

High-performance C++ reimplementation of Python FinRL's StockTradingEnv for reinforcement learning-based trading.

## Features

- **DataFrame-based data handling** using [hosseinmoein/DataFrame](https://github.com/hosseinmoein/DataFrame)
- **JSON state representation** via [nlohmann/json](https://github.com/nlohmann/json)
- **Dynamic indicator extraction** - automatically detects technical indicators from CSV columns
- **Reproducible episodes** with seed management and auto-increment
- **Multi-ticker support** - trade multiple assets simultaneously
- **Stop-loss mechanism** with configurable tolerance
- **Flexible bidding options** - default, uniform, or advanced uniform pricing

## Requirements

- C++23 compiler (GCC 13+ recommended)
- CMake 3.14+
- TBB (Threading Building Blocks)

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Quick Start

```cpp
#include "FastFinRL.hpp"

int main() {
    // Create environment
    fast_finrl::FastFinRL env("data/raw_train_df.csv");

    // Reset with tickers and seed
    auto state = env.reset({"SPY", "QQQ", "GLD"}, 42);

    // Trading loop
    while (!state["done"].get<bool>()) {
        // Actions: positive = buy, negative = sell, 0 = hold
        // Range: [-1.0, 1.0], scaled by hmax internally
        std::vector<double> actions = {0.5, -0.3, 0.0};
        state = env.step(actions);

        double reward = state["reward"].get<double>();
        double total_asset = state["portfolio"]["total_asset"].get<double>();
    }

    return 0;
}
```

## API

### Constructor

```cpp
explicit FastFinRL(const std::string& csv_path);
```

### Configuration (Public Attributes)

```cpp
double initial_amount;          // Default: 30000.0
int hmax;                       // Default: 15
double buy_cost_pct;            // Default: 0.01 (1%)
double sell_cost_pct;           // Default: 0.01 (1%)
double stop_loss_tolerance;     // Default: 0.8 (20% loss)
std::string bidding;            // "default", "uniform", "adv_uniform"
std::string stop_loss_calculation; // "close" or "low"
```

Example:
```cpp
FastFinRL env("data.csv");
env.initial_amount = 50000;
env.hmax = 20;
env.stop_loss_tolerance = 0.9;
```

### reset()

```cpp
nlohmann::json reset(const std::vector<std::string>& ticker_list, int seed);
```

- `seed >= 0`: Use specified seed
- `seed == -1`: Auto-increment from previous seed

**Returns:**
```json
{
  "day": 427,
  "date": "2021-01-08",
  "seed": 42,
  "done": false,
  "terminal": false,
  "portfolio": {
    "cash": 30000.0,
    "holdings": {
      "SPY": {"shares": 0, "avg_buy_price": 0.0}
    }
  },
  "market": {
    "SPY": {
      "open": 355.27,
      "high": 356.11,
      "low": 352.01,
      "close": 355.90,
      "indicators": {"rsi_7": 74.5, "macd": 3.67, ...}
    }
  }
}
```

### step()

```cpp
nlohmann::json step(const std::vector<double>& actions);
```

- Actions range: `[-1.0, 1.0]`
- Positive = buy, Negative = sell, Zero = hold
- Internally scaled by `hmax`

**Returns:** Same as reset() plus:
```json
{
  "reward": 0.00517,
  "info": {
    "loss_cut_amount": 0.0,
    "n_trades": 2,
    "num_stop_loss": 0
  }
}
```

### Utility Methods

```cpp
std::set<std::string> get_indicator_names() const;  // Get detected indicators
nlohmann::json get_state() const;                    // Get current state
double get_raw_value(const std::string& ticker, int day, const std::string& column) const;
```

## CSV Format

Required columns:
- `date` - Date string (used for day ranking)
- `tic` - Ticker symbol
- `open`, `high`, `low`, `close` - Price data
- `volume` - Trading volume

All other columns are treated as technical indicators.

Example:
```csv
date,close,high,low,open,volume,tic,macd,rsi_7,rsi_14
2019-05-01,355.90,356.11,352.01,355.27,12331800,SPY,3.67,74.5,68.2
2019-05-01,181.14,183.43,181.02,182.92,34797100,QQQ,2.96,62.7,69.3
```

## Testing

```bash
cd build
ctest --output-on-failure
```

15 tests covering:
- DataFrame loading and indicator extraction
- Reset with seed reproducibility
- Seed auto-increment
- Buy/sell operations
- Stop-loss mechanism
- Terminal conditions
- Multi-ticker trading
- Reward calculation

## Project Structure

```
fast_finrl/
├── CMakeLists.txt
├── README.md
├── CLAUDE.md              # Development specification
├── include/
│   └── FastFinRL.hpp      # Header file
├── src/
│   └── FastFinRL.cpp      # Implementation
├── tests/
│   └── test_fast_finrl.cpp
├── data/
│   └── raw_train_df.csv   # Training data
└── main.cpp               # Demo application
```

## Dependencies

Automatically fetched via CMake FetchContent:
- [hosseinmoein/DataFrame](https://github.com/hosseinmoein/DataFrame) - C++ DataFrame library
- [nlohmann/json](https://github.com/nlohmann/json) - JSON for Modern C++
- [GoogleTest](https://github.com/google/googletest) - Testing framework

System dependency:
- TBB (Intel Threading Building Blocks)

## License

MIT License
