# FastFinRL Architecture

## 1. 개요

FastFinRL은 금융 강화학습을 위한 고성능 C++ 환경으로, Python 바인딩을 통해 RL 알고리즘과 통합됩니다.

### 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Python Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Training   │  │   Sampling   │  │   Market     │               │
│  │     Loop     │  │    Batch     │  │    Query     │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          │ pybind11        │ pybind11        │ pybind11
          │ (zero-copy)     │ (zero-copy)     │ (zero-copy)
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          C++ Core                                    │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Environment Layer                           │  │
│  │  ┌─────────────┐         ┌─────────────────┐                  │  │
│  │  │  FastFinRL  │ ◄─────► │  VecFastFinRL   │                  │  │
│  │  │   (Uni)     │ 1:N     │     (Vec)       │                  │  │
│  │  │ - Single Env│         │ - N Parallel    │                  │  │
│  │  │ - Full API  │         │ - SoA Layout    │                  │  │
│  │  └──────┬──────┘         └───────┬─────────┘                  │  │
│  │         │                        │                             │  │
│  │         │ shared_ptr<DataFrame>  │ shared_ptr<FastFinRL>       │  │
│  │         ▼                        ▼                             │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                    DataFrame (hmdf)                      │  │  │
│  │  │  - OHLCV Data    - Technical Indicators                  │  │  │
│  │  │  - Row Index Table: ticker_row_table_[tic][day] → row    │  │  │
│  │  │  - Column Reference Cache: col_open_, col_close_, ...    │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Buffer Layer                                │  │
│  │  ┌─────────────┐         ┌─────────────────┐                  │  │
│  │  │ ReplayBuffer│ ◄─────► │ VecReplayBuffer │                  │  │
│  │  │   (Uni)     │         │     (Vec)       │                  │  │
│  │  │ - Circular  │         │ - Batch add()   │                  │  │
│  │  │ - xtensor   │         │ - env_id track  │                  │  │
│  │  └─────────────┘         └─────────────────┘                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Serialization Layer                         │  │
│  │  ┌─────────────────┐    ┌───────────────────┐                 │  │
│  │  │  StateData      │───►│ IStateSerializer  │                 │  │
│  │  │  - PortfolioState│   │ └─JsonSerializer  │                 │  │
│  │  │  - MarketState   │   └───────────────────┘                 │  │
│  │  │  - EpisodeInfo   │                                          │  │
│  │  └─────────────────┘                                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 클래스 설명

### 2.1 FastFinRL (단일 환경)

**파일**: `include/FastFinRL.hpp`

FinRL StockTradingEnv의 C++ 구현체. 단일 에피소드를 시뮬레이션합니다.

#### 핵심 구조

```cpp
class FastFinRL {
public:
    using MyDataFrame = hmdf::StdDataFrame<unsigned long>;

    // Configuration (public, 직접 수정 가능)
    double initial_amount;      // 초기 자금
    double failure_threshold;   // 종료 임계값
    int hmax;                   // 거래당 최대 주식 수
    double buy_cost_pct;        // 매수 수수료율
    double sell_cost_pct;       // 매도 수수료율
    double stop_loss_tolerance; // 손절 허용폭
    string bidding;             // 입찰 전략
    string stop_loss_calculation; // 손절 계산 방식
    ReturnFormat return_format; // Json or Vec

private:
    MyDataFrame df_;                           // 시장 데이터
    map<string, size_t> ticker_global_idx_;    // ticker → 전역 인덱스
    vector<vector<size_t>> ticker_row_table_;  // [ticker_idx][day] → row
    vector<pair<string, DoubleColRef>> indicator_cols_; // 캐시된 지표 컬럼
};
```

#### 메서드 상세

---

##### `__init__(csv_path, ...)`

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `csv_path` | str | (필수) | 데이터 파일 경로 |
| `initial_amount` | float | 30000.0 | 초기 자금 |
| `failure_threshold` | float | 25000.0 | 종료 임계값 |
| `hmax` | int | 15 | 거래당 최대 주식 수 |
| `buy_cost_pct` | float | 0.01 | 매수 수수료율 |
| `sell_cost_pct` | float | 0.01 | 매도 수수료율 |
| `stop_loss_tolerance` | float | 0.8 | 손절 허용폭 (0.8 = 20% 손실) |
| `bidding` | str | "default" | 입찰 전략 |
| `stop_loss_calculation` | str | "close" | 손절 계산 방식 |
| `tech_indicator_list` | List[str] | [] | 사용할 지표 (빈 리스트=자동 감지) |
| `macro_tickers` | List[str] | [] | 항상 포함할 기준 티커 |
| `return_format` | str | "json" | 반환 형식 ("json" 또는 "vec") |

---

##### `reset(tickers, seed, shifted_start=0)`

**환경 초기화**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `tickers` | List[str] | 거래할 ticker 리스트 |
| `seed` | int | 랜덤 시드 |
| `shifted_start` | int | 최소 시작 day 오프셋 |

**동작:**
1. tickers 설정
2. 시작 day 랜덤 선택: `[max(ticker_first_days) + shifted_start, max_day * 0.8)`
3. 포트폴리오 초기화 (cash=initial_amount, shares=0)

---

##### `reset()`

**기존 설정 유지 리셋**

- 이전 tickers 유지
- seed += 1
- day 새로 랜덤 선택

---

##### `step(actions)`

**한 스텝 실행**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `actions` | List[float] | [n_tickers] 액션 |

**반환:** dict (portfolio, market, reward, done, terminal)

---

##### `get_market_window(ticker, day, h, future)`

**단일 ticker 시장 데이터**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `ticker` | str | 티커 |
| `day` | int | 기준 day |
| `h` | int | history 길이 |
| `future` | int | future 길이 |

**반환:** dict
```python
{
    'past_ohlcv': [[o,h,l,c,v], ...],     # [h, 5]
    'past_indicators': [[...], ...],       # [h, n_ind]
    'past_mask': [1, 1, 1, 0, 0, ...],    # [h] - 유효 데이터 여부
    'current_open': 150.0,
    'current_indicators': [...],
    'future_ohlcv': [...],                 # [future, 5]
    'future_mask': [...]                   # [future]
}
```

---

##### `get_market_window_numpy(tickers, day, h, future)`

**여러 ticker 시장 데이터 (zero-copy)**

**반환:** dict[ticker] → numpy arrays

---

##### `get_all_tickers()` → `set[str]`

데이터에 있는 모든 ticker

##### `get_indicator_names()` → `set[str]`

사용 가능한 지표 이름

##### `get_max_day()` → `int`

최대 day 인덱스

##### `n_tickers()` → `int`

현재 활성 ticker 수

##### `n_indicators()` → `int`

지표 개수

#### 데이터 구조

```cpp
// 티커별 시계열 데이터 구조
struct TickerWindowData {
    // Past: [h, 5], [h, n_ind] - OHLCV + 지표
    vector<double> past_ohlcv;
    vector<double> past_indicators;
    vector<int> past_mask;      // 유효 데이터 여부

    // Current: scalar
    double current_open;
    vector<double> current_indicators;

    // Future: [f, 5], [f, n_ind]
    vector<double> future_ohlcv;
    vector<double> future_indicators;
    vector<int> future_mask;
};
```

---

### 2.2 VecFastFinRL (벡터화 환경)

**파일**: `include/VecFastFinRL.hpp`

N개의 병렬 환경을 관리합니다. DataFrame은 공유, 상태만 N배.

#### 핵심 구조

```cpp
class VecFastFinRL {
public:
    // Step 결과 (SoA 레이아웃)
    struct StepResult {
        vector<int> day;              // [N]
        vector<double> cash;          // [N]
        vector<int> shares;           // [N * n_tickers]
        vector<double> avg_buy_price; // [N * n_tickers]
        vector<double> open;          // [N * n_tickers]
        vector<double> indicators;    // [N * n_tickers * n_ind]
        vector<double> macro_open;    // [N * n_macro]
        vector<double> macro_indicators; // [N * n_macro * n_ind]
        vector<double> reward;        // [N]
        vector<uint8_t> done;         // [N]
        vector<uint8_t> terminal;     // [N]
        vector<double> total_asset;   // [N]
    };

private:
    shared_ptr<FastFinRL> base_env_;  // 시장 데이터 공유

    // Per-env 상태 (SoA layout)
    vector<int> day_;                 // [N]
    vector<double> cash_;             // [N]
    vector<int> shares_;              // [N * n_tickers]
    vector<double> avg_buy_price_;    // [N * n_tickers]
    vector<int64_t> seeds_;           // [N]
    vector<mt19937> rngs_;            // [N]
};
```

#### 메서드 상세

---

##### `__init__(csv_path, n_envs, ..., shuffle_tickers, shifted_start)`

**생성자**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `csv_path` | str | (필수) | 데이터 파일 경로 (.csv 또는 .parquet) |
| `n_envs` | int | (필수) | 병렬 환경 개수 |
| `num_tickers` | int | 0 | 환경당 ticker 수. 0=전체 사용 |
| `shuffle_tickers` | bool | False | True면 매 reset마다 ticker 재선택 |
| `shifted_start` | int | 5 | 최소 시작 day 오프셋 (history 보장용) |

**동작:**
- `shuffle_tickers=False`: 모든 env가 동일한 tickers (알파벳 순 첫 N개)
- `shuffle_tickers=True`: 각 env마다 랜덤하게 N개 선택 (initial_seed 기반)

```python
# shuffle_tickers=False (기본)
env = VecFastFinRL("data.csv", n_envs=3, num_tickers=3)
# env 0: ['AAPL', 'AMZN', 'GOOGL']
# env 1: ['AAPL', 'AMZN', 'GOOGL']  <- 동일
# env 2: ['AAPL', 'AMZN', 'GOOGL']  <- 동일

# shuffle_tickers=True
env = VecFastFinRL("data.csv", n_envs=3, num_tickers=3, shuffle_tickers=True)
# env 0: ['GOOGL', 'MSFT', 'QQQ']   <- 랜덤
# env 1: ['AAPL', 'META', 'TSLA']   <- 랜덤
# env 2: ['AMZN', 'NVDA', 'SPY']    <- 랜덤
```

---

##### `reset(tickers_list, seeds)`

**전체 리셋 (명시적 tickers + seeds)**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `tickers_list` | List[List[str]] | [N][n_tickers] - 각 env의 tickers |
| `seeds` | np.array[int64] | [N] - 각 env의 seed |

**동작:**
1. `tickers_list[i]`가 비어있으면:
   - `shuffle_tickers=True`: seed[i] 기반으로 랜덤 선택
   - `shuffle_tickers=False`: 알파벳 순 첫 N개
2. 모든 env의 ticker 개수는 동일해야 함
3. 각 env의 시작 day: `[min_start_day + shifted_start, max_day * 0.8)` 범위에서 랜덤

```python
# 명시적 tickers
env.reset([['AAPL', 'GOOGL'], ['MSFT', 'NVDA'], ['META', 'TSLA']],
          np.array([42, 43, 44]))

# 빈 tickers → shuffle_tickers 설정에 따라 자동 선택
env.reset([[], [], []], np.array([42, 43, 44]))
```

---

##### `reset(tickers_list, seed)` 또는 `reset(seed=int)`

**단일 seed 리셋**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `tickers_list` | List[List[str]] 또는 None | 생략 가능 |
| `seed` | int | 기준 seed (각 env는 seed * (i+1) * PRIME) |

**동작:**
1. `tickers_list` 생략 또는 빈 리스트:
   - `shuffle_tickers=True`: **매번 새로 shuffle** ✓
   - `shuffle_tickers=False`: 이전 tickers 유지
2. seed → 각 env별 seed 자동 생성: `(seed * (i+1) * 15485863) % 15485862`

```python
# shuffle_tickers=True일 때
env.reset(seed=42)   # tickers 새로 shuffle
env.reset(seed=100)  # tickers 다시 shuffle (다른 조합)

# shuffle_tickers=False일 때
env.reset(seed=42)   # 이전 tickers 유지
env.reset(seed=100)  # 이전 tickers 유지 (day만 변경)
```

---

##### `reset()` (no-arg)

**인자 없는 리셋**

**동작:**
1. `last_base_seed_ += 1` (자동 증가)
2. `shuffle_tickers=True`: **매번 새로 shuffle** ✓
3. `shuffle_tickers=False`: 이전 tickers 유지

```python
env.reset(seed=42)  # 초기화
env.reset()         # seed=43, shuffle_tickers면 tickers 변경
env.reset()         # seed=44, shuffle_tickers면 tickers 변경
```

---

##### `reset_indices(indices, seeds)`

**부분 리셋 (특정 env만)**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `indices` | List[int] | 리셋할 env 인덱스 |
| `seeds` | np.array[int64] | 각 인덱스의 seed |

**동작:**
- 지정된 env만 리셋
- **tickers는 변경 안 됨** (기존 유지)
- day, cash, shares 등 상태만 초기화

```python
# env 0, 2만 리셋
env.reset_indices([0, 2], np.array([100, 200]))
```

---

##### `step(actions)`

**한 스텝 실행**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `actions` | np.array[float] | [N, n_tickers] - 각 ticker 액션 |

**액션 해석:**
- `action > 0`: 매수 (최대 hmax주)
- `action < 0`: 매도 (최대 보유량)
- `action = 0`: 홀드

**동작:**
1. 각 env에서 stop loss 체크
2. 매도 우선 실행 (현금 확보)
3. 매수 실행
4. `day += 1`
5. `done` 체크: `total_asset < failure_threshold` 또는 `day >= max_day`
6. `auto_reset=True`이고 `done=True`면 자동 리셋

**반환값 (return_format에 따라):**
```python
# return_format='vec'
{
    'day': np.array([...]),           # [N]
    'cash': np.array([...]),          # [N]
    'shares': np.array([...]),        # [N, n_tickers]
    'open': np.array([...]),          # [N, n_tickers]
    'indicators': np.array([...]),    # [N, n_tickers, n_ind]
    'reward': np.array([...]),        # [N]
    'done': np.array([...]),          # [N] bool
    'terminal': np.array([...]),      # [N] bool
    'tickers': [['AAPL', ...], ...]   # [N][n_tickers]
}
```

---

##### `set_auto_reset(enabled)`

**자동 리셋 설정**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `enabled` | bool | True=done 시 자동 리셋 |

**동작:**
- `True`: step()에서 done 발생 시 해당 env 자동 리셋
- `False`: done 상태 유지, 수동으로 reset_indices() 호출 필요

---

##### `set_return_format(format)`

**반환 형식 설정**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `format` | str | 'json' 또는 'vec' |

**형식:**
- `'json'`: List[dict] - 각 env별 nested dict
- `'vec'`: dict - batched numpy arrays (GPU 학습용)

---

##### `get_tickers()`

**현재 tickers 조회**

**반환:** `List[List[str]]` - [N][n_tickers]

```python
tickers = env.get_tickers()
# [['AAPL', 'GOOGL', 'MSFT'], ['AMZN', 'META', 'NVDA'], ...]
```

---

##### `get_market_window_numpy(days, ticker_lists, h, future)`

**시장 데이터 윈도우 조회**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `days` | np.array[int] | [N] - 각 env의 day |
| `ticker_lists` | List[List[str]] | [N][n_tickers] |
| `h` | int | history 길이 |
| `future` | int | future 길이 |

**반환:** `List[dict]` - 각 env별 market window

```python
windows = env.get_market_window_numpy(
    obs['day'],
    obs['tickers'],
    h=20, future=5
)
# windows[0]['AAPL']['past_ohlcv']: [h, 5]
# windows[0]['AAPL']['past_mask']: [h]
# windows[0]['AAPL']['future_ohlcv']: [future, 5]
```

#### SoA 레이아웃

```
AoS (Array of Structs):     SoA (Struct of Arrays):
┌───────────────────┐       ┌───────────────────┐
│ env[0].day        │       │ day[0,1,2,...,N]  │
│ env[0].cash       │       ├───────────────────┤
│ env[0].shares[:]  │       │ cash[0,1,2,...,N] │
├───────────────────┤       ├───────────────────┤
│ env[1].day        │       │ shares[N*n_tic]   │
│ env[1].cash       │       └───────────────────┘
│ ...               │
└───────────────────┘       더 나은 캐시 효율성
```

---

### 2.3 ReplayBuffer (단일 버퍼)

**파일**: `include/ReplayBuffer.hpp`

경험 재생 버퍼. 순환 버퍼 방식으로 transition을 저장합니다.

#### 핵심 구조

```cpp
struct StoredTransition {
    int state_day = 0;
    vector<string> tickers;
    float state_cash = 0.0f;
    vector<int> state_shares;
    vector<float> state_avg_buy_price;

    vector<float> action;           // flat storage
    vector<float> rewards;          // multi-objective 지원
    bool done = false;
    bool terminal = false;

    int next_state_day = 0;
    float next_state_cash = 0.0f;
    vector<int> next_state_shares;
    vector<float> next_state_avg_buy_price;
};

class ReplayBuffer {
    // Sample 결과 - xtensor 배열
    struct SampleBatch {
        xt::xarray<float> s_ohlcv;         // [B, n_tic, h, 5]
        xt::xarray<float> s_indicators;    // [B, n_tic, h, n_ind]
        xt::xarray<int> s_mask;            // [B, n_tic, h]
        xt::xarray<float> s_next_ohlcv;
        xt::xarray<float> s_next_indicators;

        // Future data
        xt::xarray<float> s_future_ohlcv;  // [B, n_tic, F, 5]

        // Macro data
        xt::xarray<float> macro_ohlcv;     // [B, n_macro, h, 5]

        xt::xarray<float> actions;         // [B, ...action_shape]
        vector<vector<float>> rewards;     // [B][n_objectives]
    };

private:
    shared_ptr<const FastFinRL> env_;      // 시장 데이터 참조
    vector<StoredTransition> buffer_;      // 순환 버퍼
    size_t capacity_;
    size_t write_idx_ = 0;
    bool full_ = false;
};
```

#### 메서드 상세

---

##### `__init__(env, capacity, batch_size, seed, action_shape)`

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `env` | FastFinRL | (필수) | 환경 인스턴스 (시장 데이터 참조용) |
| `capacity` | int | 1000000 | 최대 저장 개수 |
| `batch_size` | int | 256 | 기본 배치 크기 |
| `seed` | int | 42 | 샘플링 랜덤 시드 |
| `action_shape` | tuple | None | 액션 shape (None=(n_tickers,)) |

---

##### `add(state, action, reward, next_state, done)`

**Transition 추가 (dict 형식)**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `state` | dict | env.step() 반환값 |
| `action` | np.array | 액션 |
| `reward` | float 또는 List[float] | 보상 (multi-objective 지원) |
| `next_state` | dict | 다음 상태 |
| `done` | bool | 에피소드 종료 여부 |

---

##### `add_transition(...)`

**Transition 추가 (직접 값)**

저장되는 필드:
- `state_day`, `next_state_day`
- `tickers`
- `state_cash`, `next_state_cash`
- `state_shares`, `next_state_shares`
- `state_avg_buy_price`, `next_state_avg_buy_price`
- `action`, `rewards`, `done`, `terminal`

---

##### `sample(batch_size, h, future)`

**배치 샘플링 + 시장 데이터 fetch**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `batch_size` | int | (생성자 값) | 샘플 개수 |
| `h` | int | 0 | history 길이 |
| `future` | int | 0 | future 길이 |

**반환:** tuple (states, actions, rewards, next_states, dones)

```python
s, a, r, s_next, d = buffer.sample(batch_size=256, h=20, future=5)

# s (states)
s['ohlcv']          # [B, n_tic, h, 5]
s['indicators']     # [B, n_tic, h, n_ind]
s['mask']           # [B, n_tic, h]
s['future_ohlcv']   # [B, n_tic, future, 5]
s['cash']           # [B]
s['shares']         # [B, n_tic]
s['avg_buy_price']  # [B, n_tic]

# macro (if configured)
s['macro']['ohlcv']       # [B, n_macro, h, 5]
s['macro']['indicators']  # [B, n_macro, h, n_ind]

# a (actions)
a  # [B, ...action_shape]

# r (rewards)
r  # [B, n_objectives]

# d (dones)
d  # [B]
```

---

##### `sample_indices(batch_size, min_day)`

**인덱스만 샘플링 (시장 데이터 없이)**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `batch_size` | int | 샘플 개수 |
| `min_day` | int | 최소 state_day (history 보장용) |

**반환:** `List[int]` - buffer 내 인덱스

---

##### `get(index)` → `StoredTransition`

**특정 인덱스의 transition 조회**

---

##### `get_market_data(index, h, future)`

**특정 transition의 시장 데이터 조회**

---

##### `save(path)` / `load(path)`

**바이너리 저장/로드**

- 포맷: 커스텀 바이너리 (빠른 I/O)
- 저장 내용: 모든 transitions + 메타데이터

---

##### `size()` → `int`

현재 저장된 transition 개수

##### `capacity()` → `int`

최대 저장 가능 개수

---

### 2.4 VecReplayBuffer (벡터화 버퍼)

**파일**: `include/VecReplayBuffer.hpp`

VecFastFinRL과 함께 사용하는 벡터화 버퍼. env_id로 환경을 구분합니다.

#### 핵심 구조

```cpp
struct VecStoredTransition {
    int env_id = 0;                // 환경 ID (0..N-1)

    // 나머지는 StoredTransition과 동일
    int state_day = 0;
    vector<string> tickers;
    // ...
};

class VecReplayBuffer {
private:
    vector<VecStoredTransition> buffer_;

    // 캐시된 메타데이터
    vector<string> cached_indicator_names_;
    vector<string> cached_macro_tickers_;

    // 재사용 샘플 버퍼
    mutable vector<float> sample_ohlcv_buf_;
    mutable vector<float> sample_ind_buf_;
    mutable vector<int> sample_mask_buf_;
};
```

#### 메서드 상세

---

##### `__init__(env, capacity, batch_size, seed, action_shape)`

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `env` | VecFastFinRL | (필수) | 벡터 환경 인스턴스 |
| `capacity` | int | 1000000 | 최대 저장 개수 |
| `batch_size` | int | 256 | 기본 배치 크기 |
| `seed` | int | 42 | 샘플링 랜덤 시드 |
| `action_shape` | tuple | None | 액션 shape (None=(n_tickers,)) |

---

##### `add(states, actions, rewards, next_states, dones)`

**N개 transition 일괄 추가 (자동 포맷 감지)**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `states` | dict 또는 List[dict] | vec_env.step() 반환값 |
| `actions` | np.array | [N, ...action_shape] |
| `rewards` | np.array | [N] 또는 [N, n_objectives] |
| `next_states` | dict 또는 List[dict] | 다음 상태 |
| `dones` | np.array | [N] bool |

**포맷 자동 감지:**
- `states`가 dict → `return_format='vec'`
- `states`가 List[dict] → `return_format='json'`

```python
# vec format (권장)
obs = vec_env.reset(seed=42)
next_obs = vec_env.step(actions)
buffer.add(obs, actions, rewards, next_obs, dones)

# json format
vec_env.set_return_format('json')
obs = vec_env.reset(seed=42)  # List[dict]
buffer.add(obs, actions, rewards, next_obs, dones)
```

---

##### `add_transition(env_id, state_day, ...)`

**단일 transition 추가 (직접 값)**

- `env_id`: 환경 ID (0..N-1)
- 나머지는 ReplayBuffer.add_transition과 동일

---

##### `sample(batch_size, h, future)`

**배치 샘플링**

ReplayBuffer.sample()과 동일한 인터페이스 및 반환값.

```python
s, a, r, s_next, d = buffer.sample(batch_size=256, h=20, future=5)
```

---

##### `save(path)` / `load(path)`

**바이너리 저장/로드**

---

##### `size()` → `int`

현재 저장된 transition 개수

##### `capacity()` → `int`

최대 저장 가능 개수

---

## 3. 주요 기능

### 3.1 O(1) 데이터 조회

Row Index Table을 사용한 상수 시간 조회:

```cpp
// 빌드: ticker_row_table_[ticker_idx][day] = row_index
void FastFinRL::build_index_tables() {
    for (size_t row = 0; row < df_.shape().first; ++row) {
        string ticker = col_tic_[row];
        int day = col_day_[row];

        size_t tic_idx = ticker_global_idx_[ticker];
        ticker_row_table_[tic_idx][day] = row;
    }
}

// 조회: O(1)
size_t row = ticker_row_table_[ticker_idx][day];
double close = col_close_->get()[row];
```

### 3.2 Zero-Copy Numpy 바인딩

capsule을 사용한 메모리 공유:

```cpp
// C++ 데이터를 Python numpy로 직접 노출
auto holder = make_shared<DataHolder>(data);
auto make_capsule = [holder]() {
    return py::capsule(new shared_ptr<DataHolder>(holder),
        [](void* p) { delete static_cast<shared_ptr<DataHolder>*>(p); });
};

// numpy array가 holder를 참조 - 복사 없음
py::array_t<double> arr({h, 5}, {5*sizeof(double), sizeof(double)},
                        holder->data.data(), make_capsule());
```

### 3.3 컬럼 참조 캐싱

DataFrame 컬럼 접근 최적화:

```cpp
// 해시 lookup 없이 직접 참조
using DoubleColRef = reference_wrapper<const vector<double>>;
optional<DoubleColRef> col_open_;
optional<DoubleColRef> col_close_;
vector<pair<string, DoubleColRef>> indicator_cols_;

// 초기화 시 한 번만 참조 획득
col_open_ = cref(df_.get_column<double>("open"));
col_close_ = cref(df_.get_column<double>("close"));
```

### 3.4 Return Format

두 가지 반환 형식 지원:

```cpp
enum class ReturnFormat {
    Json,  // dict/List[dict] - nested structure
    Vec    // dict with batched numpy arrays
};
```

**Json 형식** (기본값):
```python
{
    'day': 10,
    'portfolio': {
        'cash': 29500.0,
        'holdings': {'AAPL': {'shares': 5, 'avg_buy_price': 150.0}}
    },
    'market': {'AAPL': {'open': 152.0, 'indicators': {...}}}
}
```

**Vec 형식** (벡터화):
```python
{
    'day': np.array([10, 10, ...]),        # [N]
    'cash': np.array([29500, 28000, ...]), # [N]
    'shares': np.array([[5, 3], ...]),     # [N, n_tickers]
    'open': np.array([[152, 85], ...]),    # [N, n_tickers]
    'indicators': np.array([...])          # [N, n_tickers, n_ind]
}
```

### 3.5 자동 리셋

VecFastFinRL의 auto_reset 기능:

```cpp
// step 내부
if (done && auto_reset_) {
    reset_env(env_idx, seeds_[env_idx] + 1);
}
```

### 3.6 Macro Ticker

항상 상태에 포함되는 기준 티커:

```python
env = FastFinRL("data.csv", macro_tickers=["^GSPC", "^VIX"])

state = env.reset(["AAPL", "GOOGL"], seed=42)
# state['macro'] = {
#     '^GSPC': {'open': 4500.0, 'indicators': {...}},
#     '^VIX': {'open': 15.0, 'indicators': {...}}
# }
```

### 3.7 Multi-Objective Reward

여러 목표 함수 지원:

```cpp
struct StoredTransition {
    vector<float> rewards;  // size=1: scalar, size>1: multi-objective
};

// sample 결과
SampleBatch.rewards  // [batch][n_objectives]
```

---

## 4. 데이터 흐름

### 4.1 Step 실행 흐름

```
┌─────────────┐
│ Python call │  env.step(actions)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 1. Action 해석                            │
│    actions[i] → hmax 범위로 클램프         │
│    양수: 매수, 음수: 매도                  │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 2. Stop Loss 체크                         │
│    현재가 < avg_buy_price * tolerance     │
│    → 전량 매도                            │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 3. 매매 실행                              │
│    매도 우선 (현금 확보)                   │
│    매수 실행                              │
│    수수료 차감                            │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 4. 다음 day로 이동                        │
│    day_ += 1                             │
│    row_indices 업데이트                   │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 5. 상태 직렬화                            │
│    build_state_data() → StateData        │
│    IStateSerializer::serialize()          │
└──────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│ Python dict │  return state
└─────────────┘
```

### 4.2 Sample 실행 흐름

```
┌──────────────────┐
│ buffer.sample()  │
└────────┬─────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│ 1. 인덱스 샘플링                           │
│    min_day 이상인 transition만 대상        │
│    균등 랜덤 샘플링                         │
└───────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│ 2. 시장 데이터 fetch (TBB parallel)        │
│    for each index:                         │
│      get_market_window_multi(tickers, day) │
│      past: [h, 5] OHLCV                    │
│      future: [F, 5] OHLCV                  │
└───────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│ 3. xtensor 배열로 병합                     │
│    [B, n_tickers, T, 5] 형태로             │
│    메모리 연속 배치                        │
└───────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│ 4. Zero-copy numpy 반환                    │
│    capsule로 메모리 공유                   │
└───────────────────────────────────────────┘
```

---

## 5. Python API 요약

### 5.1 FastFinRL

```python
# 생성
env = FastFinRL(
    csv_path,
    initial_amount=30000.0,
    failure_threshold=25000.0,
    hmax=15,
    buy_cost_pct=0.01,
    sell_cost_pct=0.01,
    stop_loss_tolerance=0.8,
    bidding="default",              # "default", "uniform", "adv_uniform"
    stop_loss_calculation="close",  # "close" or "low"
    initial_seed=0,
    tech_indicator_list=[],         # 빈 리스트 = 자동 감지
    macro_tickers=[],
    return_format="json",           # "json" or "vec"
    num_tickers=0,                  # 0 = 전체 사용
    shuffle_tickers=False
)

# 환경 제어
state = env.reset(tickers, seed, shifted_start=0)
state = env.reset()  # 기존 ticker 유지, seed+1
state = env.step(actions)  # actions: [n_tickers]

# 조회
env.get_all_tickers()      # set[str]
env.get_indicator_names()  # set[str]
env.get_max_day()          # int
env.n_indicators()         # int
env.n_tickers()            # int

# 시장 데이터
data = env.get_market_window(ticker, day, h, future)       # dict
data = env.get_market_window_numpy(tickers, day, h, future)  # numpy
```

### 5.2 VecFastFinRL

```python
# 생성
vec_env = VecFastFinRL(
    csv_path,
    n_envs=64,           # 필수
    auto_reset=True,
    shifted_start=5,
    # ... (FastFinRL과 동일한 설정)
)

# 환경 제어
states = vec_env.reset(tickers_list, seeds)   # [N][n_tic], [N]
states = vec_env.reset(tickers_list, seed)    # 단일 seed → 자동 확장
states = vec_env.reset()                      # 기존 유지
states = vec_env.step(actions)                # actions: [N, n_tickers]

# 부분 리셋
states = vec_env.reset_indices([0, 2, 5], seeds)

# 설정
vec_env.set_auto_reset(True)
vec_env.set_return_format("vec")

# 조회
vec_env.num_envs()         # int
vec_env.n_tickers()        # int
vec_env.shifted_start      # int (read-only)
```

### 5.3 ReplayBuffer

```python
# 생성
buffer = ReplayBuffer(
    env,                    # FastFinRL 인스턴스
    capacity=1000000,
    batch_size=256,
    seed=42,
    action_shape=None       # None = (n_tickers,)
)

# 추가
buffer.add(state, action, reward, next_state, done)
buffer.add_transition(state_day, next_state_day, tickers, ...)

# 샘플링
s, a, r, s_next, done = buffer.sample(
    batch_size=256,
    history_length=20,
    future_length=5
)
# s['ohlcv']: [B, n_tic, T, 5]
# s['indicators']: [B, n_tic, T, n_ind]
# s['macro']['ohlcv']: [B, n_macro, T, 5]
# a: [B, ...action_shape]
# r: [B, n_objectives]

# 저장/로드
buffer.save("buffer.bin")
buffer.load("buffer.bin")
```

### 5.4 VecReplayBuffer

```python
# 생성
buffer = VecReplayBuffer(vec_env, capacity=1000000, ...)

# 배치 추가 (vec format 자동 감지)
buffer.add(states, actions, rewards, next_states, dones)

# 개별 추가
buffer.add_transition(env_id, state_day, ...)

# 샘플링 (ReplayBuffer와 동일)
s, a, r, s_next, done = buffer.sample(batch_size, h, future)
```

---

## 6. 메모리 레이아웃

### 6.1 SoA vs AoS

```
┌─────────────────────────────────────────────────────────────┐
│                    VecFastFinRL Memory                       │
├─────────────────────────────────────────────────────────────┤
│  day_:          [d0, d1, d2, ..., d_N-1]       contiguous   │
│  cash_:         [c0, c1, c2, ..., c_N-1]       contiguous   │
│  shares_:       [s00, s01, s10, s11, ...]     [N * n_tic]   │
│  avg_buy_price_:[a00, a01, a10, a11, ...]     [N * n_tic]   │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Flat Index 계산

```cpp
// 2D: [i, j] with stride S → i * S + j
inline size_t flat_2d(size_t i, size_t j, size_t stride) {
    return i * stride + j;
}

// 3D: [i, j, k] with strides Sj, Sk
inline size_t flat_3d(size_t i, size_t j, size_t k,
                      size_t stride_j, size_t stride_k) {
    return i * stride_j * stride_k + j * stride_k + k;
}

// 예: shares_[env_idx, ticker_idx]
shares_[flat_2d(env_idx, ticker_idx, n_tickers_)]

// 예: indicators_[env_idx, ticker_idx, ind_idx]
indicators_[flat_3d(env_idx, ticker_idx, ind_idx, n_tickers_, n_indicators_)]
```

---

## 7. 의존성

| 라이브러리 | 용도 |
|-----------|------|
| **hmdf (DataFrame)** | 시장 데이터 저장 및 조회 |
| **nlohmann/json** | JSON 직렬화 |
| **xtensor** | N차원 배열 (SampleBatch) |
| **pybind11** | Python 바인딩 |
| **TBB** | 병렬 처리 (parallel_for) |
