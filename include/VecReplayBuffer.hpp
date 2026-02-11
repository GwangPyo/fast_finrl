#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <map>
#include "FastFinRL.hpp"
#include "VecFastFinRL.hpp"
#include "xtensor.hpp"

namespace fast_finrl {

struct VecStoredTransition {
    int env_id = 0;

    // State (minimal)
    int state_day = 0;
    std::vector<std::string> tickers;
    float state_cash = 0.0f;
    std::vector<int> state_shares;
    std::vector<float> state_avg_buy_price;

    // Action (flat storage, reshape by action_shape at sample time)
    std::vector<float> action;

    // Reward & flags (supports multi-objective)
    std::vector<float> rewards;  // size=1 for scalar, size>1 for multi-objective
    bool done = false;
    bool terminal = false;

    // Next state (minimal)
    int next_state_day = 0;
    float next_state_cash = 0.0f;
    std::vector<int> next_state_shares;
    std::vector<float> next_state_avg_buy_price;
};

class VecReplayBuffer {
public:
    using MultiTickerWindowData = FastFinRL::MultiTickerWindowData;

    // Constructor - env is used for market data lookup
    // seed: default 42 for reproducibility, -1 for random_device
    // action_shape: empty = default (n_tickers,), otherwise custom shape
    explicit VecReplayBuffer(std::shared_ptr<const FastFinRL> env,
                             size_t capacity = 1000000,
                             size_t batch_size = 256,
                             int64_t seed = 42,
                             std::vector<size_t> action_shape = {});

    // Constructor from VecFastFinRL (uses internal base_env)
    explicit VecReplayBuffer(const VecFastFinRL& vec_env,
                             size_t capacity = 1000000,
                             size_t batch_size = 256,
                             int64_t seed = 42,
                             std::vector<size_t> action_shape = {});

    // Add single transition
    void add(const VecStoredTransition& transition);

    // Add batch of N transitions from VecFastFinRL step
    // This is the primary interface for vectorized environments
    void add_batch(
        int num_envs,
        const std::vector<int>& env_ids,           // [N] - can be 0..N-1 or custom
        const std::vector<int>& state_days,        // [N]
        const std::vector<int>& next_state_days,   // [N]
        const std::vector<std::vector<std::string>>& tickers_list,  // [N][n_tickers]
        const std::vector<float>& state_cash,      // [N]
        const std::vector<float>& next_state_cash, // [N]
        const int* state_shares,                   // [N * n_tickers]
        const int* next_state_shares,              // [N * n_tickers]
        const float* state_avg_buy_price,          // [N * n_tickers]
        const float* next_state_avg_buy_price,     // [N * n_tickers]
        const float* actions,                      // [N * action_flat_size]
        const std::vector<std::vector<float>>& rewards,   // [N][n_objectives]
        const std::vector<bool>& dones,            // [N]
        const std::vector<bool>& terminals,        // [N]
        int n_tickers
    );

    // Sample random indices
    // history_length: only sample transitions with state_day >= max_first_day + history_length
    std::vector<size_t> sample_indices(size_t batch_size) const;
    std::vector<size_t> sample_indices(size_t batch_size, int history_length) const;

    // Get transition by index
    const VecStoredTransition& get(size_t index) const;

    // Get market data for state or next_state
    MultiTickerWindowData get_market_data(size_t index, int h, int future, bool next_state = false) const;

    // Batch sample result - xtensor arrays [batch, n_tickers, time, ...]
    struct SampleBatch {
        // Market data: [B, n_tickers, h+1, 5]
        xt::xarray<float> s_ohlcv;
        xt::xarray<float> s_indicators;      // [B, n_tickers, h+1, n_ind]
        xt::xarray<int> s_mask;              // [B, n_tickers, h+1]
        xt::xarray<float> s_next_ohlcv;
        xt::xarray<float> s_next_indicators;
        xt::xarray<int> s_next_mask;

        // Future market data: [B, n_tickers, future, ...]
        xt::xarray<float> s_future_ohlcv;
        xt::xarray<float> s_future_indicators;
        xt::xarray<int> s_future_mask;
        xt::xarray<float> s_next_future_ohlcv;
        xt::xarray<float> s_next_future_indicators;
        xt::xarray<int> s_next_future_mask;

        // Macro ticker data: [B, n_macro, time, ...]
        xt::xarray<float> macro_ohlcv;
        xt::xarray<float> macro_indicators;
        xt::xarray<int> macro_mask;
        xt::xarray<float> macro_next_ohlcv;
        xt::xarray<float> macro_next_indicators;
        xt::xarray<int> macro_next_mask;
        xt::xarray<float> macro_future_ohlcv;
        xt::xarray<float> macro_future_indicators;
        xt::xarray<int> macro_future_mask;
        xt::xarray<float> macro_next_future_ohlcv;
        xt::xarray<float> macro_next_future_indicators;
        xt::xarray<int> macro_next_future_mask;

        std::vector<int> env_ids;                    // [batch]
        xt::xarray<float> actions;                   // [batch, ...action_shape]
        std::vector<std::vector<float>> rewards;     // [batch][n_objectives]
        std::vector<bool> dones;                     // [batch]
        int n_objectives = 1;

        // Portfolio state
        std::vector<float> state_cash;               // [batch]
        std::vector<float> next_state_cash;
        xt::xarray<int> state_shares;                // [batch, n_tickers]
        xt::xarray<int> next_state_shares;
        xt::xarray<float> state_avg_buy_price;       // [batch, n_tickers]
        xt::xarray<float> next_state_avg_buy_price;

        std::vector<std::vector<std::string>> tickers;  // [batch][n_tickers] - per-sample tickers
        std::vector<std::string> macro_tickers;         // [n_macro]
        std::vector<std::string> indicator_names;
        std::vector<size_t> action_shape;            // action shape for reshape
        int batch_size;
        int history_length;
        int future_length;
        int n_tickers;
        int n_macro_tickers = 0;
        int n_indicators;
    };

    // Sample with market data (parallel fetch)
    // history_length: 0 = no history, >0 = h history days
    // future_length: 0 = no future, >0 = future days
    SampleBatch sample(size_t batch_size, int history_length = 0, int future_length = 0) const;
    SampleBatch sample(int history_length = 0) const;  // uses default batch_size

    size_t size() const;
    size_t capacity() const { return capacity_; }
    void clear();

    // Save/Load buffer to/from file
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::shared_ptr<const FastFinRL> env_;
    std::vector<VecStoredTransition> buffer_;
    size_t capacity_;
    size_t batch_size_;
    std::vector<size_t> action_shape_;
    size_t write_idx_ = 0;
    bool full_ = false;
    mutable std::mt19937 rng_;

    // Cached metadata for faster sampling
    std::vector<std::string> cached_indicator_names_;
    std::vector<std::string> cached_macro_tickers_;
    int n_indicators_ = 0;
    int n_macro_tickers_ = 0;

    // Pre-allocated sample buffers (reused across calls)
    mutable std::vector<float> sample_ohlcv_buf_;
    mutable std::vector<float> sample_ind_buf_;
    mutable std::vector<int> sample_mask_buf_;
};

} // namespace fast_finrl
