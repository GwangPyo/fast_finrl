#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <map>
#include "FastFinRL.hpp"
#include "VecFastFinRL.hpp"

namespace fast_finrl {

struct VecStoredTransition {
    int env_id = 0;

    // State (minimal)
    int state_day = 0;
    std::vector<std::string> tickers;
    double state_cash = 0.0;
    std::vector<int> state_shares;
    std::vector<double> state_avg_buy_price;

    // Action
    std::vector<double> action;

    // Reward & flags (supports multi-objective)
    std::vector<double> rewards;  // size=1 for scalar, size>1 for multi-objective
    bool done = false;
    bool terminal = false;

    // Next state (minimal)
    int next_state_day = 0;
    double next_state_cash = 0.0;
    std::vector<int> next_state_shares;
    std::vector<double> next_state_avg_buy_price;
};

class VecReplayBuffer {
public:
    using MultiTickerWindowData = FastFinRL::MultiTickerWindowData;

    // Constructor - env is used for market data lookup
    // seed: default 42 for reproducibility, -1 for random_device
    explicit VecReplayBuffer(std::shared_ptr<const FastFinRL> env,
                             size_t capacity = 1000000,
                             size_t batch_size = 256,
                             int64_t seed = 42);

    // Constructor from VecFastFinRL (uses internal base_env)
    explicit VecReplayBuffer(const VecFastFinRL& vec_env,
                             size_t capacity = 1000000,
                             size_t batch_size = 256,
                             int64_t seed = 42);

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
        const std::vector<double>& state_cash,     // [N]
        const std::vector<double>& next_state_cash,// [N]
        const int* state_shares,                   // [N * n_tickers]
        const int* next_state_shares,              // [N * n_tickers]
        const double* state_avg_buy_price,         // [N * n_tickers]
        const double* next_state_avg_buy_price,    // [N * n_tickers]
        const double* actions,                     // [N * n_tickers]
        const std::vector<std::vector<double>>& rewards,  // [N][n_objectives]
        const std::vector<bool>& dones,            // [N]
        const std::vector<bool>& terminals,        // [N]
        int n_tickers
    );

    // Sample random indices
    std::vector<size_t> sample_indices(size_t batch_size) const;

    // Get transition by index
    const VecStoredTransition& get(size_t index) const;

    // Get market data for state or next_state
    MultiTickerWindowData get_market_data(size_t index, int h, int future, bool next_state = false) const;

    // Batch sample result
    struct SampleBatch {
        // Per-ticker market data
        std::map<std::string, std::vector<double>> s_ohlcv;     // [batch * (h+1) * 5] - OHLCV
        std::map<std::string, std::vector<double>> s_indicators;
        std::map<std::string, std::vector<int>> s_mask;

        std::map<std::string, std::vector<double>> s_next_ohlcv;
        std::map<std::string, std::vector<double>> s_next_indicators;
        std::map<std::string, std::vector<int>> s_next_mask;

        // Macro ticker market data
        std::map<std::string, std::vector<double>> macro_ohlcv;
        std::map<std::string, std::vector<double>> macro_indicators;
        std::map<std::string, std::vector<int>> macro_mask;
        std::map<std::string, std::vector<double>> macro_next_ohlcv;
        std::map<std::string, std::vector<double>> macro_next_indicators;
        std::map<std::string, std::vector<int>> macro_next_mask;

        std::vector<int> env_ids;                    // [batch]
        std::vector<double> actions;                 // [batch * n_tickers]
        std::vector<std::vector<double>> rewards;    // [batch][n_objectives]
        std::vector<bool> dones;                     // [batch]
        int n_objectives = 1;

        // Portfolio state
        std::vector<double> state_cash;
        std::vector<double> next_state_cash;
        std::vector<int> state_shares;
        std::vector<int> next_state_shares;
        std::vector<double> state_avg_buy_price;
        std::vector<double> next_state_avg_buy_price;

        std::vector<std::vector<std::string>> tickers;  // [batch][n_tickers] - per-sample tickers
        std::vector<std::string> unique_tickers;          // union of all tickers in batch
        std::vector<std::string> macro_tickers;
        std::vector<std::string> indicator_names;
        int batch_size;
        int h;
        int n_tickers;
        int n_macro_tickers = 0;
        int n_indicators;
    };

    // Sample with market data (parallel fetch)
    SampleBatch sample(int h, size_t batch_size) const;
    SampleBatch sample(int h) const;  // uses default batch_size

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
    size_t write_idx_ = 0;
    bool full_ = false;
    mutable std::mt19937 rng_;

    // Cached metadata for faster sampling
    std::vector<std::string> cached_indicator_names_;
    std::vector<std::string> cached_macro_tickers_;
    int n_indicators_ = 0;
    int n_macro_tickers_ = 0;

    // Pre-allocated sample buffers (reused across calls)
    mutable std::vector<double> sample_ohlcv_buf_;
    mutable std::vector<double> sample_ind_buf_;
    mutable std::vector<int> sample_mask_buf_;
};

} // namespace fast_finrl
