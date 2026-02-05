#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include "FastFinRL.hpp"

namespace fast_finrl {

struct StoredTransition {
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

class ReplayBuffer {
public:
    using MultiTickerWindowData = FastFinRL::MultiTickerWindowData;

    // capacity: 100K (small), 1M (default), 5M (large)
    explicit ReplayBuffer(std::shared_ptr<const FastFinRL> env, size_t capacity = 1000000, size_t batch_size = 256);

    // Add transition to buffer (circular)
    void add(const StoredTransition& transition);

    // Add transition directly from state dicts (no Python iteration)
    // rewards: single value or vector for multi-objective
    void add_transition(
        int state_day, int next_state_day,
        const std::vector<std::string>& tickers,
        double state_cash, double next_state_cash,
        const std::vector<int>& state_shares,
        const std::vector<int>& next_state_shares,
        const std::vector<double>& state_avg_buy_price,
        const std::vector<double>& next_state_avg_buy_price,
        const std::vector<double>& action,
        const std::vector<double>& rewards, bool done, bool terminal
    );

    // Sample random indices
    std::vector<size_t> sample_indices(size_t batch_size) const;

    // Get transition by index
    const StoredTransition& get(size_t index) const;

    // Get market data for state or next_state
    MultiTickerWindowData get_market_data(size_t index, int h, int future, bool next_state = false) const;

    // Batch sample result
    struct SampleBatch {
        // Per-ticker market data: ticker -> {ohlc, indicators, mask, days}
        // ohlc shape: [batch, h+1, 4] (h history + current)
        // indicators shape: [batch, h+1, n_ind]
        // mask shape: [batch, h+1] (1=valid, 0=padding)
        std::map<std::string, std::vector<double>> s_ohlc;      // [batch * (h+1) * 4]
        std::map<std::string, std::vector<double>> s_indicators;
        std::map<std::string, std::vector<int>> s_mask;         // nullptr equivalent when h=0

        std::map<std::string, std::vector<double>> s_next_ohlc;
        std::map<std::string, std::vector<double>> s_next_indicators;
        std::map<std::string, std::vector<int>> s_next_mask;

        std::vector<double> actions;                 // [batch * n_tickers]
        std::vector<std::vector<double>> rewards;    // [batch][n_objectives]
        std::vector<bool> dones;                     // [batch]
        int n_objectives = 1;

        // Portfolio state
        std::vector<double> state_cash;      // [batch]
        std::vector<double> next_state_cash; // [batch]
        std::vector<int> state_shares;       // [batch * n_tickers]
        std::vector<int> next_state_shares;  // [batch * n_tickers]
        std::vector<double> state_avg_buy_price;      // [batch * n_tickers]
        std::vector<double> next_state_avg_buy_price; // [batch * n_tickers]

        std::vector<std::string> tickers;
        std::vector<std::string> indicator_names;
        int batch_size;
        int h;
        int n_tickers;
        int n_indicators;
    };

    // Sample with market data (parallel fetch)
    SampleBatch sample(int h, size_t batch_size) const;
    SampleBatch sample(int h) const; // uses default batch_size

    size_t size() const;
    size_t capacity() const { return capacity_; }
    void clear();

    // Save/Load buffer to/from file
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::shared_ptr<const FastFinRL> env_;
    std::vector<StoredTransition> buffer_;
    size_t capacity_;
    size_t batch_size_;
    size_t write_idx_ = 0;
    bool full_ = false;
    mutable std::mt19937 rng_;
};

} // namespace fast_finrl
