#pragma once

#include <string>
#include <vector>
#include <random>
#include <memory>
#include "FastFinRL.hpp"

using namespace std;

namespace fast_finrl {

class VecFastFinRL {
public:
    // Step result containing all per-environment data (SoA layout)
    struct StepResult {
        vector<int> day;                    // [N]
        vector<double> cash;                // [N]
        vector<int> shares;                 // [N * n_tickers]
        vector<double> avg_buy_price;       // [N * n_tickers]
        vector<double> ohlc;                // [N * n_tickers * 4]
        vector<double> indicators;          // [N * n_tickers * n_ind]
        vector<double> macro_ohlc;          // [N * n_macro * 4]
        vector<double> macro_indicators;    // [N * n_macro * n_ind]
        vector<double> reward;              // [N]
        vector<uint8_t> done;               // [N] (not bool - pybind11 issue)
        vector<uint8_t> terminal;           // [N] (not bool - pybind11 issue)
        vector<double> total_asset;         // [N]
        int num_envs = 0;
        int n_tickers = 0;
        int n_indicators = 0;
        int n_macro = 0;
    };

    // Constructor
    explicit VecFastFinRL(const string& csv_path, const FastFinRLConfig& config = FastFinRLConfig{});

    // Core API
    // tickers_list: [N][n_tickers] - each env can have different tickers (same count)
    StepResult reset(const vector<vector<string>>& tickers_list, const vector<int64_t>& seeds);

    // Partial reset - reset only specified environment indices
    // indices: env indices to reset (e.g., [0, 2, 5])
    // seeds: seeds for each index (same length as indices)
    // Returns updated StepResult (only indices in the list are modified)
    StepResult reset_indices(const vector<int>& indices, const vector<int64_t>& seeds);

    StepResult step(const double* actions);  // [N, n_tickers]

    // Auto-reset control
    void set_auto_reset(bool enabled) { auto_reset_ = enabled; }

    // Accessors
    int num_envs() const { return num_envs_; }
    int n_tickers() const { return n_tickers_; }
    int n_indicators() const { return n_indicators_; }
    int n_macro() const { return n_macro_; }
    set<string> get_all_tickers() const { return base_env_->get_all_tickers(); }
    const vector<string>& get_macro_tickers() const { return base_env_->get_macro_tickers(); }
    const vector<vector<string>>& get_tickers() const { return tickers_; }

    // Configuration (read-only after construction)
    const FastFinRLConfig& config() const { return config_; }
    bool auto_reset() const { return auto_reset_; }

private:
    // Base environment for shared market data
    shared_ptr<FastFinRL> base_env_;
    FastFinRLConfig config_;
    bool auto_reset_ = true;

    // Environment dimensions
    int num_envs_ = 0;
    int n_tickers_ = 0;
    int n_indicators_ = 0;
    int n_macro_ = 0;
    int max_day_ = 0;

    // Per-env tickers (each env can have different tickers)
    vector<vector<string>> tickers_;              // [N][n_tickers]
    vector<size_t> ticker_global_idx_;            // [N * n_tickers] - flattened
    vector<int> ticker_first_day_;                // [N * n_tickers] - first available day per ticker

    // Per-env state (SoA layout for cache efficiency)
    vector<int> day_;                             // [N]
    vector<double> cash_;                         // [N]
    vector<int> shares_;                          // [N * n_tickers]
    vector<double> avg_buy_price_;                // [N * n_tickers]
    vector<int64_t> seeds_;                       // [N]
    vector<mt19937> rngs_;                        // [N]

    // Per-env episode tracking
    vector<int> num_stop_loss_;                   // [N]
    vector<int> trades_;                          // [N]
    vector<double> begin_total_asset_;            // [N]

    // Pre-allocated output buffer
    StepResult buffer_;

    // Internal helpers
    void reset_env(size_t env_idx, int64_t seed);
    void step_env(size_t env_idx, const double* actions);
    void fill_obs(size_t env_idx);

    // Market data access helpers
    size_t get_row_idx(size_t env_idx, size_t ticker_idx, int day) const;
    double get_close(size_t env_idx, size_t ticker_idx) const;
    double get_close_at_day(size_t env_idx, size_t ticker_idx, int day) const;
    double get_bid_price(size_t env_idx, size_t ticker_idx, const string& side);

    // Trading helpers (per-env)
    double calculate_total_asset(size_t env_idx) const;
    void check_stop_loss(size_t env_idx);
    int sell_stock(size_t env_idx, size_t ticker_idx, int action);
    int buy_stock(size_t env_idx, size_t ticker_idx, int action);
};

} // namespace fast_finrl
