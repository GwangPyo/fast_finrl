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
        vector<string> date;                // [N] - date string
        vector<double> cash;                // [N]
        vector<int> shares;                 // [N * n_tickers]
        vector<double> avg_buy_price;       // [N * n_tickers]
        vector<double> open;                // [N * n_tickers] - current day open only
        vector<double> indicators;          // [N * n_tickers * n_ind]
        vector<double> macro_open;          // [N * n_macro] - current day open only
        vector<double> macro_indicators;    // [N * n_macro * n_ind]
        vector<double> reward;              // [N]
        vector<uint8_t> done;               // [N] (not bool - pybind11 issue)
        vector<uint8_t> terminal;           // [N] (not bool - pybind11 issue)
        vector<double> total_asset;         // [N]
        vector<int> num_stop_loss;          // [N] - stop loss count per env
        vector<int> trades;                 // [N] - trade count per env
        vector<double> loss_cut_amount;     // [N] - stop loss amount per env
        int num_envs = 0;
        int n_tickers = 0;
        int n_indicators = 0;
        int n_macro = 0;
    };

    // Constructor - all parameters passed directly, no config struct
    explicit VecFastFinRL(
        const string& csv_path,
        int n_envs,
        double initial_amount,
        double failure_threshold,
        int hmax,
        double buy_cost_pct,
        double sell_cost_pct,
        double stop_loss_tolerance,
        const string& bidding,
        const string& stop_loss_calculation,
        int64_t initial_seed,
        const vector<string>& tech_indicator_list,
        const vector<string>& macro_tickers,
        bool auto_reset,
        ReturnFormat return_format,
        int num_tickers,
        bool shuffle_tickers,
        int shifted_start
    );

    // Core API
    // Full reset with explicit tickers and seeds
    // tickers_list: [N][n_tickers] - each env can have different tickers (same count)
    StepResult reset(const vector<vector<string>>& tickers_list, const vector<int64_t>& seeds);

    // Simplified reset: single seed, auto-expand to all envs
    // seed: base seed (each env gets seed derived via prime multiplication)
    // If tickers_list is empty, use previous tickers (or shuffle if enabled)
    StepResult reset(const vector<vector<string>>& tickers_list, int64_t seed);

    // Reset with no args: keep same tickers, auto-increment seeds
    StepResult reset();

    // Partial reset - reset only specified environment indices
    // indices: env indices to reset (e.g., [0, 2, 5])
    // seeds: seeds for each index (same length as indices)
    // Returns updated StepResult (only indices in the list are modified)
    StepResult reset_indices(const vector<int>& indices, const vector<int64_t>& seeds);

    StepResult step(const double* actions);  // [N, n_tickers]

    // Auto-reset control
    void set_auto_reset(bool enabled) { auto_reset_ = enabled; }

    // Return format control
    void set_return_format(ReturnFormat fmt) { return_format_ = fmt; }
    ReturnFormat return_format() const { return return_format_; }

    // Accessors
    int num_envs() const { return num_envs_; }
    int n_tickers() const { return n_tickers_; }
    int n_indicators() const { return n_indicators_; }
    int n_macro() const { return n_macro_; }
    set<string> get_all_tickers() const { return base_env_->get_all_tickers(); }
    set<string> get_indicator_names() const { return base_env_->get_indicator_names(); }
    const vector<string>& get_macro_tickers() const { return base_env_->get_macro_tickers(); }
    const vector<vector<string>>& get_tickers() const { return tickers_; }
    shared_ptr<const FastFinRL> get_base_env() const { return base_env_; }

    // Get current state for all environments as list of JSON
    vector<nlohmann::json> get_state() const;

    // Configuration accessors (read-only after construction)
    bool auto_reset() const { return auto_reset_; }
    double initial_amount() const { return initial_amount_; }
    double failure_threshold() const { return failure_threshold_; }
    int hmax() const { return hmax_; }
    double buy_cost_pct() const { return buy_cost_pct_; }
    double sell_cost_pct() const { return sell_cost_pct_; }
    double stop_loss_tolerance() const { return stop_loss_tolerance_; }
    const string& bidding() const { return bidding_; }
    const string& stop_loss_calculation() const { return stop_loss_calculation_; }
    bool shuffle_tickers() const { return shuffle_tickers_; }

    // Public read-only members
    int shifted_start = 0;  // start day offset for history window

private:
    // Base environment for shared market data
    shared_ptr<FastFinRL> base_env_;

    // Configuration (stored directly, no config struct)
    double initial_amount_;
    double failure_threshold_;
    int hmax_;
    double buy_cost_pct_;
    double sell_cost_pct_;
    double stop_loss_tolerance_;
    string bidding_;
    string stop_loss_calculation_;
    bool shuffle_tickers_;
    bool auto_reset_;
    ReturnFormat return_format_;

    // Environment dimensions
    int num_envs_ = 0;
    int num_tickers_config_ = 0;  // Original user setting (0 = all tickers)
    int n_tickers_ = 0;           // Actual ticker count after initialization
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
    int64_t last_base_seed_ = 0;                  // base seed from last reset

    // Per-env episode tracking
    vector<int> num_stop_loss_;                   // [N]
    vector<int> trades_;                          // [N]
    vector<double> begin_total_asset_;            // [N]
    vector<double> loss_cut_amount_;              // [N] - stop loss amount

    // Pre-allocated output buffer
    StepResult buffer_;

    // Internal helpers
    void reset_single_env(size_t env_idx, int64_t seed, const vector<string>& tickers = {});
    void step_env(size_t env_idx, const double* actions);
    void fill_obs(size_t env_idx);

    // Market data access helpers
    size_t get_row_idx(size_t env_idx, size_t ticker_idx, int day) const;
    double get_close(size_t env_idx, size_t ticker_idx) const;
    double get_close_at_day(size_t env_idx, size_t ticker_idx, int day) const;

    // Bid price functions
    using VecBidFunction = function<double(size_t env_idx, size_t ticker_idx)>;
    VecBidFunction* active_sell_bid_ = nullptr;
    VecBidFunction* active_buy_bid_ = nullptr;
    map<string, VecBidFunction> sell_bid_options_;
    map<string, VecBidFunction> buy_bid_options_;
    void init_bid_options();

    // Trading helpers (per-env)
    double calculate_total_asset(size_t env_idx) const;
    void check_stop_loss(size_t env_idx);
    int sell_stock(size_t env_idx, size_t ticker_idx, int action);
    int buy_stock(size_t env_idx, size_t ticker_idx, int action);
};

} // namespace fast_finrl
