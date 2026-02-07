#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <functional>
#include <memory>
#include <optional>
#include <nlohmann/json.hpp>
#include <DataFrame/DataFrame.h>
#include "StateSerializer.hpp"

using namespace std;

namespace fast_finrl {

// Return format for state serialization
enum class ReturnFormat {
    Json,  // dict/List[dict] with nested structure
    Vec    // dict with batched numpy arrays
};

// Configuration struct for FastFinRL constructor
struct FastFinRLConfig {
    double initial_amount = 30000.0;
    int hmax = 15;
    double buy_cost_pct = 0.01;
    double sell_cost_pct = 0.01;
    double stop_loss_tolerance = 0.8;
    string bidding = "default";
    string stop_loss_calculation = "close";
    int64_t initial_seed = 0;
    vector<string> tech_indicator_list = {};  // empty = auto-detect from CSV
    vector<string> macro_tickers = {};        // tickers always included in state.macro
    ReturnFormat return_format = ReturnFormat::Json;  // "json" or "vec"
    int num_tickers = 0;           // 0 = use all tickers provided in reset()
    bool shuffle_tickers = false;  // if true, randomly select num_tickers from all_tickers
};

class FastFinRL {
public:
    using MyDataFrame = hmdf::StdDataFrame<unsigned long>;
    using BidFunction = function<double(size_t)>;

    // Configuration attributes (public, set directly like Python)
    double initial_amount;
    int hmax;
    double buy_cost_pct;
    double sell_cost_pct;
    double stop_loss_tolerance;
    string bidding;
    string stop_loss_calculation;
    ReturnFormat return_format = ReturnFormat::Json;

    // Constructor with configuration
    explicit FastFinRL(const string& csv_path, const FastFinRLConfig& config = FastFinRLConfig{});

    // Core API
    nlohmann::json reset(const vector<string>& ticker_list, int64_t seed, int shifted_start = 0);
    nlohmann::json reset();  // No-arg reset: keep same tickers, increment seed
    nlohmann::json step(const vector<double>& actions);

    // Accessors
    set<string> get_indicator_names() const;
    set<string> get_all_tickers() const { return all_tickers_; }
    int get_max_day() const { return max_day_; }
    nlohmann::json get_state() const;
    double get_raw_value(const string& ticker, int day, const string& column) const;
    const vector<string>& get_macro_tickers() const { return macro_tickers_; }

    // Market data window query (like DB)
    // Returns past [day-h, day-1] and future [day+1, day+future] market data
    nlohmann::json get_market_window(const string& ticker, int day, int h, int future) const;

    // Fast version - returns flat arrays instead of nested JSON objects
    // Returns: {ohlc: [h+1+f, 4], indicators: [h+1+f, n], mask: [h+1+f], days: [h+1+f]}
    nlohmann::json get_market_window_flat(const string& ticker, int day, int h, int future) const;

    // Raw data struct for numpy binding (no JSON overhead)
    struct MarketWindowData {
        vector<double> ohlcv;       // flat [total_len * 5]
        vector<double> indicators;  // flat [total_len * n_indicators]
        vector<int> mask;           // [total_len]
        vector<int> days;           // [total_len]
        vector<string> indicator_names;
        int total_len;
        int n_indicators;
    };
    MarketWindowData get_market_window_raw(const string& ticker, int day, int h, int future) const;

    // Separated past/current/future data for a single ticker
    struct TickerWindowData {
        // Past: [h, 5], [h, n_ind], [h] - OHLCV
        vector<double> past_ohlcv;
        vector<double> past_indicators;
        vector<int> past_mask;
        vector<int> past_days;
        // Current: open only (scalar), indicators [n_ind]
        double current_open;
        vector<double> current_indicators;
        int current_mask;
        int current_day;
        // Future: [f, 5], [f, n_ind], [f] - OHLCV
        vector<double> future_ohlcv;
        vector<double> future_indicators;
        vector<int> future_mask;
        vector<int> future_days;
    };

    // Multi-ticker window data
    struct MultiTickerWindowData {
        map<string, TickerWindowData> tickers;
        vector<string> indicator_names;
        int h;
        int future;
        int n_indicators;
    };
    MultiTickerWindowData get_market_window_multi(const vector<string>& ticker_list, int day, int h, int future) const;

    // Batch fill for replay buffer optimization
    // Fills multiple samples directly into pre-allocated arrays
    // samples: [(global_ticker_idx, day), ...] - N samples
    // ohlcv_out: [N * time_len * 5] - pre-allocated (OHLCV)
    // ind_out: [N * time_len * n_ind] - pre-allocated
    // mask_out: [N * time_len] - pre-allocated
    void fill_market_batch(
        const vector<pair<size_t, int>>& samples,  // [(global_idx, day), ...]
        int h,  // history length (time_len = h + 1)
        double* ohlcv_out,
        double* ind_out,
        int* mask_out
    ) const;

    // Get global ticker index (for batch fill)
    size_t get_ticker_global_idx(const string& ticker) const {
        auto it = ticker_global_idx_.find(ticker);
        if (it == ticker_global_idx_.end()) {
            throw runtime_error("Ticker not found: " + ticker);
        }
        return it->second;
    }

    int get_ticker_first_day(const string& ticker) const {
        auto it = ticker_first_day_.find(ticker);
        if (it == ticker_first_day_.end()) return 0;
        return it->second;
    }

    int get_n_indicators() const { return static_cast<int>(indicator_cols_.size()); }

private:
    // DataFrame storage
    MyDataFrame df_;

    // Dynamic columns
    set<string> excluded_columns_;
    set<string> indicator_names_;
    set<string> all_tickers_;
    vector<string> tech_indicator_list_;  // user-specified indicators (empty = auto)

    // Macro tickers (always included in state.macro)
    vector<string> macro_tickers_;
    set<string> macro_ticker_set_;  // O(1) lookup

    // Shuffle tickers config
    int num_tickers_ = 0;
    bool shuffle_tickers_ = false;

    // Pre-computed index: (ticker, day) -> row_index for O(1) lookup
    map<pair<string, int>, size_t> row_index_map_;

    // Pre-computed per-ticker lookup table: ticker_row_table_[ticker_idx][day] = row_index
    // Dense array for O(1) lookup
    map<string, size_t> ticker_global_idx_;  // ticker -> index in ticker_row_table_
    vector<vector<size_t>> ticker_row_table_; // [ticker_idx][day] -> row_index
    map<string, int> ticker_first_day_;      // ticker -> first available day

    // Active ticker cache
    vector<size_t> active_global_idx_;       // global idx for active tickers
    vector<int> active_first_day_;           // first_day for active tickers

    // Cached column references for fast access
    using DoubleColRef = reference_wrapper<const vector<double>>;
    using StringColRef = reference_wrapper<const vector<string>>;
    optional<DoubleColRef> col_open_;
    optional<DoubleColRef> col_high_;
    optional<DoubleColRef> col_low_;
    optional<DoubleColRef> col_close_;
    optional<DoubleColRef> col_volume_;
    optional<StringColRef> col_date_;
    vector<pair<string, DoubleColRef>> indicator_cols_;  // vector for cache locality

    // Cached row indices for current step
    vector<size_t> active_row_indices_;

    // State variables
    int64_t current_seed_ = 0;
    mt19937 rng_;
    int day_ = 0;
    int max_day_ = 0;
    double cash_ = 0.0;

    // Per-ticker state (indexed by ticker order)
    vector<string> active_tickers_;
    map<string, size_t> ticker_to_idx_;
    vector<int> shares_;
    vector<double> avg_buy_price_;

    // Episode tracking
    int num_stop_loss_ = 0;
    int trades_ = 0;
    int trades_this_step_ = 0;
    double cost_ = 0.0;
    double begin_total_asset_ = 0.0;
    double loss_cut_amount_ = 0.0;
    bool in_step_ = false;

    // Debug: per-ticker execution info for current step
    struct TradeInfo {
        double fill_price = 0.0;      // execution price
        double cost = 0.0;            // transaction cost
        int quantity = 0;             // filled quantity (+ buy, - sell)
    };
    vector<TradeInfo> trade_info_;

    // Internal helpers
    void load_dataframe(const string& path);
    void build_index_tables();
    void extract_indicator_names();
    void setup_tickers(const vector<string>& tickers);
    void update_row_indices();

    // DataFrame accessors
    size_t find_row_index(const string& ticker, int day) const;
    double get_price(size_t ticker_idx, const string& price_type) const;
    double get_randomized_price(size_t ticker_idx, const string& option);
    double calculate_total_asset() const;
    string get_date() const;

    // Trading helpers
    int sell_stock(size_t index, int action);
    int buy_stock(size_t index, int action);
    void check_stop_loss();

    // State data builder for serialization
    StateData build_state_data(bool include_step_info, double reward = 0.0, bool done = false, bool terminal = false) const;

    // Bidding option function maps
    map<string, BidFunction> sell_bid_options_;
    map<string, BidFunction> buy_bid_options_;
    void init_bid_options();
    double get_sell_bid_price(size_t ticker_idx);
    double get_buy_bid_price(size_t ticker_idx);

    // State serializer
    unique_ptr<IStateSerializer> state_serializer_;
};

} // namespace fast_finrl
