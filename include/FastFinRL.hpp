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

using namespace std;

namespace fast_finrl {

// Configuration struct for FastFinRL constructor
struct FastFinRLConfig {
    double initial_amount = 30000.0;
    int hmax = 15;
    double buy_cost_pct = 0.01;
    double sell_cost_pct = 0.01;
    double stop_loss_tolerance = 0.8;
    string bidding = "default";
    string stop_loss_calculation = "close";
    int initial_seed = 0;
};

// Forward declaration
class FastFinRL;

class JsonHandler {
public:
    explicit JsonHandler(const FastFinRL& env) : env_(env) {}

    nlohmann::json build_state_json() const;
    nlohmann::json build_step_result_json(double reward, bool done, bool terminal) const;
    nlohmann::json build_market_json() const;
    nlohmann::json build_portfolio_json(bool include_total_asset = false) const;

private:
    const FastFinRL& env_;
};

class FastFinRL {
public:
    using MyDataFrame = hmdf::StdDataFrame<unsigned long>;
    using BidFunction = function<double(size_t)>;

    friend class JsonHandler;

    // Configuration attributes (public, set directly like Python)
    double initial_amount;
    int hmax;
    double buy_cost_pct;
    double sell_cost_pct;
    double stop_loss_tolerance;
    string bidding;
    string stop_loss_calculation;

    // Constructor with configuration
    explicit FastFinRL(const string& csv_path, const FastFinRLConfig& config = FastFinRLConfig{});

    // Core API
    nlohmann::json reset(const vector<string>& ticker_list, int seed);
    nlohmann::json step(const vector<double>& actions);

    // Accessors
    set<string> get_indicator_names() const;
    nlohmann::json get_state() const;
    double get_raw_value(const string& ticker, int day, const string& column) const;

private:
    // DataFrame storage
    MyDataFrame df_;

    // Dynamic columns
    set<string> excluded_columns_;
    set<string> indicator_names_;
    set<string> all_tickers_;

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
    optional<StringColRef> col_date_;
    vector<pair<string, DoubleColRef>> indicator_cols_;  // vector for cache locality

    // Cached row indices for current step
    vector<size_t> active_row_indices_;

    // State variables
    int current_seed_ = 0;
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
    string convert_csv_to_dataframe_format(const string& csv_path);
    void load_dataframe(const string& csv_path);
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

    // Bidding option function maps
    map<string, BidFunction> sell_bid_options_;
    map<string, BidFunction> buy_bid_options_;
    void init_bid_options();
    double get_sell_bid_price(size_t ticker_idx);
    double get_buy_bid_price(size_t ticker_idx);

    // JSON handler
    unique_ptr<JsonHandler> json_handler_;
};

} // namespace fast_finrl
