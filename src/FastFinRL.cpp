#include "FastFinRL.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

// DataFrame selector macros for filtering by ticker and day
#define DF_SELECTOR_BY_TICKER_DAY(ticker_var, day_var) \
    [&ticker_var, day_var](const unsigned long&, const string& tic, const int& d) -> bool { \
        return tic == ticker_var && d == day_var; \
    }

#define DF_SELECTOR_BY_TICKER_DAY_MEMBER(ticker_var, day_member) \
    [&ticker_var, this](const unsigned long&, const string& tic, const int& d) -> bool { \
        return tic == ticker_var && d == this->day_member; \
    }

namespace fast_finrl {

FastFinRL::FastFinRL(const string& csv_path, const FastFinRLConfig& config)
    : initial_amount(config.initial_amount)
    , hmax(config.hmax)
    , buy_cost_pct(config.buy_cost_pct)
    , sell_cost_pct(config.sell_cost_pct)
    , stop_loss_tolerance(config.stop_loss_tolerance)
    , bidding(config.bidding)
    , stop_loss_calculation(config.stop_loss_calculation)
    , current_seed_(config.initial_seed)
    , rng_(config.initial_seed)
    , tech_indicator_list_(config.tech_indicator_list)
    , json_handler_(make_unique<JsonHandler>(*this))
{
    excluded_columns_ = {"day", "date", "tic", "open", "high", "low", "close", "volume", "start"};
    load_dataframe(csv_path);
    extract_indicator_names();
    init_bid_options();
}

set<string> FastFinRL::get_indicator_names() const {
    return indicator_names_;
}

string FastFinRL::convert_csv_to_dataframe_format(const string& csv_path) {
    ifstream infile(csv_path);
    if (!infile.is_open()) {
        throw runtime_error("Cannot open file: " + csv_path);
    }

    // Read header line
    string header_line;
    getline(infile, header_line);

    // Parse column names
    vector<string> col_names;
    stringstream header_ss(header_line);
    string col_name;
    while (getline(header_ss, col_name, ',')) {
        col_names.push_back(col_name);
    }

    // Determine column types: 'date' and 'tic' are strings, rest are numeric
    vector<string> col_types;
    for (const auto& name : col_names) {
        if (name == "date" || name == "tic") {
            col_types.push_back("string");
        } else {
            col_types.push_back("double");
        }
    }

    // Read all data rows
    vector<vector<string>> data(col_names.size());
    string line;
    size_t num_rows = 0;
    while (getline(infile, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string value;
        size_t col_idx = 0;
        while (getline(ss, value, ',') && col_idx < col_names.size()) {
            data[col_idx].push_back(value);
            col_idx++;
        }
        num_rows++;
    }
    infile.close();

    // Generate DataFrame format
    stringstream output;

    // First column is always the index (INDEX:num_rows:<ulong>:0,1,2,...)
    output << "INDEX:" << num_rows << ":<ulong>:";
    for (size_t i = 0; i < num_rows; ++i) {
        output << i;
        if (i < num_rows - 1) output << ",";
    }
    output << ",\n";

    // Add each column
    for (size_t i = 0; i < col_names.size(); ++i) {
        output << col_names[i] << ":" << num_rows << ":<" << col_types[i] << ">:";
        for (size_t j = 0; j < data[i].size(); ++j) {
            output << data[i][j];
            if (j < data[i].size() - 1) output << ",";
        }
        output << ",\n";
    }

    // Write to temp file
    string temp_path = csv_path + ".df";
    ofstream outfile(temp_path);
    outfile << output.str();
    outfile.close();

    return temp_path;
}

void FastFinRL::load_dataframe(const string& csv_path) {
    // Convert standard CSV to DataFrame format and load
    string df_path = convert_csv_to_dataframe_format(csv_path);
    df_.read(df_path.c_str(), hmdf::io_format::csv);

    // Get all unique tickers using DataFrame library
    auto unique_tics = df_.get_col_unique_values<string>("tic");
    all_tickers_.insert(unique_tics.begin(), unique_tics.end());

    // Generate 'day' column from unique dates (dense ranking like Python)
    // Similar to: df['day'] = df['date'].rank(method='dense').astype(int) - 1
    const auto& date_col = df_.get_column<string>("date");

    // Get unique dates using DataFrame library, then sort
    auto sorted_dates = df_.get_col_unique_values<string>("date");
    sort(sorted_dates.begin(), sorted_dates.end());

    // Create date -> day mapping
    map<string, int> date_to_day;
    for (size_t i = 0; i < sorted_dates.size(); ++i) {
        date_to_day[sorted_dates[i]] = static_cast<int>(i);
    }

    // Generate day values for each row
    vector<int> day_values;
    day_values.reserve(date_col.size());
    for (const auto& date : date_col) {
        day_values.push_back(date_to_day[date]);
    }

    // Add or replace 'day' column in DataFrame
    if (df_.has_column("day")) {
        df_.remove_column<int>("day");
    }
    df_.load_column("day", move(day_values));

    // Set max_day
    max_day_ = static_cast<int>(sorted_dates.size());

    // Build pre-computed lookup tables
    const auto& tic_col = df_.get_column<string>("tic");
    const auto& day_col = df_.get_column<int>("day");
    row_index_map_.clear();
    ticker_first_day_.clear();
    ticker_global_idx_.clear();

    // First pass: build row_index_map and assign global indices
    size_t next_global_idx = 0;
    for (size_t i = 0; i < tic_col.size(); ++i) {
        const auto& tic = tic_col[i];
        int day = day_col[i];
        row_index_map_[{tic, day}] = i;

        if (ticker_global_idx_.find(tic) == ticker_global_idx_.end()) {
            ticker_global_idx_[tic] = next_global_idx++;
        }

        auto it = ticker_first_day_.find(tic);
        if (it == ticker_first_day_.end() || day < it->second) {
            ticker_first_day_[tic] = day;
        }
    }

    // Build dense lookup table: ticker_row_table_[global_idx][day] = row_index
    ticker_row_table_.assign(next_global_idx, vector<size_t>(max_day_, 0));
    for (size_t i = 0; i < tic_col.size(); ++i) {
        size_t global_idx = ticker_global_idx_[tic_col[i]];
        int day = day_col[i];
        ticker_row_table_[global_idx][day] = i;
    }

    // Cache column references
    col_open_.emplace(df_.get_column<double>("open"));
    col_high_.emplace(df_.get_column<double>("high"));
    col_low_.emplace(df_.get_column<double>("low"));
    col_close_.emplace(df_.get_column<double>("close"));
    col_date_.emplace(df_.get_column<string>("date"));
}

void FastFinRL::extract_indicator_names() {
    indicator_names_.clear();
    indicator_cols_.clear();

    if (!tech_indicator_list_.empty()) {
        // Use user-specified indicator list
        for (const auto& name : tech_indicator_list_) {
            indicator_names_.insert(name);
            indicator_cols_.emplace_back(name, cref(df_.get_column<double>(name.c_str())));
        }
    } else {
        // Auto-detect: all columns except excluded ones
        auto col_info = df_.get_columns_info<int, double, string, long, unsigned long>();

        for (const auto& [col_name, col_size, col_type] : col_info) {
            string name(col_name.c_str());
            if (excluded_columns_.find(name) == excluded_columns_.end()) {
                indicator_names_.insert(name);
                indicator_cols_.emplace_back(name, cref(df_.get_column<double>(name.c_str())));
            }
        }
    }
}

void FastFinRL::setup_tickers(const vector<string>& tickers) {
    active_tickers_ = tickers;
    ticker_to_idx_.clear();

    size_t n = tickers.size();
    active_global_idx_.resize(n);
    active_first_day_.resize(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& tic = tickers[i];
        if (all_tickers_.find(tic) == all_tickers_.end()) {
            throw runtime_error("Ticker not found: " + tic);
        }
        ticker_to_idx_[tic] = i;
        active_global_idx_[i] = ticker_global_idx_[tic];
        active_first_day_[i] = ticker_first_day_[tic];
    }

    shares_.assign(n, 0);
    avg_buy_price_.assign(n, 0.0);
    trade_info_.assign(n, TradeInfo{});
    active_row_indices_.resize(n);
}

void FastFinRL::update_row_indices() {
    // O(1) direct table lookup
    for (size_t i = 0; i < active_tickers_.size(); ++i) {
        active_row_indices_[i] = ticker_row_table_[active_global_idx_[i]][day_];
    }
}


size_t FastFinRL::find_row_index(const string& ticker, int day) const {
    // Use pre-computed index map for O(1) lookup
    auto it = row_index_map_.find({ticker, day});
    if (it != row_index_map_.end()) {
        return it->second;
    }
    throw runtime_error("Row not found for ticker: " + ticker + " day: " + to_string(day));
}

double FastFinRL::get_raw_value(const string& ticker, int day, const string& column) const {
    size_t row_idx = find_row_index(ticker, day);
    for (const auto& [name, col_ref] : indicator_cols_) {
        if (name == column) {
            return col_ref.get()[row_idx];
        }
    }
    // Fallback for non-cached columns
    const auto& col = df_.get_column<double>(column.c_str());
    return col[row_idx];
}

double FastFinRL::get_price(size_t ticker_idx, const string& price_type) const {
    size_t row_idx = active_row_indices_[ticker_idx];
    if (price_type == "close") return col_close_->get()[row_idx];
    if (price_type == "open") return col_open_->get()[row_idx];
    if (price_type == "high") return col_high_->get()[row_idx];
    if (price_type == "low") return col_low_->get()[row_idx];
    return col_close_->get()[row_idx];
}

string FastFinRL::get_date() const {
    if (active_tickers_.empty()) return "";
    return col_date_->get()[active_row_indices_[0]];
}

double FastFinRL::calculate_total_asset() const {
    double total = cash_;
    for (size_t i = 0; i < shares_.size(); ++i) {
        total += shares_[i] * get_price(i, "close");
    }
    return total;
}

void FastFinRL::init_bid_options() {
    // Default: use close price
    auto default_fn = [this](size_t idx) -> double {
        return get_price(idx, "close");
    };

    // Uniform: random between low and high
    auto uniform_fn = [this](size_t idx) -> double {
        double low = get_price(idx, "low");
        double high = get_price(idx, "high");
        uniform_real_distribution<double> dist(low, high);
        return dist(rng_);
    };

    // Low uniform (for sell): random between low and min(open, close)
    auto low_uniform_fn = [this](size_t idx) -> double {
        double low = get_price(idx, "low");
        double open_price = get_price(idx, "open");
        double close_price = get_price(idx, "close");
        double maximum = min(open_price, close_price);
        uniform_real_distribution<double> dist(low, maximum);
        return dist(rng_);
    };

    // High uniform (for buy): random between max(open, close) and high
    auto high_uniform_fn = [this](size_t idx) -> double {
        double high = get_price(idx, "high");
        double open_price = get_price(idx, "open");
        double close_price = get_price(idx, "close");
        double minimum = max(open_price, close_price);
        uniform_real_distribution<double> dist(minimum, high);
        return dist(rng_);
    };

    // Deterministic: use close price (fully reproducible)
    auto deterministic_fn = [this](size_t idx) -> double {
        return get_price(idx, "close");
    };

    // Sell bid options: default, uniform, adv_uniform -> low_uniform, deterministic
    sell_bid_options_["default"] = default_fn;
    sell_bid_options_["uniform"] = uniform_fn;
    sell_bid_options_["adv_uniform"] = low_uniform_fn;
    sell_bid_options_["deterministic"] = deterministic_fn;

    // Buy bid options: default, uniform, adv_uniform -> high_uniform, deterministic
    buy_bid_options_["default"] = default_fn;
    buy_bid_options_["uniform"] = uniform_fn;
    buy_bid_options_["adv_uniform"] = high_uniform_fn;
    buy_bid_options_["deterministic"] = deterministic_fn;
}

double FastFinRL::get_sell_bid_price(size_t ticker_idx) {
    auto it = sell_bid_options_.find(bidding);
    if (it != sell_bid_options_.end()) {
        return it->second(ticker_idx);
    }
    return get_price(ticker_idx, "close");
}

double FastFinRL::get_buy_bid_price(size_t ticker_idx) {
    auto it = buy_bid_options_.find(bidding);
    if (it != buy_bid_options_.end()) {
        return it->second(ticker_idx);
    }
    return get_price(ticker_idx, "close");
}

double FastFinRL::get_randomized_price(size_t ticker_idx, const string& option) {
    // Legacy method - kept for compatibility
    if (option == "sell") {
        return get_sell_bid_price(ticker_idx);
    } else if (option == "buy") {
        return get_buy_bid_price(ticker_idx);
    }
    return get_price(ticker_idx, "close");
}

nlohmann::json FastFinRL::reset(const vector<string>& ticker_list, int64_t seed) {
    // 1. Seed handling
    if (seed == -1) {
        current_seed_++;
    } else {
        current_seed_ = seed;
    }
    rng_.seed(current_seed_);

    // 2. Setup tickers
    setup_tickers(ticker_list);

    // 3. Random day selection [min_start_day, max_day * 0.8)
    // min_start_day = max of first available day among active tickers
    int min_start_day = 0;
    for (size_t i = 0; i < active_first_day_.size(); ++i) {
        if (active_first_day_[i] > min_start_day) {
            min_start_day = active_first_day_[i];
        }
    }

    int max_start_day = static_cast<int>(max_day_ * 0.8);
    if (max_start_day <= min_start_day) max_start_day = min_start_day + 1;
    uniform_int_distribution<int> dist(min_start_day, max_start_day - 1);
    day_ = dist(rng_);

    // Update cached row indices
    update_row_indices();

    // 4. Initialize portfolio
    cash_ = initial_amount;
    shares_.assign(active_tickers_.size(), 0);
    avg_buy_price_.assign(active_tickers_.size(), 0.0);

    // 5. Reset episode tracking
    num_stop_loss_ = 0;
    trades_ = 0;
    cost_ = 0.0;
    in_step_ = false;

    // 6. Build and return state JSON
    return json_handler_->build_state_json();
}

int FastFinRL::sell_stock(size_t index, int action) {
    if (shares_[index] <= 0) return 0;

    int sell_num = min(action, shares_[index]);
    double price = get_sell_bid_price(index);
    double trade_cost = price * sell_num * sell_cost_pct;
    double sell_amount = price * sell_num * (1.0 - sell_cost_pct);

    cash_ += sell_amount;
    shares_[index] -= sell_num;
    cost_ += trade_cost;
    trades_++;
    trades_this_step_ += sell_num;

    // Debug: record execution info
    trade_info_[index].fill_price = price;
    trade_info_[index].cost += trade_cost;
    trade_info_[index].quantity -= sell_num;  // negative for sell

    return sell_num;
}

int FastFinRL::buy_stock(size_t index, int action) {
    double price = get_buy_bid_price(index);
    int available = static_cast<int>(cash_ / (price * (1.0 + buy_cost_pct)));
    int buy_num = min(available, action);

    if (buy_num <= 0) return 0;

    // Calculate for moving average
    double prev_total = shares_[index] * avg_buy_price_[index];
    double trade_cost = price * buy_num * buy_cost_pct;
    double buy_amount = price * buy_num * (1.0 + buy_cost_pct);

    cash_ -= buy_amount;
    shares_[index] += buy_num;

    // Update avg_buy_price (moving average)
    int new_shares = shares_[index];
    if (new_shares > 0) {
        avg_buy_price_[index] = (prev_total + buy_amount) / new_shares;
    }

    cost_ += trade_cost;
    trades_++;
    trades_this_step_ += buy_num;

    // Debug: record execution info
    trade_info_[index].fill_price = price;
    trade_info_[index].cost += trade_cost;
    trade_info_[index].quantity += buy_num;  // positive for buy

    return buy_num;
}

void FastFinRL::check_stop_loss() {
    double before_cash = cash_;

    for (size_t i = 0; i < shares_.size(); ++i) {
        if (shares_[i] <= 0) continue;

        double price = (stop_loss_calculation == "close")
            ? get_price(i, "close")
            : get_price(i, "low");

        if (price < avg_buy_price_[i] * stop_loss_tolerance) {
            sell_stock(i, shares_[i]);
            avg_buy_price_[i] = 0.0;
            num_stop_loss_++;
        }
    }

    loss_cut_amount_ = abs(cash_ - before_cash);
}

nlohmann::json FastFinRL::step(const vector<double>& actions) {
    in_step_ = true;
    trades_this_step_ = 0;
    loss_cut_amount_ = 0.0;

    // Clear trade info for this step
    for (auto& ti : trade_info_) {
        ti = TradeInfo{};
    }

    // 1. Record begin asset
    begin_total_asset_ = calculate_total_asset();

    // 2. Scale actions: action * hmax, convert to int
    vector<int> scaled_actions(actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
        scaled_actions[i] = static_cast<int>(actions[i] * hmax);
    }

    // 3. Separate sell/buy indices
    vector<size_t> sell_indices, buy_indices;
    for (size_t i = 0; i < scaled_actions.size(); ++i) {
        if (scaled_actions[i] < 0) sell_indices.push_back(i);
        else if (scaled_actions[i] > 0) buy_indices.push_back(i);
    }

    // 4. Sort: sells ascending (most negative first), buys descending (largest first)
    sort(sell_indices.begin(), sell_indices.end(),
         [&](size_t a, size_t b) { return scaled_actions[a] < scaled_actions[b]; });
    sort(buy_indices.begin(), buy_indices.end(),
         [&](size_t a, size_t b) { return scaled_actions[a] > scaled_actions[b]; });

    // 5. Execute sells first
    for (size_t idx : sell_indices) {
        sell_stock(idx, abs(scaled_actions[idx]));
    }

    // 6. Reset avg_buy_price for stocks with 0 shares
    for (size_t i = 0; i < shares_.size(); ++i) {
        if (shares_[i] == 0) avg_buy_price_[i] = 0.0;
    }

    // 7. Execute buys
    for (size_t idx : buy_indices) {
        buy_stock(idx, scaled_actions[idx]);
    }

    // 8. Stop loss check
    check_stop_loss();

    // 9. Advance day and update cached indices
    day_++;
    update_row_indices();

    // 10. Calculate reward
    double end_total_asset = calculate_total_asset();
    double reward = log(end_total_asset / begin_total_asset_);

    // 11. Check terminal conditions
    bool terminal = (day_ >= max_day_ - 1);
    bool done = (end_total_asset <= 25000.0) || terminal;

    // 12. Build and return state
    return json_handler_->build_step_result_json(reward, done, terminal);
}

nlohmann::json FastFinRL::get_state() const {
    if (in_step_) {
        return json_handler_->build_step_result_json(0.0, false, false);
    }
    return json_handler_->build_state_json();
}

// ============================================================================
// JsonHandler Implementation
// ============================================================================

nlohmann::json JsonHandler::build_market_json() const {
    const size_t n = env_.active_tickers_.size();
    constexpr size_t OMP_THRESHOLD = 16;  // Only use OpenMP for 16+ tickers

    // Pre-allocate per-ticker JSON objects
    vector<nlohmann::json> ticker_jsons(n);

    // Build each ticker's JSON (parallel if enough tickers and OpenMP available)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n >= OMP_THRESHOLD)
    #endif
    for (size_t i = 0; i < n; ++i) {
        size_t row_idx = env_.active_row_indices_[i];
        nlohmann::json ticker_data;

        ticker_data["open"] = env_.col_open_->get()[row_idx];
        ticker_data["high"] = env_.col_high_->get()[row_idx];
        ticker_data["low"] = env_.col_low_->get()[row_idx];
        ticker_data["close"] = env_.col_close_->get()[row_idx];

        nlohmann::json indicators;
        for (const auto& [ind_name, col_ref] : env_.indicator_cols_) {
            indicators[ind_name] = col_ref.get()[row_idx];
        }
        ticker_data["indicators"] = move(indicators);

        ticker_jsons[i] = move(ticker_data);
    }

    // Merge into final market JSON (sequential)
    nlohmann::json market;
    for (size_t i = 0; i < n; ++i) {
        market[env_.active_tickers_[i]] = move(ticker_jsons[i]);
    }

    return market;
}

nlohmann::json JsonHandler::build_portfolio_json(bool include_total_asset) const {
    const size_t n = env_.active_tickers_.size();
    constexpr size_t OMP_THRESHOLD = 16;  // Only use OpenMP for 16+ tickers

    nlohmann::json portfolio;
    portfolio["cash"] = env_.cash_;

    // Build holdings (parallel if enough tickers)
    vector<nlohmann::json> holding_jsons(n);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n >= OMP_THRESHOLD)
    #endif
    for (size_t i = 0; i < n; ++i) {
        holding_jsons[i] = {
            {"shares", env_.shares_[i]},
            {"avg_buy_price", env_.avg_buy_price_[i]}
        };
    }

    // Merge holdings
    nlohmann::json holdings;
    for (size_t i = 0; i < n; ++i) {
        holdings[env_.active_tickers_[i]] = move(holding_jsons[i]);
    }
    portfolio["holdings"] = move(holdings);

    if (include_total_asset) {
        portfolio["total_asset"] = env_.calculate_total_asset();
    }

    return portfolio;
}

nlohmann::json JsonHandler::build_state_json() const {
    nlohmann::json state;
    state["day"] = env_.day_;
    state["date"] = env_.get_date();
    state["seed"] = env_.current_seed_;
    state["done"] = false;
    state["terminal"] = false;
    state["portfolio"] = build_portfolio_json(false);
    state["market"] = build_market_json();
    return state;
}

nlohmann::json JsonHandler::build_step_result_json(double reward, bool done, bool terminal) const {
    nlohmann::json state;
    state["day"] = env_.day_;
    state["date"] = env_.get_date();
    state["done"] = done;
    state["terminal"] = terminal;
    state["reward"] = reward;
    state["portfolio"] = build_portfolio_json(true);
    state["market"] = build_market_json();

    nlohmann::json info;
    info["loss_cut_amount"] = env_.loss_cut_amount_;
    info["n_trades"] = env_.trades_this_step_;
    info["num_stop_loss"] = env_.num_stop_loss_;
    state["info"] = info;

    // Debug: per-ticker execution info
    nlohmann::json debug;
    for (size_t i = 0; i < env_.active_tickers_.size(); ++i) {
        const auto& ti = env_.trade_info_[i];
        if (ti.quantity != 0) {  // Only include if there was a trade
            debug[env_.active_tickers_[i]] = {
                {"fill_price", ti.fill_price},
                {"cost", ti.cost},
                {"quantity", ti.quantity}
            };
        }
    }
    state["debug"] = debug;

    return state;
}

} // namespace fast_finrl
