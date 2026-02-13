#include "FastFinRL.hpp"
#include "DataLoader.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

FastFinRL::FastFinRL(const string& csv_path, const FastFinRLConfig& config)
    : initial_amount(config.initial_amount)
    , failure_threshold(config.failure_threshold)
    , hmax(config.hmax)
    , buy_cost_pct(config.buy_cost_pct)
    , sell_cost_pct(config.sell_cost_pct)
    , stop_loss_tolerance(config.stop_loss_tolerance)
    , bidding(config.bidding)
    , stop_loss_calculation(config.stop_loss_calculation)
    , return_format(config.return_format)
    , current_seed_(config.initial_seed)
    , rng_(config.initial_seed)
    , tech_indicator_list_(config.tech_indicator_list)
    , macro_tickers_(config.macro_tickers)
    , num_tickers_(config.num_tickers)
    , shuffle_tickers_(config.shuffle_tickers)
    , state_serializer_(make_unique<JsonStateSerializer>())
{
    excluded_columns_ = {"day", "day_idx", "date", "tic", "open", "high", "low", "close", "volume", "start"};
    load_dataframe(csv_path);
    extract_indicator_names();
    init_bid_options();

    // Validate and build macro ticker set
    for (const string& tic : macro_tickers_) {
        if (all_tickers_.find(tic) == all_tickers_.end()) {
            throw runtime_error("Macro ticker not found in data: " + tic);
        }
    }
    macro_ticker_set_.insert(macro_tickers_.begin(), macro_tickers_.end());
}

set<string> FastFinRL::get_indicator_names() const {
    return indicator_names_;
}

void FastFinRL::load_dataframe(const string& path) {
    // Enable DataFrame's internal threading for parallel operations
    hmdf::ThreadGranularity::set_optimum_thread_level();

    // Use DataLoader interface to load data
    auto loader = create_loader(path);
    loader->load(path, df_);

    // Build index tables and cache column references
    build_index_tables();
}

void FastFinRL::build_index_tables() {
    // Get all unique tickers using DataFrame library
    auto unique_tics = df_.get_col_unique_values<string>("tic");
    all_tickers_.insert(unique_tics.begin(), unique_tics.end());

    // Generate 'day_idx' column from unique dates (dense ranking like Python)
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

    // Create 'day_idx' column from sorted date timestamps
    df_.load_column("day_idx", move(day_values));

    // Set max_day
    max_day_ = static_cast<int>(sorted_dates.size());

    // Build pre-computed lookup tables
    const auto& tic_col = df_.get_column<string>("tic");
    const auto& day_col = df_.get_column<int>("day_idx");
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
    col_volume_.emplace(df_.get_column<double>("volume"));
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

nlohmann::json FastFinRL::reset(const vector<string>& ticker_list, int64_t seed, int shifted_start) {
    // 1. Seed handling
    if (seed == -1) {
        current_seed_++;
    } else {
        current_seed_ = seed;
    }
    rng_.seed(current_seed_);

    // 2. Determine effective tickers (with optional shuffle)
    vector<string> effective_tickers = ticker_list;

    if (shuffle_tickers_ && num_tickers_ > 0) {
        // Get all available tickers
        vector<string> all_tics(all_tickers_.begin(), all_tickers_.end());

        // Shuffle using RNG
        shuffle(all_tics.begin(), all_tics.end(), rng_);

        // Take first num_tickers
        int n = min(num_tickers_, static_cast<int>(all_tics.size()));
        effective_tickers = vector<string>(all_tics.begin(), all_tics.begin() + n);

        // Sort for consistency
        sort(effective_tickers.begin(), effective_tickers.end());
    }

    // 3. Setup tickers
    setup_tickers(effective_tickers);

    // 3. Random day selection [min_start_day + shifted_start, max_day * 0.8)
    // min_start_day = max of first available day among active tickers + macro tickers
    int min_start_day = 0;
    for (size_t i = 0; i < active_first_day_.size(); ++i) {
        if (active_first_day_[i] > min_start_day) {
            min_start_day = active_first_day_[i];
        }
    }
    // Also consider macro tickers
    for (const string& tic : macro_tickers_) {
        int first = ticker_first_day_.at(tic);
        if (first > min_start_day) {
            min_start_day = first;
        }
    }

    // Apply shifted_start offset
    min_start_day += shifted_start;

    int max_start_day = static_cast<int>(max_day_ * 0.8);
    if (min_start_day >= max_start_day) {
        throw runtime_error("shifted_start too large: min_start_day(" + to_string(min_start_day) +
                          ") >= max_start_day(" + to_string(max_start_day) + ")");
    }
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
    return state_serializer_->serialize(build_state_data(false), false);
}

nlohmann::json FastFinRL::reset() {
    // No-arg reset: keep same tickers (or all tickers if none set), increment seed
    vector<string> tickers_to_use;
    if (active_tickers_.empty()) {
        // Use all available tickers
        tickers_to_use = vector<string>(all_tickers_.begin(), all_tickers_.end());
        sort(tickers_to_use.begin(), tickers_to_use.end());
    } else {
        tickers_to_use = active_tickers_;
    }
    return reset(tickers_to_use, current_seed_ + 1, 0);
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
    bool done = (end_total_asset <= failure_threshold) || terminal;

    // 12. Build and return state
    return state_serializer_->serialize(build_state_data(true, reward, done, terminal), true);
}

nlohmann::json FastFinRL::get_state() const {
    if (in_step_) {
        return state_serializer_->serialize(build_state_data(true, 0.0, false, false), true);
    }
    return state_serializer_->serialize(build_state_data(false), false);
}

StateData FastFinRL::build_state_data(bool include_step_info, double reward, bool done, bool terminal) const {
    StateData state;

    // Basic state info
    state.day = day_;
    state.date = get_date();
    state.seed = current_seed_;
    state.done = done;
    state.terminal = terminal;
    state.reward = reward;

    // Portfolio
    state.portfolio.cash = cash_;
    state.portfolio.total_asset = calculate_total_asset();
    for (size_t i = 0; i < active_tickers_.size(); ++i) {
        TickerHolding holding;
        holding.shares = shares_[i];
        holding.avg_buy_price = avg_buy_price_[i];
        state.portfolio.holdings[active_tickers_[i]] = holding;
    }

    // Market data
    for (size_t i = 0; i < active_tickers_.size(); ++i) {
        size_t row_idx = active_row_indices_[i];
        TickerMarketData market_data;
        market_data.open = col_open_->get()[row_idx];
        market_data.high = col_high_->get()[row_idx];
        market_data.low = col_low_->get()[row_idx];
        market_data.close = col_close_->get()[row_idx];

        // Technical indicators
        for (const auto& [name, col_ref] : indicator_cols_) {
            market_data.indicators[name] = col_ref.get()[row_idx];
        }

        state.market.tickers[active_tickers_[i]] = move(market_data);
    }

    // Macro tickers (always included in state.macro)
    for (const string& tic : macro_tickers_) {
        size_t row_idx = find_row_index(tic, day_);
        TickerMarketData macro_data;
        macro_data.open = col_open_->get()[row_idx];
        macro_data.high = col_high_->get()[row_idx];
        macro_data.low = col_low_->get()[row_idx];
        macro_data.close = col_close_->get()[row_idx];

        for (const auto& [name, col_ref] : indicator_cols_) {
            macro_data.indicators[name] = col_ref.get()[row_idx];
        }

        state.macro.tickers[tic] = move(macro_data);
    }

    // Episode info (only meaningful in step)
    if (include_step_info) {
        state.info.loss_cut_amount = loss_cut_amount_;
        state.info.n_trades = trades_this_step_;
        state.info.num_stop_loss = num_stop_loss_;

        // Debug trade info
        for (size_t i = 0; i < active_tickers_.size(); ++i) {
            TradeDebugInfo debug;
            debug.fill_price = trade_info_[i].fill_price;
            debug.cost = trade_info_[i].cost;
            debug.quantity = trade_info_[i].quantity;
            state.debug[active_tickers_[i]] = debug;
        }
    }

    return state;
}

nlohmann::json FastFinRL::get_market_window(const string& ticker, int day, int h, int future) const {
    // Validate ticker
    auto it = ticker_global_idx_.find(ticker);
    if (it == ticker_global_idx_.end()) {
        throw runtime_error("Ticker not found: " + ticker);
    }
    size_t global_idx = it->second;

    // Get ticker's first day
    int first_day = ticker_first_day_.at(ticker);

    // Validate day range (allow querying any day, will pad if needed)
    if (day < 0 || day >= max_day_) {
        throw runtime_error("Day " + to_string(day) + " out of range [0, " + to_string(max_day_) + ")");
    }

    // Pre-allocate vectors for parallel filling
    vector<nlohmann::json> past_data(h);
    vector<int> past_mask_vec(h);
    vector<nlohmann::json> future_data(future);
    vector<int> future_mask_vec(future);

    // Cache indicator names for thread safety
    vector<string> indicator_names;
    indicator_names.reserve(indicator_cols_.size());
    for (const auto& [name, _] : indicator_cols_) {
        indicator_names.push_back(name);
    }

    // Helper lambda to build market data for a single day (thread-safe)
    auto build_day_data = [&](int d) -> nlohmann::json {
        size_t row_idx = ticker_row_table_[global_idx][d];
        nlohmann::json day_data;
        day_data["day"] = d;
        day_data["date"] = col_date_->get()[row_idx];
        day_data["open"] = col_open_->get()[row_idx];
        day_data["high"] = col_high_->get()[row_idx];
        day_data["low"] = col_low_->get()[row_idx];
        day_data["close"] = col_close_->get()[row_idx];

        nlohmann::json indicators;
        for (size_t i = 0; i < indicator_cols_.size(); ++i) {
            indicators[indicator_names[i]] = indicator_cols_[i].second.get()[row_idx];
        }
        day_data["indicators"] = move(indicators);
        return day_data;
    };

    // Helper to build padded (zero) day data (thread-safe)
    auto build_padded_data = [&]() -> nlohmann::json {
        nlohmann::json day_data;
        day_data["day"] = -1;
        day_data["date"] = "";
        day_data["open"] = 0.0;
        day_data["high"] = 0.0;
        day_data["low"] = 0.0;
        day_data["close"] = 0.0;

        nlohmann::json indicators;
        for (const auto& name : indicator_names) {
            indicators[name] = 0.0;
        }
        day_data["indicators"] = move(indicators);
        return day_data;
    };

    // Parallel build past data: [day-h, day-h+1, ..., day-1]
    if (h > 0) {
        tbb::parallel_for(tbb::blocked_range<int>(0, h),
            [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i < range.end(); ++i) {
                    int d = day - h + i;  // i=0 -> day-h, i=h-1 -> day-1
                    if (d >= first_day && d >= 0) {
                        past_data[i] = build_day_data(d);
                        past_mask_vec[i] = 1;
                    } else {
                        past_data[i] = build_padded_data();
                        past_mask_vec[i] = 0;
                    }
                }
            }
        );
    }

    // Parallel build future data: [day+1, day+2, ..., day+future]
    if (future > 0) {
        tbb::parallel_for(tbb::blocked_range<int>(0, future),
            [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i < range.end(); ++i) {
                    int d = day + 1 + i;  // i=0 -> day+1, i=future-1 -> day+future
                    if (d < max_day_) {
                        future_data[i] = build_day_data(d);
                        future_mask_vec[i] = 1;
                    } else {
                        future_data[i] = build_padded_data();
                        future_mask_vec[i] = 0;
                    }
                }
            }
        );
    }

    // Convert vectors to JSON arrays
    nlohmann::json result;
    result["past"] = nlohmann::json(past_data);
    result["past_mask"] = nlohmann::json(past_mask_vec);
    result["future"] = nlohmann::json(future_data);
    result["future_mask"] = nlohmann::json(future_mask_vec);

    // Current day
    if (day >= first_day) {
        result["current"] = build_day_data(day);
        result["current_mask"] = 1;
    } else {
        result["current"] = build_padded_data();
        result["current_mask"] = 0;
    }

    return result;
}

nlohmann::json FastFinRL::get_market_window_flat(const string& ticker, int day, int h, int future) const {
    // Validate ticker
    auto it = ticker_global_idx_.find(ticker);
    if (it == ticker_global_idx_.end()) {
        throw runtime_error("Ticker not found: " + ticker);
    }
    size_t global_idx = it->second;
    int first_day = ticker_first_day_.at(ticker);

    if (day < 0 || day >= max_day_) {
        throw runtime_error("Day " + to_string(day) + " out of range [0, " + to_string(max_day_) + ")");
    }

    int total_len = h + 1 + future;  // past + current + future
    size_t n_indicators = indicator_cols_.size();

    // Pre-allocate flat arrays
    vector<vector<double>> ohlc(total_len, vector<double>(4, 0.0));
    vector<vector<double>> indicators(total_len, vector<double>(n_indicators, 0.0));
    vector<int> mask(total_len, 0);
    vector<int> days(total_len, -1);
    vector<string> dates(total_len, "");

    // Parallel fill all data
    tbb::parallel_for(tbb::blocked_range<int>(0, total_len),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                int d;
                if (i < h) {
                    d = day - h + i;  // past: [day-h, day-1]
                } else if (i == h) {
                    d = day;          // current
                } else {
                    d = day + (i - h); // future: [day+1, day+future]
                }

                if (d >= first_day && d >= 0 && d < max_day_) {
                    size_t row_idx = ticker_row_table_[global_idx][d];
                    ohlc[i][0] = col_open_->get()[row_idx];
                    ohlc[i][1] = col_high_->get()[row_idx];
                    ohlc[i][2] = col_low_->get()[row_idx];
                    ohlc[i][3] = col_close_->get()[row_idx];

                    for (size_t j = 0; j < n_indicators; ++j) {
                        indicators[i][j] = indicator_cols_[j].second.get()[row_idx];
                    }

                    mask[i] = 1;
                    days[i] = d;
                    dates[i] = col_date_->get()[row_idx];
                }
                // else: already initialized to 0/empty (padding)
            }
        }
    );

    nlohmann::json result;
    result["ohlc"] = move(ohlc);
    result["indicators"] = move(indicators);
    result["mask"] = move(mask);
    result["days"] = move(days);
    result["dates"] = move(dates);
    result["h"] = h;
    result["future"] = future;

    // Include indicator names for reference
    vector<string> indicator_names;
    indicator_names.reserve(n_indicators);
    for (const auto& [name, _] : indicator_cols_) {
        indicator_names.push_back(name);
    }
    result["indicator_names"] = move(indicator_names);

    return result;
}

FastFinRL::MarketWindowData FastFinRL::get_market_window_raw(const string& ticker, int day, int h, int future) const {
    // Validate ticker
    auto it = ticker_global_idx_.find(ticker);
    if (it == ticker_global_idx_.end()) {
        throw runtime_error("Ticker not found: " + ticker);
    }
    size_t global_idx = it->second;
    int first_day = ticker_first_day_.at(ticker);

    if (day < 0 || day >= max_day_) {
        throw runtime_error("Day " + to_string(day) + " out of range [0, " + to_string(max_day_) + ")");
    }

    int total_len = h + 1 + future;
    size_t n_ind = indicator_cols_.size();

    MarketWindowData result;
    result.total_len = total_len;
    result.n_indicators = static_cast<int>(n_ind);

    // Pre-allocate flat arrays - OHLCV (5 values)
    result.ohlcv.resize(total_len * 5, 0.0);
    result.indicators.resize(total_len * n_ind, 0.0);
    result.mask.resize(total_len, 0);
    result.days.resize(total_len, -1);

    // Cache indicator names
    result.indicator_names.reserve(n_ind);
    for (const auto& [name, _] : indicator_cols_) {
        result.indicator_names.push_back(name);
    }

    // Parallel fill using raw pointers for thread safety
    double* ohlcv_ptr = result.ohlcv.data();
    double* ind_ptr = result.indicators.data();
    int* mask_ptr = result.mask.data();
    int* days_ptr = result.days.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, total_len),
        [=, this](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                int d;
                if (i < h) {
                    d = day - h + i;
                } else if (i == h) {
                    d = day;
                } else {
                    d = day + (i - h);
                }

                if (d >= first_day && d >= 0 && d < max_day_) {
                    size_t row_idx = ticker_row_table_[global_idx][d];
                    size_t ohlcv_base = i * 5;
                    ohlcv_ptr[ohlcv_base + 0] = col_open_->get()[row_idx];
                    ohlcv_ptr[ohlcv_base + 1] = col_high_->get()[row_idx];
                    ohlcv_ptr[ohlcv_base + 2] = col_low_->get()[row_idx];
                    ohlcv_ptr[ohlcv_base + 3] = col_close_->get()[row_idx];
                    ohlcv_ptr[ohlcv_base + 4] = col_volume_->get()[row_idx];

                    size_t ind_base = i * n_ind;
                    for (size_t j = 0; j < n_ind; ++j) {
                        ind_ptr[ind_base + j] = indicator_cols_[j].second.get()[row_idx];
                    }

                    mask_ptr[i] = 1;
                    days_ptr[i] = d;
                }
            }
        }
    );

    return result;
}

FastFinRL::MultiTickerWindowData FastFinRL::get_market_window_multi(
    const vector<string>& ticker_list, int day, int h, int future) const {

    if (day < 0 || day >= max_day_) {
        throw runtime_error("Day " + to_string(day) + " out of range [0, " + to_string(max_day_) + ")");
    }

    size_t n_ind = indicator_cols_.size();

    MultiTickerWindowData result;
    result.h = h;
    result.future = future;
    result.n_indicators = static_cast<int>(n_ind);

    // Cache indicator names
    for (const auto& [name, _] : indicator_cols_) {
        result.indicator_names.push_back(name);
    }

    // Process each ticker in parallel
    vector<pair<string, TickerWindowData>> ticker_results(ticker_list.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, ticker_list.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t t = range.begin(); t < range.end(); ++t) {
                const string& ticker = ticker_list[t];

                auto it = ticker_global_idx_.find(ticker);
                if (it == ticker_global_idx_.end()) {
                    throw runtime_error("Ticker not found: " + ticker);
                }
                size_t global_idx = it->second;
                int first_day = ticker_first_day_.at(ticker);

                TickerWindowData& td = ticker_results[t].second;
                ticker_results[t].first = ticker;

                // Allocate arrays - OHLCV (5 values)
                td.past_ohlcv.resize(h * 5, 0.0);
                td.past_indicators.resize(h * n_ind, 0.0);
                td.past_mask.resize(h, 0);
                td.past_days.resize(h, -1);

                td.current_open = 0.0;
                td.current_indicators.resize(n_ind, 0.0);
                td.current_mask = 0;
                td.current_day = -1;

                td.future_ohlcv.resize(future * 5, 0.0);
                td.future_indicators.resize(future * n_ind, 0.0);
                td.future_mask.resize(future, 0);
                td.future_days.resize(future, -1);

                // Fill past: [day-h, day-1] - OHLCV
                for (int i = 0; i < h; ++i) {
                    int d = day - h + i;
                    if (d >= first_day && d >= 0 && d < max_day_) {
                        size_t row_idx = ticker_row_table_[global_idx][d];
                        size_t base = i * 5;
                        td.past_ohlcv[base + 0] = col_open_->get()[row_idx];
                        td.past_ohlcv[base + 1] = col_high_->get()[row_idx];
                        td.past_ohlcv[base + 2] = col_low_->get()[row_idx];
                        td.past_ohlcv[base + 3] = col_close_->get()[row_idx];
                        td.past_ohlcv[base + 4] = col_volume_->get()[row_idx];

                        size_t ind_base = i * n_ind;
                        for (size_t j = 0; j < n_ind; ++j) {
                            td.past_indicators[ind_base + j] = indicator_cols_[j].second.get()[row_idx];
                        }
                        td.past_mask[i] = 1;
                        td.past_days[i] = d;
                    }
                }

                // Fill current - only open price (high/low/close are future info)
                if (day >= first_day && day < max_day_) {
                    size_t row_idx = ticker_row_table_[global_idx][day];
                    td.current_open = col_open_->get()[row_idx];

                    for (size_t j = 0; j < n_ind; ++j) {
                        td.current_indicators[j] = indicator_cols_[j].second.get()[row_idx];
                    }
                    td.current_mask = 1;
                    td.current_day = day;
                }

                // Fill future: [day+1, day+future] - OHLCV
                for (int i = 0; i < future; ++i) {
                    int d = day + 1 + i;
                    if (d >= first_day && d < max_day_) {
                        size_t row_idx = ticker_row_table_[global_idx][d];
                        size_t base = i * 5;
                        td.future_ohlcv[base + 0] = col_open_->get()[row_idx];
                        td.future_ohlcv[base + 1] = col_high_->get()[row_idx];
                        td.future_ohlcv[base + 2] = col_low_->get()[row_idx];
                        td.future_ohlcv[base + 3] = col_close_->get()[row_idx];
                        td.future_ohlcv[base + 4] = col_volume_->get()[row_idx];

                        size_t ind_base = i * n_ind;
                        for (size_t j = 0; j < n_ind; ++j) {
                            td.future_indicators[ind_base + j] = indicator_cols_[j].second.get()[row_idx];
                        }
                        td.future_mask[i] = 1;
                        td.future_days[i] = d;
                    }
                }
            }
        }
    );

    // Move results to map
    for (auto& [ticker, data] : ticker_results) {
        result.tickers[ticker] = move(data);
    }

    return result;
}

} // namespace fast_finrl
