#include "FastFinRL.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <future>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/properties.h>

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
    excluded_columns_ = {"day", "day_idx", "date", "tic", "open", "high", "low", "close", "volume", "start"};
    load_dataframe(csv_path);
    extract_indicator_names();
    init_bid_options();
}

set<string> FastFinRL::get_indicator_names() const {
    return indicator_names_;
}

hmdf::ReadParams FastFinRL::build_csv2_schema(const string& csv_path) {
    ifstream infile(csv_path);
    if (!infile.is_open()) {
        throw runtime_error("Cannot open file: " + csv_path);
    }

    // Read only the header line
    string header_line;
    getline(infile, header_line);
    infile.close();

    // Parse column names
    vector<string> col_names;
    stringstream header_ss(header_line);
    string col_name;
    while (getline(header_ss, col_name, ',')) {
        col_names.push_back(col_name);
    }

    // Build schema for csv2 format
    hmdf::ReadParams params;
    params.skip_first_line = true;  // Skip header row

    // First entry is always the index (ULONG)
    hmdf::ReadSchema index_schema;
    index_schema.col_name = hmdf::DF_INDEX_COL_NAME;
    index_schema.col_type = hmdf::file_dtypes::ULONG;
    index_schema.col_idx = 0;
    params.schema.push_back(index_schema);

    // Add schema for each column
    for (size_t i = 0; i < col_names.size(); ++i) {
        hmdf::ReadSchema col_schema;
        col_schema.col_name = col_names[i].c_str();

        // Determine type: 'date' and 'tic' are strings, rest are numeric
        if (col_names[i] == "date" || col_names[i] == "tic") {
            col_schema.col_type = hmdf::file_dtypes::STRING;
        } else {
            col_schema.col_type = hmdf::file_dtypes::DOUBLE;
        }
        col_schema.col_idx = static_cast<int>(i + 1);  // +1 because index is at 0
        params.schema.push_back(col_schema);
    }

    return params;
}

namespace {
    // Template to extract column data from Arrow chunked array
    template<typename ArrowType, typename CppType>
    vector<CppType> extract_column(const std::shared_ptr<arrow::ChunkedArray>& chunked, size_t num_rows) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        vector<CppType> col_data;
        col_data.reserve(num_rows);
        for (int c = 0; c < chunked->num_chunks(); ++c) {
            auto array = std::static_pointer_cast<ArrayType>(chunked->chunk(c));
            for (int64_t j = 0; j < array->length(); ++j) {
                if constexpr (std::is_same_v<CppType, string>) {
                    col_data.push_back(string(array->GetView(j)));
                } else {
                    col_data.push_back(static_cast<CppType>(array->Value(j)));
                }
            }
        }
        return col_data;
    }

    // Convert numeric to string
    template<typename ArrowType>
    vector<string> extract_as_string(const std::shared_ptr<arrow::ChunkedArray>& chunked, size_t num_rows) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        vector<string> col_data;
        col_data.reserve(num_rows);
        for (int c = 0; c < chunked->num_chunks(); ++c) {
            auto array = std::static_pointer_cast<ArrayType>(chunked->chunk(c));
            for (int64_t j = 0; j < array->length(); ++j) {
                col_data.push_back(to_string(array->Value(j)));
            }
        }
        return col_data;
    }
}

void FastFinRL::load_from_parquet(const string& path) {
    PARQUET_ASSIGN_OR_THROW(auto infile, arrow::io::ReadableFile::Open(path));

    // Create ParquetFileReader first
    auto parquet_reader = parquet::ParquetFileReader::Open(infile);

    // Enable parallel column reading
    parquet::ArrowReaderProperties props(true);  // use_threads = true
    props.set_pre_buffer(true);  // Pre-buffer for better I/O

    // Create FileReader with properties
    PARQUET_ASSIGN_OR_THROW(auto reader,
        parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(parquet_reader), props));
    reader->set_use_threads(true);  // Enable parallel decoding

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

    size_t num_rows = table->num_rows();
    int num_cols = table->num_columns();

    // Create index
    vector<unsigned long> index(num_rows);
    for (size_t i = 0; i < num_rows; ++i) index[i] = i;
    df_.load_index(move(index));

    // Pre-extract column info
    vector<string> col_names(num_cols);
    vector<arrow::Type::type> col_types(num_cols);
    for (int i = 0; i < num_cols; ++i) {
        col_names[i] = table->field(i)->name();
        col_types[i] = table->column(i)->type()->id();
    }

    // Extract columns in parallel using std::async
    vector<vector<double>> double_cols(num_cols);
    vector<vector<string>> string_cols(num_cols);
    vector<bool> is_string_col(num_cols, false);
    vector<std::future<void>> futures;
    futures.reserve(num_cols);

    for (int i = 0; i < num_cols; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            auto chunked = table->column(i);
            auto type_id = col_types[i];
            const auto& col_name = col_names[i];

            if (col_name == "date" || col_name == "tic") {
                is_string_col[i] = true;
                if (type_id == arrow::Type::STRING) {
                    string_cols[i] = extract_column<arrow::StringType, string>(chunked, num_rows);
                } else if (type_id == arrow::Type::INT64) {
                    string_cols[i] = extract_as_string<arrow::Int64Type>(chunked, num_rows);
                }
            } else {
                if (type_id == arrow::Type::DOUBLE) {
                    double_cols[i] = extract_column<arrow::DoubleType, double>(chunked, num_rows);
                } else if (type_id == arrow::Type::INT64) {
                    double_cols[i] = extract_column<arrow::Int64Type, double>(chunked, num_rows);
                } else if (type_id == arrow::Type::FLOAT) {
                    double_cols[i] = extract_column<arrow::FloatType, double>(chunked, num_rows);
                }
            }
        }));
    }

    // Wait for all extractions to complete
    for (auto& f : futures) f.get();

    // Load columns into DataFrame (sequential - DataFrame not thread-safe)
    for (int i = 0; i < num_cols; ++i) {
        if (is_string_col[i]) {
            df_.load_column(col_names[i].c_str(), move(string_cols[i]));
        } else {
            df_.load_column(col_names[i].c_str(), move(double_cols[i]));
        }
    }
}

void FastFinRL::load_dataframe(const string& path) {
    // Enable DataFrame's internal threading for parallel operations
    hmdf::ThreadGranularity::set_optimum_thread_level();

    // Check file extension and load accordingly
    bool is_parquet = (path.size() >= 8 && path.substr(path.size() - 8) == ".parquet");

    if (is_parquet) {
        load_from_parquet(path);
    } else {
        auto params = build_csv2_schema(path);
        df_.read(path.c_str(), hmdf::io_format::csv2, params);
    }

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

    // Create 'day' column from sorted date timestamps
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
