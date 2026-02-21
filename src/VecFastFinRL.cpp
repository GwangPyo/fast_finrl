#include "VecFastFinRL.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

// Large prime for seed generation (10^6th prime)
static constexpr int64_t SEED_PRIME = 15485863LL;

VecFastFinRL::VecFastFinRL(
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
    int shifted_start)
    : initial_amount_(initial_amount)
    , failure_threshold_(failure_threshold)
    , hmax_(hmax)
    , buy_cost_pct_(buy_cost_pct)
    , sell_cost_pct_(sell_cost_pct)
    , stop_loss_tolerance_(stop_loss_tolerance)
    , bidding_(bidding)
    , stop_loss_calculation_(stop_loss_calculation)
    , shuffle_tickers_(shuffle_tickers)
    , auto_reset_(auto_reset)
    , return_format_(return_format)
    , num_envs_(n_envs)
    , num_tickers_config_(num_tickers)
    , last_base_seed_(initial_seed)
    , shifted_start(shifted_start)
{
    if (n_envs <= 0) {
        throw runtime_error("n_envs must be > 0");
    }

    // Create config for base_env (FastFinRL still uses config)
    FastFinRLConfig config;
    config.initial_amount = initial_amount;
    config.failure_threshold = failure_threshold;
    config.hmax = hmax;
    config.buy_cost_pct = buy_cost_pct;
    config.sell_cost_pct = sell_cost_pct;
    config.stop_loss_tolerance = stop_loss_tolerance;
    config.bidding = bidding;
    config.stop_loss_calculation = stop_loss_calculation;
    config.initial_seed = initial_seed;
    config.tech_indicator_list = tech_indicator_list;
    config.macro_tickers = macro_tickers;
    config.return_format = return_format;
    config.num_tickers = num_tickers;
    config.shuffle_tickers = shuffle_tickers;

    // Create base environment for shared market data
    base_env_ = make_shared<FastFinRL>(csv_path, config);
    max_day_ = base_env_->get_max_day();
    n_indicators_ = static_cast<int>(base_env_->get_indicator_names().size());
    n_macro_ = static_cast<int>(macro_tickers.size());

    // Initialize n_tickers based on config (before reset)
    if (num_tickers > 0) {
        n_tickers_ = num_tickers;
    } else {
        n_tickers_ = static_cast<int>(base_env_->get_all_tickers().size());
    }

    // Initialize default tickers (before reset)
    const auto& all_tickers_set = base_env_->get_all_tickers();
    vector<string> all_tics(all_tickers_set.begin(), all_tickers_set.end());
    sort(all_tics.begin(), all_tics.end());

    if (num_tickers > 0) {
        if (shuffle_tickers) {
            // Shuffle with initial_seed for each env
            for (int i = 0; i < num_envs_; ++i) {
                vector<string> candidates = all_tics;
                int64_t env_seed = (initial_seed * (i + 1) * SEED_PRIME) % (SEED_PRIME - 1);
                mt19937 rng(static_cast<unsigned int>(env_seed));
                shuffle(candidates.begin(), candidates.end(), rng);
                int n = min(num_tickers, static_cast<int>(candidates.size()));
                vector<string> selected(candidates.begin(), candidates.begin() + n);
                sort(selected.begin(), selected.end());
                tickers_.push_back(selected);
            }
        } else {
            // Fixed first N tickers (alphabetical)
            int n = min(num_tickers, static_cast<int>(all_tics.size()));
            vector<string> selected(all_tics.begin(), all_tics.begin() + n);
            tickers_.assign(num_envs_, selected);
        }
    } else {
        // All tickers
        tickers_.assign(num_envs_, all_tics);
    }

    // Initialize bid option function pointers
    init_bid_options();
}

void VecFastFinRL::init_bid_options() {
    // Default/deterministic: use close price
    auto default_fn = [this](size_t env_idx, size_t ticker_idx) -> double {
        const string& tic = tickers_[env_idx][ticker_idx];
        int day = day_[env_idx];
        return base_env_->get_raw_value(tic, day, "close");
    };

    // Uniform: random between low and high
    auto uniform_fn = [this](size_t env_idx, size_t ticker_idx) -> double {
        const string& tic = tickers_[env_idx][ticker_idx];
        int day = day_[env_idx];
        double low = base_env_->get_raw_value(tic, day, "low");
        double high = base_env_->get_raw_value(tic, day, "high");
        uniform_real_distribution<double> dist(low, high);
        return dist(rngs_[env_idx]);
    };

    // Low uniform (for sell): random between low and min(open, close)
    auto low_uniform_fn = [this](size_t env_idx, size_t ticker_idx) -> double {
        const string& tic = tickers_[env_idx][ticker_idx];
        int day = day_[env_idx];
        double low = base_env_->get_raw_value(tic, day, "low");
        double open_price = base_env_->get_raw_value(tic, day, "open");
        double close = base_env_->get_raw_value(tic, day, "close");
        double maximum = min(open_price, close);
        uniform_real_distribution<double> dist(low, maximum);
        return dist(rngs_[env_idx]);
    };

    // High uniform (for buy): random between max(open, close) and high
    auto high_uniform_fn = [this](size_t env_idx, size_t ticker_idx) -> double {
        const string& tic = tickers_[env_idx][ticker_idx];
        int day = day_[env_idx];
        double high = base_env_->get_raw_value(tic, day, "high");
        double open_price = base_env_->get_raw_value(tic, day, "open");
        double close = base_env_->get_raw_value(tic, day, "close");
        double minimum = max(open_price, close);
        uniform_real_distribution<double> dist(minimum, high);
        return dist(rngs_[env_idx]);
    };

    // Sell bid options
    sell_bid_options_["default"] = default_fn;
    sell_bid_options_["uniform"] = uniform_fn;
    sell_bid_options_["adv_uniform"] = low_uniform_fn;
    sell_bid_options_["deterministic"] = default_fn;

    // Buy bid options
    buy_bid_options_["default"] = default_fn;
    buy_bid_options_["uniform"] = uniform_fn;
    buy_bid_options_["adv_uniform"] = high_uniform_fn;
    buy_bid_options_["deterministic"] = default_fn;

    // Cache active bid functions
    active_sell_bid_ = &sell_bid_options_.at(bidding_);
    active_buy_bid_ = &buy_bid_options_.at(bidding_);
}

VecFastFinRL::StepResult VecFastFinRL::reset(
    const vector<vector<string>>& tickers_list,
    const vector<int64_t>& seeds)
{
    num_envs_ = static_cast<int>(tickers_list.size());

    if (seeds.size() != static_cast<size_t>(num_envs_)) {
        throw runtime_error("seeds.size() must match tickers_list.size()");
    }

    if (num_envs_ == 0) {
        throw runtime_error("tickers_list cannot be empty");
    }

    // Handle empty tickers_list[i]: use shuffle if enabled, otherwise use all tickers
    vector<vector<string>> effective_tickers_list = tickers_list;
    const auto& all_tickers_set = base_env_->get_all_tickers();
    vector<string> all_tics(all_tickers_set.begin(), all_tickers_set.end());
    sort(all_tics.begin(), all_tics.end());

    for (int i = 0; i < num_envs_; ++i) {
        if (tickers_list[i].empty()) {
            if (num_tickers_config_ > 0) {
                if (shuffle_tickers_) {
                    // Shuffle and select num_tickers (each env different)
                    vector<string> candidates = all_tics;
                    mt19937 rng(static_cast<unsigned int>(seeds[i]));
                    shuffle(candidates.begin(), candidates.end(), rng);
                    int n = min(num_tickers_config_, static_cast<int>(candidates.size()));
                    vector<string> selected(candidates.begin(), candidates.begin() + n);
                    sort(selected.begin(), selected.end());
                    effective_tickers_list[i] = selected;
                } else {
                    // First num_tickers alphabetically (all envs same)
                    int n = min(num_tickers_config_, static_cast<int>(all_tics.size()));
                    vector<string> selected(all_tics.begin(), all_tics.begin() + n);
                    effective_tickers_list[i] = selected;
                }
            } else {
                // Use all tickers (sorted)
                effective_tickers_list[i] = all_tics;
            }
        }
    }

    // Now determine n_tickers from effective list
    n_tickers_ = static_cast<int>(effective_tickers_list[0].size());

    // Validate: all envs must have same number of tickers
    for (const auto& tickers : effective_tickers_list) {
        if (static_cast<int>(tickers.size()) != n_tickers_) {
            throw runtime_error("All envs must have same number of tickers");
        }
    }

    // Store per-env tickers
    tickers_ = effective_tickers_list;

    // Validate all tickers exist
    const auto& all_tickers = base_env_->get_all_tickers();
    for (int i = 0; i < num_envs_; ++i) {
        for (int t = 0; t < n_tickers_; ++t) {
            const string& tic = tickers_[i][t];
            if (all_tickers.find(tic) == all_tickers.end()) {
                throw runtime_error("Ticker not found: " + tic);
            }
        }
    }

    // Allocate per-env state arrays
    day_.resize(num_envs_);
    cash_.resize(num_envs_);
    shares_.resize(num_envs_ * n_tickers_, 0);
    avg_buy_price_.resize(num_envs_ * n_tickers_, 0.0);
    seeds_.resize(num_envs_);
    rngs_.resize(num_envs_);
    num_stop_loss_.resize(num_envs_, 0);
    trades_.resize(num_envs_, 0);
    begin_total_asset_.resize(num_envs_, 0.0);
    loss_cut_amount_.resize(num_envs_, 0.0);

    // Allocate output buffer
    buffer_.num_envs = num_envs_;
    buffer_.n_tickers = n_tickers_;
    buffer_.n_indicators = n_indicators_;
    buffer_.n_macro = n_macro_;

    buffer_.day.resize(num_envs_);
    buffer_.date.resize(num_envs_);
    buffer_.cash.resize(num_envs_);
    buffer_.shares.resize(num_envs_ * n_tickers_);
    buffer_.avg_buy_price.resize(num_envs_ * n_tickers_);
    buffer_.open.resize(num_envs_ * n_tickers_);
    buffer_.indicators.resize(num_envs_ * n_tickers_ * n_indicators_);
    buffer_.reward.resize(num_envs_, 0.0);
    buffer_.done.resize(num_envs_, false);
    buffer_.terminal.resize(num_envs_, false);
    buffer_.total_asset.resize(num_envs_);
    buffer_.num_stop_loss.resize(num_envs_, 0);
    buffer_.trades.resize(num_envs_, 0);
    buffer_.loss_cut_amount.resize(num_envs_, 0.0);

    if (n_macro_ > 0) {
        buffer_.macro_open.resize(num_envs_ * n_macro_);
        buffer_.macro_indicators.resize(num_envs_ * n_macro_ * n_indicators_);
    }

    // Parallel reset all environments
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_envs_),
        [this, &seeds, &effective_tickers_list](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                reset_single_env(i, seeds[i], effective_tickers_list[i]);
            }
        });

    return buffer_;
}

VecFastFinRL::StepResult VecFastFinRL::reset(
    const vector<vector<string>>& tickers_list,
    int64_t seed)
{
    // Validate tickers_list matches n_envs if provided
    if (!tickers_list.empty() && static_cast<int>(tickers_list.size()) != num_envs_) {
        throw runtime_error("tickers_list.size() must match n_envs (" + to_string(num_envs_) + ")");
    }

    // Store base seed for no-arg reset
    last_base_seed_ = seed;

    // Generate per-env seeds using prime multiplication
    vector<int64_t> seeds(num_envs_);
    for (int i = 0; i < num_envs_; ++i) {
        seeds[i] = (seed * (i + 1) * SEED_PRIME) % (SEED_PRIME - 1);
    }

    // Use previous tickers if tickers_list is empty (unless shuffle_tickers is enabled)
    vector<vector<string>> effective_tickers_list = tickers_list;
    if (tickers_list.empty()) {
        if (shuffle_tickers_) {
            // Empty vectors trigger shuffle in full reset
            effective_tickers_list.assign(num_envs_, vector<string>{});
        } else if (!tickers_.empty()) {
            // Reuse previous tickers
            effective_tickers_list = tickers_;
        } else {
            // No previous tickers, create empty for default selection
            effective_tickers_list.assign(num_envs_, vector<string>{});
        }
    }

    return reset(effective_tickers_list, seeds);
}

VecFastFinRL::StepResult VecFastFinRL::reset() {
    // No-arg reset: increment seed
    // If shuffle_tickers is true, use empty lists to trigger re-shuffle
    // Otherwise keep same tickers
    vector<vector<string>> tickers_to_use;

    if (shuffle_tickers_ || tickers_.empty()) {
        // Empty lists -> will be filled by shuffle or use all
        tickers_to_use.assign(num_envs_, vector<string>{});
    } else {
        tickers_to_use = tickers_;
    }

    // Increment base seed
    return reset(tickers_to_use, last_base_seed_ + 1);
}

void VecFastFinRL::reset_single_env(size_t env_idx, int64_t seed, const vector<string>& tickers) {
    // Initialize RNG
    seeds_[env_idx] = seed;
    rngs_[env_idx].seed(static_cast<unsigned int>(seed));

    // Ticker handling
    if (!tickers.empty()) {
        // Explicit tickers provided
        tickers_[env_idx] = tickers;
    } else if (shuffle_tickers_) {
        // Shuffle enabled - select new random tickers
        const auto& all_tickers_set = base_env_->get_all_tickers();
        vector<string> all_tics(all_tickers_set.begin(), all_tickers_set.end());
        sort(all_tics.begin(), all_tics.end());
        shuffle(all_tics.begin(), all_tics.end(), rngs_[env_idx]);
        int n = (num_tickers_config_ > 0) ? min(num_tickers_config_, static_cast<int>(all_tics.size())) : static_cast<int>(all_tics.size());
        tickers_[env_idx] = vector<string>(all_tics.begin(), all_tics.begin() + n);
        sort(tickers_[env_idx].begin(), tickers_[env_idx].end());
    }
    // else: keep existing tickers_[env_idx]

    // Calculate min_start_day for this env's tickers
    int min_start_day = 0;
    for (int t = 0; t < n_tickers_; ++t) {
        const string& tic = tickers_[env_idx][t];
        int first_day = base_env_->get_ticker_first_day(tic);
        if (first_day > min_start_day) {
            min_start_day = first_day;
        }
    }

    // Also consider macro tickers
    for (const string& tic : base_env_->get_macro_tickers()) {
        int first_day = base_env_->get_ticker_first_day(tic);
        if (first_day > min_start_day) {
            min_start_day = first_day;
        }
    }

    // Apply shifted_start offset for history window
    min_start_day += shifted_start;

    // Random day selection
    // max_start_day must be <= max_day * 0.8, but also must respect min_start_day
    int max_start_day = static_cast<int>(max_day_ * 0.8);
    if (min_start_day >= max_start_day) {
        // If ticker data starts late, use a valid range from min_start_day
        // Cap at max_day - 1 to ensure at least some episode length
        max_start_day = max_day_ - 1;
    }
    if (min_start_day > max_start_day) {
        // Not enough data for this shifted_start - fail loud
        throw runtime_error("Not enough data: min_start_day=" + to_string(min_start_day) +
                            " > max_start_day=" + to_string(max_start_day) +
                            " (max_day=" + to_string(max_day_) + ", shifted_start=" + to_string(shifted_start) + ")");
    }

    uniform_int_distribution<int> dist(min_start_day, max_start_day);
    day_[env_idx] = dist(rngs_[env_idx]);

    // Initialize portfolio
    cash_[env_idx] = initial_amount_;
    size_t base_idx = env_idx * n_tickers_;
    for (int t = 0; t < n_tickers_; ++t) {
        shares_[base_idx + t] = 0;
        avg_buy_price_[base_idx + t] = 0.0;
    }

    // Reset episode tracking
    num_stop_loss_[env_idx] = 0;
    trades_[env_idx] = 0;
    loss_cut_amount_[env_idx] = 0.0;

    // Output
    buffer_.done[env_idx] = 0;
    buffer_.terminal[env_idx] = 0;
    buffer_.reward[env_idx] = 0.0;
    buffer_.num_stop_loss[env_idx] = 0;
    buffer_.trades[env_idx] = 0;
    buffer_.loss_cut_amount[env_idx] = 0.0;

    // Fill observations
    fill_obs(env_idx);
}

void VecFastFinRL::fill_obs(size_t env_idx) {
    int day = day_[env_idx];
    size_t base_idx = env_idx * n_tickers_;

    buffer_.day[env_idx] = day;
    buffer_.date[env_idx] = base_env_->get_date_at_day_idx(day);
    buffer_.cash[env_idx] = cash_[env_idx];
    buffer_.trades[env_idx] = trades_[env_idx];
    buffer_.loss_cut_amount[env_idx] = loss_cut_amount_[env_idx];

    // Copy shares and avg_buy_price
    for (int t = 0; t < n_tickers_; ++t) {
        buffer_.shares[base_idx + t] = shares_[base_idx + t];
        buffer_.avg_buy_price[base_idx + t] = avg_buy_price_[base_idx + t];
    }

    // Fill open price and indicators for each ticker (no HLC - data leak)
    for (int t = 0; t < n_tickers_; ++t) {
        const string& tic = tickers_[env_idx][t];
        size_t open_idx = env_idx * n_tickers_ + t;
        size_t ind_base = (env_idx * n_tickers_ + t) * n_indicators_;

        buffer_.open[open_idx] = base_env_->get_raw_value(tic, day, "open");

        auto indicator_names = base_env_->get_indicator_names();
        int ind_idx = 0;
        for (const string& ind_name : indicator_names) {
            buffer_.indicators[ind_base + ind_idx] = base_env_->get_raw_value(tic, day, ind_name);
            ind_idx++;
        }
    }

    // Fill macro tickers (open only)
    if (n_macro_ > 0) {
        const auto& macro_tickers = base_env_->get_macro_tickers();
        for (int m = 0; m < n_macro_; ++m) {
            const string& tic = macro_tickers[m];
            size_t open_idx = env_idx * n_macro_ + m;
            size_t ind_base = (env_idx * n_macro_ + m) * n_indicators_;

            buffer_.macro_open[open_idx] = base_env_->get_raw_value(tic, day, "open");

            auto indicator_names = base_env_->get_indicator_names();
            int ind_idx = 0;
            for (const string& ind_name : indicator_names) {
                buffer_.macro_indicators[ind_base + ind_idx] = base_env_->get_raw_value(tic, day, ind_name);
                ind_idx++;
            }
        }
    }

    // Calculate total asset
    buffer_.total_asset[env_idx] = calculate_total_asset(env_idx);
}

double VecFastFinRL::calculate_total_asset(size_t env_idx) const {
    double total = cash_[env_idx];
    size_t base_idx = env_idx * n_tickers_;
    int day = day_[env_idx];

    for (int t = 0; t < n_tickers_; ++t) {
        if (shares_[base_idx + t] > 0) {
            const string& tic = tickers_[env_idx][t];
            double close = base_env_->get_raw_value(tic, day, "close");
            total += shares_[base_idx + t] * close;
        }
    }
    return total;
}

double VecFastFinRL::get_close(const size_t env_idx, const size_t ticker_idx) const {
    const string& tic = tickers_[env_idx][ticker_idx];
    int day = day_[env_idx];
    return base_env_->get_raw_value(tic, day, "close");
}

double VecFastFinRL::get_close_at_day(const size_t env_idx, const size_t ticker_idx, const int day) const {
    const string& tic = tickers_[env_idx][ticker_idx];
    return base_env_->get_raw_value(tic, day, "close");
}

int VecFastFinRL::sell_stock(const size_t env_idx, const size_t ticker_idx, int action) {
    size_t idx = env_idx * n_tickers_ + ticker_idx;
    if (shares_[idx] <= 0) return 0;

    int sell_num = min(action, shares_[idx]);
    double price = (*active_sell_bid_)(env_idx, ticker_idx);
    double sell_amount = price * sell_num * (1.0 - sell_cost_pct_);

    cash_[env_idx] += sell_amount;
    shares_[idx] -= sell_num;
    trades_[env_idx]++;

    if (shares_[idx] == 0) {
        avg_buy_price_[idx] = 0.0;
    }

    return sell_num;
}

int VecFastFinRL::buy_stock(size_t env_idx, size_t ticker_idx, int action) {
    size_t idx = env_idx * n_tickers_ + ticker_idx;
    double price = (*active_buy_bid_)(env_idx, ticker_idx);

    if (price <= 0) return 0;

    int available = static_cast<int>(cash_[env_idx] / (price * (1.0 + buy_cost_pct_)));
    int buy_num = min(available, action);

    if (buy_num <= 0) return 0;

    double prev_total = shares_[idx] * avg_buy_price_[idx];
    double buy_amount = price * buy_num * (1.0 + buy_cost_pct_);

    cash_[env_idx] -= buy_amount;
    shares_[idx] += buy_num;

    if (shares_[idx] > 0) {
        avg_buy_price_[idx] = (prev_total + buy_amount) / shares_[idx];
    }

    trades_[env_idx]++;
    return buy_num;
}

void VecFastFinRL::check_stop_loss(size_t env_idx) {
    size_t base_idx = env_idx * n_tickers_;
    int day = day_[env_idx];

    for (int t = 0; t < n_tickers_; ++t) {
        size_t idx = base_idx + t;
        if (shares_[idx] <= 0) continue;

        const string& tic = tickers_[env_idx][t];
        double price;
        if (stop_loss_calculation_ == "close") {
            price = base_env_->get_raw_value(tic, day, "close");
        } else {
            price = base_env_->get_raw_value(tic, day, "low");
        }

        if (price < avg_buy_price_[idx] * stop_loss_tolerance_) {
            // Calculate loss before selling (negative = loss)
            double loss = (price - avg_buy_price_[idx]) * shares_[idx];
            loss_cut_amount_[env_idx] += abs(loss);

            sell_stock(env_idx, t, shares_[idx]);
            num_stop_loss_[env_idx]++;
        }
    }
}

VecFastFinRL::StepResult VecFastFinRL::reset_indices(
    const vector<int>& indices,
    const vector<int64_t>& seeds)
{
    if (indices.size() != seeds.size()) {
        throw runtime_error("indices.size() must match seeds.size()");
    }

    if (num_envs_ == 0) {
        throw runtime_error("Must call reset() before reset_indices()");
    }

    // Validate indices
    for (int idx : indices) {
        if (idx < 0 || idx >= num_envs_) {
            throw runtime_error("Invalid env index: " + to_string(idx));
        }
    }

    // Parallel reset specified environments
    tbb::parallel_for(tbb::blocked_range<size_t>(0, indices.size()),
        [this, &indices, &seeds](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                size_t env_idx = indices[i];
                reset_single_env(env_idx, seeds[i], {});  // empty tickers → shuffle if enabled
            }
        });

    return buffer_;
}

VecFastFinRL::StepResult VecFastFinRL::step(const double* actions) {
    // Parallel step all environments
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_envs_),
        [this, actions](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                step_env(i, actions + i * n_tickers_);
            }
        });

    return buffer_;
}

void VecFastFinRL::step_env(size_t env_idx, const double* actions) {
    size_t base_idx = env_idx * n_tickers_;

    // 1. Record begin asset
    begin_total_asset_[env_idx] = calculate_total_asset(env_idx);

    // 2. Scale actions
    vector<int> scaled_actions(n_tickers_);
    for (int t = 0; t < n_tickers_; ++t) {
        scaled_actions[t] = static_cast<int>(actions[t] * hmax_);
    }

    // 3. Separate sell/buy
    vector<pair<int, int>> sells, buys;  // (ticker_idx, abs_action)
    for (int t = 0; t < n_tickers_; ++t) {
        if (scaled_actions[t] < 0) {
            sells.emplace_back(t, -scaled_actions[t]);
        } else if (scaled_actions[t] > 0) {
            buys.emplace_back(t, scaled_actions[t]);
        }
    }

    // 4. Sort by magnitude
    sort(sells.begin(), sells.end(), [](auto& a, auto& b) { return a.second > b.second; });
    sort(buys.begin(), buys.end(), [](auto& a, auto& b) { return a.second > b.second; });

    // 5. Execute sells first
    for (auto& [t, qty] : sells) {
        sell_stock(env_idx, t, qty);
    }

    // 6. Execute buys
    for (auto& [t, qty] : buys) {
        buy_stock(env_idx, t, qty);
    }

    // 8. Advance day
    day_[env_idx]++;

    // 9. Stop loss check (at new day)
    check_stop_loss(env_idx);

    // 10. Calculate reward
    double end_asset = calculate_total_asset(env_idx);
    double reward = 0.0;
    if (begin_total_asset_[env_idx] > 0) {
        reward = log(end_asset / begin_total_asset_[env_idx]);
    }
    buffer_.reward[env_idx] = reward;

    // 11. Check terminal conditions
    bool terminal = (day_[env_idx] >= max_day_ - 1);
    bool done = (end_asset <= failure_threshold_) || terminal;
    buffer_.terminal[env_idx] = terminal ? 1 : 0;
    buffer_.done[env_idx] = done ? 1 : 0;
    buffer_.num_stop_loss[env_idx] = num_stop_loss_[env_idx];

    // 12. Auto-reset if done
    if (auto_reset_ && done) {
        reset_single_env(env_idx, seeds_[env_idx] + 1, {});  // empty tickers → shuffle if enabled
        // Restore done=True so user sees the episode ended
        buffer_.done[env_idx] = 1;
    }

    // 13. Fill observation
    fill_obs(env_idx);
}

vector<nlohmann::json> VecFastFinRL::get_state() const {
    vector<nlohmann::json> states(num_envs_);

    tbb::parallel_for(tbb::blocked_range<int>(0, num_envs_),
        [this, &states](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                nlohmann::json state;

                // Basic info
                state["day"] = day_[i];
                state["seed"] = seeds_[i];
                state["done"] = false;
                state["terminal"] = false;

                // Portfolio
                nlohmann::json portfolio;
                portfolio["cash"] = cash_[i];

                // Calculate total asset
                double total_asset = cash_[i];
                int base_idx = i * n_tickers_;
                for (int t = 0; t < n_tickers_; ++t) {
                    if (shares_[base_idx + t] > 0) {
                        double close_price = get_close(i, t);
                        total_asset += shares_[base_idx + t] * close_price;
                    }
                }
                portfolio["total_asset"] = total_asset;

                // Holdings
                nlohmann::json holdings;
                for (int t = 0; t < n_tickers_; ++t) {
                    nlohmann::json h;
                    h["shares"] = shares_[base_idx + t];
                    h["avg_buy_price"] = avg_buy_price_[base_idx + t];
                    holdings[tickers_[i][t]] = h;
                }
                portfolio["holdings"] = holdings;
                state["portfolio"] = portfolio;

                // Market data
                nlohmann::json market;
                for (int t = 0; t < n_tickers_; ++t) {
                    nlohmann::json m;
                    m["open"] = base_env_->get_raw_value(tickers_[i][t], day_[i], "open");
                    m["high"] = base_env_->get_raw_value(tickers_[i][t], day_[i], "high");
                    m["low"] = base_env_->get_raw_value(tickers_[i][t], day_[i], "low");
                    m["close"] = base_env_->get_raw_value(tickers_[i][t], day_[i], "close");
                    m["volume"] = base_env_->get_raw_value(tickers_[i][t], day_[i], "volume");
                    market[tickers_[i][t]] = m;
                }
                state["market"] = market;

                states[i] = std::move(state);
            }
        });

    return states;
}

} // namespace fast_finrl
