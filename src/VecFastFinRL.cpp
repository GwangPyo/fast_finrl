#include "VecFastFinRL.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

VecFastFinRL::VecFastFinRL(const string& csv_path, const FastFinRLConfig& config)
    : config_(config)
    , auto_reset_(true)
    , return_format_(config.return_format)
{
    // Create base environment for shared market data
    base_env_ = make_shared<FastFinRL>(csv_path, config);
    max_day_ = base_env_->get_max_day();
    n_indicators_ = static_cast<int>(base_env_->get_indicator_names().size());
    n_macro_ = static_cast<int>(config.macro_tickers.size());
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

    n_tickers_ = static_cast<int>(tickers_list[0].size());

    // Validate: all envs must have same number of tickers
    for (const auto& tickers : tickers_list) {
        if (static_cast<int>(tickers.size()) != n_tickers_) {
            throw runtime_error("All envs must have same number of tickers");
        }
    }

    // Store per-env tickers
    tickers_ = tickers_list;

    // Build ticker index lookup and first_day for each env's tickers
    const auto& all_tickers = base_env_->get_all_tickers();
    ticker_global_idx_.resize(num_envs_ * n_tickers_);
    ticker_first_day_.resize(num_envs_ * n_tickers_);

    for (int i = 0; i < num_envs_; ++i) {
        for (int t = 0; t < n_tickers_; ++t) {
            const string& tic = tickers_list[i][t];
            if (all_tickers.find(tic) == all_tickers.end()) {
                throw runtime_error("Ticker not found: " + tic);
            }
            // Get global idx and first_day from base_env_ via get_raw_value trick
            // We need to access internal data, so use reset temporarily
            size_t flat_idx = i * n_tickers_ + t;
            ticker_global_idx_[flat_idx] = base_env_->get_raw_value(tic, 0, "close") >= 0 ? 0 : 0;  // placeholder
        }
    }

    // Actually, we need to properly get the ticker info. Let's reset base_env with first env's tickers
    // to access internal maps. This is a workaround - in production, we'd expose the maps.
    base_env_->reset(tickers_list[0], 0, 0);

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

    // Allocate output buffer
    buffer_.num_envs = num_envs_;
    buffer_.n_tickers = n_tickers_;
    buffer_.n_indicators = n_indicators_;
    buffer_.n_macro = n_macro_;

    buffer_.day.resize(num_envs_);
    buffer_.cash.resize(num_envs_);
    buffer_.shares.resize(num_envs_ * n_tickers_);
    buffer_.avg_buy_price.resize(num_envs_ * n_tickers_);
    buffer_.ohlc.resize(num_envs_ * n_tickers_ * 4);
    buffer_.indicators.resize(num_envs_ * n_tickers_ * n_indicators_);
    buffer_.reward.resize(num_envs_, 0.0);
    buffer_.done.resize(num_envs_, false);
    buffer_.terminal.resize(num_envs_, false);
    buffer_.total_asset.resize(num_envs_);

    if (n_macro_ > 0) {
        buffer_.macro_ohlc.resize(num_envs_ * n_macro_ * 4);
        buffer_.macro_indicators.resize(num_envs_ * n_macro_ * n_indicators_);
    }

    // Parallel reset all environments
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_envs_),
        [this, &seeds](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                reset_env(i, seeds[i]);
                fill_obs(i);
            }
        });

    return buffer_;
}

void VecFastFinRL::reset_env(size_t env_idx, int64_t seed) {
    // Initialize RNG
    seeds_[env_idx] = seed;
    rngs_[env_idx].seed(static_cast<unsigned int>(seed));

    // Calculate min_start_day for this env's tickers
    int min_start_day = 0;
    for (int t = 0; t < n_tickers_; ++t) {
        const string& tic = tickers_[env_idx][t];
        // Get first available day for this ticker via base_env
        // For now, use day 0 as fallback; proper implementation needs exposed ticker_first_day_
        int first_day = 0;
        try {
            // Try to get data from day 0 - if it fails, ticker starts later
            base_env_->get_raw_value(tic, 0, "close");
        } catch (...) {
            // Ticker not available at day 0, find first available
            for (int d = 1; d < max_day_; ++d) {
                try {
                    base_env_->get_raw_value(tic, d, "close");
                    first_day = d;
                    break;
                } catch (...) {}
            }
        }
        if (first_day > min_start_day) {
            min_start_day = first_day;
        }
    }

    // Also consider macro tickers
    for (const string& tic : base_env_->get_macro_tickers()) {
        int first_day = 0;
        try {
            base_env_->get_raw_value(tic, 0, "close");
        } catch (...) {
            for (int d = 1; d < max_day_; ++d) {
                try {
                    base_env_->get_raw_value(tic, d, "close");
                    first_day = d;
                    break;
                } catch (...) {}
            }
        }
        if (first_day > min_start_day) {
            min_start_day = first_day;
        }
    }

    // Random day selection
    int max_start_day = static_cast<int>(max_day_ * 0.8);
    if (min_start_day >= max_start_day) {
        min_start_day = 0;
    }

    uniform_int_distribution<int> dist(min_start_day, max_start_day - 1);
    day_[env_idx] = dist(rngs_[env_idx]);

    // Initialize portfolio
    cash_[env_idx] = config_.initial_amount;
    size_t base_idx = env_idx * n_tickers_;
    for (int t = 0; t < n_tickers_; ++t) {
        shares_[base_idx + t] = 0;
        avg_buy_price_[base_idx + t] = 0.0;
    }

    // Reset episode tracking
    num_stop_loss_[env_idx] = 0;
    trades_[env_idx] = 0;

    // Output
    buffer_.done[env_idx] = 0;
    buffer_.terminal[env_idx] = 0;
    buffer_.reward[env_idx] = 0.0;
}

void VecFastFinRL::fill_obs(size_t env_idx) {
    int day = day_[env_idx];
    size_t base_idx = env_idx * n_tickers_;

    buffer_.day[env_idx] = day;
    buffer_.cash[env_idx] = cash_[env_idx];

    // Copy shares and avg_buy_price
    for (int t = 0; t < n_tickers_; ++t) {
        buffer_.shares[base_idx + t] = shares_[base_idx + t];
        buffer_.avg_buy_price[base_idx + t] = avg_buy_price_[base_idx + t];
    }

    // Fill OHLC and indicators for each ticker
    for (int t = 0; t < n_tickers_; ++t) {
        const string& tic = tickers_[env_idx][t];
        size_t ohlc_base = (env_idx * n_tickers_ + t) * 4;
        size_t ind_base = (env_idx * n_tickers_ + t) * n_indicators_;

        try {
            buffer_.ohlc[ohlc_base + 0] = base_env_->get_raw_value(tic, day, "open");
            buffer_.ohlc[ohlc_base + 1] = base_env_->get_raw_value(tic, day, "high");
            buffer_.ohlc[ohlc_base + 2] = base_env_->get_raw_value(tic, day, "low");
            buffer_.ohlc[ohlc_base + 3] = base_env_->get_raw_value(tic, day, "close");

            // Indicators
            auto indicator_names = base_env_->get_indicator_names();
            int ind_idx = 0;
            for (const string& ind_name : indicator_names) {
                buffer_.indicators[ind_base + ind_idx] = base_env_->get_raw_value(tic, day, ind_name);
                ind_idx++;
            }
        } catch (...) {
            // Fill with zeros if data not available
            for (int k = 0; k < 4; ++k) buffer_.ohlc[ohlc_base + k] = 0.0;
            for (int k = 0; k < n_indicators_; ++k) buffer_.indicators[ind_base + k] = 0.0;
        }
    }

    // Fill macro tickers
    if (n_macro_ > 0) {
        const auto& macro_tickers = base_env_->get_macro_tickers();
        for (int m = 0; m < n_macro_; ++m) {
            const string& tic = macro_tickers[m];
            size_t ohlc_base = (env_idx * n_macro_ + m) * 4;
            size_t ind_base = (env_idx * n_macro_ + m) * n_indicators_;

            try {
                buffer_.macro_ohlc[ohlc_base + 0] = base_env_->get_raw_value(tic, day, "open");
                buffer_.macro_ohlc[ohlc_base + 1] = base_env_->get_raw_value(tic, day, "high");
                buffer_.macro_ohlc[ohlc_base + 2] = base_env_->get_raw_value(tic, day, "low");
                buffer_.macro_ohlc[ohlc_base + 3] = base_env_->get_raw_value(tic, day, "close");

                auto indicator_names = base_env_->get_indicator_names();
                int ind_idx = 0;
                for (const string& ind_name : indicator_names) {
                    buffer_.macro_indicators[ind_base + ind_idx] = base_env_->get_raw_value(tic, day, ind_name);
                    ind_idx++;
                }
            } catch (...) {
                for (int k = 0; k < 4; ++k) buffer_.macro_ohlc[ohlc_base + k] = 0.0;
                for (int k = 0; k < n_indicators_; ++k) buffer_.macro_indicators[ind_base + k] = 0.0;
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
            try {
                double close = base_env_->get_raw_value(tic, day, "close");
                total += shares_[base_idx + t] * close;
            } catch (...) {}
        }
    }
    return total;
}

double VecFastFinRL::get_close(size_t env_idx, size_t ticker_idx) const {
    const string& tic = tickers_[env_idx][ticker_idx];
    int day = day_[env_idx];
    try {
        return base_env_->get_raw_value(tic, day, "close");
    } catch (...) {
        return 0.0;
    }
}

double VecFastFinRL::get_close_at_day(size_t env_idx, size_t ticker_idx, int day) const {
    const string& tic = tickers_[env_idx][ticker_idx];
    try {
        return base_env_->get_raw_value(tic, day, "close");
    } catch (...) {
        return 0.0;
    }
}

double VecFastFinRL::get_bid_price(size_t env_idx, size_t ticker_idx, const string& side) {
    const string& tic = tickers_[env_idx][ticker_idx];
    int day = day_[env_idx];

    try {
        if (config_.bidding == "default" || config_.bidding == "deterministic") {
            return base_env_->get_raw_value(tic, day, "close");
        }

        double low = base_env_->get_raw_value(tic, day, "low");
        double high = base_env_->get_raw_value(tic, day, "high");
        double open_price = base_env_->get_raw_value(tic, day, "open");
        double close = base_env_->get_raw_value(tic, day, "close");

        if (config_.bidding == "uniform") {
            uniform_real_distribution<double> dist(low, high);
            return dist(rngs_[env_idx]);
        }

        if (config_.bidding == "adv_uniform") {
            if (side == "sell") {
                double maximum = min(open_price, close);
                uniform_real_distribution<double> dist(low, maximum);
                return dist(rngs_[env_idx]);
            } else {
                double minimum = max(open_price, close);
                uniform_real_distribution<double> dist(minimum, high);
                return dist(rngs_[env_idx]);
            }
        }

        return close;
    } catch (...) {
        return 0.0;
    }
}

int VecFastFinRL::sell_stock(size_t env_idx, size_t ticker_idx, int action) {
    size_t idx = env_idx * n_tickers_ + ticker_idx;
    if (shares_[idx] <= 0) return 0;

    int sell_num = min(action, shares_[idx]);
    double price = get_bid_price(env_idx, ticker_idx, "sell");
    double sell_amount = price * sell_num * (1.0 - config_.sell_cost_pct);

    cash_[env_idx] += sell_amount;
    shares_[idx] -= sell_num;
    trades_[env_idx]++;

    return sell_num;
}

int VecFastFinRL::buy_stock(size_t env_idx, size_t ticker_idx, int action) {
    size_t idx = env_idx * n_tickers_ + ticker_idx;
    double price = get_bid_price(env_idx, ticker_idx, "buy");

    if (price <= 0) return 0;

    int available = static_cast<int>(cash_[env_idx] / (price * (1.0 + config_.buy_cost_pct)));
    int buy_num = min(available, action);

    if (buy_num <= 0) return 0;

    double prev_total = shares_[idx] * avg_buy_price_[idx];
    double buy_amount = price * buy_num * (1.0 + config_.buy_cost_pct);

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
        try {
            if (config_.stop_loss_calculation == "close") {
                price = base_env_->get_raw_value(tic, day, "close");
            } else {
                price = base_env_->get_raw_value(tic, day, "low");
            }
        } catch (...) {
            continue;
        }

        if (price < avg_buy_price_[idx] * config_.stop_loss_tolerance) {
            sell_stock(env_idx, t, shares_[idx]);
            avg_buy_price_[idx] = 0.0;
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
                reset_env(env_idx, seeds[i]);
                fill_obs(env_idx);
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
        scaled_actions[t] = static_cast<int>(actions[t] * config_.hmax);
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

    // 6. Reset avg_buy_price for zero-share positions
    for (int t = 0; t < n_tickers_; ++t) {
        if (shares_[base_idx + t] == 0) {
            avg_buy_price_[base_idx + t] = 0.0;
        }
    }

    // 7. Execute buys
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
    bool done = (end_asset <= 25000.0) || terminal;
    buffer_.terminal[env_idx] = terminal ? 1 : 0;
    buffer_.done[env_idx] = done ? 1 : 0;

    // 12. Auto-reset if done
    if (auto_reset_ && done) {
        reset_env(env_idx, seeds_[env_idx] + 1);
    }

    // 13. Fill observation
    fill_obs(env_idx);
}

} // namespace fast_finrl
