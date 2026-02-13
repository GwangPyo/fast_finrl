#include "ReplayBuffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

namespace {
// Helper: copy market data from raw MarketWindowData to result arrays
inline void copy_market_data(
    xt::xarray<float>& result_ohlcv,
    xt::xarray<float>& result_ind,
    xt::xarray<int>& result_mask,
    const FastFinRL::MarketWindowData& raw,
    size_t batch_idx, size_t ticker_idx,
    size_t raw_len, int slice_start, int slice_end, size_t n_ind)
{
    xt::xarray<double> ohlcv_full = xt::adapt(raw.ohlcv, {raw_len, 5UL});
    xt::xarray<double> ind_full = xt::adapt(raw.indicators, {raw_len, n_ind});
    xt::xarray<int> mask_full = xt::adapt(raw.mask, {raw_len});

    xt::view(result_ohlcv, batch_idx, ticker_idx, xt::all(), xt::all()) =
        xt::cast<float>(xt::view(ohlcv_full, xt::range(slice_start, slice_end), xt::all()));
    xt::view(result_ind, batch_idx, ticker_idx, xt::all(), xt::all()) =
        xt::cast<float>(xt::view(ind_full, xt::range(slice_start, slice_end), xt::all()));
    xt::view(result_mask, batch_idx, ticker_idx, xt::all()) =
        xt::view(mask_full, xt::range(slice_start, slice_end));
}
} // anonymous namespace

ReplayBuffer::ReplayBuffer(std::shared_ptr<const FastFinRL> env, size_t capacity, size_t batch_size, int64_t seed, std::vector<size_t> action_shape)
    : env_(std::move(env))
    , capacity_(capacity)
    , batch_size_(batch_size)
    , action_shape_(std::move(action_shape))
    , rng_(seed >= 0 ? static_cast<unsigned int>(seed) : std::random_device{}())
{
    // Reserve up to 1M, larger buffers grow dynamically
    buffer_.reserve(std::min(capacity_, size_t(1000000)));
    std::mt19937 gen(seed); // fix seed

    // Default action_shape = (n_tickers,)
    if (action_shape_.empty() && env_) {
        action_shape_ = {static_cast<size_t>(env_->n_tickers())};
    }
}

void ReplayBuffer::add(const StoredTransition& transition) {
    if (buffer_.size() < capacity_) {
        buffer_.push_back(transition);
    } else {
        buffer_[write_idx_] = transition;
        full_ = true;
    }
    write_idx_ = (write_idx_ + 1) % capacity_;
}

void ReplayBuffer::add_transition(
    int state_day, int next_state_day,
    const std::vector<std::string>& tickers,
    float state_cash, float next_state_cash,
    const std::vector<int>& state_shares,
    const std::vector<int>& next_state_shares,
    const std::vector<float>& state_avg_buy_price,
    const std::vector<float>& next_state_avg_buy_price,
    const std::vector<float>& action,
    const std::vector<float>& rewards, bool done, bool terminal)
{
    StoredTransition t;
    t.state_day = state_day;
    t.next_state_day = next_state_day;
    t.tickers = tickers;
    t.state_cash = state_cash;
    t.next_state_cash = next_state_cash;
    t.state_shares = state_shares;
    t.next_state_shares = next_state_shares;
    t.state_avg_buy_price = state_avg_buy_price;
    t.next_state_avg_buy_price = next_state_avg_buy_price;
    t.action = action;
    t.rewards = rewards;
    t.done = done;
    t.terminal = terminal;
    add(t);
}

std::vector<size_t> ReplayBuffer::sample_indices(size_t batch_size) const {
    return sample_indices(batch_size, 0);
}

std::vector<size_t> ReplayBuffer::sample_indices(size_t batch_size, int history_length) const {
    size_t current_size = size();
    if (current_size == 0) {
        throw std::runtime_error("Cannot sample from empty buffer");
    }

    // Get macro tickers for first_day check
    const std::vector<std::string>& macro_tickers = env_->get_macro_tickers();

    // Build list of valid indices
    // For each transition, check if state_day >= max(first_day of tickers) + history_length
    std::vector<size_t> valid_indices;
    valid_indices.reserve(current_size);
    for (size_t i = 0; i < current_size; ++i) {
        const auto& t = buffer_[i];
        // Find max first_day among this transition's tickers
        int max_first_day = 0;
        for (const auto& tic : t.tickers) {
            int first_day = env_->get_ticker_first_day(tic);
            max_first_day = std::max(max_first_day, first_day);
        }
        // Also check macro_tickers' first_day
        for (const auto& tic : macro_tickers) {
            int first_day = env_->get_ticker_first_day(tic);
            max_first_day = std::max(max_first_day, first_day);
        }
        int min_day = max_first_day + history_length;
        if (t.state_day >= min_day) {
            valid_indices.push_back(i);
        }
    }

    if (valid_indices.empty()) {
        throw std::runtime_error("No valid samples with sufficient history (h=" + std::to_string(history_length) + ")");
    }

    batch_size = std::min(batch_size, valid_indices.size());
    std::vector<size_t> indices(batch_size);
    std::uniform_int_distribution<size_t> dist(0, valid_indices.size() - 1);

    for (size_t i = 0; i < batch_size; ++i) {
        indices[i] = valid_indices[dist(rng_)];
    }

    return indices;
}

const StoredTransition& ReplayBuffer::get(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Index out of range");
    }
    return buffer_[index];
}

ReplayBuffer::MultiTickerWindowData ReplayBuffer::get_market_data(
    size_t index, int h, int future, bool next_state) const
{
    const auto& t = get(index);
    int day = next_state ? t.next_state_day : t.state_day;
    return env_->get_market_window_multi(t.tickers, day, h, future);
}

ReplayBuffer::SampleBatch ReplayBuffer::sample(int history_length) const {
    return sample(batch_size_, history_length, 0);
}

ReplayBuffer::SampleBatch ReplayBuffer::sample(size_t batch_size, int history_length, int future_length) const {
    std::vector<size_t> indices = sample_indices(batch_size, history_length);
    const size_t B = indices.size();

    SampleBatch result;
    result.batch_size = static_cast<int>(B);
    result.history_length = history_length;
    result.future_length = future_length;
    result.action_shape = action_shape_;

    if (B == 0) return result;

    // Get first transition to determine structure
    const StoredTransition& first_t = buffer_[indices[0]];
    result.tickers = first_t.tickers;
    result.n_tickers = static_cast<int>(first_t.tickers.size());
    result.n_objectives = static_cast<int>(first_t.rewards.size());

    // Get indicator names from env
    std::set<std::string> indicator_set = env_->get_indicator_names();
    result.indicator_names.assign(indicator_set.begin(), indicator_set.end());
    result.n_indicators = static_cast<int>(result.indicator_names.size());

    // Macro tickers
    const std::vector<std::string>& macro_tickers = env_->get_macro_tickers();
    result.macro_tickers = macro_tickers;
    result.n_macro_tickers = static_cast<int>(macro_tickers.size());

    const int n_tic = result.n_tickers;
    const int n_macro = result.n_macro_tickers;
    const int n_ind = result.n_indicators;
    const int h = history_length;
    const int f = future_length;

    // Compute action shape with batch dim
    std::vector<size_t> action_dims = {B};
    for (size_t dim : action_shape_) action_dims.push_back(dim);

    // Allocate xtensor arrays
    result.actions = xt::zeros<float>(action_dims);
    result.rewards.resize(B);
    result.dones.resize(B);
    result.state_cash.resize(B);
    result.next_state_cash.resize(B);
    result.state_shares = xt::zeros<int>({B, static_cast<size_t>(n_tic)});
    result.next_state_shares = xt::zeros<int>({B, static_cast<size_t>(n_tic)});
    result.state_avg_buy_price = xt::zeros<float>({B, static_cast<size_t>(n_tic)});
    result.next_state_avg_buy_price = xt::zeros<float>({B, static_cast<size_t>(n_tic)});

    // Allocate market data arrays [B, n_tickers, time, ...]
    if (h > 0) {
        result.s_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(h), 5UL});
        result.s_indicators = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(h), static_cast<size_t>(n_ind)});
        result.s_mask = xt::zeros<int>({B, static_cast<size_t>(n_tic), static_cast<size_t>(h)});
        result.s_next_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(h), 5UL});
        result.s_next_indicators = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(h), static_cast<size_t>(n_ind)});
        result.s_next_mask = xt::zeros<int>({B, static_cast<size_t>(n_tic), static_cast<size_t>(h)});

        // Macro: [B, n_macro, time, ...]
        if (n_macro > 0) {
            result.macro_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(h), 5UL});
            result.macro_indicators = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(h), static_cast<size_t>(n_ind)});
            result.macro_mask = xt::zeros<int>({B, static_cast<size_t>(n_macro), static_cast<size_t>(h)});
            result.macro_next_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(h), 5UL});
            result.macro_next_indicators = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(h), static_cast<size_t>(n_ind)});
            result.macro_next_mask = xt::zeros<int>({B, static_cast<size_t>(n_macro), static_cast<size_t>(h)});
        }
    }

    // Future arrays
    if (f > 0) {
        result.s_future_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(f), 5UL});
        result.s_future_indicators = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(f), static_cast<size_t>(n_ind)});
        result.s_future_mask = xt::zeros<int>({B, static_cast<size_t>(n_tic), static_cast<size_t>(f)});
        result.s_next_future_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(f), 5UL});
        result.s_next_future_indicators = xt::zeros<float>({B, static_cast<size_t>(n_tic), static_cast<size_t>(f), static_cast<size_t>(n_ind)});
        result.s_next_future_mask = xt::zeros<int>({B, static_cast<size_t>(n_tic), static_cast<size_t>(f)});

        if (n_macro > 0) {
            result.macro_future_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(f), 5UL});
            result.macro_future_indicators = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(f), static_cast<size_t>(n_ind)});
            result.macro_future_mask = xt::zeros<int>({B, static_cast<size_t>(n_macro), static_cast<size_t>(f)});
            result.macro_next_future_ohlcv = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(f), 5UL});
            result.macro_next_future_indicators = xt::zeros<float>({B, static_cast<size_t>(n_macro), static_cast<size_t>(f), static_cast<size_t>(n_ind)});
            result.macro_next_future_mask = xt::zeros<int>({B, static_cast<size_t>(n_macro), static_cast<size_t>(f)});
        }
    }

    // Fill data per sample (parallel)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, B),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                const StoredTransition& t = buffer_[indices[i]];

                // Copy basic fields
                result.rewards[i] = t.rewards;
                result.dones[i] = t.done;
                result.state_cash[i] = t.state_cash;
                result.next_state_cash[i] = t.next_state_cash;

                // Actions [B, action_shape...] - use xt::view
                auto action_view = xt::view(result.actions, i, xt::all());
                std::copy(t.action.begin(), t.action.end(), action_view.begin());

                // Portfolio state [B, n_tickers] - use xt::view
                auto shares_view = xt::view(result.state_shares, i, xt::all());
                auto next_shares_view = xt::view(result.next_state_shares, i, xt::all());
                auto avg_view = xt::view(result.state_avg_buy_price, i, xt::all());
                auto next_avg_view = xt::view(result.next_state_avg_buy_price, i, xt::all());
                std::copy(t.state_shares.begin(), t.state_shares.end(), shares_view.begin());
                std::copy(t.next_state_shares.begin(), t.next_state_shares.end(), next_shares_view.begin());
                std::copy(t.state_avg_buy_price.begin(), t.state_avg_buy_price.end(), avg_view.begin());
                std::copy(t.next_state_avg_buy_price.begin(), t.next_state_avg_buy_price.end(), next_avg_view.begin());

                // Market data: get_market_window_raw returns (h+1) elements, slice [0,h) for history
                if (h > 0) {
                    size_t raw_len = static_cast<size_t>(h + 1);
                    for (int j = 0; j < n_tic; ++j) {
                        const std::string& tic = t.tickers[j];
                        FastFinRL::MarketWindowData raw = env_->get_market_window_raw(tic, t.state_day, h, 0);
                        FastFinRL::MarketWindowData raw_next = env_->get_market_window_raw(tic, t.next_state_day, h, 0);
                        copy_market_data(result.s_ohlcv, result.s_indicators, result.s_mask, raw, i, j, raw_len, 0, h, n_ind);
                        copy_market_data(result.s_next_ohlcv, result.s_next_indicators, result.s_next_mask, raw_next, i, j, raw_len, 0, h, n_ind);
                    }
                    for (int m = 0; m < n_macro; ++m) {
                        const std::string& tic = macro_tickers[m];
                        FastFinRL::MarketWindowData raw = env_->get_market_window_raw(tic, t.state_day, h, 0);
                        FastFinRL::MarketWindowData raw_next = env_->get_market_window_raw(tic, t.next_state_day, h, 0);
                        copy_market_data(result.macro_ohlcv, result.macro_indicators, result.macro_mask, raw, i, m, raw_len, 0, h, n_ind);
                        copy_market_data(result.macro_next_ohlcv, result.macro_next_indicators, result.macro_next_mask, raw_next, i, m, raw_len, 0, h, n_ind);
                    }
                }

                // Future data: slice [1, 1+f) to skip current day at index 0
                if (f > 0) {
                    size_t raw_future_len = static_cast<size_t>(1 + f);
                    for (int j = 0; j < n_tic; ++j) {
                        const std::string& tic = t.tickers[j];
                        FastFinRL::MarketWindowData raw = env_->get_market_window_raw(tic, t.state_day, 0, f);
                        FastFinRL::MarketWindowData raw_next = env_->get_market_window_raw(tic, t.next_state_day, 0, f);
                        copy_market_data(result.s_future_ohlcv, result.s_future_indicators, result.s_future_mask, raw, i, j, raw_future_len, 1, 1 + f, n_ind);
                        copy_market_data(result.s_next_future_ohlcv, result.s_next_future_indicators, result.s_next_future_mask, raw_next, i, j, raw_future_len, 1, 1 + f, n_ind);
                    }
                    for (int m = 0; m < n_macro; ++m) {
                        const std::string& tic = macro_tickers[m];
                        FastFinRL::MarketWindowData raw = env_->get_market_window_raw(tic, t.state_day, 0, f);
                        FastFinRL::MarketWindowData raw_next = env_->get_market_window_raw(tic, t.next_state_day, 0, f);
                        copy_market_data(result.macro_future_ohlcv, result.macro_future_indicators, result.macro_future_mask, raw, i, m, raw_future_len, 1, 1 + f, n_ind);
                        copy_market_data(result.macro_next_future_ohlcv, result.macro_next_future_indicators, result.macro_next_future_mask, raw_next, i, m, raw_future_len, 1, 1 + f, n_ind);
                    }
                }
            }
        });

    return result;
}

size_t ReplayBuffer::size() const {
    return full_ ? capacity_ : buffer_.size();
}

void ReplayBuffer::clear() {
    buffer_.clear();
    write_idx_ = 0;
    full_ = false;
}

void ReplayBuffer::save(const std::string& path) const {
    nlohmann::json j;
    j["capacity"] = capacity_;
    j["write_idx"] = write_idx_;
    j["full"] = full_;

    nlohmann::json transitions = nlohmann::json::array();
    for (const auto& t : buffer_) {
        nlohmann::json tj;
        tj["state_day"] = t.state_day;
        tj["tickers"] = t.tickers;
        tj["state_cash"] = t.state_cash;
        tj["state_shares"] = t.state_shares;
        tj["state_avg_buy_price"] = t.state_avg_buy_price;
        tj["action"] = t.action;
        tj["rewards"] = t.rewards;
        tj["done"] = t.done;
        tj["terminal"] = t.terminal;
        tj["next_state_day"] = t.next_state_day;
        tj["next_state_cash"] = t.next_state_cash;
        tj["next_state_shares"] = t.next_state_shares;
        tj["next_state_avg_buy_price"] = t.next_state_avg_buy_price;
        transitions.push_back(tj);
    }
    j["transitions"] = transitions;

    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    f << j.dump();
}

void ReplayBuffer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    nlohmann::json j;
    f >> j;

    capacity_ = j["capacity"].get<size_t>();
    write_idx_ = j["write_idx"].get<size_t>();
    full_ = j["full"].get<bool>();

    buffer_.clear();
    buffer_.reserve(capacity_);

    for (const auto& tj : j["transitions"]) {
        StoredTransition t;
        t.state_day = tj["state_day"].get<int>();
        t.tickers = tj["tickers"].get<std::vector<std::string>>();
        t.state_cash = tj["state_cash"].get<float>();
        t.state_shares = tj["state_shares"].get<std::vector<int>>();
        t.state_avg_buy_price = tj["state_avg_buy_price"].get<std::vector<float>>();
        t.action = tj["action"].get<std::vector<float>>();
        t.rewards = tj["rewards"].get<std::vector<float>>();
        t.done = tj["done"].get<bool>();
        t.terminal = tj["terminal"].get<bool>();
        t.next_state_day = tj["next_state_day"].get<int>();
        t.next_state_cash = tj["next_state_cash"].get<float>();
        t.next_state_shares = tj["next_state_shares"].get<std::vector<int>>();
        t.next_state_avg_buy_price = tj["next_state_avg_buy_price"].get<std::vector<float>>();
        buffer_.push_back(t);
    }
}

} // namespace fast_finrl
