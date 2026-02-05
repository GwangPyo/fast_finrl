#include "VecReplayBuffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

VecReplayBuffer::VecReplayBuffer(std::shared_ptr<const FastFinRL> env, size_t capacity, size_t batch_size)
    : env_(std::move(env))
    , capacity_(capacity)
    , batch_size_(batch_size)
    , rng_(std::random_device{}())
{
    buffer_.reserve(std::min(capacity_, size_t(1000000)));

    // Cache metadata for faster sampling
    auto indicator_set = env_->get_indicator_names();
    cached_indicator_names_.assign(indicator_set.begin(), indicator_set.end());
    n_indicators_ = static_cast<int>(cached_indicator_names_.size());

    cached_macro_tickers_ = env_->get_macro_tickers();
    n_macro_tickers_ = static_cast<int>(cached_macro_tickers_.size());
}

VecReplayBuffer::VecReplayBuffer(const VecFastFinRL& vec_env, size_t capacity, size_t batch_size)
    : VecReplayBuffer(vec_env.get_base_env(), capacity, batch_size)
{}

void VecReplayBuffer::add(const VecStoredTransition& transition) {
    if (buffer_.size() < capacity_) {
        buffer_.push_back(transition);
    } else {
        buffer_[write_idx_] = transition;
        full_ = true;
    }
    write_idx_ = (write_idx_ + 1) % capacity_;
}

void VecReplayBuffer::add_batch(
    int num_envs,
    const std::vector<int>& env_ids,
    const std::vector<int>& state_days,
    const std::vector<int>& next_state_days,
    const std::vector<std::vector<std::string>>& tickers_list,
    const std::vector<double>& state_cash,
    const std::vector<double>& next_state_cash,
    const int* state_shares,
    const int* next_state_shares,
    const double* state_avg_buy_price,
    const double* next_state_avg_buy_price,
    const double* actions,
    const std::vector<std::vector<double>>& rewards,
    const std::vector<bool>& dones,
    const std::vector<bool>& terminals,
    int n_tickers)
{
    for (int i = 0; i < num_envs; ++i) {
        VecStoredTransition t;
        t.env_id = env_ids[i];
        t.state_day = state_days[i];
        t.next_state_day = next_state_days[i];
        t.tickers = tickers_list[i];
        t.state_cash = state_cash[i];
        t.next_state_cash = next_state_cash[i];

        // Copy arrays
        t.state_shares.resize(n_tickers);
        t.next_state_shares.resize(n_tickers);
        t.state_avg_buy_price.resize(n_tickers);
        t.next_state_avg_buy_price.resize(n_tickers);
        t.action.resize(n_tickers);

        size_t base = i * n_tickers;
        for (int j = 0; j < n_tickers; ++j) {
            t.state_shares[j] = state_shares[base + j];
            t.next_state_shares[j] = next_state_shares[base + j];
            t.state_avg_buy_price[j] = state_avg_buy_price[base + j];
            t.next_state_avg_buy_price[j] = next_state_avg_buy_price[base + j];
            t.action[j] = actions[base + j];
        }

        t.rewards = rewards[i];
        t.done = dones[i];
        t.terminal = terminals[i];

        add(t);
    }
}

std::vector<size_t> VecReplayBuffer::sample_indices(size_t batch_size) const {
    size_t current_size = size();
    if (current_size == 0) {
        throw std::runtime_error("Cannot sample from empty buffer");
    }

    batch_size = std::min(batch_size, current_size);
    std::vector<size_t> indices(batch_size);
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);

    for (size_t i = 0; i < batch_size; ++i) {
        indices[i] = dist(rng_);
    }

    return indices;
}

const VecStoredTransition& VecReplayBuffer::get(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Index out of range");
    }
    return buffer_[index];
}

VecReplayBuffer::MultiTickerWindowData VecReplayBuffer::get_market_data(
    size_t index, int h, int future, bool next_state) const
{
    const auto& t = get(index);
    int day = next_state ? t.next_state_day : t.state_day;
    return env_->get_market_window_multi(t.tickers, day, h, future);
}

VecReplayBuffer::SampleBatch VecReplayBuffer::sample(int h) const {
    return sample(h, batch_size_);
}

VecReplayBuffer::SampleBatch VecReplayBuffer::sample(int h, size_t batch_size) const {
    auto indices = sample_indices(batch_size);
    const size_t actual_batch = indices.size();

    SampleBatch result;
    result.batch_size = static_cast<int>(actual_batch);
    result.h = h;

    if (actual_batch == 0) return result;

    // Collect per-sample tickers and build unique ticker set
    std::set<std::string> unique_ticker_set;
    result.tickers.resize(actual_batch);
    for (size_t i = 0; i < actual_batch; ++i) {
        const auto& t = buffer_[indices[i]];
        result.tickers[i] = t.tickers;
        for (const auto& tic : t.tickers) {
            unique_ticker_set.insert(tic);
        }
    }
    result.unique_tickers.assign(unique_ticker_set.begin(), unique_ticker_set.end());

    // Get structure from first transition
    const auto& first_t = buffer_[indices[0]];
    result.n_tickers = static_cast<int>(first_t.tickers.size());
    result.n_objectives = static_cast<int>(first_t.rewards.size());

    // Use cached indicator names (no repeated lookups)
    result.indicator_names = cached_indicator_names_;
    result.n_indicators = n_indicators_;

    // Use cached macro tickers
    result.macro_tickers = cached_macro_tickers_;
    result.n_macro_tickers = n_macro_tickers_;

    const int time_len = h + 1;
    const int n_tickers = result.n_tickers;
    const int n_ind = result.n_indicators;

    // Pre-allocate all arrays
    result.env_ids.resize(actual_batch);
    result.actions.resize(actual_batch * n_tickers);
    result.rewards.resize(actual_batch);
    result.dones.resize(actual_batch);
    result.state_cash.resize(actual_batch);
    result.next_state_cash.resize(actual_batch);
    result.state_shares.resize(actual_batch * n_tickers);
    result.next_state_shares.resize(actual_batch * n_tickers);
    result.state_avg_buy_price.resize(actual_batch * n_tickers);
    result.next_state_avg_buy_price.resize(actual_batch * n_tickers);

    // Allocate per-ticker market data (for all unique tickers)
    for (const auto& ticker : result.unique_tickers) {
        result.s_ohlc[ticker].resize(actual_batch * time_len * 4);
        result.s_indicators[ticker].resize(actual_batch * time_len * n_ind);
        result.s_next_ohlc[ticker].resize(actual_batch * time_len * 4);
        result.s_next_indicators[ticker].resize(actual_batch * time_len * n_ind);
        if (h > 0) {
            result.s_mask[ticker].resize(actual_batch * time_len);
            result.s_next_mask[ticker].resize(actual_batch * time_len);
        }
    }

    // Allocate macro ticker data
    for (const std::string& ticker : cached_macro_tickers_) {
        result.macro_ohlc[ticker].resize(actual_batch * time_len * 4);
        result.macro_indicators[ticker].resize(actual_batch * time_len * n_ind);
        result.macro_next_ohlc[ticker].resize(actual_batch * time_len * 4);
        result.macro_next_indicators[ticker].resize(actual_batch * time_len * n_ind);
        if (h > 0) {
            result.macro_mask[ticker].resize(actual_batch * time_len);
            result.macro_next_mask[ticker].resize(actual_batch * time_len);
        }
    }

    // Build batch sample lists for each ticker (state and next_state)
    // This allows us to use fill_market_batch for efficient batch fetching
    std::map<std::string, std::vector<std::pair<size_t, int>>> ticker_samples;       // state
    std::map<std::string, std::vector<std::pair<size_t, int>>> ticker_next_samples;  // next_state
    std::map<std::string, std::vector<std::pair<size_t, int>>> macro_samples;
    std::map<std::string, std::vector<std::pair<size_t, int>>> macro_next_samples;

    // Pre-compute global indices for all unique tickers
    std::map<std::string, size_t> ticker_global_idx;
    for (const auto& ticker : result.unique_tickers) {
        ticker_global_idx[ticker] = env_->get_ticker_global_idx(ticker);
    }
    for (const auto& ticker : cached_macro_tickers_) {
        ticker_global_idx[ticker] = env_->get_ticker_global_idx(ticker);
    }

    // Collect all (global_idx, day) pairs for batch fetch
    for (size_t i = 0; i < actual_batch; ++i) {
        const auto& t = buffer_[indices[i]];

        // Actions and portfolio (non-market data)
        result.env_ids[i] = t.env_id;
        for (int j = 0; j < n_tickers; ++j) {
            result.actions[i * n_tickers + j] = t.action[j];
            result.state_shares[i * n_tickers + j] = t.state_shares[j];
            result.next_state_shares[i * n_tickers + j] = t.next_state_shares[j];
            result.state_avg_buy_price[i * n_tickers + j] = t.state_avg_buy_price[j];
            result.next_state_avg_buy_price[i * n_tickers + j] = t.next_state_avg_buy_price[j];
        }
        result.rewards[i] = t.rewards;
        result.dones[i] = t.done;
        result.state_cash[i] = t.state_cash;
        result.next_state_cash[i] = t.next_state_cash;

        // Build sample lists for tickers
        for (const auto& ticker : t.tickers) {
            size_t global_idx = ticker_global_idx[ticker];
            ticker_samples[ticker].emplace_back(global_idx, t.state_day);
            ticker_next_samples[ticker].emplace_back(global_idx, t.next_state_day);
        }

        // Macro tickers
        for (const auto& ticker : cached_macro_tickers_) {
            size_t global_idx = ticker_global_idx[ticker];
            macro_samples[ticker].emplace_back(global_idx, t.state_day);
            macro_next_samples[ticker].emplace_back(global_idx, t.next_state_day);
        }
    }

    // Batch fetch market data for each unique ticker
    for (const auto& ticker : result.unique_tickers) {
        if (!ticker_samples[ticker].empty()) {
            env_->fill_market_batch(
                ticker_samples[ticker], h,
                result.s_ohlc[ticker].data(),
                result.s_indicators[ticker].data(),
                h > 0 ? result.s_mask[ticker].data() : nullptr
            );
            env_->fill_market_batch(
                ticker_next_samples[ticker], h,
                result.s_next_ohlc[ticker].data(),
                result.s_next_indicators[ticker].data(),
                h > 0 ? result.s_next_mask[ticker].data() : nullptr
            );
        }
    }

    // Batch fetch macro tickers
    for (const auto& ticker : cached_macro_tickers_) {
        env_->fill_market_batch(
            macro_samples[ticker], h,
            result.macro_ohlc[ticker].data(),
            result.macro_indicators[ticker].data(),
            h > 0 ? result.macro_mask[ticker].data() : nullptr
        );
        env_->fill_market_batch(
            macro_next_samples[ticker], h,
            result.macro_next_ohlc[ticker].data(),
            result.macro_next_indicators[ticker].data(),
            h > 0 ? result.macro_next_mask[ticker].data() : nullptr
        );
    }

    return result;
}

size_t VecReplayBuffer::size() const {
    return full_ ? capacity_ : buffer_.size();
}

void VecReplayBuffer::clear() {
    buffer_.clear();
    write_idx_ = 0;
    full_ = false;
}

void VecReplayBuffer::save(const std::string& path) const {
    nlohmann::json j;
    j["capacity"] = capacity_;
    j["write_idx"] = write_idx_;
    j["full"] = full_;

    nlohmann::json transitions = nlohmann::json::array();
    for (const auto& t : buffer_) {
        nlohmann::json tj;
        tj["env_id"] = t.env_id;
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

void VecReplayBuffer::load(const std::string& path) {
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
        VecStoredTransition t;
        t.env_id = tj["env_id"].get<int>();
        t.state_day = tj["state_day"].get<int>();
        t.tickers = tj["tickers"].get<std::vector<std::string>>();
        t.state_cash = tj["state_cash"].get<double>();
        t.state_shares = tj["state_shares"].get<std::vector<int>>();
        t.state_avg_buy_price = tj["state_avg_buy_price"].get<std::vector<double>>();
        t.action = tj["action"].get<std::vector<double>>();
        t.rewards = tj["rewards"].get<std::vector<double>>();
        t.done = tj["done"].get<bool>();
        t.terminal = tj["terminal"].get<bool>();
        t.next_state_day = tj["next_state_day"].get<int>();
        t.next_state_cash = tj["next_state_cash"].get<double>();
        t.next_state_shares = tj["next_state_shares"].get<std::vector<int>>();
        t.next_state_avg_buy_price = tj["next_state_avg_buy_price"].get<std::vector<double>>();
        buffer_.push_back(t);
    }
}

} // namespace fast_finrl
