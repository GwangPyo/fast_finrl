#include "ReplayBuffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

ReplayBuffer::ReplayBuffer(std::shared_ptr<const FastFinRL> env, size_t capacity, size_t batch_size)
    : env_(std::move(env))
    , capacity_(capacity)
    , batch_size_(batch_size)
    , rng_(std::random_device{}())
{
    // Reserve up to 1M, larger buffers grow dynamically
    buffer_.reserve(std::min(capacity_, size_t(1000000)));
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
    double state_cash, double next_state_cash,
    const std::vector<int>& state_shares,
    const std::vector<int>& next_state_shares,
    const std::vector<double>& state_avg_buy_price,
    const std::vector<double>& next_state_avg_buy_price,
    const std::vector<double>& action,
    const std::vector<double>& rewards, bool done, bool terminal)
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

ReplayBuffer::SampleBatch ReplayBuffer::sample(int h) const {
    return sample(h, batch_size_);
}

ReplayBuffer::SampleBatch ReplayBuffer::sample(int h, size_t batch_size) const {
    auto indices = sample_indices(batch_size);
    const size_t actual_batch = indices.size();

    SampleBatch result;
    result.batch_size = static_cast<int>(actual_batch);
    result.h = h;

    if (actual_batch == 0) return result;

    // Get first transition to determine structure
    const auto& first_t = buffer_[indices[0]];
    result.tickers = first_t.tickers;
    result.n_tickers = static_cast<int>(first_t.tickers.size());
    result.n_objectives = static_cast<int>(first_t.rewards.size());

    // Get indicator names from env
    auto indicator_set = env_->get_indicator_names();
    result.indicator_names.assign(indicator_set.begin(), indicator_set.end());
    result.n_indicators = static_cast<int>(result.indicator_names.size());

    const int time_len = h + 1;  // h history + current
    const int n_tickers = result.n_tickers;
    const int n_ind = result.n_indicators;
    const int n_obj = result.n_objectives;

    // Pre-allocate all arrays
    result.actions.resize(actual_batch * n_tickers);
    result.rewards.resize(actual_batch);
    result.dones.resize(actual_batch);
    result.state_cash.resize(actual_batch);
    result.next_state_cash.resize(actual_batch);
    result.state_shares.resize(actual_batch * n_tickers);
    result.next_state_shares.resize(actual_batch * n_tickers);
    result.state_avg_buy_price.resize(actual_batch * n_tickers);
    result.next_state_avg_buy_price.resize(actual_batch * n_tickers);

    for (const auto& ticker : result.tickers) {
        result.s_ohlc[ticker].resize(actual_batch * time_len * 4);
        result.s_indicators[ticker].resize(actual_batch * time_len * n_ind);
        result.s_next_ohlc[ticker].resize(actual_batch * time_len * 4);
        result.s_next_indicators[ticker].resize(actual_batch * time_len * n_ind);
        if (h > 0) {
            result.s_mask[ticker].resize(actual_batch * time_len);
            result.s_next_mask[ticker].resize(actual_batch * time_len);
        }
    }

    // Parallel fetch market data
    tbb::parallel_for(tbb::blocked_range<size_t>(0, actual_batch),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                const auto& t = buffer_[indices[i]];

                // Copy actions and portfolio state
                for (int j = 0; j < n_tickers; ++j) {
                    result.actions[i * n_tickers + j] = t.action[j];
                    result.state_shares[i * n_tickers + j] = t.state_shares[j];
                    result.next_state_shares[i * n_tickers + j] = t.next_state_shares[j];
                    result.state_avg_buy_price[i * n_tickers + j] = t.state_avg_buy_price[j];
                    result.next_state_avg_buy_price[i * n_tickers + j] = t.next_state_avg_buy_price[j];
                }
                // Multi-objective rewards
                result.rewards[i] = t.rewards;
                result.dones[i] = t.done;
                result.state_cash[i] = t.state_cash;
                result.next_state_cash[i] = t.next_state_cash;

                // Use get_market_window_raw for flat arrays - single memcpy per ticker
                for (const auto& ticker : t.tickers) {
                    auto raw = env_->get_market_window_raw(ticker, t.state_day, h, 0);
                    auto raw_next = env_->get_market_window_raw(ticker, t.next_state_day, h, 0);

                    size_t ohlc_size = time_len * 4;
                    size_t ind_size = time_len * n_ind;

                    // Single memcpy each
                    std::memcpy(result.s_ohlc[ticker].data() + i * ohlc_size,
                               raw.ohlc.data(), ohlc_size * sizeof(double));
                    std::memcpy(result.s_indicators[ticker].data() + i * ind_size,
                               raw.indicators.data(), ind_size * sizeof(double));
                    std::memcpy(result.s_next_ohlc[ticker].data() + i * ohlc_size,
                               raw_next.ohlc.data(), ohlc_size * sizeof(double));
                    std::memcpy(result.s_next_indicators[ticker].data() + i * ind_size,
                               raw_next.indicators.data(), ind_size * sizeof(double));

                    if (h > 0) {
                        size_t mask_size = time_len;
                        std::memcpy(result.s_mask[ticker].data() + i * mask_size,
                                   raw.mask.data(), mask_size * sizeof(int));
                        std::memcpy(result.s_next_mask[ticker].data() + i * mask_size,
                                   raw_next.mask.data(), mask_size * sizeof(int));
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
