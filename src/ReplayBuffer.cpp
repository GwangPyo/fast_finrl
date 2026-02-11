#include "ReplayBuffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

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
    auto indices = sample_indices(batch_size, history_length);
    const size_t actual_batch = indices.size();

    SampleBatch result;
    result.batch_size = static_cast<int>(actual_batch);
    result.history_length = history_length;
    result.future_length = future_length;
    result.action_shape = action_shape_;

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

    const int h = history_length;
    const int time_len = h;  // h history only (no current day to prevent lookahead)
    const int n_tickers = result.n_tickers;
    const int n_ind = result.n_indicators;

    // Compute action flat size from action_shape
    size_t action_flat_size = 1;
    for (size_t dim : action_shape_) action_flat_size *= dim;

    // Pre-allocate all arrays
    result.actions.resize(actual_batch * action_flat_size);
    result.rewards.resize(actual_batch);
    result.dones.resize(actual_batch);
    result.state_cash.resize(actual_batch);
    result.next_state_cash.resize(actual_batch);
    result.state_shares.resize(actual_batch * n_tickers);
    result.next_state_shares.resize(actual_batch * n_tickers);
    result.state_avg_buy_price.resize(actual_batch * n_tickers);
    result.next_state_avg_buy_price.resize(actual_batch * n_tickers);

    if (h > 0) {
        for (const auto& ticker : result.tickers) {
            result.s_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.s_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.s_next_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.s_next_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.s_mask[ticker].resize(actual_batch * time_len);
            result.s_next_mask[ticker].resize(actual_batch * time_len);
        }
    }

    // Future market data
    if (future_length > 0) {
        for (const auto& ticker : result.tickers) {
            result.s_future_ohlcv[ticker].resize(actual_batch * future_length * 5);
            result.s_future_indicators[ticker].resize(actual_batch * future_length * n_ind);
            result.s_future_mask[ticker].resize(actual_batch * future_length);
            result.s_next_future_ohlcv[ticker].resize(actual_batch * future_length * 5);
            result.s_next_future_indicators[ticker].resize(actual_batch * future_length * n_ind);
            result.s_next_future_mask[ticker].resize(actual_batch * future_length);
        }
    }

    // Macro tickers - get from env
    const std::vector<std::string>& macro_tickers = env_->get_macro_tickers();
    result.macro_tickers = macro_tickers;
    result.n_macro_tickers = static_cast<int>(macro_tickers.size());

    if (h > 0) {
        for (const std::string& ticker : macro_tickers) {
            result.macro_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.macro_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.macro_next_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.macro_next_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.macro_mask[ticker].resize(actual_batch * time_len);
            result.macro_next_mask[ticker].resize(actual_batch * time_len);
        }
    }

    // Parallel fetch market data
    tbb::parallel_for(tbb::blocked_range<size_t>(0, actual_batch),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                const auto& t = buffer_[indices[i]];

                // Copy actions (flat)
                for (size_t j = 0; j < action_flat_size && j < t.action.size(); ++j) {
                    result.actions[i * action_flat_size + j] = t.action[j];
                }
                // Copy portfolio state
                for (int j = 0; j < n_tickers; ++j) {
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

                // History market data - convert double to float
                if (h > 0) {
                    for (const auto& ticker : t.tickers) {
                        auto raw = env_->get_market_window_raw(ticker, t.state_day, h, 0);
                        auto raw_next = env_->get_market_window_raw(ticker, t.next_state_day, h, 0);

                        size_t ohlcv_size = time_len * 5;
                        size_t ind_size = time_len * n_ind;

                        // Convert double to float
                        for (size_t k = 0; k < ohlcv_size; ++k) {
                            result.s_ohlcv[ticker][i * ohlcv_size + k] = static_cast<float>(raw.ohlcv[k]);
                            result.s_next_ohlcv[ticker][i * ohlcv_size + k] = static_cast<float>(raw_next.ohlcv[k]);
                        }
                        for (size_t k = 0; k < ind_size; ++k) {
                            result.s_indicators[ticker][i * ind_size + k] = static_cast<float>(raw.indicators[k]);
                            result.s_next_indicators[ticker][i * ind_size + k] = static_cast<float>(raw_next.indicators[k]);
                        }
                        size_t mask_size = time_len;
                        std::memcpy(result.s_mask[ticker].data() + i * mask_size,
                                   raw.mask.data(), mask_size * sizeof(int));
                        std::memcpy(result.s_next_mask[ticker].data() + i * mask_size,
                                   raw_next.mask.data(), mask_size * sizeof(int));
                    }

                    // Macro tickers
                    for (const std::string& ticker : macro_tickers) {
                        auto raw = env_->get_market_window_raw(ticker, t.state_day, h, 0);
                        auto raw_next = env_->get_market_window_raw(ticker, t.next_state_day, h, 0);

                        size_t ohlcv_size = time_len * 5;
                        size_t ind_size = time_len * n_ind;

                        for (size_t k = 0; k < ohlcv_size; ++k) {
                            result.macro_ohlcv[ticker][i * ohlcv_size + k] = static_cast<float>(raw.ohlcv[k]);
                            result.macro_next_ohlcv[ticker][i * ohlcv_size + k] = static_cast<float>(raw_next.ohlcv[k]);
                        }
                        for (size_t k = 0; k < ind_size; ++k) {
                            result.macro_indicators[ticker][i * ind_size + k] = static_cast<float>(raw.indicators[k]);
                            result.macro_next_indicators[ticker][i * ind_size + k] = static_cast<float>(raw_next.indicators[k]);
                        }
                        size_t mask_size = time_len;
                        std::memcpy(result.macro_mask[ticker].data() + i * mask_size,
                                   raw.mask.data(), mask_size * sizeof(int));
                        std::memcpy(result.macro_next_mask[ticker].data() + i * mask_size,
                                   raw_next.mask.data(), mask_size * sizeof(int));
                    }
                }

                // Future market data - when h=0, future=N: data is in ohlcv/indicators/mask fields
                if (future_length > 0) {
                    for (const auto& ticker : t.tickers) {
                        auto raw = env_->get_market_window_raw(ticker, t.state_day, 0, future_length);
                        auto raw_next = env_->get_market_window_raw(ticker, t.next_state_day, 0, future_length);

                        size_t ohlcv_size = future_length * 5;
                        size_t ind_size = future_length * n_ind;

                        for (size_t k = 0; k < ohlcv_size; ++k) {
                            result.s_future_ohlcv[ticker][i * ohlcv_size + k] = static_cast<float>(raw.ohlcv[k]);
                            result.s_next_future_ohlcv[ticker][i * ohlcv_size + k] = static_cast<float>(raw_next.ohlcv[k]);
                        }
                        for (size_t k = 0; k < ind_size; ++k) {
                            result.s_future_indicators[ticker][i * ind_size + k] = static_cast<float>(raw.indicators[k]);
                            result.s_next_future_indicators[ticker][i * ind_size + k] = static_cast<float>(raw_next.indicators[k]);
                        }
                        size_t mask_size = future_length;
                        std::memcpy(result.s_future_mask[ticker].data() + i * mask_size,
                                   raw.mask.data(), mask_size * sizeof(int));
                        std::memcpy(result.s_next_future_mask[ticker].data() + i * mask_size,
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
