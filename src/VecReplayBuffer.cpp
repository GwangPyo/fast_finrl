#include "VecReplayBuffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace fast_finrl {

VecReplayBuffer::VecReplayBuffer(std::shared_ptr<const FastFinRL> env, size_t capacity, size_t batch_size, int64_t seed, std::vector<size_t> action_shape)
    : env_(std::move(env))
    , capacity_(capacity)
    , batch_size_(batch_size)
    , action_shape_(std::move(action_shape))
    , rng_(seed >= 0 ? static_cast<unsigned int>(seed) : std::random_device{}())
{
    buffer_.reserve(std::min(capacity_, size_t(1000000)));

    // Default action_shape = (n_tickers,)
    if (action_shape_.empty() && env_) {
        action_shape_ = {static_cast<size_t>(env_->n_tickers())};
    }

    // Cache metadata for faster sampling
    auto indicator_set = env_->get_indicator_names();
    cached_indicator_names_.assign(indicator_set.begin(), indicator_set.end());
    n_indicators_ = static_cast<int>(cached_indicator_names_.size());

    cached_macro_tickers_ = env_->get_macro_tickers();
    n_macro_tickers_ = static_cast<int>(cached_macro_tickers_.size());
}

VecReplayBuffer::VecReplayBuffer(const VecFastFinRL& vec_env, size_t capacity, size_t batch_size, int64_t seed, std::vector<size_t> action_shape)
    : VecReplayBuffer(vec_env.get_base_env(), capacity, batch_size, seed, std::move(action_shape))
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
    const std::vector<float>& state_cash,
    const std::vector<float>& next_state_cash,
    const int* state_shares,
    const int* next_state_shares,
    const float* state_avg_buy_price,
    const float* next_state_avg_buy_price,
    const float* actions,
    const std::vector<std::vector<float>>& rewards,
    const std::vector<bool>& dones,
    const std::vector<bool>& terminals,
    int n_tickers)
{
    // Compute action flat size from action_shape
    size_t action_flat_size = 1;
    for (size_t dim : action_shape_) action_flat_size *= dim;

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
        t.action.resize(action_flat_size);

        size_t share_base = i * n_tickers;
        for (int j = 0; j < n_tickers; ++j) {
            t.state_shares[j] = state_shares[share_base + j];
            t.next_state_shares[j] = next_state_shares[share_base + j];
            t.state_avg_buy_price[j] = state_avg_buy_price[share_base + j];
            t.next_state_avg_buy_price[j] = next_state_avg_buy_price[share_base + j];
        }

        size_t action_base = i * action_flat_size;
        for (size_t j = 0; j < action_flat_size; ++j) {
            t.action[j] = actions[action_base + j];
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

VecReplayBuffer::SampleBatch VecReplayBuffer::sample(int history_length) const {
    return sample(batch_size_, history_length, 0);
}

VecReplayBuffer::SampleBatch VecReplayBuffer::sample(size_t batch_size, int history_length, int future_length) const {
    auto indices = sample_indices(batch_size);
    const size_t actual_batch = indices.size();

    SampleBatch result;
    result.batch_size = static_cast<int>(actual_batch);
    result.history_length = history_length;
    result.future_length = future_length;
    result.action_shape = action_shape_;

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

    const int h = history_length;
    const int time_len = h;  // history only (no current day to prevent lookahead)
    const int n_tickers = result.n_tickers;
    const int n_ind = result.n_indicators;

    // Compute action flat size from action_shape
    size_t action_flat_size = 1;
    for (size_t dim : action_shape_) action_flat_size *= dim;

    // Pre-allocate all arrays
    result.env_ids.resize(actual_batch);
    result.actions.resize(actual_batch * action_flat_size);
    result.rewards.resize(actual_batch);
    result.dones.resize(actual_batch);
    result.state_cash.resize(actual_batch);
    result.next_state_cash.resize(actual_batch);
    result.state_shares.resize(actual_batch * n_tickers);
    result.next_state_shares.resize(actual_batch * n_tickers);
    result.state_avg_buy_price.resize(actual_batch * n_tickers);
    result.next_state_avg_buy_price.resize(actual_batch * n_tickers);

    // Allocate per-ticker market data (for all unique tickers)
    if (h > 0) {
        for (const auto& ticker : result.unique_tickers) {
            result.s_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.s_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.s_next_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.s_next_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.s_mask[ticker].resize(actual_batch * time_len);
            result.s_next_mask[ticker].resize(actual_batch * time_len);
        }

        // Allocate macro ticker data
        for (const std::string& ticker : cached_macro_tickers_) {
            result.macro_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.macro_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.macro_next_ohlcv[ticker].resize(actual_batch * time_len * 5);
            result.macro_next_indicators[ticker].resize(actual_batch * time_len * n_ind);
            result.macro_mask[ticker].resize(actual_batch * time_len);
            result.macro_next_mask[ticker].resize(actual_batch * time_len);
        }
    }

    // Future market data
    if (future_length > 0) {
        for (const auto& ticker : result.unique_tickers) {
            result.s_future_ohlcv[ticker].resize(actual_batch * future_length * 5);
            result.s_future_indicators[ticker].resize(actual_batch * future_length * n_ind);
            result.s_future_mask[ticker].resize(actual_batch * future_length);
            result.s_next_future_ohlcv[ticker].resize(actual_batch * future_length * 5);
            result.s_next_future_indicators[ticker].resize(actual_batch * future_length * n_ind);
            result.s_next_future_mask[ticker].resize(actual_batch * future_length);
        }
    }

    // Build batch sample lists for each ticker (state and next_state)
    std::map<std::string, std::vector<std::pair<size_t, int>>> ticker_samples;
    std::map<std::string, std::vector<std::pair<size_t, int>>> ticker_next_samples;
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

    // Collect all (global_idx, day) pairs for batch fetch and copy portfolio data
    for (size_t i = 0; i < actual_batch; ++i) {
        const auto& t = buffer_[indices[i]];

        // Actions (flat)
        result.env_ids[i] = t.env_id;
        for (size_t j = 0; j < action_flat_size && j < t.action.size(); ++j) {
            result.actions[i * action_flat_size + j] = t.action[j];
        }
        // Portfolio state
        for (int j = 0; j < n_tickers; ++j) {
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

    // Batch fetch history market data - need temp double buffers since fill_market_batch uses double
    if (h > 0) {
        std::vector<double> temp_ohlcv(actual_batch * time_len * 5);
        std::vector<double> temp_ind(actual_batch * time_len * n_ind);

        for (const auto& ticker : result.unique_tickers) {
            if (!ticker_samples[ticker].empty()) {
                // State
                env_->fill_market_batch(
                    ticker_samples[ticker], h,
                    temp_ohlcv.data(), temp_ind.data(),
                    result.s_mask[ticker].data()
                );
                for (size_t k = 0; k < temp_ohlcv.size(); ++k)
                    result.s_ohlcv[ticker][k] = static_cast<float>(temp_ohlcv[k]);
                for (size_t k = 0; k < temp_ind.size(); ++k)
                    result.s_indicators[ticker][k] = static_cast<float>(temp_ind[k]);

                // Next state
                env_->fill_market_batch(
                    ticker_next_samples[ticker], h,
                    temp_ohlcv.data(), temp_ind.data(),
                    result.s_next_mask[ticker].data()
                );
                for (size_t k = 0; k < temp_ohlcv.size(); ++k)
                    result.s_next_ohlcv[ticker][k] = static_cast<float>(temp_ohlcv[k]);
                for (size_t k = 0; k < temp_ind.size(); ++k)
                    result.s_next_indicators[ticker][k] = static_cast<float>(temp_ind[k]);
            }
        }

        // Macro tickers
        for (const auto& ticker : cached_macro_tickers_) {
            env_->fill_market_batch(
                macro_samples[ticker], h,
                temp_ohlcv.data(), temp_ind.data(),
                result.macro_mask[ticker].data()
            );
            for (size_t k = 0; k < temp_ohlcv.size(); ++k)
                result.macro_ohlcv[ticker][k] = static_cast<float>(temp_ohlcv[k]);
            for (size_t k = 0; k < temp_ind.size(); ++k)
                result.macro_indicators[ticker][k] = static_cast<float>(temp_ind[k]);

            env_->fill_market_batch(
                macro_next_samples[ticker], h,
                temp_ohlcv.data(), temp_ind.data(),
                result.macro_next_mask[ticker].data()
            );
            for (size_t k = 0; k < temp_ohlcv.size(); ++k)
                result.macro_next_ohlcv[ticker][k] = static_cast<float>(temp_ohlcv[k]);
            for (size_t k = 0; k < temp_ind.size(); ++k)
                result.macro_next_indicators[ticker][k] = static_cast<float>(temp_ind[k]);
        }
    }

    // Future market data - use get_market_window_raw per sample (no batch version for future)
    // When h=0, future=N: data is in ohlcv/indicators/mask fields
    if (future_length > 0) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, actual_batch),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const auto& t = buffer_[indices[i]];

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
            });
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
