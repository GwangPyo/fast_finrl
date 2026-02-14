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
    int len = slice_end - slice_start;

    for (int t = 0; t < len; ++t) {
        int src_idx = slice_start + t;
        for (int k = 0; k < 5; ++k) {
            result_ohlcv(batch_idx, ticker_idx, t, k) = static_cast<float>(raw.ohlcv[src_idx * 5 + k]);
        }
        result_mask(batch_idx, ticker_idx, t) = raw.mask[src_idx];

        if (n_ind > 0) {
            for (size_t ind = 0; ind < n_ind; ++ind) {
                result_ind(batch_idx, ticker_idx, t, ind) = static_cast<float>(raw.indicators[src_idx * n_ind + ind]);
            }
        }
    }
}
} // anonymous namespace

VecReplayBuffer::VecReplayBuffer(std::shared_ptr<const FastFinRL> env, size_t capacity, size_t batch_size, int64_t seed, std::vector<size_t> action_shape)
    : env_(std::move(env))
    , capacity_(capacity)
    , batch_size_(batch_size)
    , action_shape_(std::move(action_shape))
    , rng_(seed >= 0 ? static_cast<unsigned int>(seed) : std::random_device{}())
{
    buffer_.reserve(std::min(capacity_, size_t(1000000)));
    std::mt19937 gen(seed); // fix seed

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
    return sample_indices(batch_size, 0);
}

std::vector<size_t> VecReplayBuffer::sample_indices(size_t batch_size, int history_length) const {
    (void)history_length;  // env already validated data availability

    size_t current_size = size();
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
    auto indices = sample_indices(batch_size, history_length);
    const size_t B = indices.size();

    SampleBatch result;
    result.batch_size = static_cast<int>(B);
    result.history_length = history_length;
    result.future_length = future_length;
    result.action_shape = action_shape_;

    if (B == 0) return result;

    // Collect per-sample tickers
    result.tickers.resize(B);
    for (size_t i = 0; i < B; ++i) {
        result.tickers[i] = buffer_[indices[i]].tickers;
    }

    // Get structure from first transition
    const auto& first_t = buffer_[indices[0]];
    const int n_tic = static_cast<int>(first_t.tickers.size());
    const int n_macro = n_macro_tickers_;
    const int n_ind = n_indicators_;
    const int h = history_length;
    const int f = future_length;

    result.n_tickers = n_tic;
    result.n_objectives = static_cast<int>(first_t.rewards.size());
    result.indicator_names = cached_indicator_names_;
    result.n_indicators = n_ind;
    result.macro_tickers = cached_macro_tickers_;
    result.n_macro_tickers = n_macro;

    // Compute action shape
    std::vector<size_t> action_dims = {B};
    for (size_t dim : action_shape_) action_dims.push_back(dim);

    // Allocate xtensor arrays
    result.env_ids.resize(B);
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

    // Compute expected action flat size
    size_t action_flat_size = 1;
    for (size_t dim : action_shape_) action_flat_size *= dim;

    // Fill data per sample in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, B), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            const VecStoredTransition& t = buffer_[indices[i]];

            // Copy basic fields
            result.env_ids[i] = t.env_id;
            result.rewards[i] = t.rewards;
            result.dones[i] = t.done;
            result.state_cash[i] = t.state_cash;
            result.next_state_cash[i] = t.next_state_cash;

            // Actions [B, action_shape...]
            auto action_view = xt::view(result.actions, i, xt::all());
            std::copy(t.action.begin(), t.action.end(), action_view.begin());

            // Portfolio state [B, n_tickers]
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
                    const std::string& tic = cached_macro_tickers_[m];
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
                    const std::string& tic = cached_macro_tickers_[m];
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

size_t VecReplayBuffer::size() const {
    return full_ ? capacity_ : buffer_.size();
}

void VecReplayBuffer::clear() {
    buffer_.clear();
    write_idx_ = 0;
    full_ = false;
}

namespace {
    template<typename T>
    void write_pod(std::ostream& f, const T& val) {
        f.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    template<typename T>
    void read_pod(std::istream& f, T& val) {
        f.read(reinterpret_cast<char*>(&val), sizeof(T));
    }

    template<typename T>
    void write_vector(std::ostream& f, const std::vector<T>& vec) {
        size_t sz = vec.size();
        write_pod(f, sz);
        if (sz > 0) {
            f.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(T));
        }
    }

    template<typename T>
    void read_vector(std::istream& f, std::vector<T>& vec) {
        size_t sz;
        read_pod(f, sz);
        vec.resize(sz);
        if (sz > 0) {
            f.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(T));
        }
    }

    void write_string(std::ostream& f, const std::string& s) {
        size_t sz = s.size();
        write_pod(f, sz);
        if (sz > 0) {
            f.write(s.data(), sz);
        }
    }

    void read_string(std::istream& f, std::string& s) {
        size_t sz;
        read_pod(f, sz);
        s.resize(sz);
        if (sz > 0) {
            f.read(&s[0], sz);
        }
    }

    void write_string_vector(std::ostream& f, const std::vector<std::string>& vec) {
        size_t sz = vec.size();
        write_pod(f, sz);
        for (const auto& s : vec) {
            write_string(f, s);
        }
    }

    void read_string_vector(std::istream& f, std::vector<std::string>& vec) {
        size_t sz;
        read_pod(f, sz);
        vec.resize(sz);
        for (auto& s : vec) {
            read_string(f, s);
        }
    }
}

void VecReplayBuffer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Header
    write_pod(f, capacity_);
    write_pod(f, write_idx_);
    write_pod(f, full_);

    // Number of transitions
    size_t n_transitions = buffer_.size();
    write_pod(f, n_transitions);

    // Transitions
    for (const auto& t : buffer_) {
        write_pod(f, t.env_id);
        write_pod(f, t.state_day);
        write_string_vector(f, t.tickers);
        write_pod(f, t.state_cash);
        write_vector(f, t.state_shares);
        write_vector(f, t.state_avg_buy_price);
        write_vector(f, t.action);
        write_vector(f, t.rewards);
        write_pod(f, t.done);
        write_pod(f, t.terminal);
        write_pod(f, t.next_state_day);
        write_pod(f, t.next_state_cash);
        write_vector(f, t.next_state_shares);
        write_vector(f, t.next_state_avg_buy_price);
    }
}

void VecReplayBuffer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    // Header
    read_pod(f, capacity_);
    read_pod(f, write_idx_);
    read_pod(f, full_);

    // Number of transitions
    size_t n_transitions;
    read_pod(f, n_transitions);

    buffer_.clear();
    buffer_.reserve(n_transitions);

    // Transitions
    for (size_t i = 0; i < n_transitions; ++i) {
        VecStoredTransition t;
        read_pod(f, t.env_id);
        read_pod(f, t.state_day);
        read_string_vector(f, t.tickers);
        read_pod(f, t.state_cash);
        read_vector(f, t.state_shares);
        read_vector(f, t.state_avg_buy_price);
        read_vector(f, t.action);
        read_vector(f, t.rewards);
        read_pod(f, t.done);
        read_pod(f, t.terminal);
        read_pod(f, t.next_state_day);
        read_pod(f, t.next_state_cash);
        read_vector(f, t.next_state_shares);
        read_vector(f, t.next_state_avg_buy_price);
        buffer_.push_back(std::move(t));
    }
}

} // namespace fast_finrl
