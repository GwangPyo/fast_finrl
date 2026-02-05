#pragma once

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

namespace fast_finrl {

// State data structures for serialization (no friend class needed)
struct TickerMarketData {
    double open;
    double high;
    double low;
    double close;
    std::map<std::string, double> indicators;
};

struct TickerHolding {
    int shares;
    double avg_buy_price;
};

struct TradeDebugInfo {
    double fill_price;
    double cost;
    int quantity;
};

struct PortfolioState {
    double cash;
    double total_asset;
    std::map<std::string, TickerHolding> holdings;
};

struct MarketState {
    std::map<std::string, TickerMarketData> tickers;
};

struct EpisodeInfo {
    double loss_cut_amount;
    int n_trades;
    int num_stop_loss;
};

struct StateData {
    int day;
    std::string date;
    int64_t seed;
    bool done;
    bool terminal;
    double reward;
    PortfolioState portfolio;
    MarketState market;
    EpisodeInfo info;
    std::map<std::string, TradeDebugInfo> debug;
};

class IStateSerializer {
public:
    virtual ~IStateSerializer() = default;
    virtual nlohmann::json serialize(const StateData& state, bool include_step_info) const = 0;
};

class JsonStateSerializer : public IStateSerializer {
public:
    nlohmann::json serialize(const StateData& state, bool include_step_info) const override;
};

} // namespace fast_finrl
