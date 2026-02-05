#include "StateSerializer.hpp"

namespace fast_finrl {

using namespace std;

nlohmann::json JsonStateSerializer::serialize(const StateData& state, bool include_step_info) const {
    nlohmann::json result;

    // Basic state info
    result["day"] = state.day;
    result["day_idx"] = state.day;
    result["date"] = state.date;
    result["done"] = state.done;
    result["terminal"] = state.terminal;

    // Seed (only in reset, not in step)
    if (!include_step_info) {
        result["seed"] = state.seed;
    }

    // Reward (only in step)
    if (include_step_info) {
        result["reward"] = state.reward;
    }

    // Portfolio
    nlohmann::json portfolio;
    portfolio["cash"] = state.portfolio.cash;
    if (include_step_info) {
        portfolio["total_asset"] = state.portfolio.total_asset;
    }

    nlohmann::json holdings;
    for (const auto& [ticker, holding] : state.portfolio.holdings) {
        holdings[ticker] = {
            {"shares", holding.shares},
            {"avg_buy_price", holding.avg_buy_price}
        };
    }
    portfolio["holdings"] = move(holdings);
    result["portfolio"] = move(portfolio);

    // Market
    nlohmann::json market;
    for (const auto& [ticker, data] : state.market.tickers) {
        nlohmann::json ticker_data;
        ticker_data["open"] = data.open;
        ticker_data["high"] = data.high;
        ticker_data["low"] = data.low;
        ticker_data["close"] = data.close;

        nlohmann::json indicators;
        for (const auto& [name, value] : data.indicators) {
            indicators[name] = value;
        }
        ticker_data["indicators"] = move(indicators);
        market[ticker] = move(ticker_data);
    }
    result["market"] = move(market);

    // Macro tickers
    nlohmann::json macro;
    for (const auto& [ticker, data] : state.macro.tickers) {
        nlohmann::json ticker_data;
        ticker_data["open"] = data.open;
        ticker_data["high"] = data.high;
        ticker_data["low"] = data.low;
        ticker_data["close"] = data.close;

        nlohmann::json indicators;
        for (const auto& [name, value] : data.indicators) {
            indicators[name] = value;
        }
        ticker_data["indicators"] = move(indicators);
        macro[ticker] = move(ticker_data);
    }
    result["macro"] = move(macro);

    // Info and debug (only in step)
    if (include_step_info) {
        nlohmann::json info;
        info["loss_cut_amount"] = state.info.loss_cut_amount;
        info["n_trades"] = state.info.n_trades;
        info["num_stop_loss"] = state.info.num_stop_loss;
        result["info"] = move(info);

        nlohmann::json debug;
        for (const auto& [ticker, trade] : state.debug) {
            if (trade.quantity != 0) {
                debug[ticker] = {
                    {"fill_price", trade.fill_price},
                    {"cost", trade.cost},
                    {"quantity", trade.quantity}
                };
            }
        }
        result["debug"] = move(debug);
    }

    return result;
}

} // namespace fast_finrl
