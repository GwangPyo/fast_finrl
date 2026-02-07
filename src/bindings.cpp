#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <optional>
#include "FastFinRL.hpp"
#include "ReplayBuffer.hpp"
#include "VecFastFinRL.hpp"
#include "VecReplayBuffer.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace py = pybind11;

// Helper to convert nlohmann::json to Python object
py::object json_to_python(const nlohmann::json& j) {
    if (j.is_null()) {
        return py::none();
    } else if (j.is_boolean()) {
        return py::bool_(j.get<bool>());
    } else if (j.is_number_integer()) {
        return py::int_(j.get<int64_t>());
    } else if (j.is_number_float()) {
        return py::float_(j.get<double>());
    } else if (j.is_string()) {
        return py::str(j.get<std::string>());
    } else if (j.is_array()) {
        py::list list;
        for (const auto& elem : j) {
            list.append(json_to_python(elem));
        }
        return list;
    } else if (j.is_object()) {
        py::dict dict;
        for (auto it = j.begin(); it != j.end(); ++it) {
            dict[py::str(it.key())] = json_to_python(it.value());
        }
        return dict;
    }
    return py::none();
}

// Helper to parse return_format string
inline fast_finrl::ReturnFormat parse_return_format(const std::string& fmt) {
    if (fmt == "vec" || fmt == "vector") return fast_finrl::ReturnFormat::Vec;
    return fast_finrl::ReturnFormat::Json;
}

inline std::string return_format_to_string(fast_finrl::ReturnFormat fmt) {
    return fmt == fast_finrl::ReturnFormat::Vec ? "vec" : "json";
}

PYBIND11_MODULE(fast_finrl_py, m) {
    m.doc() = "FastFinRL - High-performance C++ implementation of FinRL StockTradingEnv";

    py::class_<fast_finrl::FastFinRL, std::shared_ptr<fast_finrl::FastFinRL>>(m, "FastFinRL")
        // Constructor with keyword arguments only
        .def(py::init([](const std::string& csv_path,
                         double initial_amount,
                         int hmax,
                         double buy_cost_pct,
                         double sell_cost_pct,
                         double stop_loss_tolerance,
                         const std::string& bidding,
                         const std::string& stop_loss_calculation,
                         int64_t initial_seed,
                         const std::vector<std::string>& tech_indicator_list,
                         const std::vector<std::string>& macro_tickers,
                         const std::string& return_format,
                         int num_tickers,
                         bool shuffle_tickers) {
            fast_finrl::FastFinRLConfig config;
            config.initial_amount = initial_amount;
            config.hmax = hmax;
            config.buy_cost_pct = buy_cost_pct;
            config.sell_cost_pct = sell_cost_pct;
            config.stop_loss_tolerance = stop_loss_tolerance;
            config.bidding = bidding;
            config.stop_loss_calculation = stop_loss_calculation;
            config.initial_seed = initial_seed;
            config.tech_indicator_list = tech_indicator_list;
            config.macro_tickers = macro_tickers;
            config.return_format = parse_return_format(return_format);
            config.num_tickers = num_tickers;
            config.shuffle_tickers = shuffle_tickers;
            return std::make_unique<fast_finrl::FastFinRL>(csv_path, config);
        }),
             py::arg("csv_path"),
             py::arg("initial_amount") = 30000.0,
             py::arg("hmax") = 15,
             py::arg("buy_cost_pct") = 0.01,
             py::arg("sell_cost_pct") = 0.01,
             py::arg("stop_loss_tolerance") = 0.8,
             py::arg("bidding") = "default",
             py::arg("stop_loss_calculation") = "close",
             py::arg("initial_seed") = 0,
             py::arg("tech_indicator_list") = std::vector<std::string>{},
             py::arg("macro_tickers") = std::vector<std::string>{},
             py::arg("return_format") = "json",
             py::arg("num_tickers") = 0,
             py::arg("shuffle_tickers") = false,
             "Create FastFinRL environment with keyword arguments")

        // Public attributes (read-write)
        .def_readwrite("initial_amount", &fast_finrl::FastFinRL::initial_amount,
                       "Initial cash amount")
        .def_readwrite("hmax", &fast_finrl::FastFinRL::hmax,
                       "Maximum shares per trade")
        .def_readwrite("buy_cost_pct", &fast_finrl::FastFinRL::buy_cost_pct,
                       "Buy transaction cost percentage")
        .def_readwrite("sell_cost_pct", &fast_finrl::FastFinRL::sell_cost_pct,
                       "Sell transaction cost percentage")
        .def_readwrite("stop_loss_tolerance", &fast_finrl::FastFinRL::stop_loss_tolerance,
                       "Stop loss tolerance")
        .def_readwrite("bidding", &fast_finrl::FastFinRL::bidding,
                       "Bidding strategy: 'default', 'uniform', 'adv_uniform'")
        .def_readwrite("stop_loss_calculation", &fast_finrl::FastFinRL::stop_loss_calculation,
                       "Stop loss calculation method: 'close' or 'low'")

        // Return format control
        .def("set_return_format", [](fast_finrl::FastFinRL& self, const std::string& fmt) {
            self.return_format = parse_return_format(fmt);
        }, py::arg("format"), "Set return format: 'json' or 'vec'")
        .def("get_return_format", [](const fast_finrl::FastFinRL& self) {
            return return_format_to_string(self.return_format);
        }, "Get current return format")

        // Core API methods
        .def("reset", [](fast_finrl::FastFinRL& self,
                         const std::vector<std::string>& ticker_list,
                         int64_t seed,
                         int shifted_start) -> py::object {
            auto json_state = self.reset(ticker_list, seed, shifted_start);

            if (self.return_format == fast_finrl::ReturnFormat::Vec) {
                // Vec format: dict with numpy arrays
                py::dict state;
                state["day"] = json_state["day"].get<int>();
                state["date"] = json_state["date"].get<std::string>();
                state["seed"] = json_state["seed"].get<int64_t>();
                state["done"] = json_state["done"].get<bool>();
                state["terminal"] = json_state["terminal"].get<bool>();
                state["reward"] = json_state.value("reward", 0.0);

                // Portfolio
                auto& portfolio = json_state["portfolio"];
                state["cash"] = portfolio["cash"].get<double>();
                state["total_asset"] = portfolio.value("total_asset", 0.0);

                // Build arrays from holdings
                auto& holdings = portfolio["holdings"];
                int n_tic = static_cast<int>(ticker_list.size());

                py::array_t<int> shares(n_tic);
                py::array_t<double> avg_buy_price(n_tic);
                int* shares_ptr = shares.mutable_data();
                double* avg_ptr = avg_buy_price.mutable_data();

                for (int i = 0; i < n_tic; ++i) {
                    const auto& h = holdings[ticker_list[i]];
                    shares_ptr[i] = h["shares"].get<int>();
                    avg_ptr[i] = h["avg_buy_price"].get<double>();
                }
                state["shares"] = shares;
                state["avg_buy_price"] = avg_buy_price;

                // Market data (open only)
                auto& market = json_state["market"];
                auto indicator_names = self.get_indicator_names();
                int n_ind = static_cast<int>(indicator_names.size());

                py::array_t<double> open_arr({n_tic});
                py::array_t<double> indicators({n_tic, n_ind});
                double* open_ptr = open_arr.mutable_data();
                double* ind_ptr = indicators.mutable_data();

                for (int i = 0; i < n_tic; ++i) {
                    const auto& m = market[ticker_list[i]];
                    open_ptr[i] = m["open"].get<double>();

                    const auto& inds = m["indicators"];
                    int j = 0;
                    for (const auto& ind_name : indicator_names) {
                        ind_ptr[i * n_ind + j] = inds[ind_name].get<double>();
                        ++j;
                    }
                }
                state["open"] = open_arr;
                state["indicators"] = indicators;
                state["tickers"] = ticker_list;

                // Macro (open only)
                if (json_state.contains("macro") && !json_state["macro"].empty()) {
                    auto& macro = json_state["macro"];
                    auto macro_tickers = self.get_macro_tickers();
                    int n_macro = static_cast<int>(macro_tickers.size());

                    py::array_t<double> macro_open({n_macro});
                    py::array_t<double> macro_ind({n_macro, n_ind});
                    double* m_open_ptr = macro_open.mutable_data();
                    double* m_ind_ptr = macro_ind.mutable_data();

                    for (int i = 0; i < n_macro; ++i) {
                        const auto& m = macro[macro_tickers[i]];
                        m_open_ptr[i] = m["open"].get<double>();

                        const auto& inds = m["indicators"];
                        int j = 0;
                        for (const auto& ind_name : indicator_names) {
                            m_ind_ptr[i * n_ind + j] = inds[ind_name].get<double>();
                            ++j;
                        }
                    }
                    state["macro_open"] = macro_open;
                    state["macro_indicators"] = macro_ind;
                    state["macro_tickers"] = macro_tickers;
                }

                // Indicator names for reference
                py::list ind_names;
                for (const auto& name : indicator_names) {
                    ind_names.append(name);
                }
                state["indicator_names"] = ind_names;

                return state;
            }

            // Json format (default)
            return json_to_python(json_state);
        }, py::arg("ticker_list"), py::arg("seed"), py::arg("shifted_start") = 0,
           "Reset environment with given tickers, seed, and shifted_start. Returns state dict.")

        // No-arg reset: keep same tickers, increment seed
        .def("reset", [](fast_finrl::FastFinRL& self) -> py::object {
            auto json_state = self.reset();

            if (self.return_format == fast_finrl::ReturnFormat::Vec) {
                // Vec format conversion (same as above)
                py::dict state;
                state["day"] = json_state["day"].get<int>();
                state["date"] = json_state["date"].get<std::string>();
                state["seed"] = json_state["seed"].get<int64_t>();
                state["done"] = json_state["done"].get<bool>();
                state["terminal"] = json_state["terminal"].get<bool>();
                state["reward"] = json_state.value("reward", 0.0);

                auto& portfolio = json_state["portfolio"];
                state["cash"] = portfolio["cash"].get<double>();
                state["total_asset"] = portfolio.value("total_asset", 0.0);

                // Get ticker list from market keys
                auto& market = json_state["market"];
                std::vector<std::string> ticker_list;
                for (auto it = market.begin(); it != market.end(); ++it) {
                    ticker_list.push_back(it.key());
                }
                std::sort(ticker_list.begin(), ticker_list.end());

                auto& holdings = portfolio["holdings"];
                int n_tic = static_cast<int>(ticker_list.size());

                py::array_t<int> shares(n_tic);
                py::array_t<double> avg_buy_price(n_tic);
                int* shares_ptr = shares.mutable_data();
                double* avg_ptr = avg_buy_price.mutable_data();

                for (int i = 0; i < n_tic; ++i) {
                    const auto& h = holdings[ticker_list[i]];
                    shares_ptr[i] = h["shares"].get<int>();
                    avg_ptr[i] = h["avg_buy_price"].get<double>();
                }
                state["shares"] = shares;
                state["avg_buy_price"] = avg_buy_price;

                auto indicator_names = self.get_indicator_names();
                int n_ind = static_cast<int>(indicator_names.size());

                py::array_t<double> open_arr({n_tic});
                py::array_t<double> indicators({n_tic, n_ind});
                double* open_ptr = open_arr.mutable_data();
                double* ind_ptr = indicators.mutable_data();

                for (int i = 0; i < n_tic; ++i) {
                    const auto& m = market[ticker_list[i]];
                    open_ptr[i] = m["open"].get<double>();

                    const auto& inds = m["indicators"];
                    int j = 0;
                    for (const auto& ind_name : indicator_names) {
                        ind_ptr[i * n_ind + j] = inds[ind_name].get<double>();
                        ++j;
                    }
                }
                state["open"] = open_arr;
                state["indicators"] = indicators;
                state["tickers"] = ticker_list;

                py::list ind_names;
                for (const auto& name : indicator_names) {
                    ind_names.append(name);
                }
                state["indicator_names"] = ind_names;

                return state;
            }

            return json_to_python(json_state);
        }, "Reset with no args: keep same tickers, increment seed.")

        .def("step", [](fast_finrl::FastFinRL& self,
                        const std::vector<double>& actions) -> py::object {
            auto json_state = self.step(actions);

            if (self.return_format == fast_finrl::ReturnFormat::Vec) {
                // Vec format: dict with numpy arrays
                py::dict state;
                state["day"] = json_state["day"].get<int>();
                state["date"] = json_state["date"].get<std::string>();
                state["done"] = json_state["done"].get<bool>();
                state["terminal"] = json_state["terminal"].get<bool>();
                state["reward"] = json_state["reward"].get<double>();

                // Portfolio
                auto& portfolio = json_state["portfolio"];
                state["cash"] = portfolio["cash"].get<double>();
                state["total_asset"] = portfolio["total_asset"].get<double>();

                // Get tickers from market keys
                auto& market = json_state["market"];
                std::vector<std::string> ticker_list;
                for (auto it = market.begin(); it != market.end(); ++it) {
                    ticker_list.push_back(it.key());
                }
                std::sort(ticker_list.begin(), ticker_list.end());  // Consistent order

                auto& holdings = portfolio["holdings"];
                int n_tic = static_cast<int>(ticker_list.size());

                py::array_t<int> shares(n_tic);
                py::array_t<double> avg_buy_price(n_tic);
                int* shares_ptr = shares.mutable_data();
                double* avg_ptr = avg_buy_price.mutable_data();

                for (int i = 0; i < n_tic; ++i) {
                    const auto& h = holdings[ticker_list[i]];
                    shares_ptr[i] = h["shares"].get<int>();
                    avg_ptr[i] = h["avg_buy_price"].get<double>();
                }
                state["shares"] = shares;
                state["avg_buy_price"] = avg_buy_price;

                // Market data (open only - no HLC to prevent data leak)
                auto indicator_names = self.get_indicator_names();
                int n_ind = static_cast<int>(indicator_names.size());

                py::array_t<double> open_arr({n_tic});
                py::array_t<double> indicators({n_tic, n_ind});
                double* open_ptr = open_arr.mutable_data();
                double* ind_ptr = indicators.mutable_data();

                for (int i = 0; i < n_tic; ++i) {
                    const auto& m = market[ticker_list[i]];
                    open_ptr[i] = m["open"].get<double>();

                    const auto& inds = m["indicators"];
                    int j = 0;
                    for (const auto& ind_name : indicator_names) {
                        ind_ptr[i * n_ind + j] = inds[ind_name].get<double>();
                        ++j;
                    }
                }
                state["open"] = open_arr;
                state["indicators"] = indicators;
                state["tickers"] = ticker_list;

                // Macro (open only)
                if (json_state.contains("macro") && !json_state["macro"].empty()) {
                    auto& macro = json_state["macro"];
                    auto macro_tickers = self.get_macro_tickers();
                    int n_macro = static_cast<int>(macro_tickers.size());

                    py::array_t<double> macro_open({n_macro});
                    py::array_t<double> macro_ind({n_macro, n_ind});
                    double* m_open_ptr = macro_open.mutable_data();
                    double* m_ind_ptr = macro_ind.mutable_data();

                    for (int i = 0; i < n_macro; ++i) {
                        const auto& m = macro[macro_tickers[i]];
                        m_open_ptr[i] = m["open"].get<double>();

                        const auto& inds = m["indicators"];
                        int j = 0;
                        for (const auto& ind_name : indicator_names) {
                            m_ind_ptr[i * n_ind + j] = inds[ind_name].get<double>();
                            ++j;
                        }
                    }
                    state["macro_open"] = macro_open;
                    state["macro_indicators"] = macro_ind;
                    state["macro_tickers"] = macro_tickers;
                }

                // Indicator names
                py::list ind_names;
                for (const auto& name : indicator_names) {
                    ind_names.append(name);
                }
                state["indicator_names"] = ind_names;

                return state;
            }

            // Json format (default)
            return json_to_python(json_state);
        }, py::arg("actions"),
           "Execute one step with given actions. Returns state dict.")

        // Accessor methods
        .def("get_indicator_names", &fast_finrl::FastFinRL::get_indicator_names,
             "Get set of technical indicator column names")

        .def("get_all_tickers", &fast_finrl::FastFinRL::get_all_tickers,
             "Get set of all available ticker symbols")

        .def("get_macro_tickers", &fast_finrl::FastFinRL::get_macro_tickers,
             "Get list of macro ticker symbols")

        .def("get_max_day", &fast_finrl::FastFinRL::get_max_day,
             "Get maximum day index")

        .def("get_state", [](const fast_finrl::FastFinRL& self) {
            return json_to_python(self.get_state());
        }, "Get current state as dict")

        .def("get_raw_value", &fast_finrl::FastFinRL::get_raw_value,
             py::arg("ticker"), py::arg("day"), py::arg("column"),
             "Get raw value from DataFrame for given ticker, day, and column")

        .def("get_market_window", [](const fast_finrl::FastFinRL& self,
                                     const std::string& ticker,
                                     int day,
                                     int h,
                                     int future) {
            return json_to_python(self.get_market_window(ticker, day, h, future));
        }, py::arg("ticker"), py::arg("day"), py::arg("h"), py::arg("future"),
           "Get market data window: past [day-h, day-1], current day, future [day+1, day+future]")

        .def("get_market_window_flat", [](const fast_finrl::FastFinRL& self,
                                          const std::string& ticker,
                                          int day,
                                          int h,
                                          int future) {
            return json_to_python(self.get_market_window_flat(ticker, day, h, future));
        }, py::arg("ticker"), py::arg("day"), py::arg("h"), py::arg("future"),
           "Fast version - returns flat arrays: ohlc[n,4], indicators[n,num_ind], mask[n], days[n]")

        .def("get_market_window_numpy", [](const fast_finrl::FastFinRL& self,
                                           const std::vector<std::string>& ticker_list,
                                           int day,
                                           int h,
                                           int future) -> py::dict {
            // Zero-copy: Store data in shared_ptr, use capsule to prevent deletion
            using DataHolder = fast_finrl::FastFinRL::MultiTickerWindowData;
            std::shared_ptr<DataHolder> holder = std::make_shared<DataHolder>(
                self.get_market_window_multi(ticker_list, day, h, future)
            );

            int n_ind = holder->n_indicators;
            py::dict result;

            // Process each ticker - zero-copy views into holder's memory
            for (auto& [ticker, td] : holder->tickers) {
                py::dict ticker_dict;

                // Capsule prevents holder deletion while arrays exist
                auto make_capsule = [holder]() {
                    return py::capsule(new std::shared_ptr<DataHolder>(holder),
                        [](void* p) { delete static_cast<std::shared_ptr<DataHolder>*>(p); });
                };

                // Past arrays - zero-copy views (OHLCV)
                ticker_dict["past_ohlcv"] = py::array_t<double>(
                    {h, 5}, {5 * sizeof(double), sizeof(double)},
                    td.past_ohlcv.data(), make_capsule());

                ticker_dict["past_indicators"] = py::array_t<double>(
                    {h, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.past_indicators.data(), make_capsule());

                ticker_dict["past_mask"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_mask.data(), make_capsule());

                ticker_dict["past_days"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_days.data(), make_capsule());

                // Current: open only (scalar)
                ticker_dict["current_open"] = td.current_open;

                ticker_dict["current_indicators"] = py::array_t<double>(
                    {n_ind}, {sizeof(double)},
                    td.current_indicators.data(), make_capsule());

                ticker_dict["current_mask"] = td.current_mask;
                ticker_dict["current_day"] = td.current_day;

                // Future arrays - zero-copy views (OHLCV)
                ticker_dict["future_ohlcv"] = py::array_t<double>(
                    {future, 5}, {5 * sizeof(double), sizeof(double)},
                    td.future_ohlcv.data(), make_capsule());

                ticker_dict["future_indicators"] = py::array_t<double>(
                    {future, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.future_indicators.data(), make_capsule());

                ticker_dict["future_mask"] = py::array_t<int>(
                    {future}, {sizeof(int)},
                    td.future_mask.data(), make_capsule());

                ticker_dict["future_days"] = py::array_t<int>(
                    {future}, {sizeof(int)},
                    td.future_days.data(), make_capsule());

                result[py::str(ticker)] = ticker_dict;
            }

            // Metadata
            py::list names;
            for (const std::string& name : holder->indicator_names) {
                names.append(py::str(name));
            }
            result["indicator_names"] = names;
            result["h"] = h;
            result["future"] = future;

            return result;
        }, py::arg("ticker_list"), py::arg("day"), py::arg("h"), py::arg("future"),
           "Multi-ticker market window with separated past/current/future numpy arrays (zero-copy)");

    // StoredTransition binding
    py::class_<fast_finrl::StoredTransition>(m, "StoredTransition")
        .def(py::init<>())
        .def_readwrite("state_day", &fast_finrl::StoredTransition::state_day)
        .def_readwrite("tickers", &fast_finrl::StoredTransition::tickers)
        .def_readwrite("state_cash", &fast_finrl::StoredTransition::state_cash)
        .def_readwrite("state_shares", &fast_finrl::StoredTransition::state_shares)
        .def_readwrite("state_avg_buy_price", &fast_finrl::StoredTransition::state_avg_buy_price)
        .def_readwrite("action", &fast_finrl::StoredTransition::action)
        .def_readwrite("rewards", &fast_finrl::StoredTransition::rewards)
        .def_readwrite("done", &fast_finrl::StoredTransition::done)
        .def_readwrite("terminal", &fast_finrl::StoredTransition::terminal)
        .def_readwrite("next_state_day", &fast_finrl::StoredTransition::next_state_day)
        .def_readwrite("next_state_cash", &fast_finrl::StoredTransition::next_state_cash)
        .def_readwrite("next_state_shares", &fast_finrl::StoredTransition::next_state_shares)
        .def_readwrite("next_state_avg_buy_price", &fast_finrl::StoredTransition::next_state_avg_buy_price);

    // ReplayBuffer binding
    py::class_<fast_finrl::ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init([](std::shared_ptr<fast_finrl::FastFinRL> env, size_t capacity, size_t batch_size, int64_t seed) {
            return std::make_unique<fast_finrl::ReplayBuffer>(
                std::const_pointer_cast<const fast_finrl::FastFinRL>(env), capacity, batch_size, seed);
        }), py::arg("env"), py::arg("capacity") = 1000000, py::arg("batch_size") = 256, py::arg("seed") = 42,
           "Create ReplayBuffer. capacity: 100K (small), 1M (default), 5M (large). seed: 42 default for reproducibility")
        .def("add", [](fast_finrl::ReplayBuffer& self,
                       const py::dict& state,
                       const std::vector<double>& action,
                       py::object reward,
                       const py::dict& next_state,
                       bool done) {
            // Extract from state dicts
            int state_day = state["day"].cast<int>();
            int next_state_day = next_state["day"].cast<int>();

            py::dict portfolio = state["portfolio"].cast<py::dict>();
            py::dict next_portfolio = next_state["portfolio"].cast<py::dict>();
            py::dict holdings = portfolio["holdings"].cast<py::dict>();
            py::dict next_holdings = next_portfolio["holdings"].cast<py::dict>();

            double state_cash = portfolio["cash"].cast<double>();
            double next_state_cash = next_portfolio["cash"].cast<double>();

            std::vector<std::string> tickers;
            std::vector<int> state_shares, next_state_shares;
            std::vector<double> state_avg, next_state_avg;

            for (auto item : holdings) {
                std::string tic = item.first.cast<std::string>();
                tickers.push_back(tic);
                py::dict h = item.second.cast<py::dict>();
                state_shares.push_back(h["shares"].cast<int>());
                state_avg.push_back(h["avg_buy_price"].cast<double>());
            }

            for (const auto& tic : tickers) {
                py::dict h = next_holdings[py::str(tic)].cast<py::dict>();
                next_state_shares.push_back(h["shares"].cast<int>());
                next_state_avg.push_back(h["avg_buy_price"].cast<double>());
            }

            // Handle scalar or multi-objective rewards
            std::vector<double> rewards;
            if (py::isinstance<py::list>(reward) || py::isinstance<py::array>(reward)) {
                rewards = reward.cast<std::vector<double>>();
            } else {
                rewards.push_back(reward.cast<double>());
            }

            self.add_transition(state_day, next_state_day, tickers,
                               state_cash, next_state_cash,
                               state_shares, next_state_shares,
                               state_avg, next_state_avg,
                               action, rewards, done, false);
        }, py::arg("state"), py::arg("action"), py::arg("reward"), py::arg("next_state"), py::arg("done"),
           "Add transition: (state, action, reward, next_state, done)")
        .def("sample_indices", &fast_finrl::ReplayBuffer::sample_indices, py::arg("batch_size"),
             "Sample random indices from buffer")
        .def("sample", [](const fast_finrl::ReplayBuffer& self, int h, std::optional<size_t> batch_size) -> py::tuple {
            // Returns: (s, a, r, s', s_mask, s'_mask)
            // s/s': dict[ticker] -> ohlcv [batch, h, 5], indicators [batch, h, n_ind] (history only, no current)
            // s_mask/s'_mask: dict[ticker] -> [batch, h] or None if h=0
            // a: [batch, n_tickers]
            // r: [batch]

            using Batch = fast_finrl::ReplayBuffer::SampleBatch;
            std::shared_ptr<Batch> holder = std::make_shared<Batch>(
                batch_size ? self.sample(h, *batch_size) : self.sample(h)
            );

            const int B = holder->batch_size;
            const int T = h;  // history only (no current day)
            const int n_ind = holder->n_indicators;
            const int n_tic = holder->n_tickers;

            auto make_capsule = [holder]() {
                return py::capsule(new std::shared_ptr<Batch>(holder),
                    [](void* p) { delete static_cast<std::shared_ptr<Batch>*>(p); });
            };

            // Build s dict
            py::dict s_dict, s_next_dict;
            py::object s_mask_dict = py::none();
            py::object s_next_mask_dict = py::none();

            if (h > 0) {
                s_mask_dict = py::dict();
                s_next_mask_dict = py::dict();
            }

            for (const auto& ticker : holder->tickers) {
                py::dict td, td_next;

                // OHLCV [B, T, 5]
                td["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->s_ohlcv[ticker].data(), make_capsule());

                td["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->s_indicators[ticker].data(), make_capsule());

                td_next["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->s_next_ohlcv[ticker].data(), make_capsule());

                td_next["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->s_next_indicators[ticker].data(), make_capsule());

                s_dict[py::str(ticker)] = td;
                s_next_dict[py::str(ticker)] = td_next;

                if (h > 0) {
                    py::cast<py::dict>(s_mask_dict)[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->s_mask[ticker].data(), make_capsule());

                    py::cast<py::dict>(s_next_mask_dict)[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->s_next_mask[ticker].data(), make_capsule());
                }
            }

            // Macro tickers - build s["macro"] and s_next["macro"]
            py::dict macro_dict, macro_next_dict;
            py::dict macro_mask_dict, macro_next_mask_dict;

            for (const std::string& ticker : holder->macro_tickers) {
                py::dict td, td_next;

                td["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->macro_ohlcv[ticker].data(), make_capsule());

                td["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->macro_indicators[ticker].data(), make_capsule());

                td_next["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->macro_next_ohlcv[ticker].data(), make_capsule());

                td_next["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->macro_next_indicators[ticker].data(), make_capsule());

                macro_dict[py::str(ticker)] = td;
                macro_next_dict[py::str(ticker)] = td_next;

                if (h > 0) {
                    macro_mask_dict[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->macro_mask[ticker].data(), make_capsule());

                    macro_next_mask_dict[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->macro_next_mask[ticker].data(), make_capsule());
                }
            }

            // Actions [B, n_tickers]
            py::array_t<double> actions({B, n_tic},
                {n_tic * sizeof(double), sizeof(double)},
                holder->actions.data(), make_capsule());

            // Rewards: always [B, reward_size]
            int n_obj = holder->n_objectives;
            py::array_t<double> rewards({B, n_obj});
            double* r_ptr = rewards.mutable_data();
            for (int i = 0; i < B; ++i) {
                for (int j = 0; j < n_obj; ++j) {
                    r_ptr[i * n_obj + j] = holder->rewards[i][j];
                }
            }

            // Dones [B, 1] - need to copy since vector<bool> is special
            py::array_t<bool> dones({B, 1});
            bool* dones_ptr = dones.mutable_data();
            for (int i = 0; i < B; ++i) {
                dones_ptr[i] = holder->dones[i];
            }

            // Portfolio state
            py::array_t<double> state_cash({B}, {sizeof(double)},
                holder->state_cash.data(), make_capsule());
            py::array_t<double> next_state_cash({B}, {sizeof(double)},
                holder->next_state_cash.data(), make_capsule());
            py::array_t<int> state_shares({B, n_tic}, {n_tic * sizeof(int), sizeof(int)},
                holder->state_shares.data(), make_capsule());
            py::array_t<int> next_state_shares({B, n_tic}, {n_tic * sizeof(int), sizeof(int)},
                holder->next_state_shares.data(), make_capsule());
            py::array_t<double> state_avg({B, n_tic}, {n_tic * sizeof(double), sizeof(double)},
                holder->state_avg_buy_price.data(), make_capsule());
            py::array_t<double> next_state_avg({B, n_tic}, {n_tic * sizeof(double), sizeof(double)},
                holder->next_state_avg_buy_price.data(), make_capsule());

            // Build portfolio dicts
            py::dict portfolio, next_portfolio;
            portfolio["cash"] = state_cash;
            portfolio["shares"] = state_shares;
            portfolio["avg_buy_price"] = state_avg;
            next_portfolio["cash"] = next_state_cash;
            next_portfolio["shares"] = next_state_shares;
            next_portfolio["avg_buy_price"] = next_state_avg;

            // Metadata
            s_dict["indicator_names"] = holder->indicator_names;
            s_dict["tickers"] = holder->tickers;
            s_dict["portfolio"] = portfolio;
            s_dict["macro"] = macro_dict;
            s_dict["macro_tickers"] = holder->macro_tickers;
            s_next_dict["indicator_names"] = holder->indicator_names;
            s_next_dict["tickers"] = holder->tickers;
            s_next_dict["portfolio"] = next_portfolio;
            s_next_dict["macro"] = macro_next_dict;
            s_next_dict["macro_tickers"] = holder->macro_tickers;

            // Add macro mask to s_mask_dict
            if (h > 0 && holder->n_macro_tickers > 0) {
                py::cast<py::dict>(s_mask_dict)["macro"] = macro_mask_dict;
                py::cast<py::dict>(s_next_mask_dict)["macro"] = macro_next_mask_dict;
            }

            return py::make_tuple(s_dict, actions, rewards, s_next_dict, dones, s_mask_dict, s_next_mask_dict);
        }, py::arg("h"), py::arg("batch_size") = py::none(),
           "Sample batch: returns (s, a, r, s', done, s_mask, s'_mask). s/s' include portfolio. Mask is None if h=0")
        .def("get", &fast_finrl::ReplayBuffer::get, py::arg("index"),
             py::return_value_policy::reference_internal,
             "Get transition by index")
        .def("get_market_data", [](const fast_finrl::ReplayBuffer& self,
                                   size_t index, int h, int future, bool next_state) -> py::dict {
            using DataHolder = fast_finrl::FastFinRL::MultiTickerWindowData;
            std::shared_ptr<DataHolder> holder = std::make_shared<DataHolder>(
                self.get_market_data(index, h, future, next_state)
            );

            int n_ind = holder->n_indicators;
            py::dict result;

            for (auto& [ticker, td] : holder->tickers) {
                py::dict ticker_dict;

                auto make_capsule = [holder]() {
                    return py::capsule(new std::shared_ptr<DataHolder>(holder),
                        [](void* p) { delete static_cast<std::shared_ptr<DataHolder>*>(p); });
                };

                ticker_dict["past_ohlcv"] = py::array_t<double>(
                    {h, 5}, {5 * sizeof(double), sizeof(double)},
                    td.past_ohlcv.data(), make_capsule());

                ticker_dict["past_indicators"] = py::array_t<double>(
                    {h, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.past_indicators.data(), make_capsule());

                ticker_dict["past_mask"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_mask.data(), make_capsule());

                ticker_dict["past_days"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_days.data(), make_capsule());

                ticker_dict["current_open"] = td.current_open;

                ticker_dict["current_indicators"] = py::array_t<double>(
                    {n_ind}, {sizeof(double)},
                    td.current_indicators.data(), make_capsule());

                ticker_dict["current_mask"] = td.current_mask;
                ticker_dict["current_day"] = td.current_day;

                ticker_dict["future_ohlcv"] = py::array_t<double>(
                    {future, 5}, {5 * sizeof(double), sizeof(double)},
                    td.future_ohlcv.data(), make_capsule());

                ticker_dict["future_indicators"] = py::array_t<double>(
                    {future, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.future_indicators.data(), make_capsule());

                ticker_dict["future_mask"] = py::array_t<int>(
                    {future}, {sizeof(int)},
                    td.future_mask.data(), make_capsule());

                ticker_dict["future_days"] = py::array_t<int>(
                    {future}, {sizeof(int)},
                    td.future_days.data(), make_capsule());

                result[py::str(ticker)] = ticker_dict;
            }

            py::list names;
            for (const std::string& name : holder->indicator_names) {
                names.append(py::str(name));
            }
            result["indicator_names"] = names;
            result["h"] = h;
            result["future"] = future;

            return result;
        }, py::arg("index"), py::arg("h"), py::arg("future"), py::arg("next_state") = false,
           "Get market data for transition (zero-copy numpy arrays)")
        .def("size", &fast_finrl::ReplayBuffer::size, "Current buffer size")
        .def("capacity", &fast_finrl::ReplayBuffer::capacity, "Buffer capacity")
        .def("clear", &fast_finrl::ReplayBuffer::clear, "Clear buffer")
        .def("save", &fast_finrl::ReplayBuffer::save, py::arg("path"), "Save buffer to file")
        .def("load", &fast_finrl::ReplayBuffer::load, py::arg("path"), "Load buffer from file");

    // Helper: Convert VecFastFinRL::StepResult to list of dicts (json format)
    auto step_result_to_list = [](const fast_finrl::VecFastFinRL::StepResult& result,
                                   const std::vector<std::vector<std::string>>& tickers_list) -> py::list {
        const int N = result.num_envs;
        const int n_tic = result.n_tickers;
        const int n_ind = result.n_indicators;
        const int n_macro = result.n_macro;

        py::list states;
        for (int i = 0; i < N; ++i) {
            py::dict state;
            state["day"] = result.day[i];
            state["cash"] = result.cash[i];
            state["total_asset"] = result.total_asset[i];
            state["done"] = (result.done[i] != 0);
            state["terminal"] = (result.terminal[i] != 0);
            state["reward"] = result.reward[i];

            py::array_t<int> shares(n_tic);
            int* shares_ptr = shares.mutable_data();
            for (int t = 0; t < n_tic; ++t) {
                shares_ptr[t] = result.shares[i * n_tic + t];
            }
            state["shares"] = shares;

            py::array_t<double> avg_buy_price(n_tic);
            double* avg_ptr = avg_buy_price.mutable_data();
            for (int t = 0; t < n_tic; ++t) {
                avg_ptr[t] = result.avg_buy_price[i * n_tic + t];
            }
            state["avg_buy_price"] = avg_buy_price;

            py::array_t<double> open_arr({n_tic});
            double* open_ptr = open_arr.mutable_data();
            for (int t = 0; t < n_tic; ++t) {
                open_ptr[t] = result.open[i * n_tic + t];
            }
            state["open"] = open_arr;

            py::array_t<double> indicators({n_tic, n_ind});
            double* ind_ptr = indicators.mutable_data();
            for (int t = 0; t < n_tic; ++t) {
                for (int k = 0; k < n_ind; ++k) {
                    ind_ptr[t * n_ind + k] = result.indicators[(i * n_tic + t) * n_ind + k];
                }
            }
            state["indicators"] = indicators;

            state["tickers"] = tickers_list[i];

            if (n_macro > 0) {
                py::array_t<double> macro_open({n_macro});
                double* m_open_ptr = macro_open.mutable_data();
                for (int m = 0; m < n_macro; ++m) {
                    m_open_ptr[m] = result.macro_open[i * n_macro + m];
                }
                state["macro_open"] = macro_open;

                py::array_t<double> macro_ind({n_macro, n_ind});
                double* m_ind_ptr = macro_ind.mutable_data();
                for (int m = 0; m < n_macro; ++m) {
                    for (int k = 0; k < n_ind; ++k) {
                        m_ind_ptr[m * n_ind + k] = result.macro_indicators[(i * n_macro + m) * n_ind + k];
                    }
                }
                state["macro_indicators"] = macro_ind;
            }

            states.append(state);
        }
        return states;
    };

    // Helper: Convert VecFastFinRL::StepResult to batched dict (vec format)
    auto step_result_to_vec = [](const fast_finrl::VecFastFinRL::StepResult& result,
                                  const std::vector<std::vector<std::string>>& tickers_list) -> py::dict {
        const int N = result.num_envs;
        const int n_tic = result.n_tickers;
        const int n_ind = result.n_indicators;
        const int n_macro = result.n_macro;

        py::dict state;

        // Scalars: [N] - use explicit shape and strides
        std::vector<ssize_t> shape_1d = {N};
        std::vector<ssize_t> stride_int = {static_cast<ssize_t>(sizeof(int))};
        std::vector<ssize_t> stride_double = {static_cast<ssize_t>(sizeof(double))};
        std::vector<ssize_t> stride_bool = {static_cast<ssize_t>(sizeof(bool))};

        py::array_t<int> day(shape_1d, stride_int);
        py::array_t<double> cash(shape_1d, stride_double);
        py::array_t<double> total_asset(shape_1d, stride_double);
        py::array_t<bool> done(shape_1d, stride_bool);
        py::array_t<bool> terminal(shape_1d, stride_bool);
        py::array_t<double> reward(shape_1d, stride_double);

        std::memcpy(day.mutable_data(), result.day.data(), N * sizeof(int));
        std::memcpy(cash.mutable_data(), result.cash.data(), N * sizeof(double));
        std::memcpy(total_asset.mutable_data(), result.total_asset.data(), N * sizeof(double));
        std::memcpy(reward.mutable_data(), result.reward.data(), N * sizeof(double));

        // done/terminal: uint8_t -> bool
        bool* done_ptr = done.mutable_data();
        bool* terminal_ptr = terminal.mutable_data();
        for (int i = 0; i < N; ++i) {
            done_ptr[i] = (result.done[i] != 0);
            terminal_ptr[i] = (result.terminal[i] != 0);
        }

        state["day"] = day;
        state["cash"] = cash;
        state["total_asset"] = total_asset;
        state["done"] = done;
        state["terminal"] = terminal;
        state["reward"] = reward;

        // shares: [N, n_tickers]
        py::array_t<int> shares({N, n_tic});
        int* shares_ptr = shares.mutable_data();
        std::memcpy(shares_ptr, result.shares.data(), N * n_tic * sizeof(int));
        state["shares"] = shares;

        // avg_buy_price: [N, n_tickers]
        py::array_t<double> avg_buy_price({N, n_tic});
        double* avg_ptr = avg_buy_price.mutable_data();
        std::memcpy(avg_ptr, result.avg_buy_price.data(), N * n_tic * sizeof(double));
        state["avg_buy_price"] = avg_buy_price;

        // open: [N, n_tickers]
        py::array_t<double> open_arr({N, n_tic});
        double* open_ptr = open_arr.mutable_data();
        std::memcpy(open_ptr, result.open.data(), N * n_tic * sizeof(double));
        state["open"] = open_arr;

        // indicators: [N, n_tickers, n_ind]
        py::array_t<double> indicators({N, n_tic, n_ind});
        double* ind_ptr = indicators.mutable_data();
        std::memcpy(ind_ptr, result.indicators.data(), N * n_tic * n_ind * sizeof(double));
        state["indicators"] = indicators;

        // tickers: List[List[str]]
        state["tickers"] = tickers_list;

        // macro: [N, n_macro] and [N, n_macro, n_ind]
        if (n_macro > 0) {
            py::array_t<double> macro_open({N, n_macro});
            double* m_open_ptr = macro_open.mutable_data();
            std::memcpy(m_open_ptr, result.macro_open.data(), N * n_macro * sizeof(double));
            state["macro_open"] = macro_open;

            py::array_t<double> macro_ind({N, n_macro, n_ind});
            double* m_ind_ptr = macro_ind.mutable_data();
            std::memcpy(m_ind_ptr, result.macro_indicators.data(), N * n_macro * n_ind * sizeof(double));
            state["macro_indicators"] = macro_ind;
        }

        // Metadata
        state["n_envs"] = N;
        state["n_tickers"] = n_tic;
        state["n_indicators"] = n_ind;
        state["n_macro"] = n_macro;

        return state;
    };

    // VecFastFinRL - Vectorized environment for N parallel environments
    py::class_<fast_finrl::VecFastFinRL>(m, "VecFastFinRL")
        .def(py::init([](const std::string& csv_path,
                         int n_envs,
                         double initial_amount,
                         int hmax,
                         double buy_cost_pct,
                         double sell_cost_pct,
                         double stop_loss_tolerance,
                         const std::string& bidding,
                         const std::string& stop_loss_calculation,
                         int64_t initial_seed,
                         const std::vector<std::string>& tech_indicator_list,
                         const std::vector<std::string>& macro_tickers,
                         bool auto_reset,
                         const std::string& return_format,
                         int num_tickers,
                         bool shuffle_tickers) {
            fast_finrl::FastFinRLConfig config;
            config.initial_amount = initial_amount;
            config.hmax = hmax;
            config.buy_cost_pct = buy_cost_pct;
            config.sell_cost_pct = sell_cost_pct;
            config.stop_loss_tolerance = stop_loss_tolerance;
            config.bidding = bidding;
            config.stop_loss_calculation = stop_loss_calculation;
            config.initial_seed = initial_seed;
            config.tech_indicator_list = tech_indicator_list;
            config.macro_tickers = macro_tickers;
            config.return_format = parse_return_format(return_format);
            config.num_tickers = num_tickers;
            config.shuffle_tickers = shuffle_tickers;
            return std::make_unique<fast_finrl::VecFastFinRL>(csv_path, n_envs, config);
        }),
             py::arg("csv_path"),
             py::arg("n_envs"),
             py::arg("initial_amount") = 30000.0,
             py::arg("hmax") = 15,
             py::arg("buy_cost_pct") = 0.01,
             py::arg("sell_cost_pct") = 0.01,
             py::arg("stop_loss_tolerance") = 0.8,
             py::arg("bidding") = "default",
             py::arg("stop_loss_calculation") = "close",
             py::arg("initial_seed") = 0,
             py::arg("tech_indicator_list") = std::vector<std::string>{},
             py::arg("macro_tickers") = std::vector<std::string>{},
             py::arg("auto_reset") = true,
             py::arg("return_format") = "json",
             py::arg("num_tickers") = 0,
             py::arg("shuffle_tickers") = false,
             "Create VecFastFinRL - vectorized environment. n_envs: number of parallel environments (required)")

        // Return format control
        .def("set_return_format", [](fast_finrl::VecFastFinRL& self, const std::string& fmt) {
            self.set_return_format(parse_return_format(fmt));
        }, py::arg("format"), "Set return format: 'json' or 'vec'")
        .def("get_return_format", [](const fast_finrl::VecFastFinRL& self) {
            return return_format_to_string(self.return_format());
        }, "Get current return format")

        // reset -> returns list of dicts (json) or single dict (vec)
        // Full reset with explicit tickers and seeds (vector)
        .def("reset", [&step_result_to_list, &step_result_to_vec](
                         fast_finrl::VecFastFinRL& self,
                         const std::vector<std::vector<std::string>>& tickers_list,
                         py::array_t<int64_t> seeds_arr) -> py::object {
            auto seeds_buf = seeds_arr.request();
            int64_t* seeds_ptr = static_cast<int64_t*>(seeds_buf.ptr);
            std::vector<int64_t> seeds(seeds_ptr, seeds_ptr + seeds_buf.size);

            auto result = self.reset(tickers_list, seeds);
            const auto& tickers = self.get_tickers();

            if (self.return_format() == fast_finrl::ReturnFormat::Vec) {
                return step_result_to_vec(result, tickers);
            }
            return step_result_to_list(result, tickers);
        }, py::arg("tickers_list"), py::arg("seeds"),
           "Reset N environments with explicit tickers and seeds.")

        // Simplified reset: single seed, auto-expand to all envs
        .def("reset", [&step_result_to_list, &step_result_to_vec](
                         fast_finrl::VecFastFinRL& self,
                         std::optional<std::vector<std::vector<std::string>>> tickers_list_opt,
                         int64_t seed) -> py::object {
            std::vector<std::vector<std::string>> tickers_list;
            if (tickers_list_opt.has_value()) {
                tickers_list = tickers_list_opt.value();
            }

            auto result = self.reset(tickers_list, seed);
            const auto& tickers = self.get_tickers();

            if (self.return_format() == fast_finrl::ReturnFormat::Vec) {
                return step_result_to_vec(result, tickers);
            }
            return step_result_to_list(result, tickers);
        }, py::arg("tickers_list") = py::none(), py::arg("seed"),
           "Simplified reset: single seed auto-expands to all envs.")

        // No-arg reset: keep same tickers, auto-increment seeds
        .def("reset", [&step_result_to_list, &step_result_to_vec](
                         fast_finrl::VecFastFinRL& self) -> py::object {
            auto result = self.reset();
            const auto& tickers = self.get_tickers();

            if (self.return_format() == fast_finrl::ReturnFormat::Vec) {
                return step_result_to_vec(result, tickers);
            }
            return step_result_to_list(result, tickers);
        }, "Reset with no args: keep same tickers, increment seeds.")

        // step -> returns list of dicts (json) or single dict (vec)
        .def("step", [&step_result_to_list, &step_result_to_vec](
                        fast_finrl::VecFastFinRL& self,
                        py::array_t<double, py::array::c_style | py::array::forcecast> actions_arr) -> py::object {
            auto actions_buf = actions_arr.request();
            if (actions_buf.ndim != 2) {
                throw std::runtime_error("actions must be 2D array [N, n_tickers]");
            }

            int expected_n_envs = self.num_envs();
            int expected_n_tickers = self.n_tickers();

            if (actions_buf.shape[0] != expected_n_envs) {
                throw std::runtime_error(
                    "actions shape[0] mismatch: expected " + std::to_string(expected_n_envs) +
                    " (n_envs), got " + std::to_string(actions_buf.shape[0]));
            }
            if (actions_buf.shape[1] != expected_n_tickers) {
                throw std::runtime_error(
                    "actions shape[1] mismatch: expected " + std::to_string(expected_n_tickers) +
                    " (n_tickers), got " + std::to_string(actions_buf.shape[1]));
            }

            const double* actions_ptr = static_cast<const double*>(actions_buf.ptr);
            auto result = self.step(actions_ptr);
            const auto& tickers = self.get_tickers();

            if (self.return_format() == fast_finrl::ReturnFormat::Vec) {
                return step_result_to_vec(result, tickers);
            }
            return step_result_to_list(result, tickers);
        }, py::arg("actions"),
           "Execute one step. Returns List[dict] (json) or dict (vec) based on return_format.")

        // Partial reset - reset only specified indices
        .def("reset_indices", [&step_result_to_list, &step_result_to_vec](
                                 fast_finrl::VecFastFinRL& self,
                                 const std::vector<int>& indices,
                                 py::array_t<int64_t> seeds_arr) -> py::object {
            auto seeds_buf = seeds_arr.request();
            int64_t* seeds_ptr = static_cast<int64_t*>(seeds_buf.ptr);
            std::vector<int64_t> seeds(seeds_ptr, seeds_ptr + seeds_buf.size);

            auto result = self.reset_indices(indices, seeds);
            const auto& tickers = self.get_tickers();

            if (self.return_format() == fast_finrl::ReturnFormat::Vec) {
                return step_result_to_vec(result, tickers);
            }
            return step_result_to_list(result, tickers);
        }, py::arg("indices"), py::arg("seeds"),
           "Reset only specified environment indices.")

        // Auto-reset control
        .def("set_auto_reset", &fast_finrl::VecFastFinRL::set_auto_reset, py::arg("enabled"),
             "Enable or disable auto-reset when environments are done")
        .def("auto_reset", &fast_finrl::VecFastFinRL::auto_reset,
             "Get current auto-reset setting")

        // Accessor methods
        .def("num_envs", &fast_finrl::VecFastFinRL::num_envs, "Number of environments")
        .def("n_tickers", &fast_finrl::VecFastFinRL::n_tickers, "Number of tickers per env")
        .def("n_indicators", &fast_finrl::VecFastFinRL::n_indicators, "Number of indicators")
        .def("n_macro", &fast_finrl::VecFastFinRL::n_macro, "Number of macro tickers")
        .def("get_all_tickers", &fast_finrl::VecFastFinRL::get_all_tickers, "Get all available tickers")
        .def("get_indicator_names", &fast_finrl::VecFastFinRL::get_indicator_names, "Get indicator names")
        .def("get_macro_tickers", &fast_finrl::VecFastFinRL::get_macro_tickers, "Get macro tickers")
        .def("get_tickers", &fast_finrl::VecFastFinRL::get_tickers, "Get tickers for each env")

        // Market window access (delegates to base_env)
        .def("get_market_window_numpy", [](const fast_finrl::VecFastFinRL& self,
                                           const std::vector<std::string>& ticker_list,
                                           int day,
                                           int h,
                                           int future) -> py::dict {
            auto base_env = self.get_base_env();

            using DataHolder = fast_finrl::FastFinRL::MultiTickerWindowData;
            std::shared_ptr<DataHolder> holder = std::make_shared<DataHolder>(
                base_env->get_market_window_multi(ticker_list, day, h, future)
            );

            int n_ind = holder->n_indicators;
            py::dict result;

            for (auto& [ticker, td] : holder->tickers) {
                py::dict ticker_dict;

                auto make_capsule = [holder]() {
                    return py::capsule(new std::shared_ptr<DataHolder>(holder),
                        [](void* p) { delete static_cast<std::shared_ptr<DataHolder>*>(p); });
                };

                ticker_dict["past_ohlcv"] = py::array_t<double>(
                    {h, 5}, {5 * sizeof(double), sizeof(double)},
                    td.past_ohlcv.data(), make_capsule());

                ticker_dict["past_indicators"] = py::array_t<double>(
                    {h, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.past_indicators.data(), make_capsule());

                ticker_dict["past_mask"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_mask.data(), make_capsule());

                ticker_dict["past_days"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_days.data(), make_capsule());

                ticker_dict["current_open"] = td.current_open;

                ticker_dict["current_indicators"] = py::array_t<double>(
                    {n_ind}, {sizeof(double)},
                    td.current_indicators.data(), make_capsule());

                ticker_dict["current_mask"] = td.current_mask;
                ticker_dict["current_day"] = td.current_day;

                ticker_dict["future_ohlcv"] = py::array_t<double>(
                    {future, 5}, {5 * sizeof(double), sizeof(double)},
                    td.future_ohlcv.data(), make_capsule());

                ticker_dict["future_indicators"] = py::array_t<double>(
                    {future, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.future_indicators.data(), make_capsule());

                ticker_dict["future_mask"] = py::array_t<int>(
                    {future}, {sizeof(int)},
                    td.future_mask.data(), make_capsule());

                ticker_dict["future_days"] = py::array_t<int>(
                    {future}, {sizeof(int)},
                    td.future_days.data(), make_capsule());

                result[py::str(ticker)] = ticker_dict;
            }

            py::list names;
            for (const std::string& name : holder->indicator_names) {
                names.append(py::str(name));
            }
            result["indicator_names"] = names;
            result["h"] = h;
            result["future"] = future;

            return result;
        }, py::arg("ticker_list"), py::arg("day"), py::arg("h"), py::arg("future"),
           "Get market window data (delegates to base environment)");

    // VecReplayBuffer - Vectorized replay buffer for N environments
    py::class_<fast_finrl::VecReplayBuffer>(m, "VecReplayBuffer")
        // Constructor from FastFinRL
        .def(py::init([](std::shared_ptr<fast_finrl::FastFinRL> env, size_t capacity, size_t batch_size, int64_t seed) {
            return std::make_unique<fast_finrl::VecReplayBuffer>(
                std::const_pointer_cast<const fast_finrl::FastFinRL>(env), capacity, batch_size, seed);
        }), py::arg("env"), py::arg("capacity") = 1000000, py::arg("batch_size") = 256, py::arg("seed") = 42,
           "Create VecReplayBuffer from FastFinRL instance. seed: 42 default for reproducibility")

        // Constructor from VecFastFinRL
        .def(py::init([](fast_finrl::VecFastFinRL& env, size_t capacity, size_t batch_size, int64_t seed) {
            return std::make_unique<fast_finrl::VecReplayBuffer>(env, capacity, batch_size, seed);
        }), py::arg("env"), py::arg("capacity") = 1000000, py::arg("batch_size") = 256, py::arg("seed") = 42,
           "Create VecReplayBuffer from VecFastFinRL instance. seed: 42 default for reproducibility")

        // add_transition - add single transition (for testing)
        .def("add_transition", [](fast_finrl::VecReplayBuffer& self,
                                  int env_id,
                                  int state_day, int next_state_day,
                                  const std::vector<std::string>& tickers,
                                  double state_cash, double next_state_cash,
                                  const std::vector<int>& state_shares,
                                  const std::vector<int>& next_state_shares,
                                  const std::vector<double>& state_avg_buy_price,
                                  const std::vector<double>& next_state_avg_buy_price,
                                  const std::vector<double>& action,
                                  py::object reward,
                                  bool done, bool terminal) {
            fast_finrl::VecStoredTransition t;
            t.env_id = env_id;
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
            if (py::isinstance<py::list>(reward) || py::isinstance<py::array>(reward)) {
                t.rewards = reward.cast<std::vector<double>>();
            } else {
                t.rewards = {reward.cast<double>()};
            }
            t.done = done;
            t.terminal = terminal;
            self.add(t);
        }, py::arg("env_id"), py::arg("state_day"), py::arg("next_state_day"),
           py::arg("tickers"), py::arg("state_cash"), py::arg("next_state_cash"),
           py::arg("state_shares"), py::arg("next_state_shares"),
           py::arg("state_avg_buy_price"), py::arg("next_state_avg_buy_price"),
           py::arg("action"), py::arg("rewards"), py::arg("done"), py::arg("terminal"),
           "Add single transition (for testing/compatibility)")

        // add - primary interface for VecFastFinRL
        .def("add", [](fast_finrl::VecReplayBuffer& self,
                             const py::list& states,
                             py::array_t<double, py::array::c_style> actions_arr,
                             const py::list& rewards_list,
                             const py::list& next_states,
                             const py::list& dones_list) {
            int num_envs = static_cast<int>(py::len(states));
            if (num_envs == 0) return;

            auto actions_buf = actions_arr.request();
            const double* actions = static_cast<const double*>(actions_buf.ptr);
            int n_tickers = static_cast<int>(actions_buf.shape[1]);

            // Extract from states
            std::vector<int> env_ids(num_envs);
            std::vector<int> state_days(num_envs);
            std::vector<int> next_state_days(num_envs);
            std::vector<std::vector<std::string>> tickers_list(num_envs);
            std::vector<double> state_cash(num_envs);
            std::vector<double> next_state_cash(num_envs);
            std::vector<int> state_shares_flat(num_envs * n_tickers);
            std::vector<int> next_state_shares_flat(num_envs * n_tickers);
            std::vector<double> state_avg_flat(num_envs * n_tickers);
            std::vector<double> next_state_avg_flat(num_envs * n_tickers);
            std::vector<std::vector<double>> rewards(num_envs);
            std::vector<bool> dones(num_envs);
            std::vector<bool> terminals(num_envs);

            for (int i = 0; i < num_envs; ++i) {
                py::object s = states[i];
                py::object ns = next_states[i];

                env_ids[i] = i;  // Default env_id = index
                state_days[i] = s.attr("day").cast<int>();
                next_state_days[i] = ns.attr("day").cast<int>();
                tickers_list[i] = s.attr("tickers").cast<std::vector<std::string>>();
                state_cash[i] = s.attr("cash").cast<double>();
                next_state_cash[i] = ns.attr("cash").cast<double>();

                // shares
                auto s_shares = s.attr("shares").cast<py::array_t<int>>();
                auto ns_shares = ns.attr("shares").cast<py::array_t<int>>();
                auto s_shares_ptr = s_shares.data();
                auto ns_shares_ptr = ns_shares.data();

                // avg_buy_price
                auto s_avg = s.attr("avg_buy_price").cast<py::array_t<double>>();
                auto ns_avg = ns.attr("avg_buy_price").cast<py::array_t<double>>();
                auto s_avg_ptr = s_avg.data();
                auto ns_avg_ptr = ns_avg.data();

                for (int j = 0; j < n_tickers; ++j) {
                    state_shares_flat[i * n_tickers + j] = s_shares_ptr[j];
                    next_state_shares_flat[i * n_tickers + j] = ns_shares_ptr[j];
                    state_avg_flat[i * n_tickers + j] = s_avg_ptr[j];
                    next_state_avg_flat[i * n_tickers + j] = ns_avg_ptr[j];
                }

                // Rewards - handle scalar or list
                py::object r = rewards_list[i];
                if (py::isinstance<py::list>(r) || py::isinstance<py::array>(r)) {
                    rewards[i] = r.cast<std::vector<double>>();
                } else {
                    rewards[i] = {r.cast<double>()};
                }

                dones[i] = dones_list[i].cast<bool>();
                terminals[i] = ns.attr("terminal").cast<bool>();
            }

            self.add_batch(
                num_envs, env_ids, state_days, next_state_days, tickers_list,
                state_cash, next_state_cash,
                state_shares_flat.data(), next_state_shares_flat.data(),
                state_avg_flat.data(), next_state_avg_flat.data(),
                actions, rewards, dones, terminals, n_tickers
            );
        }, py::arg("states"), py::arg("actions"), py::arg("rewards"),
           py::arg("next_states"), py::arg("dones"),
           "Add batch of N transitions from VecFastFinRL")

        // sample - same interface as ReplayBuffer
        .def("sample", [](const fast_finrl::VecReplayBuffer& self, int h, std::optional<size_t> batch_size) -> py::tuple {
            using Batch = fast_finrl::VecReplayBuffer::SampleBatch;
            std::shared_ptr<Batch> holder = std::make_shared<Batch>(
                batch_size ? self.sample(h, *batch_size) : self.sample(h)
            );

            const int B = holder->batch_size;
            const int T = h;  // history only (no current day)
            const int n_ind = holder->n_indicators;
            const int n_tic = holder->n_tickers;

            auto make_capsule = [holder]() {
                return py::capsule(new std::shared_ptr<Batch>(holder),
                    [](void* p) { delete static_cast<std::shared_ptr<Batch>*>(p); });
            };

            py::dict s_dict, s_next_dict;
            py::object s_mask_dict = py::none();
            py::object s_next_mask_dict = py::none();

            if (h > 0) {
                s_mask_dict = py::dict();
                s_next_mask_dict = py::dict();
            }

            // Build per-ticker data (iterate over unique_tickers)
            for (const auto& ticker : holder->unique_tickers) {
                py::dict td, td_next;

                td["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->s_ohlcv[ticker].data(), make_capsule());

                td["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->s_indicators[ticker].data(), make_capsule());

                td_next["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->s_next_ohlcv[ticker].data(), make_capsule());

                td_next["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->s_next_indicators[ticker].data(), make_capsule());

                s_dict[py::str(ticker)] = td;
                s_next_dict[py::str(ticker)] = td_next;

                if (h > 0) {
                    py::cast<py::dict>(s_mask_dict)[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->s_mask[ticker].data(), make_capsule());
                    py::cast<py::dict>(s_next_mask_dict)[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->s_next_mask[ticker].data(), make_capsule());
                }
            }

            // Macro tickers
            py::dict macro_dict, macro_next_dict;
            py::dict macro_mask_dict, macro_next_mask_dict;

            for (const std::string& ticker : holder->macro_tickers) {
                py::dict td, td_next;

                td["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->macro_ohlcv[ticker].data(), make_capsule());

                td["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->macro_indicators[ticker].data(), make_capsule());

                td_next["ohlcv"] = py::array_t<double>(
                    {B, T, 5},
                    {T * 5 * sizeof(double), 5 * sizeof(double), sizeof(double)},
                    holder->macro_next_ohlcv[ticker].data(), make_capsule());

                td_next["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->macro_next_indicators[ticker].data(), make_capsule());

                macro_dict[py::str(ticker)] = td;
                macro_next_dict[py::str(ticker)] = td_next;

                if (h > 0) {
                    macro_mask_dict[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->macro_mask[ticker].data(), make_capsule());
                    macro_next_mask_dict[py::str(ticker)] = py::array_t<int>(
                        {B, T}, {T * sizeof(int), sizeof(int)},
                        holder->macro_next_mask[ticker].data(), make_capsule());
                }
            }

            // env_ids [B]
            py::array_t<int> env_ids({B}, {sizeof(int)});
            std::copy(holder->env_ids.begin(), holder->env_ids.end(), env_ids.mutable_data());

            // Actions [B, n_tickers]
            py::array_t<double> actions({B, n_tic},
                {n_tic * sizeof(double), sizeof(double)},
                holder->actions.data(), make_capsule());

            // Rewards [B, n_objectives]
            int n_obj = holder->n_objectives;
            py::array_t<double> rewards({B, n_obj});
            double* r_ptr = rewards.mutable_data();
            for (int i = 0; i < B; ++i) {
                for (int j = 0; j < n_obj; ++j) {
                    r_ptr[i * n_obj + j] = holder->rewards[i][j];
                }
            }

            // Dones [B, 1]
            py::array_t<bool> dones({B, 1});
            bool* dones_ptr = dones.mutable_data();
            for (int i = 0; i < B; ++i) {
                dones_ptr[i] = holder->dones[i];
            }

            // Portfolio state
            py::array_t<double> state_cash({B}, {sizeof(double)},
                holder->state_cash.data(), make_capsule());
            py::array_t<double> next_state_cash({B}, {sizeof(double)},
                holder->next_state_cash.data(), make_capsule());
            py::array_t<int> state_shares({B, n_tic}, {n_tic * sizeof(int), sizeof(int)},
                holder->state_shares.data(), make_capsule());
            py::array_t<int> next_state_shares({B, n_tic}, {n_tic * sizeof(int), sizeof(int)},
                holder->next_state_shares.data(), make_capsule());
            py::array_t<double> state_avg({B, n_tic}, {n_tic * sizeof(double), sizeof(double)},
                holder->state_avg_buy_price.data(), make_capsule());
            py::array_t<double> next_state_avg({B, n_tic}, {n_tic * sizeof(double), sizeof(double)},
                holder->next_state_avg_buy_price.data(), make_capsule());

            // Build portfolio dicts
            py::dict portfolio, next_portfolio;
            portfolio["cash"] = state_cash;
            portfolio["shares"] = state_shares;
            portfolio["avg_buy_price"] = state_avg;
            next_portfolio["cash"] = next_state_cash;
            next_portfolio["shares"] = next_state_shares;
            next_portfolio["avg_buy_price"] = next_state_avg;

            // Metadata
            s_dict["indicator_names"] = holder->indicator_names;
            s_dict["tickers"] = holder->tickers;  // [batch][n_tickers] - per-sample tickers
            s_dict["unique_tickers"] = holder->unique_tickers;  // union of all tickers
            s_dict["portfolio"] = portfolio;
            s_dict["macro"] = macro_dict;
            s_dict["macro_tickers"] = holder->macro_tickers;
            s_dict["env_ids"] = env_ids;

            s_next_dict["indicator_names"] = holder->indicator_names;
            s_next_dict["tickers"] = holder->tickers;  // [batch][n_tickers]
            s_next_dict["unique_tickers"] = holder->unique_tickers;
            s_next_dict["portfolio"] = next_portfolio;
            s_next_dict["macro"] = macro_next_dict;
            s_next_dict["macro_tickers"] = holder->macro_tickers;

            if (h > 0 && holder->n_macro_tickers > 0) {
                py::cast<py::dict>(s_mask_dict)["macro"] = macro_mask_dict;
                py::cast<py::dict>(s_next_mask_dict)["macro"] = macro_next_mask_dict;
            }

            return py::make_tuple(s_dict, actions, rewards, s_next_dict, dones, s_mask_dict, s_next_mask_dict);
        }, py::arg("h"), py::arg("batch_size") = py::none(),
           "Sample batch with env_ids. Returns (s, a, r, s', done, s_mask, s'_mask)")

        .def("size", &fast_finrl::VecReplayBuffer::size, "Current buffer size")
        .def("capacity", &fast_finrl::VecReplayBuffer::capacity, "Buffer capacity")
        .def("clear", &fast_finrl::VecReplayBuffer::clear, "Clear buffer")
        .def("save", &fast_finrl::VecReplayBuffer::save, py::arg("path"), "Save buffer to file")
        .def("load", &fast_finrl::VecReplayBuffer::load, py::arg("path"), "Load buffer from file");
}
