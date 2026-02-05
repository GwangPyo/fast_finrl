#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <optional>
#include "FastFinRL.hpp"
#include "ReplayBuffer.hpp"
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
                         const std::vector<std::string>& tech_indicator_list) {
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

        // Core API methods
        .def("reset", [](fast_finrl::FastFinRL& self,
                         const std::vector<std::string>& ticker_list,
                         int64_t seed,
                         int shifted_start) {
            return json_to_python(self.reset(ticker_list, seed, shifted_start));
        }, py::arg("ticker_list"), py::arg("seed"), py::arg("shifted_start") = 0,
           "Reset environment with given tickers, seed, and shifted_start. Returns state dict with day_idx.")

        .def("step", [](fast_finrl::FastFinRL& self,
                        const std::vector<double>& actions) {
            return json_to_python(self.step(actions));
        }, py::arg("actions"),
           "Execute one step with given actions. Returns state dict with reward, done, terminal.")

        // Accessor methods
        .def("get_indicator_names", &fast_finrl::FastFinRL::get_indicator_names,
             "Get set of technical indicator column names")

        .def("get_all_tickers", &fast_finrl::FastFinRL::get_all_tickers,
             "Get set of all available ticker symbols")

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

                // Past arrays - zero-copy views
                ticker_dict["past_ohlc"] = py::array_t<double>(
                    {h, 4}, {4 * sizeof(double), sizeof(double)},
                    td.past_ohlc.data(), make_capsule());

                ticker_dict["past_indicators"] = py::array_t<double>(
                    {h, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.past_indicators.data(), make_capsule());

                ticker_dict["past_mask"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_mask.data(), make_capsule());

                ticker_dict["past_days"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_days.data(), make_capsule());

                // Current arrays - zero-copy views
                ticker_dict["current_ohlc"] = py::array_t<double>(
                    {4}, {sizeof(double)},
                    td.current_ohlc.data(), make_capsule());

                ticker_dict["current_indicators"] = py::array_t<double>(
                    {n_ind}, {sizeof(double)},
                    td.current_indicators.data(), make_capsule());

                ticker_dict["current_mask"] = td.current_mask;
                ticker_dict["current_day"] = td.current_day;

                // Future arrays - zero-copy views
                ticker_dict["future_ohlc"] = py::array_t<double>(
                    {future, 4}, {4 * sizeof(double), sizeof(double)},
                    td.future_ohlc.data(), make_capsule());

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
        .def(py::init([](std::shared_ptr<fast_finrl::FastFinRL> env, size_t capacity, size_t batch_size) {
            return std::make_unique<fast_finrl::ReplayBuffer>(
                std::const_pointer_cast<const fast_finrl::FastFinRL>(env), capacity, batch_size);
        }), py::arg("env"), py::arg("capacity") = 1000000, py::arg("batch_size") = 256,
           "Create ReplayBuffer. capacity: 100K (small), 1M (default), 5M (large)")
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
            // s/s': dict[ticker] -> ohlc [batch, h+1, 4], indicators [batch, h+1, n_ind]
            // s_mask/s'_mask: dict[ticker] -> [batch, h+1] or None if h=0
            // a: [batch, n_tickers]
            // r: [batch]

            using Batch = fast_finrl::ReplayBuffer::SampleBatch;
            std::shared_ptr<Batch> holder = std::make_shared<Batch>(
                batch_size ? self.sample(h, *batch_size) : self.sample(h)
            );

            const int B = holder->batch_size;
            const int T = h + 1;
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

                // OHLC [B, T, 4]
                td["ohlc"] = py::array_t<double>(
                    {B, T, 4},
                    {T * 4 * sizeof(double), 4 * sizeof(double), sizeof(double)},
                    holder->s_ohlc[ticker].data(), make_capsule());

                td["indicators"] = py::array_t<double>(
                    {B, T, n_ind},
                    {T * n_ind * sizeof(double), n_ind * sizeof(double), sizeof(double)},
                    holder->s_indicators[ticker].data(), make_capsule());

                td_next["ohlc"] = py::array_t<double>(
                    {B, T, 4},
                    {T * 4 * sizeof(double), 4 * sizeof(double), sizeof(double)},
                    holder->s_next_ohlc[ticker].data(), make_capsule());

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
            s_next_dict["indicator_names"] = holder->indicator_names;
            s_next_dict["tickers"] = holder->tickers;
            s_next_dict["portfolio"] = next_portfolio;

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

                ticker_dict["past_ohlc"] = py::array_t<double>(
                    {h, 4}, {4 * sizeof(double), sizeof(double)},
                    td.past_ohlc.data(), make_capsule());

                ticker_dict["past_indicators"] = py::array_t<double>(
                    {h, n_ind}, {n_ind * sizeof(double), sizeof(double)},
                    td.past_indicators.data(), make_capsule());

                ticker_dict["past_mask"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_mask.data(), make_capsule());

                ticker_dict["past_days"] = py::array_t<int>(
                    {h}, {sizeof(int)},
                    td.past_days.data(), make_capsule());

                ticker_dict["current_ohlc"] = py::array_t<double>(
                    {4}, {sizeof(double)},
                    td.current_ohlc.data(), make_capsule());

                ticker_dict["current_indicators"] = py::array_t<double>(
                    {n_ind}, {sizeof(double)},
                    td.current_indicators.data(), make_capsule());

                ticker_dict["current_mask"] = td.current_mask;
                ticker_dict["current_day"] = td.current_day;

                ticker_dict["future_ohlc"] = py::array_t<double>(
                    {future, 4}, {4 * sizeof(double), sizeof(double)},
                    td.future_ohlc.data(), make_capsule());

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
}
