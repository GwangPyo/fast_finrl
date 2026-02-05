#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include "FastFinRL.hpp"
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

    py::class_<fast_finrl::FastFinRL>(m, "FastFinRL")
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
            auto holder = std::make_shared<DataHolder>(
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
            for (const auto& name : holder->indicator_names) {
                names.append(py::str(name));
            }
            result["indicator_names"] = names;
            result["h"] = h;
            result["future"] = future;

            return result;
        }, py::arg("ticker_list"), py::arg("day"), py::arg("h"), py::arg("future"),
           "Multi-ticker market window with separated past/current/future numpy arrays (zero-copy)");
}
