#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "FastFinRL.hpp"

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
                         int initial_seed) {
            fast_finrl::FastFinRLConfig config;
            config.initial_amount = initial_amount;
            config.hmax = hmax;
            config.buy_cost_pct = buy_cost_pct;
            config.sell_cost_pct = sell_cost_pct;
            config.stop_loss_tolerance = stop_loss_tolerance;
            config.bidding = bidding;
            config.stop_loss_calculation = stop_loss_calculation;
            config.initial_seed = initial_seed;
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
                         int seed) {
            return json_to_python(self.reset(ticker_list, seed));
        }, py::arg("ticker_list"), py::arg("seed"),
           "Reset environment with given tickers and seed. Returns state dict.")

        .def("step", [](fast_finrl::FastFinRL& self,
                        const std::vector<double>& actions) {
            return json_to_python(self.step(actions));
        }, py::arg("actions"),
           "Execute one step with given actions. Returns state dict with reward, done, terminal.")

        // Accessor methods
        .def("get_indicator_names", &fast_finrl::FastFinRL::get_indicator_names,
             "Get set of technical indicator column names")

        .def("get_state", [](const fast_finrl::FastFinRL& self) {
            return json_to_python(self.get_state());
        }, "Get current state as dict")

        .def("get_raw_value", &fast_finrl::FastFinRL::get_raw_value,
             py::arg("ticker"), py::arg("day"), py::arg("column"),
             "Get raw value from DataFrame for given ticker, day, and column");
}
