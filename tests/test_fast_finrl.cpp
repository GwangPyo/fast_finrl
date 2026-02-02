#include <gtest/gtest.h>
#include "FastFinRL.hpp"
#include <filesystem>

using namespace fast_finrl;

// Helper to get data path - use real data file
std::string get_data_path() {
    // Try multiple paths for real data
    std::vector<std::string> paths = {
        "data/raw_train_df.csv",
        "../data/raw_train_df.csv",
        "../../data/raw_train_df.csv"
    };

    for (const auto& path : paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    // Return default and let test fail with meaningful error
    return "data/raw_train_df.csv";
}

// Real tickers from raw_train_df.csv
const std::vector<std::string> REAL_TICKERS = {"SPY", "QQQ", "GLD", "TLT"};
const std::string TICKER_1 = "SPY";
const std::string TICKER_2 = "QQQ";
const std::string TICKER_3 = "GLD";

// Test 1: DataFrame Load and Column Extraction
TEST(FastFinRL, LoadDataFrame) {
    FastFinRL env(get_data_path());

    auto indicators = env.get_indicator_names();

    // Basic columns should not be included
    EXPECT_TRUE(indicators.find("day") == indicators.end());
    EXPECT_TRUE(indicators.find("date") == indicators.end());
    EXPECT_TRUE(indicators.find("tic") == indicators.end());
    EXPECT_TRUE(indicators.find("open") == indicators.end());
    EXPECT_TRUE(indicators.find("close") == indicators.end());

    // Technical indicators should be included
    EXPECT_TRUE(indicators.find("macd") != indicators.end());
    EXPECT_TRUE(indicators.find("rsi_7") != indicators.end());
    EXPECT_TRUE(indicators.find("rsi_14") != indicators.end());
}

// Test 2: Reset Basic Operation
TEST(FastFinRL, ResetBasic) {
    FastFinRL env(get_data_path());

    auto state = env.reset({TICKER_1, TICKER_2}, 42);

    // Structure verification
    EXPECT_TRUE(state.contains("day"));
    EXPECT_TRUE(state.contains("date"));
    EXPECT_TRUE(state.contains("seed"));
    EXPECT_TRUE(state.contains("portfolio"));
    EXPECT_TRUE(state.contains("market"));

    // Seed return verification
    EXPECT_EQ(state["seed"].get<int>(), 42);

    // Initial capital verification
    EXPECT_DOUBLE_EQ(state["portfolio"]["cash"].get<double>(), 30000.0);

    // Initial holdings verification
    EXPECT_EQ(state["portfolio"]["holdings"][TICKER_1]["shares"].get<int>(), 0);
    EXPECT_DOUBLE_EQ(state["portfolio"]["holdings"][TICKER_1]["avg_buy_price"].get<double>(), 0.0);

    // Market data existence verification
    EXPECT_TRUE(state["market"].contains(TICKER_1));
    EXPECT_TRUE(state["market"][TICKER_1].contains("open"));
    EXPECT_TRUE(state["market"][TICKER_1].contains("indicators"));
}

// Test 3: Reset Seed Reproducibility
TEST(FastFinRL, ResetSeedReproducibility) {
    FastFinRL env1(get_data_path());
    FastFinRL env2(get_data_path());

    auto state1 = env1.reset({TICKER_1}, 42);
    auto state2 = env2.reset({TICKER_1}, 42);

    EXPECT_EQ(state1["day"].get<int>(), state2["day"].get<int>());
    EXPECT_EQ(state1["date"].get<std::string>(), state2["date"].get<std::string>());
    EXPECT_EQ(state1["seed"].get<int>(), state2["seed"].get<int>());
}

// Test 4: Reset Seed Auto Increment (seed = -1)
TEST(FastFinRL, ResetSeedAutoIncrement) {
    FastFinRL env(get_data_path());

    // Initial seed setting
    auto state1 = env.reset({TICKER_1}, 42);
    EXPECT_EQ(state1["seed"].get<int>(), 42);

    // seed = -1: auto increment
    auto state2 = env.reset({TICKER_1}, -1);
    EXPECT_EQ(state2["seed"].get<int>(), 43);

    auto state3 = env.reset({TICKER_1}, -1);
    EXPECT_EQ(state3["seed"].get<int>(), 44);

    // Explicit seed reset
    auto state4 = env.reset({TICKER_1}, 100);
    EXPECT_EQ(state4["seed"].get<int>(), 100);

    // Auto increment again
    auto state5 = env.reset({TICKER_1}, -1);
    EXPECT_EQ(state5["seed"].get<int>(), 101);
}

// Test 5: Reset Auto Increment Reproducibility
TEST(FastFinRL, ResetAutoIncrementReproducibility) {
    FastFinRL env1(get_data_path());
    FastFinRL env2(get_data_path());

    // Same reset sequence
    env1.reset({TICKER_1}, 42);
    env1.reset({TICKER_1}, -1);
    auto state1 = env1.reset({TICKER_1}, -1);

    env2.reset({TICKER_1}, 42);
    env2.reset({TICKER_1}, -1);
    auto state2 = env2.reset({TICKER_1}, -1);

    // Same results
    EXPECT_EQ(state1["day"].get<int>(), state2["day"].get<int>());
    EXPECT_EQ(state1["seed"].get<int>(), state2["seed"].get<int>());
    EXPECT_EQ(state1["seed"].get<int>(), 44);
}

// Test 6: Step Buy Operation
TEST(FastFinRL, StepBuy) {
    FastFinRL env(get_data_path());
    env.reset({TICKER_1}, 42);

    // Maximum buy (action = 1.0 * hmax)
    auto state = env.step({1.0});

    // Cash decrease verification
    EXPECT_LT(state["portfolio"]["cash"].get<double>(), 30000.0);

    // Stock holding verification
    EXPECT_GT(state["portfolio"]["holdings"][TICKER_1]["shares"].get<int>(), 0);

    // Average buy price set verification
    EXPECT_GT(state["portfolio"]["holdings"][TICKER_1]["avg_buy_price"].get<double>(), 0.0);
}

// Test 7: Step Sell Operation
TEST(FastFinRL, StepSell) {
    FastFinRL env(get_data_path());
    env.reset({TICKER_1}, 42);

    // First buy
    env.step({1.0});
    auto before_state = env.get_state();
    double before_cash = before_state["portfolio"]["cash"].get<double>();
    int before_shares = before_state["portfolio"]["holdings"][TICKER_1]["shares"].get<int>();

    // Sell all (action = -1.0)
    auto state = env.step({-1.0});

    // Cash increase verification
    EXPECT_GT(state["portfolio"]["cash"].get<double>(), before_cash);

    // Shares decrease verification
    EXPECT_LT(state["portfolio"]["holdings"][TICKER_1]["shares"].get<int>(), before_shares);
}

// Test 8: Stop Loss
TEST(FastFinRL, StopLoss) {
    FastFinRL env(get_data_path());
    env.stop_loss_tolerance = 0.8;  // 20% loss triggers stop loss
    env.reset({TICKER_1}, 42);

    // Buy stock
    env.step({1.0});

    // Continue stepping and check for stop loss
    auto state = env.step({0.0});

    // If stop loss occurred, verify behavior
    if (state["info"]["num_stop_loss"].get<int>() > 0) {
        // Either shares are 0 or loss_cut_amount is > 0
        EXPECT_TRUE(
            state["portfolio"]["holdings"][TICKER_1]["shares"].get<int>() == 0 ||
            state["info"]["loss_cut_amount"].get<double>() > 0.0
        );
    }
}

// Test 9: Terminal Condition
TEST(FastFinRL, TerminalCondition) {
    FastFinRL env(get_data_path());
    // Use seed 999 to start near end of data (80% of days = ~1142, random from 0-1142)
    // With a high seed, we get deterministic behavior
    env.reset({TICKER_1}, 999);

    // Step until terminal (real data has ~1428 days, so we may need many steps)
    nlohmann::json state;
    int max_steps = 2000;  // Increased for real data
    int steps = 0;

    while (steps < max_steps) {
        state = env.step({0.0});
        if (state["terminal"].get<bool>() || state["done"].get<bool>()) break;
        steps++;
    }

    // Should eventually reach terminal or done
    EXPECT_TRUE(state["terminal"].get<bool>() || state["done"].get<bool>());
}

// Test 10: Done Condition (Bankruptcy)
TEST(FastFinRL, DoneConditionBankrupt) {
    FastFinRL env(get_data_path());
    env.initial_amount = 26000;  // Start close to bankruptcy threshold
    env.reset({TICKER_1}, 42);

    auto state = env.step({1.0});

    // If total asset drops below 25000, done should be true
    if (state["portfolio"]["total_asset"].get<double>() <= 25000) {
        EXPECT_TRUE(state["done"].get<bool>());
    }
}

// Test 11: Indicators Match DataFrame
TEST(FastFinRL, IndicatorsMatchDataFrame) {
    FastFinRL env(get_data_path());
    auto state = env.reset({TICKER_1}, 42);

    int day = state["day"].get<int>();
    auto& indicators = state["market"][TICKER_1]["indicators"];

    // Verify indicator value matches raw DataFrame value
    double json_macd = indicators["macd"].get<double>();
    double df_macd = env.get_raw_value(TICKER_1, day, "macd");
    EXPECT_DOUBLE_EQ(json_macd, df_macd);
}

// Test 12: Multi-Ticker Operation
TEST(FastFinRL, MultiTicker) {
    FastFinRL env(get_data_path());
    auto state = env.reset({TICKER_1, TICKER_2, TICKER_3}, 42);

    EXPECT_EQ(state["market"].size(), 3);
    EXPECT_EQ(state["portfolio"]["holdings"].size(), 3);

    // Independent trading for each ticker
    // Buy TICKER_1, sell TICKER_2 (no shares so 0), hold TICKER_3
    auto next = env.step({1.0, -0.5, 0.0});

    EXPECT_GT(next["portfolio"]["holdings"][TICKER_1]["shares"].get<int>(), 0);
    EXPECT_EQ(next["portfolio"]["holdings"][TICKER_2]["shares"].get<int>(), 0);
    EXPECT_EQ(next["portfolio"]["holdings"][TICKER_3]["shares"].get<int>(), 0);
}

// Test 13: Continuous Episode Training
TEST(FastFinRL, ContinuousEpisodeTraining) {
    FastFinRL env(get_data_path());

    // Training loop simulation
    env.reset({TICKER_1}, 0);  // First episode, seed=0

    for (int episode = 1; episode <= 10; episode++) {
        auto state = env.reset({TICKER_1}, -1);  // Auto increment
        EXPECT_EQ(state["seed"].get<int>(), episode);

        // Run episode
        int steps = 0;
        while (!state["done"].get<bool>() && !state["terminal"].get<bool>() && steps < 50) {
            state = env.step({0.0});
            steps++;
        }
    }
}

// Additional test: Reward Calculation
TEST(FastFinRL, RewardCalculation) {
    FastFinRL env(get_data_path());
    env.reset({TICKER_1}, 42);

    auto state = env.step({0.0});  // Hold action

    // Reward should be present
    EXPECT_TRUE(state.contains("reward"));

    // Reward is log(end/begin), so should be finite
    double reward = state["reward"].get<double>();
    EXPECT_TRUE(std::isfinite(reward));
}

// Additional test: Info Dictionary
TEST(FastFinRL, InfoDictionary) {
    FastFinRL env(get_data_path());
    env.reset({TICKER_1}, 42);

    auto state = env.step({1.0});

    // Info should contain required fields
    EXPECT_TRUE(state.contains("info"));
    EXPECT_TRUE(state["info"].contains("loss_cut_amount"));
    EXPECT_TRUE(state["info"].contains("n_trades"));
    EXPECT_TRUE(state["info"].contains("num_stop_loss"));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
