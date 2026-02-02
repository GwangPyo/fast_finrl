#include <iostream>
#include <iomanip>
#include "FastFinRL.hpp"

int main() {
    try {
        std::cout << "FastFinRL Demo with Real Data\n";
        std::cout << "==============================\n\n";

        // Create environment with real data
        fast_finrl::FastFinRL env("./data/raw_train_df.csv");

        // Show available indicators
        std::cout << "Available indicators (" << env.get_indicator_names().size() << "):\n  ";
        int count = 0;
        for (const auto& ind : env.get_indicator_names()) {
            std::cout << ind << " ";
            if (++count % 8 == 0) std::cout << "\n  ";
        }
        std::cout << "\n\n";

        // Use real tickers from raw_train_df.csv
        std::vector<std::string> tickers = {"SPY", "QQQ", "GLD", "TLT"};

        // Reset with specific seed
        auto state = env.reset(tickers, 42);
        std::cout << "Initial state:\n";
        std::cout << "  Day: " << state["day"] << "\n";
        std::cout << "  Date: " << state["date"].get<std::string>() << "\n";
        std::cout << "  Seed: " << state["seed"] << "\n";
        std::cout << "  Cash: $" << std::fixed << std::setprecision(2) << state["portfolio"]["cash"].get<double>() << "\n";
        std::cout << "  Market data for SPY:\n";
        std::cout << "    Open:  $" << state["market"]["SPY"]["open"].get<double>() << "\n";
        std::cout << "    High:  $" << state["market"]["SPY"]["high"].get<double>() << "\n";
        std::cout << "    Low:   $" << state["market"]["SPY"]["low"].get<double>() << "\n";
        std::cout << "    Close: $" << state["market"]["SPY"]["close"].get<double>() << "\n";
        std::cout << "    RSI_7: " << state["market"]["SPY"]["indicators"]["rsi_7"].get<double>() << "\n";
        std::cout << "    MACD:  " << state["market"]["SPY"]["indicators"]["macd"].get<double>() << "\n\n";

        // Run trading simulation
        std::cout << "Running 10 steps of trading simulation...\n";
        std::cout << "Actions: buy SPY(0.8), buy QQQ(0.5), hold GLD(0), sell TLT(-0.3)\n\n";

        std::vector<double> actions = {0.8, 0.5, 0.0, -0.3};

        for (int i = 0; i < 10; i++) {
            state = env.step(actions);

            std::cout << "Step " << std::setw(2) << (i + 1) << ": ";
            std::cout << "Day=" << state["day"].get<int>() << " ";
            std::cout << "Asset=$" << std::setw(10) << state["portfolio"]["total_asset"].get<double>() << " ";
            std::cout << "Reward=" << std::setw(8) << std::setprecision(5) << state["reward"].get<double>() << " ";
            std::cout << "SPY=" << std::setw(3) << state["portfolio"]["holdings"]["SPY"]["shares"].get<int>() << " ";
            std::cout << "QQQ=" << std::setw(3) << state["portfolio"]["holdings"]["QQQ"]["shares"].get<int>() << "\n";

            if (state["done"].get<bool>() || state["terminal"].get<bool>()) {
                std::cout << "\nEpisode ended. Terminal=" << state["terminal"].get<bool>() << "\n";
                break;
            }
        }

        // Test seed auto-increment
        std::cout << "\n--- Testing seed auto-increment ---\n";
        state = env.reset(tickers, -1);
        std::cout << "Reset with seed=-1: got seed=" << state["seed"].get<int>() << " day=" << state["day"].get<int>() << "\n";
        state = env.reset(tickers, -1);
        std::cout << "Reset with seed=-1: got seed=" << state["seed"].get<int>() << " day=" << state["day"].get<int>() << "\n";
        state = env.reset(tickers, -1);
        std::cout << "Reset with seed=-1: got seed=" << state["seed"].get<int>() << " day=" << state["day"].get<int>() << "\n";

        std::cout << "\nDemo completed successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
