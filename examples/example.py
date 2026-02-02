"""
FastFinRL Python Example
========================
High-performance C++ implementation of FinRL StockTradingEnv with Python bindings.
"""

import sys
sys.path.insert(0, "../build")

from fast_finrl_py import FastFinRL, FastFinRLConfig

DATA_PATH = "../data/raw_train_df.csv"
TICKERS = ["SPY", "QQQ", "GLD", "TLT"]


def basic_usage():
    """Basic environment usage with default settings."""
    print("=" * 50)
    print("Basic Usage")
    print("=" * 50)

    env = FastFinRL(DATA_PATH)
    state = env.reset(TICKERS, seed=42)

    print(f"Day: {state['day']}, Date: {state['date']}")
    print(f"Cash: ${state['portfolio']['cash']:,.2f}")
    print(f"Tickers: {list(state['market'].keys())}")

    # Show market data for first ticker
    ticker = TICKERS[0]
    market = state['market'][ticker]
    print(f"\n{ticker} Market Data:")
    print(f"  Open: ${market['open']:.2f}")
    print(f"  High: ${market['high']:.2f}")
    print(f"  Low: ${market['low']:.2f}")
    print(f"  Close: ${market['close']:.2f}")
    print(f"  Indicators: {list(market['indicators'].keys())[:5]}...")


def custom_config():
    """Using custom configuration."""
    print("\n" + "=" * 50)
    print("Custom Configuration")
    print("=" * 50)

    config = FastFinRLConfig(
        initial_amount=30000.0,
        hmax=30,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        stop_loss_tolerance=0.85,
        bidding="uniform",
        stop_loss_calculation="close",
        initial_seed=100
    )

    print(f"Config: {config}")

    env = FastFinRL(DATA_PATH, config)
    state = env.reset(TICKERS, seed=-1)  # Auto-increment from initial_seed

    print(f"Seed used: {state['seed']}")  # Should be 101
    print(f"Initial cash: ${state['portfolio']['cash']:,.2f}")


def keyword_args():
    """Using keyword arguments instead of config object."""
    print("\n" + "=" * 50)
    print("Keyword Arguments")
    print("=" * 50)

    env = FastFinRL(
        DATA_PATH,
        initial_amount=50000.0,
        hmax=20,
        buy_cost_pct=0.005,
        sell_cost_pct=0.005
    )

    state = env.reset(["SPY", "QQQ"], seed=42)
    print(f"Cash: ${state['portfolio']['cash']:,.2f}")
    print(f"hmax: {env.hmax}")


def trading_episode():
    """Run a complete trading episode."""
    from pprint import pprint
    print("\n" + "=" * 50)
    print("Trading Episode")
    print("=" * 50)

    env = FastFinRL(DATA_PATH, initial_amount=30000.0, hmax=10, bidding='deterministic')
    state = env.reset(["SPY", "QQQ"], seed=42)
    pprint(state)
    state['market']['QQQ'].pop('indicators')
    state['market']['SPY'].pop('indicators')

    print("initial state")
    pprint(state)

    total_reward = 0.0
    steps = 0
    max_steps = 100
    total_trades = 0

    print(f"Starting episode at day {state['day']}")
    print(f"Initial cash: ${state['portfolio']['cash']:,.2f}")

    while steps < max_steps:
        # Simple strategy: buy on even steps, sell on multiples of 5

        actions = [1.0, 1.0]  # Max buy both
        action_type = "BUY"

        prev_trades = total_trades
        state = env.step(actions)
        state['market']['QQQ'].pop('indicators')
        state['market']['SPY'].pop('indicators')

        print("next state")
        pprint(state)
        total_reward += state['reward']
        total_trades = state['info']['n_trades']
        steps += 1
        break
        # Log significant actions
        if action_type != "HOLD" and total_trades > prev_trades:
            holdings = state['portfolio']['holdings']
            spy_shares = holdings['SPY']['shares']
            qqq_shares = holdings['QQQ']['shares']
            print(f"  Step {steps}: {action_type} - SPY:{spy_shares}, QQQ:{qqq_shares}, "
                  f"Cash: ${state['portfolio']['cash']:,.2f}")

        if state['done'] or state['terminal']:
            break

    print(f"\nEpisode finished after {steps} steps")
    print(f"Final day: {state['day']}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final cash: ${state['portfolio']['cash']:,.2f}")
    print(f"Total asset: ${state['portfolio']['total_asset']:,.2f}")
    print(f"Trades: {state['info']['n_trades']}")
    print(f"Stop losses: {state['info']['num_stop_loss']}")

    # Show final holdings
    print("\nFinal Holdings:")
    for ticker, holding in state['portfolio']['holdings'].items():
        print(f"  {ticker}: {holding['shares']} shares @ ${holding['avg_buy_price']:.2f}")


def seed_reproducibility():
    """Demonstrate seed reproducibility."""
    print("\n" + "=" * 50)
    print("Seed Reproducibility")
    print("=" * 50)

    env1 = FastFinRL(DATA_PATH)
    env2 = FastFinRL(DATA_PATH)

    # Same seed should produce same results
    state1 = env1.reset(["SPY"], seed=42)
    state2 = env2.reset(["SPY"], seed=42)

    print(f"Env1 - Day: {state1['day']}, Date: {state1['date']}")
    print(f"Env2 - Day: {state2['day']}, Date: {state2['date']}")
    print(f"Match: {state1['day'] == state2['day'] and state1['date'] == state2['date']}")

    # Auto-increment seed
    print("\nAuto-increment seed (-1):")
    state1 = env1.reset(["SPY"], seed=-1)
    state2 = env1.reset(["SPY"], seed=-1)
    print(f"First reset: seed={state1['seed']}")
    print(f"Second reset: seed={state2['seed']}")


def multi_episode_training():
    """Simulate multi-episode training loop."""
    print("\n" + "=" * 50)
    print("Multi-Episode Training")
    print("=" * 50)

    env = FastFinRL(DATA_PATH, initial_seed=0)
    env.reset(TICKERS, seed=0)  # Initialize seed counter

    episode_rewards = []

    for episode in range(5):
        state = env.reset(TICKERS, seed=-1)  # Auto-increment
        current_seed = state['seed']
        episode_reward = 0.0
        steps = 0

        while steps < 50:
            # Random-ish actions
            actions = [0.1 * (episode % 3 - 1) for _ in TICKERS]
            state = env.step(actions)
            episode_reward += state['reward']
            steps += 1

            if state['done'] or state['terminal']:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: seed={current_seed}, "
              f"reward={episode_reward:.4f}, steps={steps}")

    print(f"\nAverage reward: {sum(episode_rewards) / len(episode_rewards):.4f}")


def inspect_indicators():
    """Inspect available technical indicators."""
    print("\n" + "=" * 50)
    print("Technical Indicators")
    print("=" * 50)

    env = FastFinRL(DATA_PATH)
    indicators = env.get_indicator_names()

    print(f"Available indicators ({len(indicators)}):")
    for i, name in enumerate(sorted(indicators)):
        print(f"  {name}", end="")
        if (i + 1) % 5 == 0:
            print()
    print()

    # Show indicator values for a specific day
    state = env.reset(["SPY"], seed=42)
    print(f"\nSPY indicators on {state['date']}:")
    for name, value in list(state['market']['SPY']['indicators'].items())[:10]:
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    basic_usage()
    custom_config()
    keyword_args()
    trading_episode()
    seed_reproducibility()
    multi_episode_training()
    inspect_indicators()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)
