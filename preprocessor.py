"""
Observation Preprocessor for FastFinRL environments.

Converts raw state dicts from FastFinRL/VecFastFinRL into
ReplayBuffer-compatible dictionary format.
"""

import numpy as np
from typing import Dict, List, Union, Any


class ObservationPreprocessor:
    """
    Preprocesses observations from FastFinRL environments into a unified format.

    For FastFinRL (single env): returns dict without batch dimension
    For VecFastFinRL (N envs): returns dict with batch dimension N

    Output format matches ReplayBuffer.sample() structure.
    """

    def __init__(
        self,
        env: Any,
        h: int = 0,
        include_portfolio: bool = True,
        include_macro: bool = True,
    ):
        """
        Args:
            env: FastFinRL or VecFastFinRL instance
            h: History length for market window (0 = current only)
            include_portfolio: Include portfolio state (cash, shares, avg_buy_price)
            include_macro: Include macro ticker data if available
        """
        self.env = env
        self.h = h
        self.include_portfolio = include_portfolio
        self.include_macro = include_macro

        # Detect env type
        self._is_vec = hasattr(env, 'num_envs')

        # Cache indicator names
        self.indicator_names = list(env.get_indicator_names())
        self.n_indicators = len(self.indicator_names)

        # Cache macro tickers
        self.macro_tickers = list(env.get_macro_tickers()) if hasattr(env, 'get_macro_tickers') else []
        self.n_macro = len(self.macro_tickers)

    def process(self, state: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """
        Process state(s) into ReplayBuffer-compatible format.

        Args:
            state: Single state dict (FastFinRL) or List of state dicts (VecFastFinRL)

        Returns:
            Processed observation dict with structure:
            - For single env: {ticker: {"ohlc": (h+1, 4), "indicators": (h+1, n_ind)}, ...}
            - For vec env: {ticker: {"ohlc": (N, h+1, 4), "indicators": (N, h+1, n_ind)}, ...}
        """
        if self._is_vec:
            return self._process_vec(state)
        else:
            return self._process_single(state)

    def _process_single(self, state: Dict) -> Dict[str, Any]:
        """Process single FastFinRL state (no batch dimension)."""
        result = {}

        # Get tickers from state
        if "market" in state:
            # FastFinRL format
            tickers = list(state["market"].keys())
            result["tickers"] = tickers

            for ticker in tickers:
                market = state["market"][ticker]
                td = {}

                # OHLC and indicators
                if self.h > 0:
                    # Need historical data - use get_market_window_numpy
                    day = state["day"]
                    window = self.env.get_market_window_numpy(tickers, day, self.h, 0)

                    # Combine past + current: (h+1, 4), (h+1, n_ind)
                    past_ohlc = window[ticker]["past_ohlc"]  # (h, 4)
                    current_ohlc = window[ticker]["current_ohlc"]  # (4,)
                    td["ohlc"] = np.vstack([past_ohlc, current_ohlc[np.newaxis, :]])

                    past_ind = window[ticker]["past_indicators"]  # (h, n_ind)
                    current_ind = window[ticker]["current_indicators"]  # (n_ind,)
                    td["indicators"] = np.vstack([past_ind, current_ind[np.newaxis, :]])

                    td["mask"] = np.concatenate([
                        window[ticker]["past_mask"],
                        [window[ticker]["current_mask"]]
                    ])
                else:
                    # Current only: (4,), (n_ind,)
                    td["ohlc"] = np.array([
                        market["open"], market["high"],
                        market["low"], market["close"]
                    ])
                    td["indicators"] = np.array([
                        market["indicators"][ind] for ind in self.indicator_names
                    ])

                result[ticker] = td

            # Macro tickers
            if self.include_macro and "macro" in state and state["macro"]:
                result["macro"] = {}
                for ticker in state["macro"]:
                    market = state["macro"][ticker]
                    td = {}

                    if self.h > 0:
                        day = state["day"]
                        window = self.env.get_market_window_numpy([ticker], day, self.h, 0)
                        past_ohlc = window[ticker]["past_ohlc"]
                        current_ohlc = window[ticker]["current_ohlc"]
                        td["ohlc"] = np.vstack([past_ohlc, current_ohlc[np.newaxis, :]])

                        past_ind = window[ticker]["past_indicators"]
                        current_ind = window[ticker]["current_indicators"]
                        td["indicators"] = np.vstack([past_ind, current_ind[np.newaxis, :]])
                        td["mask"] = np.concatenate([
                            window[ticker]["past_mask"],
                            [window[ticker]["current_mask"]]
                        ])
                    else:
                        td["ohlc"] = np.array([
                            market["open"], market["high"],
                            market["low"], market["close"]
                        ])
                        td["indicators"] = np.array([
                            market["indicators"][ind] for ind in self.indicator_names
                        ])

                    result["macro"][ticker] = td
                result["macro_tickers"] = list(state["macro"].keys())

            # Portfolio
            if self.include_portfolio and "portfolio" in state:
                portfolio = state["portfolio"]
                result["portfolio"] = {
                    "cash": portfolio["cash"],
                    "shares": np.array([
                        portfolio["holdings"][t]["shares"] for t in tickers
                    ]),
                    "avg_buy_price": np.array([
                        portfolio["holdings"][t]["avg_buy_price"] for t in tickers
                    ]),
                }
                if "total_asset" in portfolio:
                    result["portfolio"]["total_asset"] = portfolio["total_asset"]

        else:
            # VecFastFinRL single state format (shouldn't happen for single env)
            tickers = state.get("tickers", [])
            result["tickers"] = tickers

            ohlc = state["ohlc"]  # (n_tickers, 4)
            indicators = state["indicators"]  # (n_tickers, n_ind)

            for i, ticker in enumerate(tickers):
                result[ticker] = {
                    "ohlc": ohlc[i],
                    "indicators": indicators[i],
                }

            if self.include_portfolio:
                result["portfolio"] = {
                    "cash": state["cash"],
                    "shares": state["shares"],
                    "avg_buy_price": state["avg_buy_price"],
                    "total_asset": state.get("total_asset", 0.0),
                }

            if self.include_macro and "macro_ohlc" in state:
                result["macro"] = {}
                for i, ticker in enumerate(self.macro_tickers):
                    result["macro"][ticker] = {
                        "ohlc": state["macro_ohlc"][i],
                        "indicators": state["macro_indicators"][i],
                    }
                result["macro_tickers"] = self.macro_tickers

        # Metadata
        result["day"] = state["day"]
        result["done"] = state.get("done", False)
        result["terminal"] = state.get("terminal", False)
        result["reward"] = state.get("reward", 0.0)
        result["indicator_names"] = self.indicator_names
        result["h"] = self.h

        return result

    def _process_vec(self, states: Union[List[Dict], Dict]) -> Dict[str, Any]:
        """Process VecFastFinRL states (batch dimension N).

        Supports both:
        - List[Dict] format (return_format='json')
        - Dict format with batched arrays (return_format='vec')
        """
        # Detect format: vec format has 'n_envs' key
        if isinstance(states, dict) and 'n_envs' in states:
            return self._process_vec_format(states)

        # List[Dict] format (json)
        N = len(states)
        result = {}

        # Collect all unique tickers
        all_tickers = set()
        tickers_per_env = []
        for s in states:
            tickers = s.get("tickers", [])
            tickers_per_env.append(tickers)
            all_tickers.update(tickers)

        unique_tickers = sorted(all_tickers)
        result["tickers"] = tickers_per_env  # List[List[str]] (N, n_tickers)
        result["unique_tickers"] = unique_tickers

        # First state to get dimensions
        first = states[0]
        n_tic = len(first.get("tickers", []))

        if self.h > 0:
            # h > 0: Use env's market window method
            time_len = self.h + 1

            for ticker in unique_tickers:
                ohlc_list = []
                ind_list = []
                mask_list = []

                for i, s in enumerate(states):
                    tickers = s.get("tickers", [])
                    if ticker in tickers:
                        day = s["day"]
                        window = self.env.get_market_window_numpy([ticker], day, self.h, 0)

                        # Combine past + current: (h+1, 4), (h+1, n_ind)
                        past_ohlc = window[ticker]["past_ohlc"]  # (h, 4)
                        current_ohlc = window[ticker]["current_ohlc"]  # (4,)
                        ohlc = np.vstack([past_ohlc, current_ohlc[np.newaxis, :]])

                        past_ind = window[ticker]["past_indicators"]  # (h, n_ind)
                        current_ind = window[ticker]["current_indicators"]  # (n_ind,)
                        ind = np.vstack([past_ind, current_ind[np.newaxis, :]])

                        mask = np.concatenate([
                            window[ticker]["past_mask"],
                            [window[ticker]["current_mask"]]
                        ])

                        ohlc_list.append(ohlc)
                        ind_list.append(ind)
                        mask_list.append(mask)
                    else:
                        # Ticker not in this env - fill with zeros
                        ohlc_list.append(np.zeros((time_len, 4)))
                        ind_list.append(np.zeros((time_len, self.n_indicators)))
                        mask_list.append(np.zeros(time_len, dtype=np.int32))

                result[ticker] = {
                    "ohlc": np.stack(ohlc_list),       # (N, h+1, 4)
                    "indicators": np.stack(ind_list),  # (N, h+1, n_ind)
                    "mask": np.stack(mask_list),       # (N, h+1)
                }

            # Portfolio
            if self.include_portfolio:
                result["portfolio"] = {
                    "cash": np.array([s["cash"] for s in states]),
                    "shares": np.stack([s["shares"] for s in states]),
                    "avg_buy_price": np.stack([s["avg_buy_price"] for s in states]),
                    "total_asset": np.array([s.get("total_asset", 0.0) for s in states]),
                }

            # Macro tickers
            if self.include_macro and self.n_macro > 0 and "macro_ohlc" in first:
                result["macro"] = {}
                for ticker in self.macro_tickers:
                    ohlc_list = []
                    ind_list = []
                    mask_list = []

                    for s in states:
                        day = s["day"]
                        window = self.env.get_market_window_numpy([ticker], day, self.h, 0)

                        past_ohlc = window[ticker]["past_ohlc"]
                        current_ohlc = window[ticker]["current_ohlc"]
                        ohlc = np.vstack([past_ohlc, current_ohlc[np.newaxis, :]])

                        past_ind = window[ticker]["past_indicators"]
                        current_ind = window[ticker]["current_indicators"]
                        ind = np.vstack([past_ind, current_ind[np.newaxis, :]])

                        mask = np.concatenate([
                            window[ticker]["past_mask"],
                            [window[ticker]["current_mask"]]
                        ])

                        ohlc_list.append(ohlc)
                        ind_list.append(ind)
                        mask_list.append(mask)

                    result["macro"][ticker] = {
                        "ohlc": np.stack(ohlc_list),
                        "indicators": np.stack(ind_list),
                        "mask": np.stack(mask_list),
                    }
                result["macro_tickers"] = self.macro_tickers

            # Metadata
            result["day"] = np.array([s["day"] for s in states])
            result["done"] = np.array([s.get("done", False) for s in states])
            result["terminal"] = np.array([s.get("terminal", False) for s in states])
            result["reward"] = np.array([s.get("reward", 0.0) for s in states])
            result["indicator_names"] = self.indicator_names
            result["h"] = self.h
            result["n_envs"] = N

            return result

        # h = 0: current data only
        for ticker in unique_tickers:
            ohlc_list = []
            ind_list = []

            for i, s in enumerate(states):
                tickers = s.get("tickers", [])
                if ticker in tickers:
                    idx = tickers.index(ticker)
                    ohlc_list.append(s["ohlc"][idx])  # (4,)
                    ind_list.append(s["indicators"][idx])  # (n_ind,)
                else:
                    # Ticker not in this env - fill with zeros
                    ohlc_list.append(np.zeros(4))
                    ind_list.append(np.zeros(self.n_indicators))

            result[ticker] = {
                "ohlc": np.stack(ohlc_list),  # (N, 4)
                "indicators": np.stack(ind_list),  # (N, n_ind)
            }

        # Macro tickers
        if self.include_macro and self.n_macro > 0 and "macro_ohlc" in first:
            result["macro"] = {}
            for i, ticker in enumerate(self.macro_tickers):
                ohlc_list = [s["macro_ohlc"][i] for s in states]
                ind_list = [s["macro_indicators"][i] for s in states]
                result["macro"][ticker] = {
                    "ohlc": np.stack(ohlc_list),  # (N, 4)
                    "indicators": np.stack(ind_list),  # (N, n_ind)
                }
            result["macro_tickers"] = self.macro_tickers

        # Portfolio: (N,), (N, n_tickers), (N, n_tickers)
        if self.include_portfolio:
            result["portfolio"] = {
                "cash": np.array([s["cash"] for s in states]),
                "shares": np.stack([s["shares"] for s in states]),
                "avg_buy_price": np.stack([s["avg_buy_price"] for s in states]),
                "total_asset": np.array([s.get("total_asset", 0.0) for s in states]),
            }

        # Metadata: (N,) arrays
        result["day"] = np.array([s["day"] for s in states])
        result["done"] = np.array([s.get("done", False) for s in states])
        result["terminal"] = np.array([s.get("terminal", False) for s in states])
        result["reward"] = np.array([s.get("reward", 0.0) for s in states])
        result["indicator_names"] = self.indicator_names
        result["h"] = self.h
        result["n_envs"] = N

        return result

    def _process_vec_format(self, state: Dict) -> Dict[str, Any]:
        """Process VecFastFinRL batched dict format (return_format='vec').

        If h=0: return state as-is (already in batched format)
        If h>0: add market window history to ohlc/indicators
        """
        if self.h == 0:
            # Already in vec format, just pass through with minimal additions
            result = dict(state)  # shallow copy
            result["indicator_names"] = self.indicator_names
            result["h"] = self.h
            return result

        # h > 0: Need to add historical data
        N = state['n_envs']
        n_tickers = state['n_tickers']
        time_len = self.h + 1
        days = np.asarray(state['day'])  # (N,)
        tickers_per_env = state['tickers']  # List[List[str]]

        # Build (N, time_len, n_tickers, 4) ohlc and (N, time_len, n_tickers, n_ind) indicators
        ohlc_out = np.zeros((N, time_len, n_tickers, 4))
        ind_out = np.zeros((N, time_len, n_tickers, self.n_indicators))
        mask_out = np.zeros((N, time_len, n_tickers), dtype=np.int32)

        for i in range(N):
            tickers = tickers_per_env[i]
            day = int(days[i])

            for t_idx, ticker in enumerate(tickers):
                window = self.env.get_market_window_numpy([ticker], day, self.h, 0)

                past_ohlc = window[ticker]["past_ohlc"]  # (h, 4)
                current_ohlc = window[ticker]["current_ohlc"]  # (4,)
                ohlc_out[i, :, t_idx, :] = np.vstack([past_ohlc, current_ohlc[np.newaxis, :]])

                past_ind = window[ticker]["past_indicators"]  # (h, n_ind)
                current_ind = window[ticker]["current_indicators"]  # (n_ind,)
                ind_out[i, :, t_idx, :] = np.vstack([past_ind, current_ind[np.newaxis, :]])

                mask_out[i, :, t_idx] = np.concatenate([
                    window[ticker]["past_mask"],
                    [window[ticker]["current_mask"]]
                ])

        # Build result
        result = {
            "ohlc": ohlc_out,              # (N, time_len, n_tickers, 4)
            "indicators": ind_out,          # (N, time_len, n_tickers, n_ind)
            "mask": mask_out,               # (N, time_len, n_tickers)
            "day": days,
            "cash": np.asarray(state["cash"]),
            "shares": np.asarray(state["shares"]),
            "avg_buy_price": np.asarray(state["avg_buy_price"]),
            "total_asset": np.asarray(state.get("total_asset", np.zeros(N))),
            "done": np.asarray(state.get("done", np.zeros(N, dtype=bool))),
            "terminal": np.asarray(state.get("terminal", np.zeros(N, dtype=bool))),
            "reward": np.asarray(state.get("reward", np.zeros(N))),
            "tickers": tickers_per_env,
            "n_envs": N,
            "n_tickers": n_tickers,
            "n_indicators": self.n_indicators,
            "indicator_names": self.indicator_names,
            "h": self.h,
        }

        # Macro if present
        if self.include_macro and self.n_macro > 0 and "macro_ohlc" in state:
            macro_ohlc_out = np.zeros((N, time_len, self.n_macro, 4))
            macro_ind_out = np.zeros((N, time_len, self.n_macro, self.n_indicators))
            macro_mask_out = np.zeros((N, time_len, self.n_macro), dtype=np.int32)

            for i in range(N):
                day = int(days[i])
                for m_idx, ticker in enumerate(self.macro_tickers):
                    window = self.env.get_market_window_numpy([ticker], day, self.h, 0)

                    past_ohlc = window[ticker]["past_ohlc"]
                    current_ohlc = window[ticker]["current_ohlc"]
                    macro_ohlc_out[i, :, m_idx, :] = np.vstack([past_ohlc, current_ohlc[np.newaxis, :]])

                    past_ind = window[ticker]["past_indicators"]
                    current_ind = window[ticker]["current_indicators"]
                    macro_ind_out[i, :, m_idx, :] = np.vstack([past_ind, current_ind[np.newaxis, :]])

                    macro_mask_out[i, :, m_idx] = np.concatenate([
                        window[ticker]["past_mask"],
                        [window[ticker]["current_mask"]]
                    ])

            result["macro_ohlc"] = macro_ohlc_out
            result["macro_indicators"] = macro_ind_out
            result["macro_mask"] = macro_mask_out
            result["n_macro"] = self.n_macro
            result["macro_tickers"] = self.macro_tickers

        return result

    def __call__(self, state: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Alias for process()."""
        return self.process(state)


def preprocess(
    env: Any,
    state: Union[Dict, List[Dict]],
    h: int = 0,
    include_portfolio: bool = True,
    include_macro: bool = True,
) -> Dict[str, Any]:
    """
    Functional interface for preprocessing.

    Args:
        env: FastFinRL or VecFastFinRL instance
        state: State dict or list of state dicts
        h: History length (0 = current only)
        include_portfolio: Include portfolio data
        include_macro: Include macro ticker data

    Returns:
        Processed observation dict

    Example:
        ```python
        from fast_finrl_py import FastFinRL, VecFastFinRL
        from preprocessor import preprocess, ObservationPreprocessor

        # === Single env ===
        env = FastFinRL("data.csv")
        state = env.reset(["AAPL", "GOOGL"], seed=42)
        obs = preprocess(env, state)
        # obs["AAPL"]["ohlc"] shape: (4,)
        # obs["AAPL"]["indicators"] shape: (n_ind,)
        # obs["portfolio"]["cash"]: float
        # obs["portfolio"]["shares"] shape: (2,)

        # With history (h > 0)
        obs = preprocess(env, state, h=10)
        # obs["AAPL"]["ohlc"] shape: (11, 4)
        # obs["AAPL"]["indicators"] shape: (11, n_ind)
        # obs["AAPL"]["mask"] shape: (11,)

        # === Vec env ===
        vec_env = VecFastFinRL("data.csv")
        tickers_list = [["AAPL", "GOOGL"], ["MSFT", "AMZN"]]
        states = vec_env.reset(tickers_list, np.arange(2, dtype=np.int64))
        obs = preprocess(vec_env, states)
        # obs["AAPL"]["ohlc"] shape: (2, 4)  # N=2 envs
        # obs["AAPL"]["indicators"] shape: (2, n_ind)
        # obs["portfolio"]["cash"] shape: (2,)
        # obs["portfolio"]["shares"] shape: (2, 2)  # (N, n_tickers)
        # obs["tickers"]: [["AAPL", "GOOGL"], ["MSFT", "AMZN"]]
        # obs["unique_tickers"]: ["AAPL", "AMZN", "GOOGL", "MSFT"]

        # === Reusable preprocessor ===
        preprocessor = ObservationPreprocessor(vec_env, h=0)
        for step in range(100):
            obs = preprocessor(states)
            actions = policy(obs)
            states = vec_env.step(actions)
        ```
    """
    preprocessor = ObservationPreprocessor(
        env, h=h, include_portfolio=include_portfolio, include_macro=include_macro
    )
    return preprocessor(state)
