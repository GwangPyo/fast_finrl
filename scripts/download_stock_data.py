#!/usr/bin/env python3
"""
Download stock data using yfinance and compute technical indicators.
No finrl dependency required.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime
import sys
import os

# Tickers to download
TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

# Date range
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'


def download_data(tickers, start_date, end_date):
    """Download OHLCV data from Yahoo Finance."""
    all_data = []

    for tic in tickers:
        print(f"Downloading {tic}...")
        try:
            df = yf.download(tic, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"  Warning: No data for {tic}")
                continue

            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df['tic'] = tic
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })
            all_data.append(df)
            print(f"  Downloaded {len(df)} rows")
        except Exception as e:
            print(f"  Error downloading {tic}: {e}")

    if not all_data:
        raise ValueError("No data downloaded!")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['date', 'tic']).reset_index(drop=True)

    # Add day index
    dates = combined['date'].unique()
    date_to_day = {d: i for i, d in enumerate(sorted(dates))}
    combined['day'] = combined['date'].map(date_to_day)

    # Convert date to string
    combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')

    return combined


def add_technical_indicators(df):
    """Add technical indicators using ta library."""
    print("Computing technical indicators...")

    result_dfs = []

    for tic in df['tic'].unique():
        tic_df = df[df['tic'] == tic].copy().sort_values('day')

        close = tic_df['close']
        high = tic_df['high']
        low = tic_df['low']
        volume = tic_df['volume']

        # Moving averages
        tic_df['close_5_ema'] = ta.trend.ema_indicator(close, window=5)
        tic_df['close_10_ema'] = ta.trend.ema_indicator(close, window=10)
        tic_df['close_20_ema'] = ta.trend.ema_indicator(close, window=20)
        tic_df['close_50_ema'] = ta.trend.ema_indicator(close, window=50)
        tic_df['close_20_sma'] = ta.trend.sma_indicator(close, window=20)

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        tic_df['boll'] = bb.bollinger_mavg()
        tic_df['boll_ub'] = bb.bollinger_hband()
        tic_df['boll_lb'] = bb.bollinger_lband()

        # ATR
        tic_df['atr_14'] = ta.volatility.average_true_range(high, low, close, window=14)

        # RSI
        tic_df['rsi_7'] = ta.momentum.rsi(close, window=7)
        tic_df['rsi_14'] = ta.momentum.rsi(close, window=14)

        # MACD
        macd = ta.trend.MACD(close)
        tic_df['macd'] = macd.macd()
        tic_df['macd_signal'] = macd.macd_signal()
        tic_df['macd_hist'] = macd.macd_diff()

        # CCI
        tic_df['cci_14'] = ta.trend.cci(high, low, close, window=14)

        # MFI
        tic_df['mfi_14'] = ta.volume.money_flow_index(high, low, close, volume, window=14)

        # ADX
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        tic_df['adx'] = adx.adx()
        tic_df['pdi'] = adx.adx_pos()
        tic_df['ndi'] = adx.adx_neg()

        # ROC
        tic_df['close_10_roc'] = ta.momentum.roc(close, window=10)

        result_dfs.append(tic_df)

    result = pd.concat(result_dfs, ignore_index=True)
    result = result.sort_values(['day', 'tic']).reset_index(drop=True)

    # Drop NaN rows (first ~50 days due to indicators)
    n_before = len(result)
    result = result.dropna().reset_index(drop=True)
    n_after = len(result)
    print(f"  Dropped {n_before - n_after} rows with NaN values")

    # Recalculate day index after dropping NaN
    dates = result['date'].unique()
    date_to_day = {d: i for i, d in enumerate(sorted(dates))}
    result['day'] = result['date'].map(date_to_day)

    return result


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'data/stock_data.csv'

    print(f"Downloading data for: {TICKERS}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    # Download
    df = download_data(TICKERS, START_DATE, END_DATE)
    print(f"Downloaded {len(df)} total rows, {len(df['tic'].unique())} tickers")

    # Add indicators
    df = add_technical_indicators(df)

    # Reorder columns
    cols = ['day', 'date', 'tic', 'open', 'high', 'low', 'close', 'volume']
    indicator_cols = [c for c in df.columns if c not in cols + ['adj_close']]
    df = df[cols + sorted(indicator_cols)]

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Tickers: {sorted(df['tic'].unique())}")
    print(f"Day range: {df['day'].min()} - {df['day'].max()}")


if __name__ == '__main__':
    main()
