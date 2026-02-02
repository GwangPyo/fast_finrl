#!/usr/bin/env python3
"""Simple downloader for 39 tickers (9 original + 30 Dow Jones)"""
import pandas as pd
import yfinance as yf
from stockstats import wrap

# 9 original tickers
ORIGINAL = ['XLE', 'GLD', 'SLV', 'TLT', 'TIP', 'JNK', 'SPY', 'QQQ', 'SOXX']

# Dow Jones 30
DOW_30 = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

ALL_TICKERS = list(set(ORIGINAL + DOW_30))
print(f"Downloading {len(ALL_TICKERS)} tickers...")

INDICATORS = [
    "close_10_ema", "close_5_ema", "close_20_ema", "close_50_ema", "close_100_ema",
    "close_20_sma", "boll", "boll_ub", "boll_lb", "atr", "atr_14",
    "adx", "adxr", "cci_7", "cci_14", "mfi", "pdi", "ndi",
    "close_10_roc", "macd", "rsi_7", "rsi_14", "mfi_14",
]

# Download data
data = yf.download(ALL_TICKERS, start='2019-01-01', end='2024-12-31', group_by='ticker', auto_adjust=True)
print(f"Downloaded shape: {data.shape}")

# Convert to long format
dfs = []
for tic in ALL_TICKERS:
    try:
        if len(ALL_TICKERS) > 1:
            df = data[tic].copy()
        else:
            df = data.copy()
        df['tic'] = tic
        df = df.reset_index()
        df.columns = [c.lower() if c != 'Date' else 'date' for c in df.columns]
        df = df.rename(columns={'date': 'date'})
        if 'date' not in df.columns and 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        dfs.append(df)
    except Exception as e:
        print(f"Error processing {tic}: {e}")

df = pd.concat(dfs, ignore_index=True)
df = df.dropna()
print(f"Combined shape: {df.shape}")

# Add technical indicators using stockstats
print("Adding technical indicators...")
result_dfs = []
for tic in df['tic'].unique():
    tic_df = df[df['tic'] == tic].copy().sort_values('date')
    stock = wrap(tic_df)

    for ind in INDICATORS:
        try:
            _ = stock[ind]
        except:
            pass

    result_dfs.append(stock)

df = pd.concat(result_dfs, ignore_index=True)

# Create day column (dense ranking)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date', 'tic'])
date_to_day = {d: i for i, d in enumerate(sorted(df['date'].unique()))}
df['day'] = df['date'].map(date_to_day)
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Shift indicators (no future data leakage)
cols_to_shift = ['open', 'high', 'low', 'close', 'volume']
for c in cols_to_shift:
    df[f'prev_{c}'] = df.groupby('tic')[c].shift(1)

for ind in INDICATORS:
    if ind in df.columns:
        df[ind] = df.groupby('tic')[ind].shift(1)

df = df.dropna()
print(f"Final shape: {df.shape}")
print(f"Tickers: {sorted(df['tic'].unique())}")
print(f"Ticker count: {len(df['tic'].unique())}")

# Save
df.to_csv('data/raw_39_tickers.csv', index=False)
print("Saved to data/raw_39_tickers.csv")
