import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, GroupByScaler, data_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf

TRAIN_START_DATE = '2019-05-01'
TRAIN_END_DATE = '2024-12-31'
VALID_START_DATE = '2025-01-01'
VALID_END_DATE = '2025-06-30'
TEST_START_DATE = '2025-06-30'
TEST_END_DATE = '2025-11-14'

INDICATORS = [
    "close_10_ema",
    "close_5_ema",
    "close_20_ema",
    "close_50_ema",
    "close_100_ema",
    'close_20_sma',  # sma_20
    "boll", "boll_ub", "boll_lb",
    "atr", "atr_14",
    "adx", "adxr",
    "cci_7", "cci_14",
    "mfi",
    "pdi", "ndi",
    "close_10_roc",
    'macd', 'rsi_7', 'rsi_14',   'mfi_14',
]
'''
    "rsi_7",
    "rsi_14",
    "cci_7",
    "cci_14",
    "mfi_14",
    "adxr_14",
    "close_10_ema",
    "close_5_ema",
    "close_20_ema",
    "close_50_ema",
    "close_100_ema",
'''

asset_class = { 'Commodities': ['XLE', 'GLD', 'SLV'],
                'Bonds': ['TLT', 'TIP', 'JNK'],
                'Equities': ['SPY', 'QQQ', 'SOXX']
                }

# Dow Jones 30 stocks
DOW_30 = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

# Combine: 9 original + 30 Dow Jones = 39 tickers
tics = list(set(sum(list(asset_class.values()), []) + DOW_30))


class YahooDownloaderHours(YahooDownloader):

    def fetch_data(self, proxy=None, auto_adjust=False) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                interval='1h',
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                auto_adjust=auto_adjust,
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

            if not auto_adjust:
                data_df = self._adjust_prices(data_df)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)

        data_df["date"] = data_df["Datetime"].dt.date
        data_df["Time"] = data_df["Datetime"].dt.time

        data_df["day"] = data_df["Datetime"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.Datetime.apply(lambda x: x.strftime("%Y-%m-%d-%H-%M"))
        # drop missing data
        # data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df


def download():
    df = YahooDownloader(start_date='2016-01-01',
                         end_date='2025-11-15',
                         ticker_list=tics).fetch_data()
    df_gold = YahooDownloader(start_date='2016-12-01',
                              end_date='2025-11-15',
                              ticker_list=['GC=F']).fetch_data()
    return df, df_gold


def load(path_df='./asset_df.csv', path_gold='./gold_df.csv'):
    """
    try:
        df = pd.read_csv(path_df)
        df_gold = pd.read_csv(path_gold)
    except FileNotFoundError:
    """
    df, df_gold = download()
    df.to_csv("asset_df.csv", index=False)
    df_gold.to_csv('gold_df.csv', index=False)
    return df, df_gold


class LogStandardScaler(StandardScaler):
    def fit(self, X, y=None):
        return super().fit(np.log1p(X))

    def transform(self, X):
        return super().transform(np.log1p(X))

    def inverse_transform(self, X_scaled):
        return np.expm1(super().inverse_transform(X_scaled))


if __name__ == '__main__':
    df, df_gold = load()
    print(df.tic.unique())

    gc_open = df_gold[["date", "open"]].rename(columns={ "open": "gc_open" })
    gc_open['gc_30_ema'] = gc_open['gc_open'].ewm(span=30, adjust=False).mean()
    gc_open['gc_30_ewmstd'] = gc_open['gc_open'].ewm(span=30, adjust=False).std(bias=False)

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_vix=False,
                         use_turbulence=False,
                         user_defined_feature=False)
    df = fe.preprocess_data(df)
    cols = ["open", "high", "low", "close", "volume"]
    """
    for c in cols:
        # deflate and standardization to moving average and std
        df["gold_norm_" + c] = (df[c] - df['gc_30_ema']) / np.sqrt(df['gc_30_ewmstd'] * df["gc_open"])
    emas = ["close_10_ema", "close_5_ema", "close_20_ema", "close_50_ema", "close_100_ema"]

    for e in emas:
        df[e] = (df[e] - df['gc_30_ema']) / np.sqrt(df['gc_30_ewmstd'] * df["gc_open"])
    """

    tech_indicators = [    "close_10_ema",
                           "close_5_ema",
                           "close_20_ema",
                           "close_50_ema",
                           "close_100_ema",
                           'close_20_sma',  # sma_20
                           "boll", "boll_ub", "boll_lb",
                           "atr", "atr_14",
                           "adx", "adxr",
                           "cci_7", "cci_14",
                           "mfi",
                           "pdi", "ndi",
                           "close_10_roc",
                           'macd', 'rsi_7', 'rsi_14',   'mfi_14',
                           ]

    # 'gold_norm_high', 'gold_norm_low', 'gold_norm_close', 'gc_open']

    tech_indicator_val = df[tech_indicators].values
    # shift technical indicators so that a model cannot leverage future data
    renames = ['prev_' + c for c in cols]
    df[tech_indicators] = df.groupby('tic', sort=False)[tech_indicators].shift(1)
    df[renames] = df.groupby('tic', sort=False)[cols].shift(1)
    # df['volume'] = np.log1p(df['volume']) / 5
    # df = df.rename(columns={ "volume": "log_volume",})
    log_transform_targets = [    "close_10_ema",
                                 "close_5_ema",
                                 "close_20_ema",
                                 "close_50_ema",
                                 "close_100_ema",
                                 'close_20_sma',  # sma_20
                                 "boll", "boll_ub", "boll_lb",
                                 'tic']
    rename_log_scale_target = renames
    renamed = ['log_' + v for v in rename_log_scale_target]
    renamed += ['tic']
    rename_log_scale_target = rename_log_scale_target + ['tic']
    linear_transform_targets = [
        "atr", "atr_14",
        "adx", "adxr",
        "cci_7", "cci_14",
        "mfi",
        "pdi", "ndi",
        "close_10_roc",
        'macd', 'rsi_7', 'rsi_14', 'mfi_14', 'tic'
    ]

    train_df = data_split(df, TRAIN_START_DATE,
                          TRAIN_END_DATE)  # truncation happens. The leakage point (the first day) is truncated out.
    log_scaler = GroupByScaler(by='tic', scaler=LogStandardScaler)
    rename_log_scaler = GroupByScaler(by='tic', scaler=LogStandardScaler)
    linear_scaler = GroupByScaler(by='tic', scaler=StandardScaler)

    # train_df[log_transform_targets] = log_scaler.fit_transform(train_df[log_transform_targets])

    # train_df[renamed] = rename_log_scaler.fit_transform(train_df[rename_log_scale_target])

    # train_df[linear_transform_targets] = linear_scaler.fit_transform(train_df[linear_transform_targets])

    valid_df = data_split(df, VALID_START_DATE, VALID_END_DATE)
    # valid_df[log_transform_targets] = log_scaler.transform(valid_df[log_transform_targets])
    # valid_df[renamed] = rename_log_scaler.transform(valid_df[rename_log_scale_target])
    # valid_df[linear_transform_targets] = linear_scaler.transform(valid_df[linear_transform_targets])

    test_df = data_split(df, TEST_START_DATE, TEST_END_DATE)
    # test_df[log_transform_targets] = log_scaler.transform(test_df[log_transform_targets])
    # test_df[renamed] = rename_log_scaler.transform(test_df[rename_log_scale_target])
    # test_df[linear_transform_targets] = linear_scaler.transform(test_df[linear_transform_targets])

    train_df.to_csv("raw_train_df.csv", index=False)
    print(train_df[train_df.tic == 'SOXX'])
    print(train_df.isna().any())
    valid_df.to_csv("raw_valid_df.csv", index=False)
    print(valid_df.isna().any())
    test_df.to_csv("raw_test_df.csv", index=False)
    print(test_df.isna().any())
