
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BENCHMARK_SYMBOL, TRADING_DAYS_PER_YEAR


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()

    # Base returns
    df["price_lag1"] = df.groupby("symbol")["close"].shift(1)
    df["daily_return_decimal"] = df.groupby("symbol")["close"].pct_change()
    df["daily_return_pct"] = df["daily_return_decimal"] * 100.0

    # Trend
    df["ma_7"] = df.groupby("symbol")["close"].transform(
        lambda s: s.rolling(window=7, min_periods=7).mean()
    )
    df["ma_30"] = df.groupby("symbol")["close"].transform(
        lambda s: s.rolling(window=30, min_periods=30).mean()
    )
    df["ma_spread_pct"] = ((df["ma_7"] / df["ma_30"]) - 1.0) * 100.0

    # Volatility
    df["volatility_7d"] = df.groupby("symbol")["daily_return_pct"].transform(
        lambda s: s.rolling(window=7, min_periods=7).std()
    )
    df["volatility_30d_annualized"] = df.groupby("symbol")["daily_return_decimal"].transform(
        lambda s: s.rolling(window=30, min_periods=30).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    )

    # Volume
    vol_mean_30 = df.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(window=30, min_periods=30).mean()
    )
    vol_std_30 = df.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(window=30, min_periods=30).std()
    )
    df["volume_z_30"] = (df["volume"] - vol_mean_30) / vol_std_30.replace(0, np.nan)

    # RSI(14)
    delta = df.groupby("symbol")["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.groupby(df["symbol"]).transform(
        lambda s: s.rolling(window=14, min_periods=14).mean()
    )
    avg_loss = loss.groupby(df["symbol"]).transform(
        lambda s: s.rolling(window=14, min_periods=14).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_14"] = 50

    # Cumulative return
    df["cumulative_return_pct"] = df.groupby("symbol")["daily_return_decimal"].transform(
        lambda s: ((1 + s.fillna(0)).cumprod() - 1) * 100.0
    )

    # Drawdown
    rolling_peak = df.groupby("symbol")["close"].transform(lambda s: s.cummax())
    df["drawdown_pct"] = ((df["close"] - rolling_peak) / rolling_peak) * 100.0

    # Rolling risk-adjusted return
    rolling_mean_30 = df.groupby("symbol")["daily_return_decimal"].transform(
        lambda s: s.rolling(window=30, min_periods=30).mean()
    )
    rolling_std_30 = df.groupby("symbol")["daily_return_decimal"].transform(
        lambda s: s.rolling(window=30, min_periods=30).std()
    )
    df["rolling_risk_adjusted_30"] = (
        (rolling_mean_30 / rolling_std_30.replace(0, np.nan)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    # Multi-period returns
    df["return_7d_pct"] = df.groupby("symbol")["close"].transform(
        lambda s: ((s / s.shift(7)) - 1) * 100.0
    )
    df["return_30d_pct"] = df.groupby("symbol")["close"].transform(
        lambda s: ((s / s.shift(30)) - 1) * 100.0
    )
    df["return_90d_pct"] = df.groupby("symbol")["close"].transform(
        lambda s: ((s / s.shift(90)) - 1) * 100.0
    )

    # Regime
    df["trend_regime"] = np.where(
        df["ma_7"] > df["ma_30"],
        "bullish",
        np.where(df["ma_7"] < df["ma_30"], "bearish", "neutral")
    )

    # Benchmark-relative features
    bench = df[df["symbol"] == BENCHMARK_SYMBOL][
        ["date", "return_30d_pct", "return_90d_pct", "ma_spread_pct"]
    ].copy()

    bench = bench.rename(
        columns={
            "return_30d_pct": "benchmark_return_30d_pct",
            "return_90d_pct": "benchmark_return_90d_pct",
            "ma_spread_pct": "benchmark_ma_spread_pct",
        }
    )

    df = df.merge(bench, on="date", how="left")

    df["rel_return_30d_pct"] = df["return_30d_pct"] - df["benchmark_return_30d_pct"]
    df["rel_return_90d_pct"] = df["return_90d_pct"] - df["benchmark_return_90d_pct"]

    # Clean self-reference for benchmark symbol itself
    bench_mask = df["symbol"] == BENCHMARK_SYMBOL
    df.loc[bench_mask, "rel_return_30d_pct"] = 0.0
    df.loc[bench_mask, "rel_return_90d_pct"] = 0.0

    return df
