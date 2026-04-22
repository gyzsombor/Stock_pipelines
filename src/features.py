from __future__ import annotations

import numpy as np
import pandas as pd

from config import ASSET_CLASS_MAP, BENCHMARK_SYMBOL, TRADING_DAYS_PER_YEAR


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = _safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def _asset_class_flags(symbol: str) -> dict:
    cls = ASSET_CLASS_MAP.get(symbol, "unknown")
    return {
        "is_equity": 1 if cls == "equity" else 0,
        "is_fund": 1 if cls == "fund" else 0,
        "is_crypto": 1 if cls == "crypto" else 0,
        "is_commodity": 1 if cls == "commodity" else 0,
    }


def _add_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["market_regime_bull"] = ((out["ma_spread_pct"] > 1.0) & (out["asset_vs_spy_20d"] > 0)).astype(int)
    out["market_regime_bear"] = ((out["ma_spread_pct"] < -1.0) & (out["asset_vs_spy_20d"] < 0)).astype(int)
    out["market_regime_sideways"] = (
        (out["market_regime_bull"] == 0) & (out["market_regime_bear"] == 0)
    ).astype(int)

    out["vol_regime_high"] = (out["volatility_20d_annualized"] >= 35).astype(int)
    out["vol_regime_normal"] = (out["volatility_20d_annualized"] < 35).astype(int)

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    benchmark = out[out["symbol"] == BENCHMARK_SYMBOL][["date", "close"]].copy()
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    benchmark["spy_return_1d"] = benchmark["close"].pct_change()
    benchmark["spy_return_5d"] = benchmark["close"].pct_change(5)
    benchmark["spy_return_20d"] = benchmark["close"].pct_change(20)
    benchmark["spy_volatility_20d"] = benchmark["spy_return_1d"].rolling(20).std()
    benchmark = benchmark[
        ["date", "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_volatility_20d"]
    ]

    pieces = []

    for symbol, g in out.groupby("symbol"):
        temp = g.sort_values("date").copy()

        temp["daily_return_decimal"] = temp["close"].pct_change()
        temp["daily_return_pct"] = temp["daily_return_decimal"] * 100

        temp["return_3d_pct"] = temp["close"].pct_change(3) * 100
        temp["return_5d_pct"] = temp["close"].pct_change(5) * 100
        temp["return_10d_pct"] = temp["close"].pct_change(10) * 100
        temp["return_20d_pct"] = temp["close"].pct_change(20) * 100

        temp["ma_10"] = temp["close"].rolling(10).mean()
        temp["ma_20"] = temp["close"].rolling(20).mean()
        temp["ma_50"] = temp["close"].rolling(50).mean()

        temp["ma_spread_pct"] = _safe_divide(temp["ma_10"] - temp["ma_20"], temp["ma_20"]) * 100
        temp["trend_strength_20_50"] = _safe_divide(temp["ma_20"] - temp["ma_50"], temp["ma_50"]) * 100

        temp["volatility_10d_decimal"] = temp["daily_return_decimal"].rolling(10).std()
        temp["volatility_20d_decimal"] = temp["daily_return_decimal"].rolling(20).std()

        temp["volatility_10d_annualized"] = temp["volatility_10d_decimal"] * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        temp["volatility_20d_annualized"] = temp["volatility_20d_decimal"] * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        temp["volatility_ratio_10_20"] = _safe_divide(
            temp["volatility_10d_annualized"], temp["volatility_20d_annualized"]
        )

        temp["rsi_14"] = _rsi(temp["close"], 14)

        rolling_peak = temp["close"].cummax()
        temp["drawdown_pct"] = ((temp["close"] / rolling_peak) - 1.0) * 100

        temp["volume_ma_30"] = temp["volume"].rolling(30).mean()
        temp["volume_std_30"] = temp["volume"].rolling(30).std()
        temp["volume_z_30"] = _safe_divide(temp["volume"] - temp["volume_ma_30"], temp["volume_std_30"])

        temp = temp.merge(benchmark, on="date", how="left")

        temp["asset_vs_spy_1d"] = (temp["daily_return_decimal"] - temp["spy_return_1d"]) * 100
        temp["asset_vs_spy_5d"] = ((temp["return_5d_pct"] / 100) - temp["spy_return_5d"]) * 100
        temp["asset_vs_spy_20d"] = ((temp["return_20d_pct"] / 100) - temp["spy_return_20d"]) * 100

        # rolling beta-like and correlation to benchmark
        rolling_cov = temp["daily_return_decimal"].rolling(20).cov(temp["spy_return_1d"])
        rolling_var = temp["spy_return_1d"].rolling(20).var()
        temp["beta_like_20d"] = _safe_divide(rolling_cov, rolling_var)
        temp["corr_to_spy_20d"] = temp["daily_return_decimal"].rolling(20).corr(temp["spy_return_1d"])

        flags = _asset_class_flags(symbol)
        for k, v in flags.items():
            temp[k] = v

        pieces.append(temp)

    featured = pd.concat(pieces, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    featured = _add_regime_flags(featured)

    # backward compatibility
    featured["volatility_30d_annualized"] = featured["volatility_20d_annualized"]

    return featured


def add_post_news_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # event flags from merged news columns
    out["macro_event_flag_3d"] = (out["news_macro_event_count_3d"] > 0).astype(int)
    out["earnings_event_flag_3d"] = (
        (out.get("news_company_event_count_3d", 0) > 0) & (out.get("news_high_impact_count_3d", 0) > 0)
    ).astype(int)
    out["company_event_flag_3d"] = (out["news_company_event_count_3d"] > 0).astype(int)
    out["high_impact_flag_3d"] = (out["news_high_impact_count_3d"] > 0).astype(int)
    out["market_macro_flag_3d"] = (out["market_macro_event_count_3d"] > 0).astype(int)

    return out