
from __future__ import annotations

import numpy as np
import pandas as pd


def add_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bullish = df["ma_7"] > df["ma_30"]
    bearish = df["ma_7"] < df["ma_30"]

    low_vol = df["volatility_30d_annualized"] < 25
    med_vol = (df["volatility_30d_annualized"] >= 25) & (df["volatility_30d_annualized"] < 45)
    high_vol = df["volatility_30d_annualized"] >= 45

    df["rsi_status"] = np.select(
        [
            df["rsi_14"] < 30,
            df["rsi_14"].between(30, 70, inclusive="both"),
            df["rsi_14"] > 70,
        ],
        [
            "oversold",
            "neutral",
            "overbought",
        ],
        default="unknown",
    )

    df["volatility_status"] = np.select(
        [
            low_vol,
            med_vol,
            high_vol,
        ],
        [
            "low",
            "medium",
            "high",
        ],
        default="unknown",
    )

    score = np.zeros(len(df), dtype=float)

    # Trend / structure
    score += np.where(bullish, 2, 0)
    score += np.where(bearish, -2, 0)
    score += np.where(df["ma_spread_pct"] > 1.0, 1, 0)
    score += np.where(df["ma_spread_pct"] < -1.0, -1, 0)

    # Momentum
    score += np.where(df["return_30d_pct"] > 5, 1, 0)
    score += np.where(df["return_30d_pct"] < -5, -1, 0)
    score += np.where(df["rsi_14"].between(55, 70, inclusive="both"), 1, 0)
    score += np.where(df["rsi_14"].between(30, 45, inclusive="both"), -1, 0)
    score += np.where(df["rsi_14"] > 75, -1, 0)
    score += np.where(df["rsi_14"] < 25, 1, 0)

    # Risk
    score += np.where(low_vol, 1, 0)
    score += np.where(med_vol, 0, 0)
    score += np.where(high_vol, -2, 0)

    # Participation / confirmation
    score += np.where((df["volume_z_30"] > 1.0) & (df["daily_return_pct"] > 0), 1, 0)
    score += np.where((df["volume_z_30"] > 1.0) & (df["daily_return_pct"] < 0), -1, 0)

    df["signal_score"] = score.astype(int)

    df["signal"] = np.select(
        [
            df["signal_score"] >= 5,
            df["signal_score"].between(2, 4, inclusive="both"),
            df["signal_score"].between(-1, 1, inclusive="both"),
            df["signal_score"].between(-4, -2, inclusive="both"),
            df["signal_score"] <= -5,
        ],
        [
            "STRONG_BUY",
            "BUY",
            "HOLD",
            "SELL",
            "STRONG_SELL",
        ],
        default="HOLD",
    )

    # If volatility is too high and conviction is weak, downgrade to watch states
    weak_conviction = df["signal_score"].between(-2, 2, inclusive="both")
    df.loc[high_vol & weak_conviction & (df["signal_score"] >= 0), "signal"] = "WATCH"
    df.loc[high_vol & weak_conviction & (df["signal_score"] < 0), "signal"] = "WATCH_RISK"

    abs_score = df["signal_score"].abs()
    df["signal_strength"] = np.select(
        [
            abs_score >= 5,
            abs_score >= 2,
            abs_score < 2,
        ],
        [
            "strong",
            "moderate",
            "weak",
        ],
        default="weak",
    )

    return df
