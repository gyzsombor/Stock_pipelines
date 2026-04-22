import numpy as np
import pandas as pd


def add_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    trend_up = out["ma_10"] > out["ma_20"]
    trend_down = out["ma_10"] < out["ma_20"]

    strong_trend_up = out["ma_20"] > out["ma_50"]
    strong_trend_down = out["ma_20"] < out["ma_50"]

    rsi_bull = (out["rsi_14"] >= 50) & (out["rsi_14"] <= 70)
    rsi_bear = out["rsi_14"] < 45
    rsi_overbought = out["rsi_14"] >= 75

    rel_bull = out["asset_vs_spy_20d"] > 0
    rel_bear = out["asset_vs_spy_20d"] < 0

    vol_ok = out["volatility_20d_annualized"] < 35
    vol_high = out["volatility_20d_annualized"] >= 35

    score = np.zeros(len(out), dtype=float)

    score += trend_up.astype(int) * 1.5
    score += strong_trend_up.astype(int) * 1.5
    score += rsi_bull.astype(int) * 1.0
    score += rel_bull.astype(int) * 1.0
    score += vol_ok.astype(int) * 0.5

    score -= trend_down.astype(int) * 1.5
    score -= strong_trend_down.astype(int) * 1.5
    score -= rsi_bear.astype(int) * 1.0
    score -= rel_bear.astype(int) * 1.0
    score -= vol_high.astype(int) * 0.5
    score -= rsi_overbought.astype(int) * 0.5

    out["signal_score"] = score

    out["signal"] = np.select(
        [
            out["signal_score"] >= 3.0,
            out["signal_score"] >= 1.5,
            out["signal_score"] <= -3.0,
            out["signal_score"] <= -1.5,
        ],
        [
            "STRONG_BUY",
            "BUY",
            "STRONG_SELL",
            "SELL",
        ],
        default="HOLD",
    )

    out["trend_regime"] = np.select(
        [
            strong_trend_up,
            strong_trend_down,
        ],
        [
            "Bullish trend",
            "Bearish trend",
        ],
        default="Sideways / mixed",
    )

    return out