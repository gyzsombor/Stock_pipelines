
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MAX_WEIGHT_PER_ASSET, ROLLING_OPT_HOLD_WINDOW, ROLLING_OPT_TRAIN_WINDOW, TRADING_DAYS_PER_YEAR


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, MAX_WEIGHT_PER_ASSET)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def equal_weight_weights(symbols: list[str]) -> pd.DataFrame:
    w = np.repeat(1.0 / len(symbols), len(symbols))
    return pd.DataFrame({"symbol": symbols, "weight": w})


def min_vol_weights(returns_df: pd.DataFrame) -> pd.DataFrame:
    cov = returns_df.cov().values
    n = cov.shape[0]
    diag = np.diag(cov)
    inv_diag = np.where(diag > 0, 1.0 / diag, 0.0)
    w = _normalize_weights(inv_diag)
    return pd.DataFrame({"symbol": returns_df.columns.tolist(), "weight": w})


def max_sharpe_like_weights(returns_df: pd.DataFrame) -> pd.DataFrame:
    mu = returns_df.mean().values * TRADING_DAYS_PER_YEAR
    cov = returns_df.cov().values * TRADING_DAYS_PER_YEAR
    diag = np.diag(cov)
    score = np.where(diag > 0, mu / np.sqrt(diag), 0.0)
    score = np.clip(score, 0, None)
    if score.sum() <= 0:
        score = np.ones_like(score)
    w = _normalize_weights(score)
    return pd.DataFrame({"symbol": returns_df.columns.tolist(), "weight": w})


def rolling_optimized_portfolio(
    returns_df: pd.DataFrame,
    mode: str = "max_sharpe_like",
    train_window: int = ROLLING_OPT_TRAIN_WINDOW,
    hold_window: int = ROLLING_OPT_HOLD_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode not in {"equal_weight", "min_vol", "max_sharpe_like"}:
        raise ValueError("mode must be equal_weight, min_vol, or max_sharpe_like")

    if len(returns_df) < train_window + hold_window:
        raise ValueError("Not enough rows for rolling optimization.")

    results = []
    weight_rows = []
    cols = returns_df.columns.tolist()

    start = 0
    while start + train_window < len(returns_df):
        train_slice = returns_df.iloc[start:start + train_window].copy()
        hold_slice = returns_df.iloc[start + train_window:start + train_window + hold_window].copy()
        if hold_slice.empty:
            break

        if mode == "equal_weight":
            w_df = equal_weight_weights(cols)
        elif mode == "min_vol":
            w_df = min_vol_weights(train_slice)
        else:
            w_df = max_sharpe_like_weights(train_slice)

        w = w_df.set_index("symbol")["weight"].reindex(cols).fillna(0.0).values
        hold_returns = hold_slice.values @ w

        for dt, ret in zip(hold_slice.index, hold_returns):
            results.append({"date": dt, "portfolio_return_decimal": float(ret), "mode": mode})

        for _, row in w_df.iterrows():
            weight_rows.append({
                "rebalance_date": hold_slice.index.min(),
                "symbol": row["symbol"],
                "weight": float(row["weight"]),
                "mode": mode,
            })

        start += hold_window

    result_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    if not result_df.empty:
        result_df["portfolio_equity"] = (1.0 + result_df["portfolio_return_decimal"]).cumprod()

    weights_df = pd.DataFrame(weight_rows)
    return result_df, weights_df
