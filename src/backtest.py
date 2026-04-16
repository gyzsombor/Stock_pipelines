
from __future__ import annotations

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR


def _signal_to_position(signal: pd.Series, strategy_mode: str = "long_only") -> pd.Series:
    mapping = {
        "STRONG_BUY": 1.0,
        "BUY": 1.0,
        "HOLD": 0.0,
        "WATCH": 0.0,
        "WATCH_RISK": 0.0,
        "SELL": -1.0,
        "STRONG_SELL": -1.0,
    }
    pos = signal.map(mapping).fillna(0.0)
    if strategy_mode == "long_only":
        pos = pos.clip(lower=0.0)
    elif strategy_mode != "long_short":
        raise ValueError("strategy_mode must be 'long_only' or 'long_short'.")
    return pos


def _apply_costs(returns: pd.Series, position: pd.Series, transaction_cost_bps: float, slippage_bps: float) -> pd.Series:
    position_change = position.diff().abs().fillna(position.abs())
    total_cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0
    cost_drag = position_change * total_cost_rate
    return returns - cost_drag


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    return float(drawdown.min() * 100.0)


def _metrics_from_returns(returns: pd.Series) -> dict:
    returns = returns.fillna(0.0)
    total_return = (1.0 + returns).prod() - 1.0
    ann_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / max(len(returns), 1)) - 1.0
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_like = np.nan if ann_vol == 0 or np.isnan(ann_vol) else ann_return / ann_vol
    equity = (1.0 + returns).cumprod()
    positive = (returns > 0).sum()
    negative = (returns < 0).sum()
    win_rate = positive / max((positive + negative), 1)
    return {
        "total_return_pct": total_return * 100.0,
        "annualized_return_pct": ann_return * 100.0,
        "annualized_vol_pct": ann_vol * 100.0,
        "sharpe_like": sharpe_like,
        "max_drawdown_pct": _max_drawdown(equity),
        "win_rate_pct": win_rate * 100.0,
        "days": int(len(returns)),
    }


def run_symbol_backtest(
    df: pd.DataFrame,
    symbol: str,
    strategy_mode: str = "long_only",
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df["symbol"] == symbol].sort_values("date").copy()
    if subset.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    subset["position_raw"] = _signal_to_position(subset["signal"], strategy_mode=strategy_mode)
    subset["position"] = subset["position_raw"].shift(1).fillna(0.0)

    gross_strategy = subset["position"] * subset["daily_return_decimal"].fillna(0.0)
    subset["strategy_return_decimal"] = _apply_costs(gross_strategy, subset["position"], transaction_cost_bps, slippage_bps)
    subset["buyhold_return_decimal"] = subset["daily_return_decimal"].fillna(0.0)

    subset["strategy_equity"] = (1.0 + subset["strategy_return_decimal"]).cumprod()
    subset["buyhold_equity"] = (1.0 + subset["buyhold_return_decimal"]).cumprod()

    strategy_metrics = _metrics_from_returns(subset["strategy_return_decimal"])
    buyhold_metrics = _metrics_from_returns(subset["buyhold_return_decimal"])

    metrics = pd.DataFrame([
        {"strategy": "Signal Strategy", **strategy_metrics},
        {"strategy": "Buy & Hold", **buyhold_metrics},
    ])
    return subset, metrics


def _pick_rebalance_dates(index: pd.Index, frequency: str) -> pd.Series:
    idx = pd.to_datetime(index)
    if frequency == "daily":
        return pd.Series(True, index=idx)
    if frequency == "weekly":
        return pd.Series(idx.to_series().dt.to_period("W").astype(str)).ne(
            pd.Series(idx.to_series().shift(1).dt.to_period("W").astype(str)).values
        ).set_axis(idx)
    if frequency == "monthly":
        return pd.Series(idx.to_series().dt.to_period("M").astype(str)).ne(
            pd.Series(idx.to_series().shift(1).dt.to_period("M").astype(str)).values
        ).set_axis(idx)
    raise ValueError("frequency must be daily, weekly, or monthly")


def run_portfolio_backtest(
    df: pd.DataFrame,
    symbols: list[str],
    use_signals: bool = False,
    strategy_mode: str = "long_only",
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    weights: dict | None = None,
    rebalance_frequency: str = "monthly",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df["symbol"].isin(symbols)].sort_values(["date", "symbol"]).copy()
    if subset.empty:
        raise ValueError("No portfolio data found for selected symbols.")

    pivot_returns = subset.pivot(index="date", columns="symbol", values="daily_return_decimal").sort_index().fillna(0.0)
    pivot_weights = pd.DataFrame(0.0, index=pivot_returns.index, columns=pivot_returns.columns)

    if weights is None:
        base_weights = {col: 1.0 / len(pivot_returns.columns) for col in pivot_returns.columns}
    else:
        base_weights = {col: float(weights.get(col, 0.0)) for col in pivot_returns.columns}
        total = sum(base_weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        base_weights = {k: v / total for k, v in base_weights.items()}

    if use_signals:
        subset["position_raw"] = _signal_to_position(subset["signal"], strategy_mode=strategy_mode)
        subset["position"] = subset.groupby("symbol")["position_raw"].shift(1).fillna(0.0)
        pivot_positions = subset.pivot(index="date", columns="symbol", values="position").sort_index().fillna(0.0)
    else:
        pivot_positions = pd.DataFrame(1.0, index=pivot_returns.index, columns=pivot_returns.columns)

    rebalance_mask = _pick_rebalance_dates(pivot_returns.index, rebalance_frequency)

    current_weights = pd.Series(base_weights).reindex(pivot_returns.columns).fillna(0.0)
    for date in pivot_returns.index:
        if rebalance_mask.loc[date]:
            active = pivot_positions.loc[date]
            w = current_weights * active
            total = w.abs().sum() if strategy_mode == "long_short" else w.sum()
            if total == 0:
                w = pd.Series(0.0, index=current_weights.index)
            else:
                w = w / total
            current_weights = w
        pivot_weights.loc[date] = current_weights.values

    gross_returns = (pivot_returns * pivot_weights).sum(axis=1)
    turnover = pivot_weights.diff().abs().sum(axis=1).fillna(pivot_weights.abs().sum(axis=1))
    total_cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0
    net_returns = gross_returns - turnover * total_cost_rate

    equity = (1.0 + net_returns).cumprod()

    out = pd.DataFrame({
        "date": pivot_returns.index,
        "portfolio_return_decimal": net_returns.values,
        "portfolio_equity": equity.values,
        "turnover": turnover.values,
    })

    label = "Signal-Filtered Portfolio" if use_signals else "Portfolio"
    metrics = pd.DataFrame([{ "strategy": label, **_metrics_from_returns(net_returns) }])
    metrics["avg_turnover"] = float(turnover.mean())
    metrics["rebalance_frequency"] = rebalance_frequency
    metrics["transaction_cost_bps"] = transaction_cost_bps
    metrics["slippage_bps"] = slippage_bps
    return out, metrics


def run_hybrid_recommendation_backtest(
    df: pd.DataFrame,
    recommendation_history: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recommendation_history.empty:
        raise ValueError("Recommendation history is empty.")

    merged = recommendation_history.merge(
        df[["date", "symbol", "daily_return_decimal"]].copy(),
        on=["date", "symbol"],
        how="left",
    ).sort_values("date").copy()

    mapping = {
        "STRONG BUY": 1.0,
        "BUY": 1.0,
        "WATCH": 0.0,
        "AVOID": 0.0,
        "STRONG AVOID": 0.0,
    }
    merged["position_raw"] = merged["recommendation"].map(mapping).fillna(0.0)
    merged["position"] = merged["position_raw"].shift(1).fillna(0.0)

    gross = merged["position"] * merged["daily_return_decimal"].fillna(0.0)
    merged["strategy_return_decimal"] = _apply_costs(gross, merged["position"], transaction_cost_bps, slippage_bps)
    merged["strategy_equity"] = (1.0 + merged["strategy_return_decimal"]).cumprod()

    metrics = pd.DataFrame([{ "strategy": "Hybrid Recommendation Strategy", **_metrics_from_returns(merged["strategy_return_decimal"]) }])
    return merged, metrics
