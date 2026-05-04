
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision_engine import compute_position_size

from src.config import TRADING_DAYS_PER_YEAR


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
    symbol_df: pd.DataFrame,
    recommendation_history: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    shorting_allowed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import numpy as np
    import pandas as pd

    if symbol_df.empty:
        raise ValueError("symbol_df is empty.")

    if recommendation_history.empty:
        raise ValueError("recommendation_history is empty.")

    df = symbol_df.copy()
    rec = recommendation_history.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rec["date"] = pd.to_datetime(rec["date"], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    rec = rec.sort_values("date").reset_index(drop=True)

    merged = df.merge(
        rec[["date", "recommendation", "recommendation_score", "confidence_score", "model_agreement_score"]],
        on="date",
        how="inner",
    ).copy()

    if merged.empty:
        raise ValueError("No overlapping dates between symbol_df and recommendation_history.")

    # Forward one-day realized return
    if "daily_return_decimal" in merged.columns:
        merged["forward_return_decimal"] = merged["daily_return_decimal"].shift(-1)
    else:
        merged["forward_return_decimal"] = merged["close"].pct_change().shift(-1)

    merged = merged.dropna(subset=["forward_return_decimal"]).copy()

   
    raw_score = merged["recommendation_score"].copy()

    # Position sizing using shared logic
    merged["position_size"] = merged.apply(
        lambda row: compute_position_size(
            row["recommendation_score"], row["confidence_score"], shorting_allowed, row.get("model_agreement_score", 0.0)
        ),
        axis=1
    )

    total_cost_decimal = (transaction_cost_bps + slippage_bps) / 10000.0
    merged["prev_position_size"] = merged["position_size"].shift(1).fillna(0.0)
    merged["turnover"] = (merged["position_size"] - merged["prev_position_size"]).abs()
    merged["cost_decimal"] = merged["turnover"] * total_cost_decimal

    merged["strategy_return_decimal"] = (
        merged["position_size"] * merged["forward_return_decimal"]
        - merged["cost_decimal"]
    )

    merged["buyhold_return_decimal"] = merged["forward_return_decimal"]

    merged["strategy_equity"] = (1.0 + merged["strategy_return_decimal"].fillna(0.0)).cumprod()
    merged["buyhold_equity"] = (1.0 + merged["buyhold_return_decimal"].fillna(0.0)).cumprod()

    def _max_drawdown(series: pd.Series) -> float:
        running_peak = series.cummax()
        drawdown = (series / running_peak) - 1.0
        return float(drawdown.min() * 100.0)

    def _metrics(returns: pd.Series, name: str) -> dict:
        returns = returns.fillna(0.0)
        total_return = (1.0 + returns).prod() - 1.0
        annualized_return = (1.0 + total_return) ** (252 / max(len(returns), 1)) - 1.0
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_like = annualized_return / annualized_vol if annualized_vol and not np.isnan(annualized_vol) else np.nan
        win_rate = float((returns > 0).sum() / max(((returns > 0) | (returns < 0)).sum(), 1))

        equity = (1.0 + returns).cumprod()

        return {
            "strategy": name,
            "total_return_pct": total_return * 100.0,
            "annualized_return_pct": annualized_return * 100.0,
            "annualized_vol_pct": annualized_vol * 100.0,
            "sharpe_like": sharpe_like,
            "max_drawdown_pct": _max_drawdown(equity),
            "win_rate_pct": win_rate * 100.0,
            "days": int(len(returns)),
            "number_of_trades": int((returns != 0).sum()) if name == "Hybrid Recommendation Strategy" else 0,
        }

    metrics_df = pd.DataFrame(
        [
            _metrics(merged["strategy_return_decimal"], "Hybrid Recommendation Strategy"),
            _metrics(merged["buyhold_return_decimal"], "Buy & Hold"),
        ]
    )

    return merged, metrics_df


def _generate_recommendation_history(df: pd.DataFrame, symbol: str, shorting_allowed: bool, wf_predictions: pd.DataFrame, wf_metrics: pd.DataFrame, calibration_model, version: str = "V3") -> pd.DataFrame:
    from modeling import build_analyst_engine
    
    symbol_df = df[df["symbol"] == symbol].copy()
    recs = []
    
    for _, row in symbol_df.iterrows():
        date_preds = wf_predictions[(wf_predictions["date"] == row["date"]) & (wf_predictions["symbol"] == symbol)]
        if date_preds.empty:
            continue
        
        # Use all model predictions for this date
        latest_preds = date_preds[["model", "predicted_up_probability"]].rename(columns={"predicted_up_probability": "latest_up_probability"})
        
        analyst = build_analyst_engine(row, latest_preds, wf_metrics, calibration_model, shorting_allowed, version)
        
        recs.append({
            "date": row["date"],
            "recommendation": analyst["recommendation"],
            "recommendation_score": analyst["recommendation_score"],
            "confidence_score": analyst["confidence_score"],
            "model_agreement_score": analyst.get("model_agreement_score", 0.0),
        })
    
    return pd.DataFrame(recs)


@pd.api.extensions.register_dataframe_accessor("cached")
class CachedAccessor:
    def __init__(self, df):
        self.df = df

    def hash(self):
        return hash(tuple(sorted(self.df.to_dict().items())))


def run_v2_v3_comparison_backtest(
    df: pd.DataFrame,
    symbols: list[str],
    train_window: int = 100,
    test_window: int = 20,
    prob_threshold: float = 0.55,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    from modeling import run_walk_forward_models
    
    results = []
    skipped_symbols = []
    debug_stats = {}
    
    for version in ["V2", "V3"]:
        shorting_allowed = (version == "V3")  # V2: long only, V3: allows shorting
        for symbol in symbols:
            try:
                # Run walk-forward for this symbol
                wf_predictions, wf_metrics, wf_backtest, calibration_model = run_walk_forward_models(
                    df, symbol, train_window, test_window, prob_threshold
                )
                
                rec_hist = _generate_recommendation_history(df, symbol, shorting_allowed, wf_predictions, wf_metrics, calibration_model, version)
                if rec_hist.empty:
                    skipped_symbols.append(f"{symbol} ({version}: no recommendations)")
                    continue
                
                symbol_df = df[df["symbol"] == symbol]
                backtest_df, metrics_df = run_hybrid_recommendation_backtest(
                    symbol_df, rec_hist, transaction_cost_bps, slippage_bps, shorting_allowed
                )
                
                # Track position generation for debugging
                nonzero_positions = (backtest_df["position_size"].abs() > 0.001).sum() if "position_size" in backtest_df.columns else 0
                avg_confidence = rec_hist["confidence_score"].mean() if "confidence_score" in rec_hist.columns else 0
                avg_agreement = rec_hist["model_agreement_score"].mean() if "model_agreement_score" in rec_hist.columns else 0
                
                debug_key = f"{symbol}_{version}"
                debug_stats[debug_key] = {
                    "wf_pred_rows": len(wf_predictions),
                    "rec_hist_rows": len(rec_hist),
                    "nonzero_positions": nonzero_positions,
                    "avg_confidence": float(avg_confidence),
                    "avg_agreement": float(avg_agreement),
                }
                
                strategy_metrics = metrics_df[metrics_df["strategy"] == "Hybrid Recommendation Strategy"].iloc[0]
                results.append({
                    "version": version,
                    "symbol": symbol,
                    "total_return_pct": strategy_metrics["total_return_pct"],
                    "sharpe_like": strategy_metrics["sharpe_like"],
                    "max_drawdown_pct": strategy_metrics["max_drawdown_pct"],
                    "win_rate_pct": strategy_metrics["win_rate_pct"],
                    "number_of_trades": strategy_metrics["number_of_trades"],
                    "avg_confidence": float(avg_confidence),
                    "avg_agreement": float(avg_agreement),
                })
            except Exception as e:
                skipped_symbols.append(f"{symbol} ({version}: {str(e)})")
                continue
    
    if not results:
        raise ValueError(f"All symbols failed walk-forward validation. Skipped: {skipped_symbols}")
    
    # Create per-symbol DataFrame
    per_symbol_df = pd.DataFrame(results)
    
    # Add interpretation labels
    def get_interpretation(row):
        trades = row["number_of_trades"]
        sharpe = row["sharpe_like"]
        ret = row["total_return_pct"]
        
        if trades > 10 and sharpe > 0.5 and ret > 0:
            return "Strong active signal"
        elif trades <= 5 and sharpe > 0.3:
            return "Selective / low-frequency"
        elif trades > 10 and (sharpe < 0.2 or ret < 0):
            return "High activity but weak performance"
        else:
            return "No clear edge"
    
    per_symbol_df["interpretation"] = per_symbol_df.apply(get_interpretation, axis=1)
    
    # Add warnings for high activity with weak performance
    def get_warning(row):
        trades = row["number_of_trades"]
        sharpe = row["sharpe_like"]
        ret = row["total_return_pct"]
        
        if trades > 10 and (sharpe < 0.2 or ret < 0):
            return "⚠️ High trade count with weak performance"
        return ""
    
    per_symbol_df["warning"] = per_symbol_df.apply(get_warning, axis=1)
    
    # Create aggregate comparison
    comparison_df = per_symbol_df.groupby("version").agg({
        "total_return_pct": "mean",
        "sharpe_like": "mean",
        "max_drawdown_pct": "mean",
        "win_rate_pct": "mean",
        "number_of_trades": "mean",
    }).reset_index().assign(num_symbols=len(symbols) - len(skipped_symbols))
    
    return comparison_df, per_symbol_df, skipped_symbols, debug_stats
