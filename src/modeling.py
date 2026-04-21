from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_PROB_THRESHOLD,
    MIN_MODEL_ROWS,
    MODEL_BACKTEST_EXPORT_PATH,
    MODEL_FEATURES,
    PREDICTION_HORIZON_DAYS,
    PREDICTIONS_EXPORT_PATH,
    RECENT_MODEL_WINDOW,
    TARGET_RETURN_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
    WALK_FORWARD_TEST_WINDOW,
    WALK_FORWARD_TRAIN_WINDOW,
)


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _prepare_symbol_model_data(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df["symbol"] == symbol].sort_values("date").copy()

    if subset.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    subset["target_forward_return_decimal"] = (
        subset["close"].shift(-PREDICTION_HORIZON_DAYS) / subset["close"]
    ) - 1.0

    subset["target_up_move"] = (
        subset["target_forward_return_decimal"] > TARGET_RETURN_THRESHOLD
    ).astype(float)

    subset = subset.dropna(subset=MODEL_FEATURES).copy()
    subset = subset.tail(RECENT_MODEL_WINDOW).copy()
    labeled = subset.dropna(subset=["target_up_move", "target_forward_return_decimal"]).copy()

    if len(labeled) < MIN_MODEL_ROWS:
        raise ValueError(
            f"Not enough labeled rows to model {symbol}. Need at least {MIN_MODEL_ROWS}, found {len(labeled)}."
        )

    return subset, labeled


def _chronological_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1.0 - test_fraction))
    split_idx = max(split_idx, 1)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    if test.empty:
        raise ValueError("Test split is empty. Need more rows.")

    return train, test


def _train_valid_split(train_df: pd.DataFrame, valid_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(train_df) * (1.0 - valid_fraction))
    split_idx = max(split_idx, 1)
    subtrain = train_df.iloc[:split_idx].copy()
    valid = train_df.iloc[split_idx:].copy()

    if valid.empty:
        raise ValueError("Validation split is empty. Need more rows.")

    return subtrain, valid


def _make_logistic_model(C: float, class_weight):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42, C=C, class_weight=class_weight)),
        ]
    )


def _make_mlp_model(hidden_layer_sizes: tuple, alpha: float):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )


def _make_rf_model(n_estimators: int, max_depth: int | None):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _make_gb_model(n_estimators: int, learning_rate: float, max_depth: int):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                ),
            ),
        ]
    )


def _score_probs(y_true: pd.Series, probs: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int)
    return f1_score(y_true, preds, zero_division=0)


def _evaluate_binary_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "avg_predicted_up_probability": float(np.mean(probs)),
    }


def _tune_logistic(X_subtrain, y_subtrain, X_valid, y_valid):
    candidates = [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 3.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 3.0, "class_weight": "balanced"},
    ]
    best_model, best_cfg, best_score = None, None, -1.0

    for cfg in candidates:
        model = _make_logistic_model(cfg["C"], cfg["class_weight"])
        model.fit(X_subtrain, y_subtrain)
        score = _score_probs(y_valid, model.predict_proba(X_valid)[:, 1])
        if score > best_score:
            best_model, best_cfg, best_score = model, cfg, score

    return best_model, best_cfg, best_score


def _tune_mlp(X_subtrain, y_subtrain, X_valid, y_valid):
    candidates = [
        {"hidden_layer_sizes": (16,), "alpha": 0.0001},
        {"hidden_layer_sizes": (32, 16), "alpha": 0.0001},
        {"hidden_layer_sizes": (32, 16), "alpha": 0.001},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.0001},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.001},
    ]
    best_model, best_cfg, best_score = None, None, -1.0

    for cfg in candidates:
        model = _make_mlp_model(cfg["hidden_layer_sizes"], cfg["alpha"])
        model.fit(X_subtrain, y_subtrain)
        score = _score_probs(y_valid, model.predict_proba(X_valid)[:, 1])
        if score > best_score:
            best_model, best_cfg, best_score = model, cfg, score

    return best_model, best_cfg, best_score


def _tune_rf(X_subtrain, y_subtrain, X_valid, y_valid):
    candidates = [
        {"n_estimators": 100, "max_depth": 3},
        {"n_estimators": 200, "max_depth": 4},
        {"n_estimators": 300, "max_depth": 5},
        {"n_estimators": 200, "max_depth": None},
    ]
    best_model, best_cfg, best_score = None, None, -1.0

    for cfg in candidates:
        model = _make_rf_model(cfg["n_estimators"], cfg["max_depth"])
        model.fit(X_subtrain, y_subtrain)
        score = _score_probs(y_valid, model.predict_proba(X_valid)[:, 1])
        if score > best_score:
            best_model, best_cfg, best_score = model, cfg, score

    return best_model, best_cfg, best_score


def _tune_gb(X_subtrain, y_subtrain, X_valid, y_valid):
    candidates = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 2},
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
    ]
    best_model, best_cfg, best_score = None, None, -1.0

    for cfg in candidates:
        model = _make_gb_model(cfg["n_estimators"], cfg["learning_rate"], cfg["max_depth"])
        model.fit(X_subtrain, y_subtrain)
        score = _score_probs(y_valid, model.predict_proba(X_valid)[:, 1])
        if score > best_score:
            best_model, best_cfg, best_score = model, cfg, score

    return best_model, best_cfg, best_score


def _train_tuned_models(train_df: pd.DataFrame):
    subtrain, valid = _train_valid_split(train_df, valid_fraction=0.2)

    X_subtrain = subtrain[MODEL_FEATURES]
    y_subtrain = subtrain["target_up_move"].astype(int)
    X_valid = valid[MODEL_FEATURES]
    y_valid = valid["target_up_move"].astype(int)

    _, log_cfg, log_f1 = _tune_logistic(X_subtrain, y_subtrain, X_valid, y_valid)
    _, mlp_cfg, mlp_f1 = _tune_mlp(X_subtrain, y_subtrain, X_valid, y_valid)
    _, rf_cfg, rf_f1 = _tune_rf(X_subtrain, y_subtrain, X_valid, y_valid)
    _, gb_cfg, gb_f1 = _tune_gb(X_subtrain, y_subtrain, X_valid, y_valid)

    X_full = train_df[MODEL_FEATURES]
    y_full = train_df["target_up_move"].astype(int)

    models = {
        "Logistic Regression": _make_logistic_model(log_cfg["C"], log_cfg["class_weight"]),
        "Neural Net (MLP)": _make_mlp_model(mlp_cfg["hidden_layer_sizes"], mlp_cfg["alpha"]),
        "Random Forest": _make_rf_model(rf_cfg["n_estimators"], rf_cfg["max_depth"]),
        "Gradient Boosting": _make_gb_model(gb_cfg["n_estimators"], gb_cfg["learning_rate"], gb_cfg["max_depth"]),
    }

    for model in models.values():
        model.fit(X_full, y_full)

    tuning_info = pd.DataFrame(
        [
            {"model": "Logistic Regression", "best_validation_f1": log_f1, "best_config": str(log_cfg)},
            {"model": "Neural Net (MLP)", "best_validation_f1": mlp_f1, "best_config": str(mlp_cfg)},
            {"model": "Random Forest", "best_validation_f1": rf_f1, "best_config": str(rf_cfg)},
            {"model": "Gradient Boosting", "best_validation_f1": gb_f1, "best_config": str(gb_cfg)},
        ]
    )

    return models, tuning_info


def run_symbol_models(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset_all, labeled = _prepare_symbol_model_data(df, symbol)
    train, test = _chronological_split(labeled, test_fraction=0.2)

    X_test = test[MODEL_FEATURES]
    y_test = test["target_up_move"].astype(int)

    models, tuning_info = _train_tuned_models(train)

    metrics_rows = []
    latest_rows = []
    coef_df = pd.DataFrame(columns=["feature", "coefficient"])

    latest_feature_row = subset_all[MODEL_FEATURES].tail(1)

    for model_name, model in models.items():
        base_metrics = _evaluate_binary_model(model_name, model, X_test, y_test)
        tune_row = tuning_info[tuning_info["model"] == model_name].iloc[0]
        base_metrics["best_validation_f1"] = float(tune_row["best_validation_f1"])
        base_metrics["best_config"] = str(tune_row["best_config"])
        metrics_rows.append(base_metrics)

        latest_prob = float(model.predict_proba(latest_feature_row)[:, 1][0])
        latest_pred = int(latest_prob >= 0.5)

        latest_rows.append(
            {
                "model": model_name,
                "latest_up_probability": latest_prob,
                "latest_predicted_class": latest_pred,
            }
        )

        if model_name == "Logistic Regression":
            coef_values = model.named_steps["model"].coef_[0]
            coef_df = pd.DataFrame(
                {
                    "feature": MODEL_FEATURES,
                    "coefficient": coef_values,
                }
            ).sort_values("coefficient", ascending=False).reset_index(drop=True)

    return pd.DataFrame(metrics_rows), pd.DataFrame(latest_rows), coef_df


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    return float(drawdown.min() * 100.0)


def _strategy_metrics(returns: pd.Series) -> dict:
    returns = returns.fillna(0.0)
    total_return = (1.0 + returns).prod() - 1.0
    ann_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / max(len(returns), 1)) - 1.0
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_like = np.nan if ann_vol == 0 or np.isnan(ann_vol) else ann_return / ann_vol
    hit_rate = float((returns > 0).sum() / max(((returns > 0) | (returns < 0)).sum(), 1))
    return {
        "total_return_pct": total_return * 100.0,
        "annualized_return_pct": ann_return * 100.0,
        "annualized_vol_pct": ann_vol * 100.0,
        "sharpe_like": sharpe_like,
        "max_drawdown_pct": _max_drawdown_from_returns(returns),
        "hit_rate_pct": hit_rate * 100.0,
        "num_predictions": int(len(returns)),
    }


def run_walk_forward_models(
    df: pd.DataFrame,
    symbol: str,
    train_window: int = WALK_FORWARD_TRAIN_WINDOW,
    test_window: int = WALK_FORWARD_TEST_WINDOW,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, labeled = _prepare_symbol_model_data(df, symbol)

    if len(labeled) < train_window + test_window:
        raise ValueError(
            f"Not enough rows for walk-forward on {symbol}. Need at least {train_window + test_window}, found {len(labeled)}."
        )

    prediction_rows = []
    metric_rows = []
    backtest_rows = []

    start = 0
    while start + train_window < len(labeled):
        train_slice = labeled.iloc[start : start + train_window].copy()
        test_slice = labeled.iloc[start + train_window : start + train_window + test_window].copy()
        if test_slice.empty:
            break

        models, tuning_info = _train_tuned_models(train_slice)
        X_test = test_slice[MODEL_FEATURES]
        y_test = test_slice["target_up_move"].astype(int)

        for model_name, model in models.items():
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= prob_threshold).astype(int)

            temp = test_slice[
                [
                    "date",
                    "symbol",
                    "close",
                    "signal_score",
                    "volatility_30d_annualized",
                    "drawdown_pct",
                    "news_headline_count",
                    "news_avg_sentiment",
                    "news_positive_ratio",
                    "news_negative_ratio",
                    "news_sentiment_3d",
                    "news_sentiment_7d",
                    "news_impact_score_3d",
                    "news_decayed_sentiment_7d",
                    "news_high_impact_count_3d",
                    "news_macro_event_count_3d",
                    "news_company_event_count_3d",
                    "market_news_sentiment_3d",
                    "market_news_sentiment_7d",
                    "market_news_impact_score_3d",
                    "market_macro_event_count_3d",
                    "target_up_move",
                    "target_forward_return_decimal",
                ]
            ].copy()

            temp["model"] = model_name
            temp["predicted_up_probability"] = probs
            temp["predicted_class"] = preds
            temp["actual_class"] = y_test.values
            temp["strategy_return_decimal"] = np.where(
                temp["predicted_up_probability"] >= prob_threshold,
                temp["target_forward_return_decimal"],
                0.0,
            )
            temp["buyhold_return_decimal"] = temp["target_forward_return_decimal"]
            temp["trade_flag"] = (temp["predicted_up_probability"] >= prob_threshold).astype(int)

            prediction_rows.append(temp)

        start += test_window

    if not prediction_rows:
        raise ValueError("Walk-forward produced no prediction rows.")

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(["model", "date"]).reset_index(drop=True)

    full_train, _ = _chronological_split(labeled, test_fraction=0.2)
    _, global_tuning = _train_tuned_models(full_train)

    for model_name, group in predictions.groupby("model"):
        y_true = group["actual_class"].astype(int)
        y_pred = group["predicted_class"].astype(int)
        tune_row = global_tuning[global_tuning["model"] == model_name].iloc[0]

        metric_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "avg_predicted_up_probability": float(group["predicted_up_probability"].mean()),
                "trade_rate_pct": float(group["trade_flag"].mean() * 100.0),
                "avg_news_sentiment_seen": float(group["news_avg_sentiment"].mean()),
                "pct_rows_with_news": float((group["news_headline_count"] > 0).mean() * 100.0),
                "best_validation_f1": float(tune_row["best_validation_f1"]),
                "best_config": str(tune_row["best_config"]),
                "train_window": int(train_window),
                "test_window": int(test_window),
                "prob_threshold": float(prob_threshold),
            }
        )

        strat = _strategy_metrics(group["strategy_return_decimal"])
        bench = _strategy_metrics(group["buyhold_return_decimal"])
        backtest_rows.append({"model": model_name, "strategy_name": "Model-Driven Strategy", **strat})
        backtest_rows.append({"model": model_name, "strategy_name": "Buy & Hold Over Prediction Horizon", **bench})

    metrics_df = pd.DataFrame(metric_rows).sort_values("f1", ascending=False).reset_index(drop=True)
    backtest_df = pd.DataFrame(backtest_rows).reset_index(drop=True)

    return predictions, metrics_df, backtest_df


def export_walk_forward_outputs(
    predictions_df: pd.DataFrame,
    backtest_metrics_df: pd.DataFrame,
    predictions_path: str = PREDICTIONS_EXPORT_PATH,
    backtest_path: str = MODEL_BACKTEST_EXPORT_PATH,
) -> tuple[str, str]:
    pred_path = Path(predictions_path)
    bt_path = Path(backtest_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(pred_path, index=False)
    backtest_metrics_df.to_csv(bt_path, index=False)
    return str(pred_path), str(bt_path)


def detect_market_regime(row: pd.Series) -> dict:
    ma_spread = safe_float(row.get("ma_spread_pct", 0.0))
    vol = safe_float(row.get("volatility_30d_annualized", 0.0))
    rel30 = safe_float(row.get("rel_return_30d_pct", 0.0))
    rel90 = safe_float(row.get("rel_return_90d_pct", 0.0))

    if ma_spread > 1.0 and rel30 > 0:
        trend_regime = "Bullish trend"
    elif ma_spread < -1.0 and rel30 < 0:
        trend_regime = "Bearish trend"
    else:
        trend_regime = "Mixed / sideways"

    if vol >= 45:
        vol_regime = "High volatility"
    elif vol >= 25:
        vol_regime = "Moderate volatility"
    else:
        vol_regime = "Calmer conditions"

    if rel90 > 5:
        strength = "Outperforming benchmark"
    elif rel90 < -5:
        strength = "Underperforming benchmark"
    else:
        strength = "Benchmark-like performance"

    return {"trend_regime": trend_regime, "vol_regime": vol_regime, "strength_regime": strength}


def confidence_band(score: float) -> str:
    if score >= 0.72:
        return "High"
    if score >= 0.58:
        return "Moderate"
    return "Low"


def risk_level_from_penalty(penalty: float) -> str:
    if penalty >= 0.12:
        return "High"
    if penalty >= 0.06:
        return "Moderate"
    return "Low"


def model_agreement_label(probs: list[float]) -> str:
    if not probs:
        return "Unknown"
    gap = max(probs) - min(probs)
    if gap <= 0.08:
        return "High"
    if gap <= 0.17:
        return "Moderate"
    return "Low"


def news_support_label(row: pd.Series) -> str:
    count_3d = safe_float(row.get("news_count_3d", 0))
    impact_3d = abs(safe_float(row.get("news_impact_score_3d", 0.0)))
    market_count_3d = safe_float(row.get("market_news_count_3d", 0))
    if count_3d >= 3 or impact_3d >= 0.12 or market_count_3d >= 4:
        return "High"
    if count_3d >= 1 or market_count_3d >= 2:
        return "Moderate"
    return "Low"


def compute_technical_probability(row: pd.Series) -> tuple[float, list[str]]:
    score = safe_float(row.get("signal_score", 0.0))
    ma_spread = safe_float(row.get("ma_spread_pct", 0.0))
    rsi = safe_float(row.get("rsi_14", 50.0))
    rel30 = safe_float(row.get("rel_return_30d_pct", 0.0))
    rel90 = safe_float(row.get("rel_return_90d_pct", 0.0))
    drawdown = safe_float(row.get("drawdown_pct", 0.0))

    prob = 0.50
    reasons = []

    prob += np.clip(score / 14.0, -0.18, 0.18)
    prob += np.clip(ma_spread / 20.0, -0.08, 0.08)
    prob += np.clip(rel30 / 30.0, -0.08, 0.08)
    prob += np.clip(rel90 / 50.0, -0.06, 0.06)

    if 55 <= rsi <= 70:
        prob += 0.04
        reasons.append("Momentum is supportive without being extremely overextended")
    elif rsi >= 75:
        prob -= 0.05
        reasons.append("Momentum looks extended and may be vulnerable to pullback")
    elif rsi <= 30:
        prob += 0.02
        reasons.append("RSI is depressed, which may support a recovery setup")

    if drawdown <= -20:
        prob -= 0.04
        reasons.append("Deep drawdown still adds caution")

    if score >= 3:
        reasons.append("Technical signal structure is bullish")
    elif score <= -2:
        reasons.append("Technical signal structure is bearish")

    if rel30 > 0:
        reasons.append("Recent relative performance versus the benchmark is positive")
    elif rel30 < 0:
        reasons.append("Recent relative performance versus the benchmark is weak")

    return clip01(prob), reasons[:5]


def compute_adaptive_model_probability(latest_preds: pd.DataFrame, wf_metrics: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    if latest_preds.empty:
        breakdown = pd.DataFrame(
            {
                "model": ["No model outputs"],
                "probability": [0.50],
                "weight": [1.00],
                "weighted_probability": [0.50],
            }
        )
        return 0.50, breakdown

    probs = latest_preds[["model", "latest_up_probability"]].copy()

    if wf_metrics.empty or "f1" not in wf_metrics.columns:
        probs["weight"] = 1.0
    else:
        metric_weights = wf_metrics[["model", "f1"]].copy()
        metric_weights["f1"] = metric_weights["f1"].fillna(0.0).clip(lower=0.01)
        probs = probs.merge(metric_weights, on="model", how="left")
        probs["weight"] = probs["f1"].fillna(0.10)

    if probs["weight"].sum() <= 0:
        probs["weight"] = 1.0

    probs["weight"] = probs["weight"] / probs["weight"].sum()
    probs["weighted_probability"] = probs["latest_up_probability"] * probs["weight"]

    return float(probs["weighted_probability"].sum()), probs.rename(columns={"latest_up_probability": "probability"})


def compute_context_probability(row: pd.Series) -> tuple[float, list[str], str]:
    news_count_3d = safe_float(row.get("news_count_3d", 0.0))
    symbol_sent_3d = safe_float(row.get("news_sentiment_3d", 0.0))
    symbol_sent_7d = safe_float(row.get("news_sentiment_7d", 0.0))
    symbol_impact = safe_float(row.get("news_impact_score_3d", 0.0))
    decayed_sent = safe_float(row.get("news_decayed_sentiment_7d", 0.0))
    high_impact_count = safe_float(row.get("news_high_impact_count_3d", 0.0))
    macro_event_count = safe_float(row.get("news_macro_event_count_3d", 0.0))
    company_event_count = safe_float(row.get("news_company_event_count_3d", 0.0))

    market_sent_3d = safe_float(row.get("market_news_sentiment_3d", 0.0))
    market_sent_7d = safe_float(row.get("market_news_sentiment_7d", 0.0))
    market_impact = safe_float(row.get("market_news_impact_score_3d", 0.0))
    market_count = safe_float(row.get("market_news_count_3d", 0.0))
    market_macro_count = safe_float(row.get("market_macro_event_count_3d", 0.0))

    prob = 0.50
    reasons = []

    if news_count_3d == 0 and market_count == 0:
        return 0.50, ["Recent news support is limited, so the context layer stays neutral"], "Low"

    prob += np.clip(symbol_sent_3d * 0.18, -0.07, 0.07)
    prob += np.clip(symbol_sent_7d * 0.12, -0.05, 0.05)
    prob += np.clip(decayed_sent * 0.16, -0.06, 0.06)
    prob += np.clip(symbol_impact * 0.12, -0.06, 0.06)
    prob += np.clip(market_sent_3d * 0.15, -0.05, 0.05)
    prob += np.clip(market_sent_7d * 0.10, -0.04, 0.04)
    prob += np.clip(market_impact * 0.10, -0.04, 0.04)

    prob += np.clip(high_impact_count * 0.02, 0.0, 0.06)
    prob += np.clip(company_event_count * 0.015, 0.0, 0.04)
    prob += np.clip(macro_event_count * 0.015, 0.0, 0.04)
    prob += np.clip(market_macro_count * 0.01, 0.0, 0.03)

    if symbol_impact > 0.08:
        reasons.append("Recent asset-specific headlines appear meaningful enough to matter")
    if symbol_sent_3d > 0.08 or decayed_sent > 0.08:
        reasons.append("Recent asset-specific news tone is supportive")
    elif symbol_sent_3d < -0.08 or decayed_sent < -0.08:
        reasons.append("Recent asset-specific news tone is negative")

    if market_sent_3d > 0.08:
        reasons.append("Broader market news is supportive")
    elif market_sent_3d < -0.08:
        reasons.append("Broader market news is creating headwinds")

    support = news_support_label(row)
    return clip01(prob), reasons[:4], support


def compute_risk_penalty(row: pd.Series) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons = []

    vol = safe_float(row.get("volatility_30d_annualized", 0.0))
    drawdown = safe_float(row.get("drawdown_pct", 0.0))
    rsi = safe_float(row.get("rsi_14", 50.0))
    rel30 = safe_float(row.get("rel_return_30d_pct", 0.0))
    rel90 = safe_float(row.get("rel_return_90d_pct", 0.0))

    if vol >= 55:
        penalty += 0.10
        reasons.append("Volatility is unusually high")
    elif vol >= 40:
        penalty += 0.06
        reasons.append("Volatility is elevated")

    if drawdown <= -20:
        penalty += 0.08
        reasons.append("The asset remains in a deep drawdown")
    elif drawdown <= -10:
        penalty += 0.04
        reasons.append("The asset is still below prior highs")

    if rsi >= 75:
        penalty += 0.03
        reasons.append("Momentum is stretched")

    if rel30 < -5:
        penalty += 0.03
        reasons.append("Recent benchmark-relative performance is weak")

    if rel90 < -10:
        penalty += 0.04
        reasons.append("Longer-term benchmark-relative performance is weak")

    return penalty, reasons[:5]


def compute_quality_score(wf_metrics: pd.DataFrame) -> tuple[float, str]:
    if wf_metrics.empty or "f1" not in wf_metrics.columns:
        return 0.50, "Limited validation"

    best_f1 = float(wf_metrics["f1"].max())
    if best_f1 >= 0.62:
        return 0.85, "Strong validation"
    if best_f1 >= 0.54:
        return 0.65, "Acceptable validation"
    if best_f1 >= 0.48:
        return 0.50, "Mixed validation"
    return 0.35, "Weak validation"


def final_recommendation_label(score: float) -> str:
    if score >= 0.68:
        return "STRONG BUY"
    if score >= 0.58:
        return "BUY"
    if score >= 0.45:
        return "WATCH"
    if score >= 0.35:
        return "AVOID"
    return "STRONG AVOID"


def build_analyst_engine(latest_row: pd.Series, latest_preds: pd.DataFrame, wf_metrics: pd.DataFrame) -> dict:
    technical_prob, technical_reasons = compute_technical_probability(latest_row)
    model_prob, model_breakdown = compute_adaptive_model_probability(latest_preds, wf_metrics)
    context_prob, context_reasons, news_support = compute_context_probability(latest_row)
    risk_penalty, risk_reasons = compute_risk_penalty(latest_row)
    quality_score, quality_label = compute_quality_score(wf_metrics)

    probs = [technical_prob, model_prob]
    if news_support != "Low":
        probs.append(context_prob)

    agreement = model_agreement_label(probs)

    raw_score = (0.43 * technical_prob) + (0.42 * model_prob) + (0.15 * context_prob)
    final_score = clip01(raw_score - (0.40 * risk_penalty))

    technical_strength = abs(technical_prob - 0.5) * 2
    model_strength = abs(model_prob - 0.5) * 2
    agreement_score = {"High": 0.88, "Moderate": 0.68, "Low": 0.44}.get(agreement, 0.50)
    support_bonus = {"High": 0.85, "Moderate": 0.65, "Low": 0.50}.get(news_support, 0.50)

    confidence = (
        0.28 * agreement_score
        + 0.26 * quality_score
        + 0.22 * technical_strength
        + 0.16 * model_strength
        + 0.08 * support_bonus
    )
    confidence -= 0.18 * risk_penalty
    if agreement == "Low":
        confidence *= 0.85
    confidence = clip01(confidence)

    recommendation = final_recommendation_label(final_score)
    regime = detect_market_regime(latest_row)

    reasons = []
    reasons.extend(technical_reasons[:2])
    reasons.extend(context_reasons[:2])
    if agreement == "High":
        reasons.append("Model outputs are broadly aligned")
    elif agreement == "Low":
        reasons.append("Model outputs are not tightly aligned")
    reasons.extend(risk_reasons[:2])

    deduped = []
    for reason in reasons:
        if reason and reason not in deduped:
            deduped.append(reason)

    component_table = pd.DataFrame(
        [
            {"component": "Technical Layer", "score": technical_prob, "comment": "Trend, momentum, relative performance"},
            {"component": "Model Layer", "score": model_prob, "comment": "Adaptive weighting based on recent validation"},
            {"component": "Context Layer", "score": context_prob, "comment": "Symbol news, macro news, event impact"},
            {"component": "Risk Penalty", "score": risk_penalty, "comment": "Volatility, drawdown, overstretch"},
            {"component": "Validation Quality", "score": quality_score, "comment": quality_label},
        ]
    )

    return {
        "recommendation": recommendation,
        "recommendation_score": final_score,
        "confidence_score": confidence,
        "confidence_band": confidence_band(confidence),
        "technical_probability": technical_prob,
        "model_probability": model_prob,
        "context_probability": context_prob,
        "model_agreement": agreement,
        "news_support": news_support,
        "quality_label": quality_label,
        "risk_penalty": risk_penalty,
        "risk_level": risk_level_from_penalty(risk_penalty),
        "regime": regime,
        "model_breakdown": model_breakdown,
        "component_table": component_table,
        "reasons": deduped[:6],
    }


def build_analyst_memo(symbol: str, latest_row: pd.Series, analyst: dict) -> str:
    trend_regime = analyst["regime"]["trend_regime"]
    vol_regime = analyst["regime"]["vol_regime"]
    strength_regime = analyst["regime"]["strength_regime"]

    decision = analyst["recommendation"]
    confidence = analyst["confidence_band"]
    agreement = analyst["model_agreement"]
    news_support = analyst["news_support"]
    risk_level = analyst["risk_level"]

    return (
        f"For {symbol}, the engine currently assigns a {decision} view with {confidence.lower()} confidence. "
        f"Technically, the asset is in a {trend_regime.lower()} environment with {vol_regime.lower()}, while relative performance remains "
        f"{strength_regime.lower()}. Model agreement is {agreement.lower()}, which means the predictive layer is "
        f"{'providing clear support' if agreement == 'High' else 'not fully aligned yet' if agreement == 'Moderate' else 'still mixed'}. "
        f"News support is {news_support.lower()}, and the current risk backdrop is {risk_level.lower()}. "
        f"Overall, this should be interpreted as a structured analyst-style recommendation rather than a guess."
    )
