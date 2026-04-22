from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    ASSET_CLASS_MAP,
    ASSET_CLASS_TARGET_FLOOR,
    DEFAULT_PROB_THRESHOLD,
    MIN_MODEL_ROWS,
    MODEL_BACKTEST_EXPORT_PATH,
    MODEL_FEATURES,
    PREDICTION_HORIZON_DAYS,
    PREDICTIONS_EXPORT_PATH,
    RECENT_MODEL_WINDOW,
    TRADING_DAYS_PER_YEAR,
    VOL_TARGET_MULTIPLIER,
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


def _asset_class(symbol: str) -> str:
    return ASSET_CLASS_MAP.get(symbol, "unknown")


def _target_floor(symbol: str) -> float:
    cls = _asset_class(symbol)
    return ASSET_CLASS_TARGET_FLOOR.get(cls, ASSET_CLASS_TARGET_FLOOR["unknown"])


def _prepare_symbol_model_data(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df["symbol"] == symbol].sort_values("date").copy()

    if subset.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    subset["target_forward_return_decimal"] = (
        subset["close"].shift(-PREDICTION_HORIZON_DAYS) / subset["close"]
    ) - 1.0

    # volatility-adjusted target
    vol_decimal = subset["volatility_20d_annualized"] / 100 / np.sqrt(TRADING_DAYS_PER_YEAR)
    floor = _target_floor(symbol)
    subset["dynamic_target_threshold"] = np.maximum(floor, vol_decimal * np.sqrt(PREDICTION_HORIZON_DAYS) * VOL_TARGET_MULTIPLIER)

    subset["target_up_move"] = (
        subset["target_forward_return_decimal"] > subset["dynamic_target_threshold"]
    ).astype(float)

    subset = subset.dropna(subset=MODEL_FEATURES).copy()
    subset = subset.tail(RECENT_MODEL_WINDOW).copy()
    labeled = subset.dropna(
        subset=["target_up_move", "target_forward_return_decimal", "dynamic_target_threshold"]
    ).copy()

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


def _fit_calibration_model(predictions_df: pd.DataFrame) -> IsotonicRegression:
    """
    Fit isotonic regression for probability calibration using walk-forward predictions.
    """
    if predictions_df.empty or "predicted_up_probability" not in predictions_df.columns or "actual_class" not in predictions_df.columns:
        # Return identity calibration if no data
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit([0.0, 1.0], [0.0, 1.0])
        return iso
    
    # Use ensemble predictions (average across models for each date)
    ensemble_preds = predictions_df.groupby("date").agg({
        "predicted_up_probability": "mean",
        "actual_class": "first"  # Assuming same actual for all models
    }).reset_index()
    
    y_prob = ensemble_preds["predicted_up_probability"].values
    y_true = ensemble_preds["actual_class"].values
    
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso


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

            temp["position_size"] = np.clip((temp["predicted_up_probability"] - 0.5) * 2, 0, 1)
            temp["strategy_return_decimal"] = temp["position_size"] * temp["target_forward_return_decimal"]

            temp["buyhold_return_decimal"] = temp["target_forward_return_decimal"]
            temp["trade_flag"] = (temp["position_size"] > 0).astype(int)

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
                "brier_score": brier_score_loss(y_true, group["predicted_up_probability"]),
                "avg_predicted_up_probability": float(group["predicted_up_probability"].mean()),
                "avg_position_size": float(group["position_size"].mean()),
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

    # Fit calibration model using walk-forward predictions
    calibration_model = _fit_calibration_model(predictions)

    return predictions, metrics_df, backtest_df, calibration_model


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
    vol = safe_float(row.get("volatility_20d_annualized", 0.0))
    rel20 = safe_float(row.get("asset_vs_spy_20d", 0.0))

    if ma_spread > 1.0 and rel20 > 0:
        trend_regime = "Bullish trend"
    elif ma_spread < -1.0 and rel20 < 0:
        trend_regime = "Bearish trend"
    else:
        trend_regime = "Mixed / sideways"

    if vol >= 45:
        vol_regime = "High volatility"
    elif vol >= 25:
        vol_regime = "Moderate volatility"
    else:
        vol_regime = "Calmer conditions"

    if rel20 > 5:
        strength = "Outperforming benchmark"
    elif rel20 < -5:
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
    if gap <= 0.05:
        return "High"
    if gap <= 0.12:
        return "Moderate"
    return "Low"


def news_support_label(row: pd.Series) -> str:
    count_3d = safe_float(row.get("news_count_3d", 0))
    impact_3d = abs(safe_float(row.get("news_impact_score_3d", 0.0)))
    market_count_3d = safe_float(row.get("market_macro_event_count_3d", 0))
    if count_3d >= 3 or impact_3d >= 0.12 or market_count_3d >= 2:
        return "High"
    if count_3d >= 1:
        return "Moderate"
    return "Low"


def compute_technical_probability(row: pd.Series) -> tuple[float, list[str]]:
    ma_spread = safe_float(row.get("ma_spread_pct", 0.0))
    rsi = safe_float(row.get("rsi_14", 50.0))
    rel20 = safe_float(row.get("asset_vs_spy_20d", 0.0))
    drawdown = safe_float(row.get("drawdown_pct", 0.0))
    beta_like = safe_float(row.get("beta_like_20d", 1.0))

    prob = 0.50
    reasons = []

    prob += np.clip(ma_spread / 18.0, -0.10, 0.10)
    prob += np.clip(rel20 / 25.0, -0.10, 0.10)

    if 52 <= rsi <= 68:
        prob += 0.04
        reasons.append("Momentum is supportive without being extreme")
    elif rsi >= 75:
        prob -= 0.04
        reasons.append("Momentum looks stretched")
    elif rsi <= 30:
        prob += 0.02
        reasons.append("Deep weakness may support recovery")

    if drawdown <= -20:
        prob -= 0.04
        reasons.append("Deep drawdown still adds caution")

    if beta_like > 1.8:
        prob -= 0.02
        reasons.append("High market sensitivity adds uncertainty")

    if rel20 > 0:
        reasons.append("Recent benchmark-relative performance is positive")
    elif rel20 < 0:
        reasons.append("Recent benchmark-relative performance is weak")

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

    # live ensemble: GB + RF lead, Logistic small, MLP excluded from live decision
    live_models = ["Gradient Boosting", "Random Forest", "Logistic Regression"]
    probs = latest_preds[latest_preds["model"].isin(live_models)][["model", "latest_up_probability"]].copy()

    if probs.empty:
        probs = latest_preds[["model", "latest_up_probability"]].copy()

    if wf_metrics.empty or "f1" not in wf_metrics.columns:
        probs["weight"] = 1.0
    else:
        metric_weights = wf_metrics[["model", "f1"]].copy()
        probs = probs.merge(metric_weights, on="model", how="left")

        base_weight_map = {
            "Gradient Boosting": 1.35,
            "Random Forest": 1.15,
            "Logistic Regression": 0.75,
        }
        probs["base_weight"] = probs["model"].map(base_weight_map).fillna(0.50)
        probs["weight"] = probs["f1"].fillna(0.10) * probs["base_weight"]
        probs["weight"] = probs["weight"].clip(0.15, 0.75)

    if probs["weight"].sum() <= 0:
        probs["weight"] = 1.0

    probs["weight"] = probs["weight"] / probs["weight"].sum()
    probs["weighted_probability"] = probs["latest_up_probability"] * probs["weight"]

    return float(probs["weighted_probability"].sum()), probs.rename(columns={"latest_up_probability": "probability"})


def compute_context_probability(row: pd.Series) -> tuple[float, list[str], str]:
    prob = 0.50
    reasons = []

    symbol_sent_3d = safe_float(row.get("news_sentiment_3d", 0.0))
    decayed_sent = safe_float(row.get("news_decayed_sentiment_7d", 0.0))
    symbol_impact = safe_float(row.get("news_impact_score_3d", 0.0))
    market_sent_3d = safe_float(row.get("market_news_sentiment_3d", 0.0))
    market_impact = safe_float(row.get("market_news_impact_score_3d", 0.0))

    macro_flag = safe_float(row.get("macro_event_flag_3d", 0))
    earnings_flag = safe_float(row.get("earnings_event_flag_3d", 0))
    company_flag = safe_float(row.get("company_event_flag_3d", 0))
    high_impact_flag = safe_float(row.get("high_impact_flag_3d", 0))
    market_macro_flag = safe_float(row.get("market_macro_flag_3d", 0))

    prob += np.clip(symbol_sent_3d * 0.15, -0.05, 0.05)
    prob += np.clip(decayed_sent * 0.15, -0.05, 0.05)
    prob += np.clip(symbol_impact * 0.12, -0.05, 0.05)
    prob += np.clip(market_sent_3d * 0.12, -0.04, 0.04)
    prob += np.clip(market_impact * 0.10, -0.04, 0.04)

    prob += macro_flag * 0.01
    prob += earnings_flag * 0.02
    prob += company_flag * 0.01
    prob += high_impact_flag * 0.02
    prob += market_macro_flag * 0.01

    if high_impact_flag:
        reasons.append("Recent high-impact news is affecting the setup")
    if earnings_flag:
        reasons.append("Recent company-level event flow may matter")
    if market_macro_flag:
        reasons.append("Broader macro context is active")

    support = news_support_label(row)
    return clip01(prob), reasons[:4], support


def compute_risk_penalty(row: pd.Series) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons = []

    vol = safe_float(row.get("volatility_20d_annualized", 0.0))
    drawdown = safe_float(row.get("drawdown_pct", 0.0))
    rsi = safe_float(row.get("rsi_14", 50.0))
    rel20 = safe_float(row.get("asset_vs_spy_20d", 0.0))
    beta_like = safe_float(row.get("beta_like_20d", 1.0))

    if vol >= 55:
        penalty += 0.08
        reasons.append("Volatility is unusually high")
    elif vol >= 40:
        penalty += 0.05
        reasons.append("Volatility is elevated")

    if drawdown <= -20:
        penalty += 0.06
        reasons.append("The asset remains in a deep drawdown")

    if rsi >= 75:
        penalty += 0.03
        reasons.append("Momentum is stretched")

    if rel20 < -5:
        penalty += 0.03
        reasons.append("Relative performance is weak")

    if beta_like > 1.8:
        penalty += 0.03
        reasons.append("Market sensitivity is elevated")

    return penalty, reasons[:5]


def compute_quality_score(wf_metrics: pd.DataFrame) -> tuple[float, str]:
    if wf_metrics.empty or "f1" not in wf_metrics.columns:
        return 0.50, "Limited validation"

    lead = wf_metrics[wf_metrics["model"].isin(["Gradient Boosting", "Random Forest"])].copy()
    if lead.empty:
        lead = wf_metrics.copy()

    best_f1 = float(lead["f1"].max())
    mean_f1 = float(lead["f1"].mean())

    score = 0.50 + (best_f1 - 0.50) * 1.6 + (mean_f1 - 0.45) * 0.8
    score = clip01(score)

    if score >= 0.75:
        label = "Strong validation"
    elif score >= 0.60:
        label = "Acceptable validation"
    elif score >= 0.48:
        label = "Mixed validation"
    else:
        label = "Weak validation"

    return score, label


def final_recommendation_label(score: float) -> str:
    if score >= 0.64:
        return "STRONG BUY"
    if score >= 0.57:
        return "BUY"
    if score >= 0.47:
        return "WATCH"
    if score >= 0.40:
        return "AVOID"
    return "STRONG AVOID"


def build_analyst_engine(latest_row: pd.Series, latest_preds: pd.DataFrame, wf_metrics: pd.DataFrame, calibration_model: IsotonicRegression | None = None) -> dict:
    # -------------------------
    # Live ensemble weights
    # -------------------------
    weights = {
        "Gradient Boosting": 1.5,
        "Random Forest": 1.1,
    }

    model_outputs = {}
    for _, row in latest_preds.iterrows():
        model_name = row["model"]
        prob = row["latest_up_probability"]
        if model_name in weights:
            model_outputs[model_name] = float(prob)

    weighted_probs = []
    total_weight = 0.0
    for model_name, prob in model_outputs.items():
        w = weights[model_name]
        weighted_probs.append(w * prob)
        total_weight += w

    ensemble_prob = 0.5 if total_weight == 0 else sum(weighted_probs) / total_weight
    score = (ensemble_prob - 0.5) * 2.0  # centered score in [-1, 1]

    # -------------------------
    # Agreement
    # -------------------------
    directions = [1 if p > 0.5 else -1 for p in model_outputs.values()]
    agreement = abs(sum(directions)) / len(directions) if directions else 0.0

    if agreement > 0.7:
        agreement_label = "High"
    elif agreement > 0.4:
        agreement_label = "Moderate"
    else:
        agreement_label = "Low"

    # -------------------------
    # Risk
    # -------------------------
    vol = safe_float(latest_row.get("volatility_20d_annualized", 20.0), 20.0)
    risk_penalty = min(1.0, vol / 40.0)

    if risk_penalty > 0.7:
        risk_label = "High"
    elif risk_penalty > 0.4:
        risk_label = "Moderate"
    else:
        risk_label = "Low"

    # -------------------------
    # News support
    # -------------------------
    news_strength = max(
        abs(safe_float(latest_row.get("news_sentiment_3d", 0.0), 0.0)),
        abs(safe_float(latest_row.get("news_decayed_sentiment_7d", 0.0), 0.0)),
        abs(safe_float(latest_row.get("news_impact_score_3d", 0.0), 0.0)),
    )

    if news_strength > 0.20:
        news_label = "High"
    elif news_strength > 0.05:
        news_label = "Moderate"
    else:
        news_label = "Low"

    # -------------------------
    # Recommendation
    # -------------------------
    if score > 0.20:
        decision = "BUY"
    elif score > 0.05:
        decision = "WEAK BUY"
    elif score < -0.20:
        decision = "AVOID"
    elif score < -0.05:
        decision = "WEAK AVOID"
    else:
        decision = "HOLD"

    # -------------------------
    # Confidence
    # -------------------------
    raw_confidence = (
        0.6 * abs(score) +
        0.25 * agreement +
        0.15 * (1 - risk_penalty)
    )
    raw_confidence = min(float(raw_confidence), 0.85)

    # Apply calibration if available
    if calibration_model is not None:
        calibrated_confidence = float(calibration_model.predict([raw_confidence])[0])
        calibrated_confidence = clip01(calibrated_confidence)  # Ensure [0,1]
    else:
        calibrated_confidence = raw_confidence

    confidence = calibrated_confidence  # Use calibrated for display

    if confidence >= 0.72:
        confidence_band = "High"
    elif confidence >= 0.58:
        confidence_band = "Moderate"
    else:
        confidence_band = "Low"

    # -------------------------
    # Dynamic Position Sizing
    # Scales position size with calibrated confidence and reduces for risk
    # Formula: direction * calibrated_confidence * (1 - risk_penalty), bounded [-0.4, 0.4]
    # -------------------------
    if decision in ["BUY", "WEAK BUY"]:
        direction = 1.0
    elif decision in ["AVOID", "WEAK AVOID"]:
        direction = -1.0
    else:
        direction = 0.0

    # Conservative sizing: calibrated confidence drives size, risk penalty reduces it
    base_size = calibrated_confidence * direction
    risk_adjusted_size = base_size * (1 - risk_penalty)
    recommended_position_size = float(np.clip(risk_adjusted_size, -0.4, 0.4))  # Conservative bounds

    # -------------------------
    # Validation quality
    # -------------------------
    if wf_metrics is not None and not wf_metrics.empty and "f1" in wf_metrics.columns:
        best_f1 = float(wf_metrics["f1"].max())
        if best_f1 >= 0.60:
            quality_label = "Strong"
        elif best_f1 >= 0.50:
            quality_label = "Moderate"
        else:
            quality_label = "Weak"
    else:
        quality_label = "Unknown"

    # -------------------------
    # Regime
    # -------------------------
    ma_spread = safe_float(latest_row.get("ma_spread_pct", 0.0), 0.0)
    rel20 = safe_float(latest_row.get("asset_vs_spy_20d", 0.0), 0.0)

    if ma_spread > 1.0 and rel20 > 0:
        trend_regime = "Bullish trend"
    elif ma_spread < -1.0 and rel20 < 0:
        trend_regime = "Bearish trend"
    else:
        trend_regime = "Mixed / sideways"

    if vol >= 45:
        vol_regime = "High volatility"
    elif vol >= 25:
        vol_regime = "Moderate volatility"
    else:
        vol_regime = "Calmer conditions"

    if rel20 > 5:
        strength_regime = "Outperforming benchmark"
    elif rel20 < -5:
        strength_regime = "Underperforming benchmark"
    else:
        strength_regime = "Benchmark-like performance"

    # -------------------------
    # Reasons
    # -------------------------
    reasons = []

    if decision in ["BUY", "WEAK BUY"]:
        reasons.append("Lead models lean bullish")
    elif decision in ["AVOID", "WEAK AVOID"]:
        reasons.append("Lead models lean defensive")
    else:
        reasons.append("Lead models are near neutral")

    if agreement_label == "High":
        reasons.append("Model direction is aligned")
    elif agreement_label == "Low":
        reasons.append("Model direction is mixed")

    if risk_label == "High":
        reasons.append("Volatility risk is elevated")
    elif risk_label == "Low":
        reasons.append("Risk conditions are relatively contained")

    if news_label == "High":
        reasons.append("Recent news flow is meaningful")

    # -------------------------
    # Tables expected by App.py
    # -------------------------
    model_breakdown = pd.DataFrame(
        [
            {"model": k, "probability": v, "weight": weights[k]}
            for k, v in model_outputs.items()
        ]
    )

    if not model_breakdown.empty:
        model_breakdown["weight"] = model_breakdown["weight"] / model_breakdown["weight"].sum()
        model_breakdown["weighted_probability"] = (
            model_breakdown["probability"] * model_breakdown["weight"]
        )

    component_table = pd.DataFrame(
        [
            {"component": "Technical Layer", "score": float(0.5 + 0.5 * score), "comment": "Trend and relative-strength proxy"},
            {"component": "Model Layer", "score": float(ensemble_prob), "comment": "GB + RF led live ensemble"},
            {"component": "Context Layer", "score": float(news_strength), "comment": "News and event context"},
            {"component": "Risk Penalty", "score": float(risk_penalty), "comment": risk_label},
            {"component": "Validation Quality", "score": float(best_f1) if (wf_metrics is not None and not wf_metrics.empty and "f1" in wf_metrics.columns) else np.nan, "comment": quality_label},
        ]
    )

    return {
        "recommendation": decision,
        "recommendation_score": float(ensemble_prob),
        "confidence_score": float(confidence),
        "raw_confidence": float(raw_confidence),
        "calibrated_confidence": float(calibrated_confidence),
        "recommended_position_size": recommended_position_size,
        "confidence_band": confidence_band,
        "technical_probability": float(0.5 + 0.5 * score),
        "model_probability": float(ensemble_prob),
        "context_probability": float(news_strength),
        "model_agreement": agreement_label,
        "risk_level": risk_label,
        "risk_penalty": float(risk_penalty),
        "news_support": news_label,
        "quality_label": quality_label,
        "regime": {
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
            "strength_regime": strength_regime,
        },
        "reasons": reasons,
        "model_breakdown": model_breakdown,
        "component_table": component_table,
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
        f"Technically, the asset is in a {trend_regime.lower()} environment with {vol_regime.lower()}, while relative strength remains "
        f"{strength_regime.lower()}. Lead-model agreement is {agreement.lower()}, and the current news backdrop is "
        f"{news_support.lower()}. The present risk environment is {risk_level.lower()}, so this should be interpreted "
        f"as a structured analyst recommendation rather than a guarantee."
    )