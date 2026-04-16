
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
    ENSEMBLE_GB_WEIGHT,
    ENSEMBLE_LOGISTIC_WEIGHT,
    ENSEMBLE_MLP_WEIGHT,
    ENSEMBLE_NEWS_WEIGHT,
    ENSEMBLE_RF_WEIGHT,
    ENSEMBLE_SIGNAL_WEIGHT,
    MIN_MODEL_ROWS,
    MODEL_BACKTEST_EXPORT_PATH,
    MODEL_FEATURES,
    PREDICTIONS_EXPORT_PATH,
    TRADING_DAYS_PER_YEAR,
    WALK_FORWARD_TEST_WINDOW,
    WALK_FORWARD_TRAIN_WINDOW,
)


def _prepare_symbol_model_data(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df["symbol"] == symbol].sort_values("date").copy()

    if subset.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    subset["target_next_day_up"] = (subset["close"].shift(-1) > subset["close"]).astype(float)
    subset["target_next_day_return_decimal"] = (subset["close"].shift(-1) / subset["close"]) - 1.0

    subset = subset.dropna(subset=MODEL_FEATURES).copy()
    labeled = subset.dropna(subset=["target_next_day_up", "target_next_day_return_decimal"]).copy()

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
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42, C=C, class_weight=class_weight)),
    ])


def _make_mlp_model(hidden_layer_sizes: tuple, alpha: float):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )),
    ])


def _make_rf_model(n_estimators: int, max_depth: int | None):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _make_gb_model(n_estimators: int, learning_rate: float, max_depth: int):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )),
    ])


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
    y_subtrain = subtrain["target_next_day_up"].astype(int)
    X_valid = valid[MODEL_FEATURES]
    y_valid = valid["target_next_day_up"].astype(int)

    results = []

    _, log_cfg, log_f1 = _tune_logistic(X_subtrain, y_subtrain, X_valid, y_valid)
    _, mlp_cfg, mlp_f1 = _tune_mlp(X_subtrain, y_subtrain, X_valid, y_valid)
    _, rf_cfg, rf_f1 = _tune_rf(X_subtrain, y_subtrain, X_valid, y_valid)
    _, gb_cfg, gb_f1 = _tune_gb(X_subtrain, y_subtrain, X_valid, y_valid)

    X_full = train_df[MODEL_FEATURES]
    y_full = train_df["target_next_day_up"].astype(int)

    models = {
        "Logistic Regression": _make_logistic_model(log_cfg["C"], log_cfg["class_weight"]),
        "Neural Net (MLP)": _make_mlp_model(mlp_cfg["hidden_layer_sizes"], mlp_cfg["alpha"]),
        "Random Forest": _make_rf_model(rf_cfg["n_estimators"], rf_cfg["max_depth"]),
        "Gradient Boosting": _make_gb_model(gb_cfg["n_estimators"], gb_cfg["learning_rate"], gb_cfg["max_depth"]),
    }

    for model in models.values():
        model.fit(X_full, y_full)

    tuning_info = pd.DataFrame([
        {"model": "Logistic Regression", "best_validation_f1": log_f1, "best_config": str(log_cfg)},
        {"model": "Neural Net (MLP)", "best_validation_f1": mlp_f1, "best_config": str(mlp_cfg)},
        {"model": "Random Forest", "best_validation_f1": rf_f1, "best_config": str(rf_cfg)},
        {"model": "Gradient Boosting", "best_validation_f1": gb_f1, "best_config": str(gb_cfg)},
    ])

    return models, tuning_info


def run_symbol_models(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset_all, labeled = _prepare_symbol_model_data(df, symbol)
    train, test = _chronological_split(labeled, test_fraction=0.2)

    X_test = test[MODEL_FEATURES]
    y_test = test["target_next_day_up"].astype(int)

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

        latest_rows.append({
            "model": model_name,
            "latest_up_probability": latest_prob,
            "latest_predicted_class": latest_pred,
        })

        if model_name == "Logistic Regression":
            coef_values = model.named_steps["model"].coef_[0]
            coef_df = pd.DataFrame({
                "feature": MODEL_FEATURES,
                "coefficient": coef_values,
            }).sort_values("coefficient", ascending=False).reset_index(drop=True)

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
        train_slice = labeled.iloc[start:start + train_window].copy()
        test_slice = labeled.iloc[start + train_window:start + train_window + test_window].copy()
        if test_slice.empty:
            break

        models, tuning_info = _train_tuned_models(train_slice)
        X_test = test_slice[MODEL_FEATURES]
        y_test = test_slice["target_next_day_up"].astype(int)

        for model_name, model in models.items():
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= prob_threshold).astype(int)

            temp = test_slice[[
                "date", "symbol", "close", "signal_score", "volatility_30d_annualized",
                "drawdown_pct", "news_headline_count", "news_avg_sentiment",
                "news_positive_ratio", "news_negative_ratio",
                "target_next_day_up", "target_next_day_return_decimal"
            ]].copy()

            temp["model"] = model_name
            temp["predicted_up_probability"] = probs
            temp["predicted_class"] = preds
            temp["actual_class"] = y_test.values
            temp["strategy_return_decimal"] = np.where(
                temp["predicted_up_probability"] >= prob_threshold,
                temp["target_next_day_return_decimal"],
                0.0,
            )
            temp["buyhold_return_decimal"] = temp["target_next_day_return_decimal"]
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

        metric_rows.append({
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
        })

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


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _signal_to_probability(signal_score: float) -> float:
    return _clip01(0.5 + (float(signal_score) / 12.0))


def _news_to_probability(news_avg_sentiment: float, news_positive_ratio: float, news_negative_ratio: float) -> float:
    sentiment_component = 0.5 + (float(news_avg_sentiment) * 0.35)
    ratio_component = 0.5 + ((float(news_positive_ratio) - float(news_negative_ratio)) * 0.25)
    return _clip01((sentiment_component + ratio_component) / 2.0)


def _risk_penalty(row: pd.Series) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons = []
    vol = float(row.get("volatility_30d_annualized", 0.0))
    dd = float(row.get("drawdown_pct", 0.0))
    rsi = float(row.get("rsi_14", 50.0))
    rel30 = float(row.get("rel_return_30d_pct", 0.0))
    rel90 = float(row.get("rel_return_90d_pct", 0.0))

    if vol >= 55:
        penalty += 0.12
        reasons.append("very high volatility")
    elif vol >= 40:
        penalty += 0.07
        reasons.append("elevated volatility")

    if dd <= -20:
        penalty += 0.10
        reasons.append("deep drawdown")
    elif dd <= -10:
        penalty += 0.05
        reasons.append("moderate drawdown")

    if rsi >= 75:
        penalty += 0.04
        reasons.append("overbought momentum risk")

    if rel30 < -5:
        penalty += 0.04
        reasons.append("recent underperformance vs benchmark")

    if rel90 < -10:
        penalty += 0.05
        reasons.append("longer-term underperformance vs benchmark")

    return penalty, reasons


def _recommendation_label(score: float) -> str:
    if score >= 0.70:
        return "STRONG BUY"
    if score >= 0.60:
        return "BUY"
    if score >= 0.52:
        return "WATCH"
    if score >= 0.45:
        return "AVOID"
    return "STRONG AVOID"


def explain_hybrid_contributions(latest_row: pd.Series, latest_preds: pd.DataFrame) -> pd.DataFrame:
    prob_map = {row["model"]: float(row["latest_up_probability"]) for _, row in latest_preds.iterrows()} if not latest_preds.empty else {}

    signal_prob = _signal_to_probability(float(latest_row.get("signal_score", 0)))
    news_prob = _news_to_probability(
        float(latest_row.get("news_avg_sentiment", 0.0)),
        float(latest_row.get("news_positive_ratio", 0.0)),
        float(latest_row.get("news_negative_ratio", 0.0)),
    )

    rows = [
        {"component": "Rule Signal", "raw_value": signal_prob, "weight": ENSEMBLE_SIGNAL_WEIGHT, "weighted_contribution": signal_prob * ENSEMBLE_SIGNAL_WEIGHT},
        {"component": "Logistic Regression", "raw_value": prob_map.get("Logistic Regression", 0.50), "weight": ENSEMBLE_LOGISTIC_WEIGHT, "weighted_contribution": prob_map.get("Logistic Regression", 0.50) * ENSEMBLE_LOGISTIC_WEIGHT},
        {"component": "Neural Net (MLP)", "raw_value": prob_map.get("Neural Net (MLP)", 0.50), "weight": ENSEMBLE_MLP_WEIGHT, "weighted_contribution": prob_map.get("Neural Net (MLP)", 0.50) * ENSEMBLE_MLP_WEIGHT},
        {"component": "Random Forest", "raw_value": prob_map.get("Random Forest", 0.50), "weight": ENSEMBLE_RF_WEIGHT, "weighted_contribution": prob_map.get("Random Forest", 0.50) * ENSEMBLE_RF_WEIGHT},
        {"component": "Gradient Boosting", "raw_value": prob_map.get("Gradient Boosting", 0.50), "weight": ENSEMBLE_GB_WEIGHT, "weighted_contribution": prob_map.get("Gradient Boosting", 0.50) * ENSEMBLE_GB_WEIGHT},
        {"component": "News", "raw_value": news_prob, "weight": ENSEMBLE_NEWS_WEIGHT, "weighted_contribution": news_prob * ENSEMBLE_NEWS_WEIGHT},
    ]
    return pd.DataFrame(rows).sort_values("weighted_contribution", ascending=False).reset_index(drop=True)


def build_hybrid_recommendation(
    latest_row: pd.Series,
    latest_preds: pd.DataFrame,
    wf_metrics: pd.DataFrame | None = None,
) -> dict:
    prob_map = {row["model"]: float(row["latest_up_probability"]) for _, row in latest_preds.iterrows()} if not latest_preds.empty else {}

    logistic_prob = prob_map.get("Logistic Regression", 0.50)
    mlp_prob = prob_map.get("Neural Net (MLP)", 0.50)
    rf_prob = prob_map.get("Random Forest", 0.50)
    gb_prob = prob_map.get("Gradient Boosting", 0.50)

    signal_prob = _signal_to_probability(float(latest_row.get("signal_score", 0)))
    news_prob = _news_to_probability(
        float(latest_row.get("news_avg_sentiment", 0.0)),
        float(latest_row.get("news_positive_ratio", 0.0)),
        float(latest_row.get("news_negative_ratio", 0.0)),
    )

    base_score = (
        ENSEMBLE_SIGNAL_WEIGHT * signal_prob +
        ENSEMBLE_LOGISTIC_WEIGHT * logistic_prob +
        ENSEMBLE_MLP_WEIGHT * mlp_prob +
        ENSEMBLE_RF_WEIGHT * rf_prob +
        ENSEMBLE_GB_WEIGHT * gb_prob +
        ENSEMBLE_NEWS_WEIGHT * news_prob
    )

    penalty, penalty_reasons = _risk_penalty(latest_row)
    final_score = _clip01(base_score - penalty)

    model_probs = [logistic_prob, mlp_prob, rf_prob, gb_prob]
    agreement_gap = max(model_probs) - min(model_probs)
    model_agreement = "High" if agreement_gap <= 0.07 else ("Medium" if agreement_gap <= 0.15 else "Low")

    reasons = []
    if float(latest_row.get("signal_score", 0)) >= 3:
        reasons.append("rule-based signal is bullish")
    elif float(latest_row.get("signal_score", 0)) <= -2:
        reasons.append("rule-based signal is bearish")

    if float(latest_row.get("rel_return_30d_pct", 0.0)) > 0:
        reasons.append("asset is outperforming the benchmark over 30d")
    elif float(latest_row.get("rel_return_30d_pct", 0.0)) < 0:
        reasons.append("asset is underperforming the benchmark over 30d")

    if logistic_prob >= 0.60:
        reasons.append("logistic model is supportive")
    if mlp_prob >= 0.60:
        reasons.append("neural net model is supportive")
    if rf_prob >= 0.60:
        reasons.append("random forest is supportive")
    if gb_prob >= 0.60:
        reasons.append("gradient boosting is supportive")

    if float(latest_row.get("news_avg_sentiment", 0.0)) >= 0.15:
        reasons.append("recent news tone is positive")
    elif float(latest_row.get("news_avg_sentiment", 0.0)) <= -0.15:
        reasons.append("recent news tone is negative")

    if float(latest_row.get("ma_spread_pct", 0.0)) > 0:
        reasons.append("short-term trend is above medium-term trend")

    if wf_metrics is not None and not wf_metrics.empty:
        best_f1 = float(wf_metrics["f1"].max())
        if best_f1 >= 0.55:
            reasons.append("walk-forward model performance is acceptable")
        else:
            reasons.append("walk-forward model performance is modest")

    reasons.extend(penalty_reasons)
    if not reasons:
        reasons.append("signals are mixed and conviction is limited")

    return {
        "recommendation": _recommendation_label(final_score),
        "confidence_score": final_score,
        "base_score_before_risk": base_score,
        "risk_penalty": penalty,
        "model_agreement": model_agreement,
        "logistic_up_probability": logistic_prob,
        "mlp_up_probability": mlp_prob,
        "rf_up_probability": rf_prob,
        "gb_up_probability": gb_prob,
        "signal_implied_probability": signal_prob,
        "news_implied_probability": news_prob,
        "reason_text": "; ".join(reasons[:8]) + ".",
    }
