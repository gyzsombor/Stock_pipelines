from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from visuals import (
    plot_backtest_equity,
    plot_compare_close,
    plot_compare_normalized,
    plot_cumulative_return,
    plot_daily_returns,
    plot_drawdown,
    plot_logistic_coefficients,
    plot_price_ma,
    plot_risk_adjusted_return,
    plot_rsi,
    plot_signals,
    plot_volatility,
    plot_vs_benchmark,
)

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.backtest import run_hybrid_recommendation_backtest, run_portfolio_backtest, run_symbol_backtest
from src.config import (
    APP_TITLE,
    BENCHMARK_SYMBOL,
    DEFAULT_PROB_THRESHOLD,
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRANSACTION_COST_BPS,
    MAX_COMPARE_DEFAULT,
    NEWS_HEADLINES_PATH,
    NEWS_SUMMARY_PATH,
    OUTPUT_CSV_PATH,
    PREDICTIONS_EXPORT_PATH,
    ROLLING_OPT_HOLD_WINDOW,
    ROLLING_OPT_TRAIN_WINDOW,
)
from src.llm_explainer import generate_analyst_memo_llm
from src.modeling import (
    build_analyst_engine,
    confidence_band,
    export_walk_forward_outputs,
    risk_level_from_penalty,
    run_symbol_models,
    run_walk_forward_models,
)
from src.portfolio import equal_weight_weights, max_sharpe_like_weights, min_vol_weights, rolling_optimized_portfolio

DATA_PATH = BASE_DIR / OUTPUT_CSV_PATH
NEWS_HEADLINES_FILE = BASE_DIR / NEWS_HEADLINES_PATH
NEWS_SUMMARY_FILE = BASE_DIR / NEWS_SUMMARY_PATH
PIPELINE_PATH = SRC_DIR / "pipeline.py"

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .app-card {
        border: 1px solid #e7e7e7;
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        background: #ffffff;
        min-height: 128px;
    }
    .app-card-label {
        font-size: 0.95rem;
        color: #666666;
        margin-bottom: 0.65rem;
    }
    .app-card-value {
        font-size: 2.0rem;
        line-height: 1.15;
        font-weight: 700;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    .app-card-sub {
        font-size: 0.86rem;
        color: #666666;
        margin-top: 0.4rem;
    }
    .memo-box {
        border: 1px solid #ececec;
        border-radius: 18px;
        padding: 18px;
        background: #fafafa;
    }
    .pill-box {
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 12px 14px;
        background: #fafafa;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_title(title: str, help_text: str) -> None:
    c1, c2 = st.columns([25, 1])
    with c1:
        st.markdown(f"### {title}")
    with c2:
        st.markdown("ℹ️", help=help_text)


def card_html(label: str, value: str, subtitle: str | None = None) -> str:
    subtitle_html = f"<div class='app-card-sub'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div class="app-card">
        <div class="app-card-label">{label}</div>
        <div class="app-card-value">{value}</div>
        {subtitle_html}
    </div>
    """


def pill_html(title: str, value: str) -> str:
    return f"""
    <div class="pill-box">
        <div style="font-size:0.85rem;color:#666;">{title}</div>
        <div style="font-size:1.05rem;font-weight:600;margin-top:0.25rem;">{value}</div>
    </div>
    """


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def news_support_label(row: pd.Series) -> str:
    count_3d = safe_float(row.get("news_count_3d", 0))
    impact_3d = abs(safe_float(row.get("news_impact_score_3d", 0.0)))
    market_count_3d = safe_float(row.get("market_news_count_3d", 0))
    if count_3d >= 3 or impact_3d >= 0.12 or market_count_3d >= 4:
        return "High"
    if count_3d >= 1 or market_count_3d >= 2:
        return "Moderate"
    return "Low"


def run_pipeline() -> tuple[bool, str]:
    if not PIPELINE_PATH.exists():
        return False, f"Pipeline file not found: {PIPELINE_PATH}"
    try:
        result = subprocess.run(
            [sys.executable, str(PIPELINE_PATH)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout if result.stdout else "Pipeline finished successfully."
    except subprocess.CalledProcessError as e:
        return False, e.stderr or e.stdout or str(e)
    except Exception as e:
        return False, str(e)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Click the refresh button to run the pipeline.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


@st.cache_data
def load_news_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "latest_headline_time" in df.columns:
        df["latest_headline_time"] = pd.to_datetime(df["latest_headline_time"], errors="coerce")
    return df


@st.cache_data
def load_news_headlines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.sort_values("date").groupby("symbol", as_index=False).tail(1).copy()
    cols = [
        "symbol",
        "date",
        "close",
        "return_7d_pct",
        "return_30d_pct",
        "return_90d_pct",
        "rel_return_30d_pct",
        "rel_return_90d_pct",
        "volatility_30d_annualized",
        "rsi_14",
        "ma_spread_pct",
        "trend_regime",
        "signal",
        "signal_score",
        "news_headline_count",
        "news_avg_sentiment",
        "news_count_3d",
        "news_impact_score_3d",
        "market_news_sentiment_3d",
    ]
    existing = [c for c in cols if c in latest.columns]
    latest = latest[existing].sort_values(["signal_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    latest = latest.rename(columns={"date": "latest_date", "close": "latest_close", "volatility_30d_annualized": "volatility_30d_ann_pct"})
    return latest


def build_data_health_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date")
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(group)),
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "pct_rows_with_news": float((group["news_headline_count"] > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def plot_confidence_history(hist_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    if not hist_df.empty:
        fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["confidence_score"], mode="lines", name="Confidence"))
    fig.update_layout(
        title=f"{symbol} Confidence Trend",
        xaxis_title="Date",
        yaxis_title="Confidence",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_recommendation_history(symbol_df: pd.DataFrame, latest_preds: pd.DataFrame, wf_metrics: pd.DataFrame) -> pd.DataFrame:
    recent = symbol_df.sort_values("date").tail(90).copy()
    rows = []

    for _, row in recent.iterrows():
        analyst = build_analyst_engine(row, latest_preds, wf_metrics)
        rows.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "recommendation": analyst["recommendation"],
                "confidence_score": analyst["confidence_score"],
                "risk_penalty": analyst["risk_penalty"],
                "model_agreement": analyst["model_agreement"],
            }
        )

    return pd.DataFrame(rows)


def recommendation_to_num(label: str) -> float:
    mapping = {"STRONG BUY": 5, "BUY": 4, "WATCH": 3, "AVOID": 2, "STRONG AVOID": 1}
    return mapping.get(label, 3)


def plot_recommendation_history(hist_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    if not hist_df.empty:
        temp = hist_df.copy()
        temp["recommendation_num"] = temp["recommendation"].map(recommendation_to_num)
        fig.add_trace(go.Scatter(x=temp["date"], y=temp["recommendation_num"], mode="lines+markers", name="Recommendation"))
    fig.update_layout(
        title=f"{symbol} Recommendation History",
        xaxis_title="Date",
        yaxis_title="Recommendation Scale",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_walkforward_equity(predictions_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for model_name, group in predictions_df.groupby("model"):
        temp = group.sort_values("date").copy()
        temp["strategy_equity"] = (1.0 + temp["strategy_return_decimal"].fillna(0.0)).cumprod()
        temp["buyhold_equity"] = (1.0 + temp["buyhold_return_decimal"].fillna(0.0)).cumprod()
        fig.add_trace(go.Scatter(x=temp["date"], y=temp["strategy_equity"], mode="lines", name=f"{model_name} Strategy"))
        fig.add_trace(go.Scatter(x=temp["date"], y=temp["buyhold_equity"], mode="lines", name=f"{model_name} Buy & Hold"))
    fig.update_layout(
        title="Model Strategy vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


st.title(APP_TITLE)
st.caption("AI-assisted analyst engine with explainable decision logic, technical validation, news context, and portfolio research.")

st.sidebar.header("Main Controls")

if st.sidebar.button("Run / Refresh Pipeline", use_container_width=True):
    with st.spinner("Running pipeline..."):
        success, message = run_pipeline()
    if success:
        st.cache_data.clear()
        st.sidebar.success("Pipeline refreshed successfully.")
        st.text_area("Pipeline Output", message, height=220)
        st.rerun()
    else:
        st.sidebar.error("Pipeline failed.")
        st.text_area("Pipeline Error", message, height=220)
        st.stop()

data = load_data(DATA_PATH)
news_summary = load_news_summary(NEWS_SUMMARY_FILE)
news_headlines = load_news_headlines(NEWS_HEADLINES_FILE)

symbols = sorted(data["symbol"].dropna().unique().tolist())
min_date = data["date"].min().date()
max_date = data["date"].max().date()

selected_dates = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = min_date, max_date

filtered_data = data[(data["date"].dt.date >= start_date) & (data["date"].dt.date <= end_date)].copy()
symbols_filtered = sorted(filtered_data["symbol"].dropna().unique().tolist())
selected_symbol = st.sidebar.selectbox("Select Asset", symbols_filtered, index=0)

default_compare = symbols_filtered[: min(MAX_COMPARE_DEFAULT, len(symbols_filtered))]
compare_symbols = st.sidebar.multiselect("Compare Assets", symbols_filtered, default=default_compare)
portfolio_symbols = st.sidebar.multiselect("Portfolio Assets", symbols_filtered, default=default_compare if default_compare else symbols_filtered[:1])

benchmark_symbol = BENCHMARK_SYMBOL if BENCHMARK_SYMBOL in symbols_filtered else (symbols_filtered[0] if symbols_filtered else BENCHMARK_SYMBOL)

with st.sidebar.expander("Optional Settings", expanded=False):
    prob_threshold = st.slider(
        "Model Confidence Threshold",
        min_value=0.50,
        max_value=0.80,
        value=float(DEFAULT_PROB_THRESHOLD),
        step=0.01,
        help="Higher threshold means the system requires stronger model support before acting.",
    )
    transaction_cost_bps = st.number_input(
        "Estimated Trading Cost (bps)",
        min_value=0,
        max_value=100,
        value=DEFAULT_TRANSACTION_COST_BPS,
        step=1,
        help="Applied in backtesting to keep performance estimates more realistic.",
    )
    slippage_bps = st.number_input(
        "Estimated Execution Slippage (bps)",
        min_value=0,
        max_value=100,
        value=DEFAULT_SLIPPAGE_BPS,
        step=1,
        help="Represents friction between expected and executed trade price.",
    )

with st.sidebar.expander("Advanced Research Settings", expanded=False):
    strategy_mode = st.selectbox("Strategy Mode", ["long_only", "long_short"], index=0)
    use_signal_filter = st.checkbox("Use Signal Filter In Portfolio", value=False)
    train_window = st.number_input("Train Window", min_value=40, max_value=120, value=100, step=10)
    test_window = st.number_input("Test Window", min_value=5, max_value=30, value=20, step=5)
    rebalance_frequency = st.selectbox(
        "Rebalance Frequency",
        ["daily", "weekly", "monthly"],
        index=["daily", "weekly", "monthly"].index(DEFAULT_REBALANCE_FREQUENCY),
    )
    rolling_train_window = st.number_input("Rolling Train Window", min_value=60, max_value=400, value=ROLLING_OPT_TRAIN_WINDOW, step=21)
    rolling_hold_window = st.number_input("Rolling Hold Window", min_value=5, max_value=63, value=ROLLING_OPT_HOLD_WINDOW, step=5)

summary_table = build_summary_table(filtered_data)
latest_row = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date").iloc[-1]
data_health = build_data_health_table(filtered_data)

model_metrics = pd.DataFrame()
latest_preds = pd.DataFrame()
coef_df = pd.DataFrame()
wf_predictions = pd.DataFrame()
wf_metrics = pd.DataFrame()
wf_backtest = pd.DataFrame()

model_error = None
wf_error = None

try:
    model_metrics, latest_preds, coef_df = run_symbol_models(filtered_data, selected_symbol)
except Exception as e:
    model_error = str(e)

try:
    wf_predictions, wf_metrics, wf_backtest = run_walk_forward_models(
        filtered_data,
        selected_symbol,
        train_window=int(train_window),
        test_window=int(test_window),
        prob_threshold=float(prob_threshold),
    )
except Exception as e:
    wf_error = str(e)

analyst_error = None
analyst = None

try:
    analyst = build_analyst_engine(latest_row, latest_preds, wf_metrics)
except Exception as e:
    analyst_error = str(e)

section_title(
    "Final Recommendation",
    "This is the main analyst output. It uses technical structure, machine learning, recent news context, and risk control to produce a disciplined view rather than a guess.",
)

if analyst_error is not None:
    st.error(f"Analyst engine error: {analyst_error}")

if analyst is not None:
    llm_note = generate_analyst_memo_llm(selected_symbol, latest_row, analyst)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(card_html("Decision", analyst["recommendation"]), unsafe_allow_html=True)
    with c2:
        st.markdown(card_html("Confidence", f'{analyst["confidence_score"]:.2f}', analyst["confidence_band"]), unsafe_allow_html=True)
    with c3:
        st.markdown(card_html("Model Agreement", analyst["model_agreement"]), unsafe_allow_html=True)
    with c4:
        st.markdown(card_html("Risk Level", analyst["risk_level"]), unsafe_allow_html=True)
    with c5:
        st.markdown(card_html("News Support", analyst["news_support"]), unsafe_allow_html=True)

    section_title(
        "Analyst Memo",
        "This is the professional-language explanation layer. It turns model outputs, technical structure, and news context into readable analyst-style guidance.",
    )
    st.markdown(f"<div class='memo-box'>{llm_note['memo']}</div>", unsafe_allow_html=True)
    st.caption(f"Memo mode: {llm_note.get('mode', 'unknown')}")

    section_title(
        "Key Reasons",
        "These are the main evidence points behind the current recommendation.",
    )
    for reason in analyst["reasons"]:
        st.markdown(f"- {reason}")

    with st.expander("Decision components", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown(pill_html("Technical Probability", f'{analyst["technical_probability"]:.2f}'), unsafe_allow_html=True)
            st.markdown(pill_html("Model Probability", f'{analyst["model_probability"]:.2f}'), unsafe_allow_html=True)
        with right:
            st.markdown(pill_html("Context Probability", f'{analyst["context_probability"]:.2f}'), unsafe_allow_html=True)
            st.markdown(pill_html("Validation Quality", analyst["quality_label"]), unsafe_allow_html=True)

        st.dataframe(analyst["component_table"], use_container_width=True)

        if not analyst["model_breakdown"].empty:
            st.dataframe(
                analyst["model_breakdown"][["model", "probability", "weight", "weighted_probability"]],
                use_container_width=True,
            )

section_title(
    "Market Summary",
    "This table gives a quick ranking view across assets using recent trend, momentum, relative strength, and news context.",
)
st.dataframe(summary_table, use_container_width=True)

tabs = st.tabs(["Executive View", "News & Context", "Strategy Validation", "Portfolio Lab", "Advanced Research"])
overview_tab, news_tab, strategy_tab, portfolio_tab, advanced_tab = tabs

with overview_tab:
    top1, top2 = st.columns(2)
    with top1:
        st.plotly_chart(plot_price_ma(filtered_data, selected_symbol), use_container_width=True)
    with top2:
        st.plotly_chart(plot_vs_benchmark(filtered_data, selected_symbol, benchmark_symbol), use_container_width=True)

    bottom1, bottom2 = st.columns(2)
    with bottom1:
        st.plotly_chart(plot_cumulative_return(filtered_data, selected_symbol), use_container_width=True)
    with bottom2:
        try:
            symbol_hist = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date").copy()
            rec_hist = build_recommendation_history(symbol_hist, latest_preds, wf_metrics)
            if not rec_hist.empty:
                st.plotly_chart(plot_confidence_history(rec_hist, selected_symbol), use_container_width=True)
        except Exception as e:
            st.info(f"Confidence history unavailable: {e}")

    section_title(
        "Regime Snapshot",
        "This summarizes how the current market environment looks for the selected asset.",
    )
    if analyst is not None:
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(pill_html("Trend Regime", analyst["regime"]["trend_regime"]), unsafe_allow_html=True)
        with r2:
            st.markdown(pill_html("Volatility Regime", analyst["regime"]["vol_regime"]), unsafe_allow_html=True)
        with r3:
            st.markdown(pill_html("Relative Strength", analyst["regime"]["strength_regime"]), unsafe_allow_html=True)

    with st.expander("Technical Detail", expanded=False):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(plot_rsi(filtered_data, selected_symbol), use_container_width=True)
            st.plotly_chart(plot_signals(filtered_data, selected_symbol), use_container_width=True)
        with right:
            st.plotly_chart(plot_volatility(filtered_data, selected_symbol), use_container_width=True)
            st.plotly_chart(plot_drawdown(filtered_data, selected_symbol), use_container_width=True)

with news_tab:
    symbol_news_summary = news_summary[news_summary["symbol"] == selected_symbol].copy() if not news_summary.empty else pd.DataFrame()
    symbol_news = news_headlines[news_headlines["symbol"] == selected_symbol].copy() if not news_headlines.empty else pd.DataFrame()

    section_title(
        "News Context",
        "This section separates symbol-specific headlines from broader market context so the user can see whether the decision is supported by meaningful external information.",
    )

    left, right = st.columns(2)
    with left:
        if not symbol_news_summary.empty:
            row = symbol_news_summary.iloc[0]
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Headline Count", int(row["headline_count"]))
            n2.metric("Average Sentiment", f"{row['avg_headline_sentiment']:.3f}")
            n3.metric("News Label", str(row["news_sentiment_label"]).title())
            latest_time = row["latest_headline_time"]
            n4.metric("Latest Headline Time", str(latest_time)[:19] if pd.notna(latest_time) else "N/A")
        else:
            st.info("No recent symbol-specific news summary is available for the selected asset.")

    with right:
        metric_cols = [
            "news_headline_count",
            "news_sentiment_3d",
            "news_sentiment_7d",
            "news_impact_score_3d",
            "news_decayed_sentiment_7d",
            "news_high_impact_count_3d",
            "news_macro_event_count_3d",
            "news_company_event_count_3d",
            "market_news_sentiment_3d",
            "market_news_impact_score_3d",
            "market_macro_event_count_3d",
        ]
        available = [c for c in metric_cols if c in latest_row.index]
        if available:
            st.dataframe(
                pd.DataFrame({"metric": available, "value": [latest_row[c] for c in available]}),
                use_container_width=True,
                hide_index=True,
            )

    if not symbol_news.empty:
        section_title(
            "Recent Headlines",
            "These are the headlines feeding the context layer. This helps the user see whether the recommendation is reacting to meaningful events or operating with limited news support.",
        )
        st.dataframe(
            symbol_news[
                [
                    "published_at",
                    "headline",
                    "source",
                    "headline_sentiment",
                    "headline_sentiment_label",
                    "event_type",
                    "event_scope",
                    "impact_bucket",
                    "link",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No recent headlines were captured for this symbol.")

with strategy_tab:
    section_title(
        "Strategy Validation",
        "This section tests how the system behaves when translated into actual strategy rules. It is a validation layer, not a promise of future returns.",
    )

    try:
        bt_df, bt_metrics = run_symbol_backtest(
            filtered_data,
            selected_symbol,
            strategy_mode=strategy_mode,
            transaction_cost_bps=float(transaction_cost_bps),
            slippage_bps=float(slippage_bps),
        )
        st.dataframe(bt_metrics, use_container_width=True)
        st.plotly_chart(plot_backtest_equity(bt_df, selected_symbol), use_container_width=True)
    except Exception as e:
        st.warning(f"Signal strategy backtest unavailable: {e}")

    try:
        symbol_hist = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date").copy()
        rec_hist = build_recommendation_history(symbol_hist, latest_preds, wf_metrics)
        hybrid_bt_df, hybrid_bt_metrics = run_hybrid_recommendation_backtest(
            filtered_data[filtered_data["symbol"] == selected_symbol].copy(),
            rec_hist,
            transaction_cost_bps=float(transaction_cost_bps),
            slippage_bps=float(slippage_bps),
        )
        st.markdown("**Recommendation Strategy Backtest**")
        st.dataframe(hybrid_bt_metrics, use_container_width=True)
    except Exception as e:
        st.info(f"Recommendation backtest unavailable: {e}")

    if not wf_metrics.empty:
        st.markdown("**Walk-Forward Validation**")
        st.dataframe(wf_metrics, use_container_width=True)
        st.plotly_chart(plot_walkforward_equity(wf_predictions), use_container_width=True)
    elif wf_error is not None:
        st.info(f"Walk-forward validation unavailable: {wf_error}")

with portfolio_tab:
    section_title(
        "Portfolio Lab",
        "This section compares multiple allocation styles so the user can move beyond single-asset analysis and evaluate how ideas behave inside a portfolio.",
    )

    if portfolio_symbols:
        try:
            pivot_returns = (
                filtered_data[filtered_data["symbol"].isin(portfolio_symbols)]
                .pivot(index="date", columns="symbol", values="daily_return_decimal")
                .dropna(how="all")
                .fillna(0.0)
            )

            eq_w = equal_weight_weights(portfolio_symbols)
            mv_w = min_vol_weights(pivot_returns[portfolio_symbols])
            ms_w = max_sharpe_like_weights(pivot_returns[portfolio_symbols])

            st.markdown("**Suggested Portfolio Weights**")
            w1, w2, w3 = st.columns(3)
            with w1:
                st.markdown("**Equal Weight**")
                st.dataframe(eq_w, use_container_width=True)
            with w2:
                st.markdown("**Minimum Volatility**")
                st.dataframe(mv_w, use_container_width=True)
            with w3:
                st.markdown("**Max Sharpe-like**")
                st.dataframe(ms_w, use_container_width=True)

            results = []
            curves = []

            for name, wdf in {
                "Equal Weight": eq_w,
                "Minimum Volatility": mv_w,
                "Max Sharpe-like": ms_w,
            }.items():
                weight_map = dict(zip(wdf["symbol"], wdf["weight"]))
                port_df, port_metrics = run_portfolio_backtest(
                    filtered_data,
                    portfolio_symbols,
                    use_signals=use_signal_filter,
                    strategy_mode=strategy_mode,
                    transaction_cost_bps=float(transaction_cost_bps),
                    slippage_bps=float(slippage_bps),
                    weights=weight_map,
                    rebalance_frequency=rebalance_frequency,
                )
                port_metrics["portfolio_type"] = name
                results.append(port_metrics)
                temp = port_df.copy()
                temp["portfolio_type"] = name
                curves.append(temp)

            result_df = pd.concat(results, ignore_index=True)
            curve_df = pd.concat(curves, ignore_index=True)

            st.dataframe(result_df, use_container_width=True)

            fig = go.Figure()
            for name, group in curve_df.groupby("portfolio_type"):
                fig.add_trace(go.Scatter(x=group["date"], y=group["portfolio_equity"], mode="lines", name=name))
            fig.update_layout(
                title="Portfolio Comparison",
                xaxis_title="Date",
                yaxis_title="Growth of $1",
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Portfolio analysis unavailable: {e}")

with advanced_tab:
    section_title(
        "Advanced Research",
        "This area keeps the research layer visible for analyst users without crowding the main decision experience.",
    )

    if model_error is not None:
        st.warning(f"Model metrics unavailable: {model_error}")
    if wf_error is not None:
        st.warning(f"Walk-forward metrics unavailable: {wf_error}")

    a1, a2 = st.columns(2)

    with a1:
        if not model_metrics.empty:
            st.markdown("**Standard Model Metrics**")
            st.dataframe(model_metrics, use_container_width=True)

        if not latest_preds.empty:
            st.markdown("**Latest Model Outputs**")
            st.dataframe(latest_preds, use_container_width=True)

        if not coef_df.empty:
            st.markdown("**Logistic Feature Influence**")
            st.plotly_chart(plot_logistic_coefficients(coef_df, selected_symbol), use_container_width=True)

    with a2:
        st.markdown("**Data Health**")
        st.dataframe(data_health, use_container_width=True)

        if compare_symbols:
            st.markdown("**Normalized Comparison**")
            st.plotly_chart(plot_compare_normalized(filtered_data, compare_symbols), use_container_width=True)

    with st.expander("More Technical Charts", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.plotly_chart(plot_daily_returns(filtered_data, selected_symbol), use_container_width=True)
            st.plotly_chart(plot_compare_close(filtered_data, compare_symbols), use_container_width=True)
        with t2:
            st.plotly_chart(plot_risk_adjusted_return(filtered_data, selected_symbol), use_container_width=True)
            try:
                symbol_hist = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date").copy()
                rec_hist = build_recommendation_history(symbol_hist, latest_preds, wf_metrics)
                if not rec_hist.empty:
                    st.plotly_chart(plot_recommendation_history(rec_hist, selected_symbol), use_container_width=True)
                    st.dataframe(rec_hist.tail(20), use_container_width=True)
            except Exception as e:
                st.info(f"Recommendation history unavailable: {e}")

    with st.expander("Exports", expanded=False):
        if not wf_predictions.empty and not wf_backtest.empty:
            pred_csv = wf_predictions.to_csv(index=False).encode("utf-8")
            bt_csv = wf_backtest.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Walk-Forward Predictions CSV",
                data=pred_csv,
                file_name=f"{selected_symbol.lower()}_walkforward_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download Model Backtest Metrics CSV",
                data=bt_csv,
                file_name=f"{selected_symbol.lower()}_model_backtest_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if st.button("Export Walk-Forward Files To data/ Folder", use_container_width=True):
                pred_path, bt_path = export_walk_forward_outputs(
                    wf_predictions,
                    wf_backtest,
                    predictions_path=PREDICTIONS_EXPORT_PATH,
                    backtest_path=PREDICTIONS_EXPORT_PATH.replace("walkforward_predictions.csv", "model_backtest_metrics.csv"),
                )
                st.success(f"Exported: {pred_path} and {bt_path}")
        else:
            st.info("No walk-forward export files are available right now.")

    with st.expander("Raw Data", expanded=False):
        symbol_data = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date", ascending=False).reset_index(drop=True)
        st.dataframe(symbol_data, use_container_width=True)