from pathlib import Path
import subprocess
import sys

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

from backtest import run_hybrid_recommendation_backtest, run_portfolio_backtest, run_symbol_backtest
from config import (
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
from modeling import (
    build_hybrid_recommendation,
    explain_hybrid_contributions,
    export_walk_forward_outputs,
    run_symbol_models,
    run_walk_forward_models,
)
from portfolio import equal_weight_weights, max_sharpe_like_weights, min_vol_weights, rolling_optimized_portfolio

DATA_PATH = BASE_DIR / OUTPUT_CSV_PATH
NEWS_HEADLINES_FILE = BASE_DIR / NEWS_HEADLINES_PATH
NEWS_SUMMARY_FILE = BASE_DIR / NEWS_SUMMARY_PATH
PIPELINE_PATH = SRC_DIR / "pipeline.py"


def section_title(title: str, help_text: str):
    col1, col2 = st.columns([20, 1])
    with col1:
        st.markdown(f"### {title}")
    with col2:
        st.markdown("ℹ️", help=help_text)


def confidence_band(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.58:
        return "Moderate"
    return "Low"


def risk_level(penalty: float) -> str:
    if penalty >= 0.12:
        return "High"
    if penalty >= 0.06:
        return "Moderate"
    return "Low"


def news_support_label(row: pd.Series) -> str:
    count_3d = float(row.get("news_count_3d", 0))
    impact_3d = float(row.get("news_impact_score_3d", 0.0))
    market_count = float(row.get("market_news_count_3d", 0))
    if count_3d >= 3 or abs(impact_3d) >= 0.12 or market_count >= 4:
        return "High"
    if count_3d >= 1 or market_count >= 2:
        return "Moderate"
    return "Low"


def clean_reason_lines(reason_text: str) -> list[str]:
    if not isinstance(reason_text, str) or not reason_text.strip():
        return ["Signals are mixed and conviction is limited."]
    parts = [p.strip().capitalize() for p in reason_text.replace(".", "").split(";") if p.strip()]
    return parts[:8] if parts else ["Signals are mixed and conviction is limited."]


def card_html(label: str, value: str, subtitle: str | None = None) -> str:
    subtitle_html = f"<div style='font-size:0.85rem;color:#666;margin-top:0.35rem;'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div style="
        border:1px solid #e6e6e6;
        border-radius:16px;
        padding:18px 18px 14px 18px;
        background:#ffffff;
        min-height:120px;
    ">
        <div style="font-size:0.95rem;color:#555;margin-bottom:0.75rem;">{label}</div>
        <div style="
            font-size:2.2rem;
            line-height:1.15;
            font-weight:700;
            word-break:break-word;
            overflow-wrap:anywhere;
        ">{value}</div>
        {subtitle_html}
    </div>
    """


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
    existing_cols = [c for c in cols if c in latest.columns]
    latest = latest[existing_cols].sort_values(["signal_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    latest = latest.rename(
        columns={
            "date": "latest_date",
            "close": "latest_close",
            "volatility_30d_annualized": "volatility_30d_ann_pct",
        }
    )
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
        card = build_hybrid_recommendation(row, latest_preds, wf_metrics)
        rows.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "recommendation": card["recommendation"],
                "confidence_score": card["confidence_score"],
                "risk_penalty": card["risk_penalty"],
                "model_agreement": card["model_agreement"],
                "note": "Uses current model outputs with historical market features",
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


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption("Market decision-support dashboard combining technical signals, machine learning, news context, and portfolio analytics.")

# Sidebar
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
        help="Higher threshold means the system needs stronger model support before acting.",
    )
    transaction_cost_bps = st.number_input(
        "Estimated Trading Cost (bps)",
        min_value=0,
        max_value=100,
        value=DEFAULT_TRANSACTION_COST_BPS,
        step=1,
        help="Applied in backtesting to keep strategy results more realistic.",
    )
    slippage_bps = st.number_input(
        "Estimated Execution Slippage (bps)",
        min_value=0,
        max_value=100,
        value=DEFAULT_SLIPPAGE_BPS,
        step=1,
        help="Represents price movement or execution friction during trading.",
    )

with st.sidebar.expander("Advanced Research Settings", expanded=False):
    strategy_mode = st.selectbox("Strategy Mode", ["long_only", "long_short"], index=0)
    use_signal_filter = st.checkbox("Use Signal Filter In Portfolio", value=False)
    train_window = st.number_input("Train Window", min_value=40, max_value=120, value=80, step=10)
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

with st.expander("How to read this dashboard", expanded=False):
    st.markdown(
        """
        - **Final Recommendation** combines technical signals, machine learning outputs, news context, and risk penalties.  
        - **Confidence** shows how strong the overall recommendation is.  
        - **Model Agreement** shows whether the models broadly support the same direction.  
        - **News Support** reflects how much recent news context is available and how impactful it appears to be.  
        - **Advanced tabs** are available for deeper validation, but the main screen is designed for fast decision support.
        """
    )

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

hybrid_card = None
contrib_df = pd.DataFrame()
hybrid_error = None

try:
    hybrid_card = build_hybrid_recommendation(latest_row, latest_preds, wf_metrics)
    contrib_df = explain_hybrid_contributions(latest_row, latest_preds)
except Exception as e:
    hybrid_error = str(e)

section_title(
    "Final Recommendation",
    "This is the main decision output. It combines technical structure, machine learning, recent news context, and risk penalties.",
)

if hybrid_error is not None:
    st.error(f"Recommendation engine error: {hybrid_error}")

if hybrid_card is not None:
    confidence_text = confidence_band(float(hybrid_card["confidence_score"]))
    risk_text = risk_level(float(hybrid_card["risk_penalty"]))
    news_support = news_support_label(latest_row)

    card1, card2, card3, card4, card5 = st.columns(5)

    with card1:
        st.markdown(
            card_html("Decision", hybrid_card["recommendation"]),
            unsafe_allow_html=True,
        )
    with card2:
        st.markdown(
            card_html("Confidence", f'{hybrid_card["confidence_score"]:.2f}', confidence_text),
            unsafe_allow_html=True,
        )
    with card3:
        st.markdown(
            card_html("Model Agreement", hybrid_card["model_agreement"]),
            unsafe_allow_html=True,
        )
    with card4:
        st.markdown(
            card_html("Risk Level", risk_text),
            unsafe_allow_html=True,
        )
    with card5:
        st.markdown(
            card_html("News Support", news_support),
            unsafe_allow_html=True,
        )

    section_title(
        "Why this decision?",
        "These are the main reasons driving the current recommendation. They summarize technical structure, model support, news context, and risk concerns.",
    )

    reason_lines = clean_reason_lines(hybrid_card["reason_text"])
    for reason in reason_lines:
        st.markdown(f"- {reason}")

    if not contrib_df.empty:
        with st.expander("Decision components", expanded=False):
            st.dataframe(
                contrib_df[["component", "raw_value", "weighted_contribution"]],
                use_container_width=True,
            )

section_title(
    "Market Summary",
    "This table ranks the latest state of each asset using recent trend, momentum, risk, and news context.",
)
st.dataframe(summary_table, use_container_width=True)

tabs = st.tabs(["Overview", "News & Context", "Strategy", "Portfolio", "Advanced"])
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
        "This section shows whether recent headlines are supportive, negative, or limited. It helps explain whether the recommendation has news support or relies mostly on technical and model signals.",
    )

    row_left, row_right = st.columns(2)
    with row_left:
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

    with row_right:
        news_metric_cols = [
            "news_headline_count",
            "news_sentiment_3d",
            "news_sentiment_7d",
            "news_impact_score_3d",
            "news_decayed_sentiment_7d",
            "market_news_sentiment_3d",
            "market_news_impact_score_3d",
        ]
        existing_news_metric_cols = [c for c in news_metric_cols if c in latest_row.index]
        if existing_news_metric_cols:
            st.dataframe(
                pd.DataFrame(
                    {"metric": existing_news_metric_cols, "value": [latest_row[c] for c in existing_news_metric_cols]}
                ),
                use_container_width=True,
                hide_index=True,
            )

    if not symbol_news.empty:
        section_title(
            "Recent Headlines",
            "These are the latest headlines used as part of the news context layer. They help explain whether the system is reacting to supportive, neutral, or negative information.",
        )
        st.dataframe(
            symbol_news[["published_at", "headline", "source", "headline_sentiment", "headline_sentiment_label", "link"]],
            use_container_width=True,
        )
    else:
        st.info("No recent headlines were captured for this symbol.")

with strategy_tab:
    section_title(
        "Strategy Validation",
        "This section shows how the system behaves when translated into trading rules. It is not a guarantee of future performance, but it helps validate consistency and realism.",
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
        "Portfolio Analytics",
        "This section compares different allocation styles so the user can evaluate not only one asset, but also how multiple assets behave together in a portfolio.",
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
        "Advanced Validation",
        "This area is for deeper inspection of model quality, feature influence, raw outputs, and research diagnostics.",
    )

    if model_error is not None:
        st.warning(f"Model metrics unavailable: {model_error}")
    if wf_error is not None:
        st.warning(f"Walk-forward metrics unavailable: {wf_error}")

    adv1, adv2 = st.columns(2)

    with adv1:
        if not model_metrics.empty:
            st.markdown("**Standard Model Metrics**")
            st.dataframe(model_metrics, use_container_width=True)

        if not latest_preds.empty:
            st.markdown("**Latest Model Outputs**")
            st.dataframe(latest_preds, use_container_width=True)

        if not coef_df.empty:
            st.markdown("**Logistic Feature Influence**")
            st.plotly_chart(plot_logistic_coefficients(coef_df, selected_symbol), use_container_width=True)

    with adv2:
        st.markdown("**Data Health**")
        st.dataframe(data_health, use_container_width=True)

        if compare_symbols:
            st.markdown("**Asset Comparison**")
            st.plotly_chart(plot_compare_normalized(filtered_data, compare_symbols), use_container_width=True)

    with st.expander("More Technical Charts", expanded=False):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.plotly_chart(plot_daily_returns(filtered_data, selected_symbol), use_container_width=True)
            st.plotly_chart(plot_compare_close(filtered_data, compare_symbols), use_container_width=True)
        with tc2:
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