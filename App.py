
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
    plot_portfolio_equity,
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
    MODEL_BACKTEST_EXPORT_PATH,
    NEWS_HEADLINES_PATH,
    NEWS_SUMMARY_PATH,
    OUTPUT_CSV_PATH,
    PREDICTIONS_EXPORT_PATH,
    ROLLING_OPT_HOLD_WINDOW,
    ROLLING_OPT_TRAIN_WINDOW,
    WALK_FORWARD_TEST_WINDOW,
    WALK_FORWARD_TRAIN_WINDOW,
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


def info_label(label: str, info: str):
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown(f"**{label}**")
    with col2:
        st.markdown("??", help=info)


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)
st.caption("Advanced market research dashboard with signals, ensemble ML, recommendation engine, portfolio optimization, rebalancing, costs, slippage, and explainability.")


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
        "symbol", "date", "close", "return_7d_pct", "return_30d_pct", "return_90d_pct",
        "rel_return_30d_pct", "rel_return_90d_pct", "volatility_30d_annualized",
        "rsi_14", "ma_spread_pct", "trend_regime", "rsi_status", "volatility_status",
        "signal_score", "signal_strength", "signal", "news_headline_count", "news_avg_sentiment",
        "news_positive_ratio", "news_negative_ratio"
    ]
    latest = latest[cols].sort_values(["signal_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    latest = latest.rename(columns={
        "date": "latest_date",
        "close": "latest_close",
        "volatility_30d_annualized": "volatility_30d_ann_pct",
    })
    return latest


def interpretation_text(row: pd.Series) -> str:
    return (
        f"Trend: {row['trend_regime']}. RSI: {row['rsi_status']}. Volatility: {row['volatility_status']}. "
        f"Signal: {row['signal']}. Signal score: {row['signal_score']}."
    )


def build_data_health_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date")
        rows.append({
            "symbol": symbol,
            "rows": int(len(group)),
            "start_date": group["date"].min(),
            "end_date": group["date"].max(),
            "pct_rows_with_news": float((group["news_headline_count"] > 0).mean() * 100.0),
        })
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def plot_walkforward_equity(predictions_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for model_name, group in predictions_df.groupby("model"):
        temp = group.sort_values("date").copy()
        temp["strategy_equity"] = (1.0 + temp["strategy_return_decimal"].fillna(0.0)).cumprod()
        temp["buyhold_equity"] = (1.0 + temp["buyhold_return_decimal"].fillna(0.0)).cumprod()
        fig.add_trace(go.Scatter(x=temp["date"], y=temp["strategy_equity"], mode="lines", name=f"{model_name} Strategy"))
        fig.add_trace(go.Scatter(x=temp["date"], y=temp["buyhold_equity"], mode="lines", name=f"{model_name} BuyHold"))
    fig.update_layout(title="Walk-Forward Model-Driven Equity Curves", xaxis_title="Date", yaxis_title="Growth of $1", template="plotly_white", hovermode="x unified")
    return fig


def plot_confidence_history(hist_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    if not hist_df.empty:
        fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["confidence_score"], mode="lines", name="Hybrid Confidence"))
    fig.update_layout(title=f"{symbol} Hybrid Confidence Over Time", xaxis_title="Date", yaxis_title="Confidence", template="plotly_white", hovermode="x unified")
    return fig


def build_recommendation_history(symbol_df: pd.DataFrame, latest_preds: pd.DataFrame, wf_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in symbol_df.sort_values("date").tail(90).iterrows():
        card = build_hybrid_recommendation(row, latest_preds, wf_metrics)
        rows.append({
            "date": row["date"],
            "symbol": row["symbol"],
            "recommendation": card["recommendation"],
            "confidence_score": card["confidence_score"],
            "risk_penalty": card["risk_penalty"],
            "model_agreement": card["model_agreement"],
        })
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
    fig.update_layout(title=f"{symbol} Recommendation History", xaxis_title="Date", yaxis_title="Recommendation Scale", template="plotly_white", hovermode="x unified")
    return fig


st.sidebar.header("Controls")
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

st.sidebar.markdown("---")
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
selected_symbol = st.sidebar.selectbox("Select Symbol", symbols_filtered, index=0)

default_compare = symbols_filtered[: min(MAX_COMPARE_DEFAULT, len(symbols_filtered))]
compare_symbols = st.sidebar.multiselect("Compare Symbols", symbols_filtered, default=default_compare)
benchmark_symbol = BENCHMARK_SYMBOL if BENCHMARK_SYMBOL in symbols_filtered else (symbols_filtered[0] if symbols_filtered else BENCHMARK_SYMBOL)

strategy_mode = st.sidebar.selectbox("Strategy Mode", ["long_only", "long_short"], index=0)
portfolio_symbols = st.sidebar.multiselect("Portfolio Symbols", symbols_filtered, default=default_compare if default_compare else symbols_filtered[:1])
use_signal_filter = st.sidebar.checkbox("Use Signal Filter In Portfolio", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Walk-Forward Settings")
train_window = st.sidebar.number_input("Train Window", min_value=60, max_value=400, value=WALK_FORWARD_TRAIN_WINDOW, step=20)
test_window = st.sidebar.number_input("Test Window", min_value=5, max_value=60, value=WALK_FORWARD_TEST_WINDOW, step=5)
prob_threshold = st.sidebar.slider("Probability Threshold", min_value=0.50, max_value=0.80, value=float(DEFAULT_PROB_THRESHOLD), step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Execution Frictions")
transaction_cost_bps = st.sidebar.number_input("Transaction Cost (bps)", min_value=0, max_value=100, value=DEFAULT_TRANSACTION_COST_BPS, step=1)
slippage_bps = st.sidebar.number_input("Slippage (bps)", min_value=0, max_value=100, value=DEFAULT_SLIPPAGE_BPS, step=1)
rebalance_frequency = st.sidebar.selectbox("Rebalance Frequency", ["daily", "weekly", "monthly"], index=["daily", "weekly", "monthly"].index(DEFAULT_REBALANCE_FREQUENCY))

st.sidebar.markdown("---")
st.sidebar.subheader("Rolling Optimization")
rolling_train_window = st.sidebar.number_input("Rolling Train Window", min_value=60, max_value=400, value=ROLLING_OPT_TRAIN_WINDOW, step=21)
rolling_hold_window = st.sidebar.number_input("Rolling Hold Window", min_value=5, max_value=63, value=ROLLING_OPT_HOLD_WINDOW, step=5)

summary_table = build_summary_table(filtered_data)
latest_row = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date").iloc[-1]
data_health = build_data_health_table(filtered_data)

with st.expander("Project Overview", expanded=False):
    st.markdown(
        '''
        This version includes:
        - batch market pipeline
        - rule-based signal engine
        - ensemble ML models
        - hybrid recommendation engine
        - confidence and recommendation history
        - portfolio optimization
        - transaction cost + slippage aware backtests
        - rebalancing comparison
        - hybrid recommendation backtest
        - feature contribution view
        '''
    )

with st.expander("Metric Guide", expanded=False):
    st.markdown(
        '''
        **Rule-Based Signal** = technical view only.  
        **Final Buy Decision** = combined signal + ML + news + risk penalties.  
        **Confidence Score** = final hybrid score from 0 to 1.  
        **Model Agreement** = how closely the model probabilities align.  
        **rel_return_30d_pct / rel_return_90d_pct** = performance relative to SPY.  
        **Transaction Cost / Slippage** = execution frictions applied in backtests.  
        **Equal Weight / Min-Vol / Max-Sharpe-like** = different portfolio construction styles.  
        '''
    )

with st.expander("Data Health", expanded=False):
    st.dataframe(data_health, use_container_width=True)

model_metrics = pd.DataFrame()
latest_preds = pd.DataFrame()
coef_df = pd.DataFrame()
wf_predictions = pd.DataFrame()
wf_metrics = pd.DataFrame()
wf_backtest = pd.DataFrame()

try:
    model_metrics, latest_preds, coef_df = run_symbol_models(filtered_data, selected_symbol)
except Exception:
    pass

try:
    wf_predictions, wf_metrics, wf_backtest = run_walk_forward_models(
        filtered_data, selected_symbol, train_window=int(train_window), test_window=int(test_window), prob_threshold=float(prob_threshold)
    )
except Exception:
    pass

hybrid_card = None
contrib_df = pd.DataFrame()
try:
    hybrid_card = build_hybrid_recommendation(latest_row, latest_preds, wf_metrics)
    contrib_df = explain_hybrid_contributions(latest_row, latest_preds)
except Exception:
    pass

st.subheader("Decision Engine")
if hybrid_card is not None:
    info_label("Rule-Based Signal", "This is the technical signal based only on price, momentum, volatility, and volume. It does NOT include ML models or news.")
    st.write(f"Signal: {latest_row['signal']} (Score: {latest_row['signal_score']})")

    info_label("Final Buy Decision", "This combines rule-based signals, ensemble ML models, news sentiment, and risk penalties to answer: should I buy this now?")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Decision", hybrid_card["recommendation"])
    r2.metric("Confidence", f'{hybrid_card["confidence_score"]:.2f}')
    r3.metric("Model Agreement", hybrid_card["model_agreement"])
    r4.metric("Risk Penalty", f'{hybrid_card["risk_penalty"]:.2f}')

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Logistic Prob", f'{hybrid_card["logistic_up_probability"]:.2f}')
    s2.metric("MLP Prob", f'{hybrid_card["mlp_up_probability"]:.2f}')
    s3.metric("RF Prob", f'{hybrid_card["rf_up_probability"]:.2f}')
    s4.metric("GB Prob", f'{hybrid_card["gb_up_probability"]:.2f}')

    st.success(hybrid_card["reason_text"])

    if not contrib_df.empty:
        st.markdown("**Feature Contribution View**")
        st.dataframe(contrib_df, use_container_width=True)

st.subheader("Market Summary")
st.dataframe(summary_table, use_container_width=True)

tabs = st.tabs([
    "Overview", "Risk", "Signals", "Comparison", "Strategy", "Portfolio",
    "Optimization", "Modeling", "Decision History", "News", "Rankings", "Raw Data"
])

overview_tab, risk_tab, signals_tab, comparison_tab, strategy_tab, portfolio_tab, optimization_tab, modeling_tab, decision_history_tab, news_tab, rankings_tab, raw_tab = tabs

with overview_tab:
    st.plotly_chart(plot_price_ma(filtered_data, selected_symbol), use_container_width=True)
    st.plotly_chart(plot_cumulative_return(filtered_data, selected_symbol), use_container_width=True)

with risk_tab:
    st.plotly_chart(plot_volatility(filtered_data, selected_symbol), use_container_width=True)
    st.plotly_chart(plot_drawdown(filtered_data, selected_symbol), use_container_width=True)
    st.plotly_chart(plot_risk_adjusted_return(filtered_data, selected_symbol), use_container_width=True)

with signals_tab:
    st.plotly_chart(plot_rsi(filtered_data, selected_symbol), use_container_width=True)
    st.plotly_chart(plot_daily_returns(filtered_data, selected_symbol), use_container_width=True)
    st.plotly_chart(plot_signals(filtered_data, selected_symbol), use_container_width=True)

with comparison_tab:
    if compare_symbols:
        ctab1, ctab2, ctab3 = st.tabs(["Close Price", "Normalized", "Vs Benchmark"])
        with ctab1:
            st.plotly_chart(plot_compare_close(filtered_data, compare_symbols), use_container_width=True)
        with ctab2:
            st.plotly_chart(plot_compare_normalized(filtered_data, compare_symbols), use_container_width=True)
        with ctab3:
            st.plotly_chart(plot_vs_benchmark(filtered_data, selected_symbol, benchmark_symbol), use_container_width=True)

with strategy_tab:
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
        st.error(str(e))

    st.markdown("**Hybrid Recommendation Backtest**")
    try:
        symbol_hist = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date").copy()
        rec_hist = build_recommendation_history(symbol_hist, latest_preds, wf_metrics)
        hybrid_bt_df, hybrid_bt_metrics = run_hybrid_recommendation_backtest(
            filtered_data[filtered_data["symbol"] == selected_symbol].copy(),
            rec_hist,
            transaction_cost_bps=float(transaction_cost_bps),
            slippage_bps=float(slippage_bps),
        )
        st.dataframe(hybrid_bt_metrics, use_container_width=True)
    except Exception as e:
        st.info(f"Hybrid recommendation backtest unavailable: {e}")

with portfolio_tab:
    if portfolio_symbols:
        try:
            pivot_returns = filtered_data[filtered_data["symbol"].isin(portfolio_symbols)].pivot(index="date", columns="symbol", values="daily_return_decimal").dropna(how="all").fillna(0.0)
            eq_w = equal_weight_weights(portfolio_symbols)
            mv_w = min_vol_weights(pivot_returns[portfolio_symbols])
            ms_w = max_sharpe_like_weights(pivot_returns[portfolio_symbols])

            st.markdown("**Optimized Weights**")
            w1, w2, w3 = st.columns(3)
            with w1:
                st.markdown("**Equal Weight**")
                st.dataframe(eq_w, use_container_width=True)
            with w2:
                st.markdown("**Min-Vol**")
                st.dataframe(mv_w, use_container_width=True)
            with w3:
                st.markdown("**Max-Sharpe-like**")
                st.dataframe(ms_w, use_container_width=True)

            results = []
            curves = []

            for name, wdf in {
                "Equal Weight": eq_w,
                "Min-Vol": mv_w,
                "Max-Sharpe-like": ms_w,
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

            st.markdown("**Portfolio Comparison**")
            st.dataframe(result_df, use_container_width=True)

            fig = go.Figure()
            for name, group in curve_df.groupby("portfolio_type"):
                fig.add_trace(go.Scatter(x=group["date"], y=group["portfolio_equity"], mode="lines", name=name))
            fig.update_layout(title="Portfolio Optimization Equity Curves", xaxis_title="Date", yaxis_title="Growth of $1", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(str(e))

with optimization_tab:
    if portfolio_symbols:
        try:
            pivot_returns = filtered_data[filtered_data["symbol"].isin(portfolio_symbols)].pivot(index="date", columns="symbol", values="daily_return_decimal").dropna(how="all").fillna(0.0)

            st.markdown("**Rolling Portfolio Optimization**")
            roll_results = []
            roll_weights = []

            for mode in ["equal_weight", "min_vol", "max_sharpe_like"]:
                res, wts = rolling_optimized_portfolio(
                    pivot_returns[portfolio_symbols],
                    mode=mode,
                    train_window=int(rolling_train_window),
                    hold_window=int(rolling_hold_window),
                )
                if not res.empty:
                    roll_results.append(res)
                if not wts.empty:
                    roll_weights.append(wts)

            if roll_results:
                roll_df = pd.concat(roll_results, ignore_index=True)
                fig = go.Figure()
                for mode, group in roll_df.groupby("mode"):
                    fig.add_trace(go.Scatter(x=group["date"], y=group["portfolio_equity"], mode="lines", name=mode))
                fig.update_layout(title="Rolling Optimized Portfolio Equity Curves", xaxis_title="Date", yaxis_title="Growth of $1", template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                summary = []
                for mode, group in roll_df.groupby("mode"):
                    rets = group["portfolio_return_decimal"]
                    total_return = (1 + rets).prod() - 1
                    ann_vol = rets.std() * (252 ** 0.5)
                    summary.append({
                        "mode": mode,
                        "total_return_pct": total_return * 100,
                        "annualized_vol_pct": ann_vol * 100,
                    })
                st.dataframe(pd.DataFrame(summary), use_container_width=True)

            if roll_weights:
                st.markdown("**Rolling Rebalance Weights**")
                st.dataframe(pd.concat(roll_weights, ignore_index=True).head(100), use_container_width=True)

        except Exception as e:
            st.error(str(e))

with modeling_tab:
    if not model_metrics.empty:
        st.markdown("**Standard Model Metrics**")
        st.dataframe(model_metrics, use_container_width=True)
        st.markdown("**Latest Predictions**")
        st.dataframe(latest_preds, use_container_width=True)
        if not coef_df.empty:
            st.markdown("**Logistic Regression Coefficients**")
            st.plotly_chart(plot_logistic_coefficients(coef_df, selected_symbol), use_container_width=True)

    if not wf_metrics.empty:
        st.markdown("**Walk-Forward Classification Metrics**")
        st.dataframe(wf_metrics, use_container_width=True)
        st.markdown("**Model-Driven Backtest Metrics**")
        st.dataframe(wf_backtest, use_container_width=True)
        st.plotly_chart(plot_walkforward_equity(wf_predictions), use_container_width=True)

        pred_csv = wf_predictions.to_csv(index=False).encode("utf-8")
        bt_csv = wf_backtest.to_csv(index=False).encode("utf-8")

        st.download_button("Download Walk-Forward Predictions CSV", data=pred_csv, file_name=f"{selected_symbol.lower()}_walkforward_predictions_professional.csv", mime="text/csv", use_container_width=True)
        st.download_button("Download Model Backtest Metrics CSV", data=bt_csv, file_name=f"{selected_symbol.lower()}_model_backtest_metrics_professional.csv", mime="text/csv", use_container_width=True)

        if st.button("Export Walk-Forward Files To data/ Folder", use_container_width=True):
            pred_path, bt_path = export_walk_forward_outputs(
                wf_predictions,
                wf_backtest,
                predictions_path=PREDICTIONS_EXPORT_PATH,
                backtest_path=MODEL_BACKTEST_EXPORT_PATH,
            )
            st.success(f"Exported: {pred_path} and {bt_path}")

with decision_history_tab:
    try:
        symbol_hist = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date").copy()
        rec_hist = build_recommendation_history(symbol_hist, latest_preds, wf_metrics)
        if not rec_hist.empty:
            st.plotly_chart(plot_confidence_history(rec_hist, selected_symbol), use_container_width=True)
            st.plotly_chart(plot_recommendation_history(rec_hist, selected_symbol), use_container_width=True)
            st.dataframe(rec_hist.tail(30), use_container_width=True)
    except Exception as e:
        st.info(f"Decision history unavailable: {e}")

with news_tab:
    symbol_news_summary = news_summary[news_summary["symbol"] == selected_symbol].copy() if not news_summary.empty else pd.DataFrame()
    symbol_news = news_headlines[news_headlines["symbol"] == selected_symbol].copy() if not news_headlines.empty else pd.DataFrame()

    if not symbol_news_summary.empty:
        row = symbol_news_summary.iloc[0]
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Headline Count", int(row["headline_count"]))
        n2.metric("Avg Sentiment", f"{row['avg_headline_sentiment']:.3f}")
        n3.metric("News Label", str(row["news_sentiment_label"]).title())
        latest_time = row["latest_headline_time"]
        n4.metric("Latest Headline Time", str(latest_time)[:19] if pd.notna(latest_time) else "N/A")

    if not symbol_news.empty:
        st.dataframe(symbol_news[["published_at", "headline", "source", "headline_sentiment", "headline_sentiment_label", "link"]], use_container_width=True)

with rankings_tab:
    st.markdown("**Top Opportunities**")
    top_cols = [
        "symbol", "signal", "signal_score", "signal_strength", "return_30d_pct", "return_90d_pct",
        "rel_return_30d_pct", "rel_return_90d_pct", "volatility_30d_ann_pct", "rsi_14", "ma_spread_pct",
        "news_headline_count", "news_avg_sentiment"
    ]
    st.dataframe(summary_table[top_cols].head(10), use_container_width=True)

with raw_tab:
    symbol_data = filtered_data[filtered_data["symbol"] == selected_symbol].sort_values("date", ascending=False).reset_index(drop=True)
    st.dataframe(symbol_data, use_container_width=True)