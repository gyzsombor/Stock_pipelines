
import pandas as pd
import plotly.graph_objs as go


def _subset_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return df[df["symbol"] == symbol].sort_values("date").copy()


def _base_layout(fig: go.Figure, title: str, yaxis_title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
    )
    return fig


def plot_price_ma(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["ma_7"], mode="lines", name="MA 7"))
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["ma_30"], mode="lines", name="MA 30"))
    return _base_layout(fig, f"{symbol} Price & Moving Averages", "Price")


def plot_daily_returns(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=subset["date"], y=subset["daily_return_pct"], name="Daily Return (%)"))
    return _base_layout(fig, f"{symbol} Daily Returns", "Daily Return (%)")


def plot_volatility(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["volatility_30d_annualized"], mode="lines", name="30d Annualized Volatility (%)"))
    return _base_layout(fig, f"{symbol} 30-day Annualized Volatility", "Volatility (%)")


def plot_rsi(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["rsi_14"], mode="lines", name="RSI 14"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    return _base_layout(fig, f"{symbol} RSI 14", "RSI")


def plot_signals(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["close"], mode="lines", name="Close"))

    buys = subset[subset["signal"].isin(["BUY", "STRONG_BUY"])]
    sells = subset[subset["signal"].isin(["SELL", "STRONG_SELL"])]

    fig.add_trace(go.Scatter(
        x=buys["date"], y=buys["close"], mode="markers", name="Buy-side",
        marker=dict(size=10, symbol="triangle-up")
    ))
    fig.add_trace(go.Scatter(
        x=sells["date"], y=sells["close"], mode="markers", name="Sell-side",
        marker=dict(size=10, symbol="triangle-down")
    ))

    return _base_layout(fig, f"{symbol} Price & Signals", "Price")


def plot_cumulative_return(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["cumulative_return_pct"], mode="lines", name="Cumulative Return (%)"))
    return _base_layout(fig, f"{symbol} Cumulative Return", "Return (%)")


def plot_drawdown(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["drawdown_pct"], mode="lines", name="Drawdown (%)", fill="tozeroy"))
    return _base_layout(fig, f"{symbol} Drawdown from Peak", "Drawdown (%)")


def plot_risk_adjusted_return(df: pd.DataFrame, symbol: str) -> go.Figure:
    subset = _subset_symbol(df, symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["rolling_risk_adjusted_30"], mode="lines", name="30d Risk-Adjusted Return"))
    return _base_layout(fig, f"{symbol} Rolling Risk-Adjusted Return (30d)", "Ratio")


def plot_compare_close(df: pd.DataFrame, symbols: list[str]) -> go.Figure:
    fig = go.Figure()
    for symbol in symbols:
        subset = df[df["symbol"] == symbol].sort_values("date")
        fig.add_trace(go.Scatter(x=subset["date"], y=subset["close"], mode="lines", name=symbol))
    return _base_layout(fig, "Compare Closing Prices", "Close Price")


def plot_compare_normalized(df: pd.DataFrame, symbols: list[str]) -> go.Figure:
    fig = go.Figure()
    for symbol in symbols:
        subset = df[df["symbol"] == symbol].sort_values("date").copy()
        if subset.empty:
            continue
        first_close = subset["close"].iloc[0]
        if first_close == 0:
            continue
        subset["normalized_close"] = (subset["close"] / first_close) * 100
        fig.add_trace(go.Scatter(x=subset["date"], y=subset["normalized_close"], mode="lines", name=symbol))
    return _base_layout(fig, "Normalized Performance (Base = 100)", "Normalized Close")


def plot_vs_benchmark(df: pd.DataFrame, symbol: str, benchmark_symbol: str) -> go.Figure:
    fig = go.Figure()

    asset = df[df["symbol"] == symbol].sort_values("date").copy()
    bench = df[df["symbol"] == benchmark_symbol].sort_values("date").copy()

    if asset.empty or bench.empty:
        return _base_layout(fig, f"{symbol} vs {benchmark_symbol}", "Base = 100")

    asset["normalized"] = (asset["close"] / asset["close"].iloc[0]) * 100
    bench["normalized"] = (bench["close"] / bench["close"].iloc[0]) * 100

    fig.add_trace(go.Scatter(x=asset["date"], y=asset["normalized"], mode="lines", name=symbol))
    fig.add_trace(go.Scatter(x=bench["date"], y=bench["normalized"], mode="lines", name=benchmark_symbol))

    return _base_layout(fig, f"{symbol} vs {benchmark_symbol} (Normalized Performance)", "Base = 100")


def plot_backtest_equity(backtest_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["strategy_equity"], mode="lines", name="Signal Strategy"))
    fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["buyhold_equity"], mode="lines", name="Buy & Hold"))
    return _base_layout(fig, f"{symbol} Strategy Equity Curve", "Growth of $1")


def plot_portfolio_equity(portfolio_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_df["date"], y=portfolio_df["portfolio_equity"], mode="lines", name="Portfolio"))
    return _base_layout(fig, title, "Growth of $1")


def plot_logistic_coefficients(coef_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=coef_df["feature"], y=coef_df["coefficient"], name="Coefficient"))
    return _base_layout(fig, f"{symbol} Logistic Regression Feature Coefficients", "Coefficient")
