from __future__ import annotations

import duckdb
import pandas as pd

from config import (
    DB_PATH,
    INTERVAL,
    NEWS_DAILY_FEATURES_PATH,
    NEWS_HEADLINES_PATH,
    NEWS_MAX_HEADLINES_PER_SYMBOL,
    NEWS_SUMMARY_PATH,
    OUTPUT_CSV_PATH,
    PERIOD,
    RAW_CACHE_PATH,
    REFRESH_AFTER_HOURS,
    TABLE_NAME,
    WATCHLIST_PATH,
    WINDOW_DAYS,
)
from features import add_features
from io_utils import ensure_dirs, fetch_ohlcv_batch, load_watchlist
from news_features import (
    build_daily_news_features,
    fetch_market_news,
    fetch_symbol_news,
    merge_news_features_into_market,
    summarize_news,
)
from signals import add_signal


REQUIRED_FEATURE_COLUMNS = [
    "daily_return_pct",
    "ma_30",
    "ma_spread_pct",
    "volatility_7d",
    "volatility_30d_annualized",
    "rsi_14",
    "cumulative_return_pct",
    "drawdown_pct",
    "rolling_risk_adjusted_30",
    "return_7d_pct",
    "return_30d_pct",
    "return_90d_pct",
    "rel_return_30d_pct",
    "rel_return_90d_pct",
]


def write_duckdb(df: pd.DataFrame) -> None:
    with duckdb.connect(DB_PATH) as con:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        con.register("df_view", df)
        con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df_view")


def prepare_raw_data(symbols: list[str]) -> pd.DataFrame:
    raw = fetch_ohlcv_batch(
        symbols=symbols,
        period=PERIOD,
        interval=INTERVAL,
        cache_path=RAW_CACHE_PATH,
        refresh_after_hours=REFRESH_AFTER_HOURS,
    )

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
    raw = raw.dropna(subset=["date"]).copy()

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=WINDOW_DAYS)
    raw = raw.loc[raw["date"] >= cutoff].copy()

    if raw.empty:
        raise RuntimeError("No rows remain after applying the date window.")

    return raw


def add_news_data(clean: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    try:
        print("\nFetching news headlines...")
        news_headlines = fetch_symbol_news(
            symbols=symbols,
            max_headlines_per_symbol=NEWS_MAX_HEADLINES_PER_SYMBOL,
        )
        market_headlines = fetch_market_news()
        news_summary = summarize_news(news_headlines)

        news_daily = build_daily_news_features(
            market_df=clean[["symbol", "date"]].copy(),
            symbol_news_df=news_headlines,
            market_news_df=market_headlines,
        )

        news_headlines.to_csv(NEWS_HEADLINES_PATH, index=False)
        news_summary.to_csv(NEWS_SUMMARY_PATH, index=False)
        news_daily.to_csv(NEWS_DAILY_FEATURES_PATH, index=False)

        print(f"News headlines saved: {NEWS_HEADLINES_PATH}")
        print(f"News summary saved: {NEWS_SUMMARY_PATH}")
        print(f"News daily features saved: {NEWS_DAILY_FEATURES_PATH}")

        return merge_news_features_into_market(clean, news_daily)

    except Exception as exc:
        print(f"News fetch failed: {exc}")
        clean = clean.copy()
        clean["news_headline_count"] = 0
        clean["news_avg_sentiment"] = 0.0
        clean["news_positive_ratio"] = 0.0
        clean["news_negative_ratio"] = 0.0
        clean["news_latest_sentiment"] = 0.0
        clean["news_has_any"] = 0
        clean["news_sentiment_3d"] = 0.0
        clean["news_sentiment_7d"] = 0.0
        clean["news_count_3d"] = 0
        clean["news_count_7d"] = 0
        clean["news_impact_score_3d"] = 0.0
        clean["news_decayed_sentiment_7d"] = 0.0
        clean["market_news_sentiment_3d"] = 0.0
        clean["market_news_sentiment_7d"] = 0.0
        clean["market_news_impact_score_3d"] = 0.0
        clean["market_news_count_3d"] = 0
        return clean

def main() -> None:
    ensure_dirs()

    watchlist = load_watchlist(WATCHLIST_PATH)
    symbols = watchlist["symbol"].tolist()

    print(f"Loaded {len(symbols)} symbols from: {WATCHLIST_PATH}")
    print(symbols)

    raw = prepare_raw_data(symbols)
    clean = add_signal(add_features(raw))
    clean = add_news_data(clean, symbols)

    clean = clean.dropna(subset=REQUIRED_FEATURE_COLUMNS).reset_index(drop=True)

    if clean.empty:
        raise RuntimeError("No rows remain after feature engineering. Need more price history.")

    clean.to_csv(OUTPUT_CSV_PATH, index=False)
    write_duckdb(clean)

    print("\nPipeline finished successfully")
    print(f"Rows saved: {len(clean):,}")
    print(f"Symbols processed: {clean['symbol'].nunique()}")
    print(f"CSV refreshed: {OUTPUT_CSV_PATH}")
    print(f"DuckDB refreshed: {DB_PATH} | table: {TABLE_NAME}")


if __name__ == "__main__":
    main()