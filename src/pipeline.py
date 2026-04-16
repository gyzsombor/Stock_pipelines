
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
    fetch_symbol_news,
    merge_news_features_into_market,
    summarize_news,
)
from signals import add_signal


def write_duckdb(df: pd.DataFrame) -> None:
    con = duckdb.connect(DB_PATH)

    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
    con.register("df_view", df)

    # Create schema directly from the current dataframe so new columns never break inserts
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df_view WHERE 1=0;")
    con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM df_view;")

    con.close()


def main() -> None:
    ensure_dirs()

    wl = load_watchlist(WATCHLIST_PATH)
    symbols = wl["symbol"].tolist()

    print(f"Loaded {len(symbols)} symbols from: {WATCHLIST_PATH}")
    print(symbols)

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

    clean = add_features(raw)
    clean = add_signal(clean)

    try:
        print("\nFetching news headlines...")
        news_headlines = fetch_symbol_news(
            symbols=symbols,
            max_headlines_per_symbol=NEWS_MAX_HEADLINES_PER_SYMBOL,
        )
        news_summary = summarize_news(news_headlines)
        news_daily = build_daily_news_features(news_headlines)

        news_headlines.to_csv(NEWS_HEADLINES_PATH, index=False)
        news_summary.to_csv(NEWS_SUMMARY_PATH, index=False)
        news_daily.to_csv(NEWS_DAILY_FEATURES_PATH, index=False)

        clean = merge_news_features_into_market(clean, news_daily)

        print(f"News headlines saved: {NEWS_HEADLINES_PATH}")
        print(f"News summary saved: {NEWS_SUMMARY_PATH}")
        print(f"News daily features saved: {NEWS_DAILY_FEATURES_PATH}")
    except Exception as e:
        print(f"News fetch failed: {e}")
        clean["news_headline_count"] = 0
        clean["news_avg_sentiment"] = 0.0
        clean["news_positive_ratio"] = 0.0
        clean["news_negative_ratio"] = 0.0
        clean["news_latest_sentiment"] = 0.0
        clean["news_has_any"] = 0

    clean = clean.dropna(
        subset=[
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
    ).reset_index(drop=True)

    if clean.empty:
        raise RuntimeError("No rows remain after feature engineering. Need more price history.")

    clean.to_csv(OUTPUT_CSV_PATH, index=False)
    write_duckdb(clean)

    print("\nPipeline finished successfully")
    print(f"Rows saved: {len(clean):,}")
    print(f"Symbols processed: {clean['symbol'].nunique()}")
    print("CSV refreshed:", OUTPUT_CSV_PATH)
    print("DuckDB refreshed:", DB_PATH, "| table:", TABLE_NAME)


if __name__ == "__main__":
    main()
