
from __future__ import annotations

from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


SEARCH_NAME_MAP = {
    "BTC-USD": "Bitcoin",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "SPY": "S&P 500 ETF",
}


def _symbol_search_term(symbol: str) -> str:
    return SEARCH_NAME_MAP.get(symbol, symbol)


def _label_sentiment(score: float) -> str:
    if score >= 0.20:
        return "positive"
    if score <= -0.20:
        return "negative"
    return "neutral"


def _to_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_localize(None)


def fetch_symbol_news(symbols: list[str], max_headlines_per_symbol: int = 8) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    for symbol in symbols:
        term = _symbol_search_term(symbol)
        query = quote_plus(f"{term} stock OR price market")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        feed = feedparser.parse(url)
        entries = getattr(feed, "entries", [])[:max_headlines_per_symbol]

        for entry in entries:
            title = str(getattr(entry, "title", "")).strip()
            link = str(getattr(entry, "link", "")).strip()

            source = ""
            if hasattr(entry, "source") and entry.source:
                source = str(getattr(entry.source, "title", "")).strip()

            published_at = pd.NaT
            published_raw = getattr(entry, "published", None)
            if published_raw:
                try:
                    published_at = parsedate_to_datetime(published_raw)
                except Exception:
                    published_at = pd.NaT

            published_at = _to_naive_timestamp(published_at)
            sentiment = analyzer.polarity_scores(title).get("compound", 0.0)

            rows.append(
                {
                    "symbol": symbol,
                    "search_term": term,
                    "published_at": published_at,
                    "headline": title,
                    "source": source,
                    "link": link,
                    "headline_sentiment": float(sentiment),
                    "headline_sentiment_label": _label_sentiment(float(sentiment)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "search_term",
                "published_at",
                "headline",
                "source",
                "link",
                "headline_sentiment",
                "headline_sentiment_label",
            ]
        )

    news = pd.DataFrame(rows)
    news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce")
    news["news_date"] = pd.to_datetime(news["published_at"], errors="coerce").dt.normalize()

    news = news.drop_duplicates(subset=["symbol", "headline"]).sort_values(
        ["symbol", "published_at"], ascending=[True, False]
    ).reset_index(drop=True)
    return news


def summarize_news(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "headline_count",
                "avg_headline_sentiment",
                "positive_headlines",
                "negative_headlines",
                "neutral_headlines",
                "latest_headline_time",
                "news_sentiment_label",
            ]
        )

    summary = (
        news_df.groupby("symbol", as_index=False)
        .agg(
            headline_count=("headline", "count"),
            avg_headline_sentiment=("headline_sentiment", "mean"),
            positive_headlines=("headline_sentiment_label", lambda s: int((s == "positive").sum())),
            negative_headlines=("headline_sentiment_label", lambda s: int((s == "negative").sum())),
            neutral_headlines=("headline_sentiment_label", lambda s: int((s == "neutral").sum())),
            latest_headline_time=("published_at", "max"),
        )
    )

    summary["news_sentiment_label"] = summary["avg_headline_sentiment"].apply(_label_sentiment)
    return summary.sort_values(["avg_headline_sentiment", "headline_count"], ascending=[False, False]).reset_index(drop=True)


def build_daily_news_features(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "news_headline_count",
                "news_avg_sentiment",
                "news_positive_ratio",
                "news_negative_ratio",
                "news_latest_sentiment",
                "news_has_any",
            ]
        )

    valid = news_df.dropna(subset=["news_date"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "news_headline_count",
                "news_avg_sentiment",
                "news_positive_ratio",
                "news_negative_ratio",
                "news_latest_sentiment",
                "news_has_any",
            ]
        )

    valid["news_date"] = pd.to_datetime(valid["news_date"], errors="coerce").dt.tz_localize(None)
    valid["is_positive"] = (valid["headline_sentiment_label"] == "positive").astype(int)
    valid["is_negative"] = (valid["headline_sentiment_label"] == "negative").astype(int)

    daily = (
        valid.groupby(["symbol", "news_date"], as_index=False)
        .agg(
            news_headline_count=("headline", "count"),
            news_avg_sentiment=("headline_sentiment", "mean"),
            news_positive_ratio=("is_positive", "mean"),
            news_negative_ratio=("is_negative", "mean"),
            news_latest_sentiment=("headline_sentiment", "last"),
        )
        .rename(columns={"news_date": "date"})
    )

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.tz_localize(None)
    daily["news_has_any"] = 1
    return daily.sort_values(["symbol", "date"]).reset_index(drop=True)


def merge_news_features_into_market(clean_df: pd.DataFrame, news_daily_df: pd.DataFrame) -> pd.DataFrame:
    out = clean_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    if news_daily_df.empty:
        out["news_headline_count"] = 0
        out["news_avg_sentiment"] = 0.0
        out["news_positive_ratio"] = 0.0
        out["news_negative_ratio"] = 0.0
        out["news_latest_sentiment"] = 0.0
        out["news_has_any"] = 0
        return out

    news_daily_df = news_daily_df.copy()
    news_daily_df["date"] = pd.to_datetime(news_daily_df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    out = out.merge(
        news_daily_df,
        on=["symbol", "date"],
        how="left",
    )

    fill_map = {
        "news_headline_count": 0,
        "news_avg_sentiment": 0.0,
        "news_positive_ratio": 0.0,
        "news_negative_ratio": 0.0,
        "news_latest_sentiment": 0.0,
        "news_has_any": 0,
    }
    for col, val in fill_map.items():
        if col not in out.columns:
            out[col] = val
        out[col] = out[col].fillna(val)

    return out
