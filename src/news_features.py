from __future__ import annotations

from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import MARKET_NEWS_MAX_HEADLINES, NEWS_LOOKBACK_DAYS


SEARCH_ALIASES = {
    "AAPL": ["Apple", "Apple stock", "Apple earnings"],
    "MSFT": ["Microsoft", "Microsoft stock", "Azure"],
    "NVDA": ["NVIDIA", "NVIDIA stock", "AI chips"],
    "AMZN": ["Amazon", "Amazon stock", "AWS"],
    "META": ["Meta", "Meta stock", "Facebook parent"],
    "GOOGL": ["Google", "Alphabet stock", "Google earnings"],
    "TSLA": ["Tesla", "Tesla stock", "EV market"],
    "NFLX": ["Netflix", "Netflix stock"],
    "SPY": ["S&P 500 ETF", "SPY", "US stock market"],
    "QQQ": ["Nasdaq 100 ETF", "QQQ", "technology stocks"],
    "BTC-USD": ["Bitcoin", "BTC", "crypto market"],
    "GC=F": ["Gold futures", "Gold market"],
    "SI=F": ["Silver futures", "Silver market"],
}

TIER1_SOURCES = {
    "Reuters",
    "Bloomberg",
    "The Wall Street Journal",
    "CNBC",
    "Financial Times",
    "MarketWatch",
    "Barron's",
    "Yahoo Finance",
}

MACRO_KEYWORDS = {
    "fed",
    "federal reserve",
    "interest rate",
    "rates",
    "inflation",
    "cpi",
    "recession",
    "gdp",
    "tariff",
    "jobs report",
    "payrolls",
    "treasury",
}

EARNINGS_KEYWORDS = {
    "earnings",
    "guidance",
    "forecast",
    "revenue",
    "profit",
    "eps",
    "beat",
    "miss",
}

ANALYST_KEYWORDS = {
    "upgrade",
    "downgrade",
    "rating",
    "target price",
    "price target",
}

CORPORATE_KEYWORDS = {
    "merger",
    "acquisition",
    "lawsuit",
    "bankruptcy",
    "layoffs",
    "sec",
    "fda",
    "ceo",
    "launch",
    "buyback",
}

MARKET_NEWS_QUERY = (
    "stock market OR Federal Reserve OR interest rates OR inflation OR CPI "
    "OR recession OR tariffs OR earnings OR treasury yields"
)


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


def _event_type(title: str) -> str:
    text = str(title).lower()

    if any(k in text for k in MACRO_KEYWORDS):
        return "macro"
    if any(k in text for k in EARNINGS_KEYWORDS):
        return "earnings"
    if any(k in text for k in ANALYST_KEYWORDS):
        return "analyst"
    if any(k in text for k in CORPORATE_KEYWORDS):
        return "corporate"
    return "general"


def _event_scope(symbol: str | None, title: str) -> str:
    if symbol == "MARKET":
        return "market_wide"

    text = str(title).lower()
    if any(k in text for k in MACRO_KEYWORDS):
        return "market_wide"
    if any(k in text for k in EARNINGS_KEYWORDS | ANALYST_KEYWORDS | CORPORATE_KEYWORDS):
        return "company_specific"
    return "unclear"


def _impact_weight(title: str, source: str, sentiment: float) -> float:
    title_lower = str(title).lower()
    source = str(source).strip()

    weight = 1.0

    if any(keyword in title_lower for keyword in EARNINGS_KEYWORDS):
        weight += 0.75

    if any(keyword in title_lower for keyword in ANALYST_KEYWORDS):
        weight += 0.40

    if any(keyword in title_lower for keyword in CORPORATE_KEYWORDS):
        weight += 0.55

    if any(keyword in title_lower for keyword in MACRO_KEYWORDS):
        weight += 1.25

    if source in TIER1_SOURCES:
        weight += 0.30

    weight += min(abs(float(sentiment)), 0.75)
    return float(weight)


def _impact_bucket(weight: float) -> str:
    if weight >= 2.6:
        return "high"
    if weight >= 1.8:
        return "medium"
    return "low"


def _symbol_queries(symbol: str) -> list[str]:
    aliases = SEARCH_ALIASES.get(symbol, [symbol])
    return [f"{alias} stock OR earnings OR price OR market" for alias in aliases[:2]]


def _fetch_google_news_rows(query: str, max_items: int, symbol: str | None = None) -> list[dict]:
    analyzer = SentimentIntensityAnalyzer()
    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(url)
    entries = getattr(feed, "entries", [])[:max_items]

    rows: list[dict] = []

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
        event_type = _event_type(title)
        event_scope = _event_scope(symbol, title)
        impact_weight = _impact_weight(title, source, sentiment)

        rows.append(
            {
                "symbol": symbol,
                "published_at": published_at,
                "headline": title,
                "source": source,
                "link": link,
                "headline_sentiment": float(sentiment),
                "headline_sentiment_label": _label_sentiment(float(sentiment)),
                "event_type": event_type,
                "event_scope": event_scope,
                "impact_weight": float(impact_weight),
                "impact_bucket": _impact_bucket(float(impact_weight)),
            }
        )

    return rows


def fetch_symbol_news(symbols: list[str], max_headlines_per_symbol: int = 35) -> pd.DataFrame:
    rows = []
    per_query = max(8, int(max_headlines_per_symbol / 2))

    for symbol in symbols:
        for query in _symbol_queries(symbol):
            rows.extend(_fetch_google_news_rows(query, per_query, symbol=symbol))

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "published_at",
                "headline",
                "source",
                "link",
                "headline_sentiment",
                "headline_sentiment_label",
                "event_type",
                "event_scope",
                "impact_weight",
                "impact_bucket",
                "news_date",
            ]
        )

    news = pd.DataFrame(rows)
    news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce")
    news["news_date"] = pd.to_datetime(news["published_at"], errors="coerce").dt.normalize()

    news = (
        news.drop_duplicates(subset=["symbol", "headline"])
        .sort_values(["symbol", "published_at"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return news


def fetch_market_news(max_headlines: int = MARKET_NEWS_MAX_HEADLINES) -> pd.DataFrame:
    rows = _fetch_google_news_rows(MARKET_NEWS_QUERY, max_headlines, symbol="MARKET")

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "published_at",
                "headline",
                "source",
                "link",
                "headline_sentiment",
                "headline_sentiment_label",
                "event_type",
                "event_scope",
                "impact_weight",
                "impact_bucket",
                "news_date",
            ]
        )

    news = pd.DataFrame(rows)
    news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce")
    news["news_date"] = pd.to_datetime(news["published_at"], errors="coerce").dt.normalize()

    news = (
        news.drop_duplicates(subset=["headline"])
        .sort_values(["published_at"], ascending=False)
        .reset_index(drop=True)
    )
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
                "avg_impact_weight",
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
            avg_impact_weight=("impact_weight", "mean"),
        )
    )

    summary["news_sentiment_label"] = summary["avg_headline_sentiment"].apply(_label_sentiment)
    return summary.sort_values(["avg_headline_sentiment", "headline_count"], ascending=[False, False]).reset_index(drop=True)


def _window_slice(df: pd.DataFrame, end_date: pd.Timestamp, days: int) -> pd.DataFrame:
    start_date = end_date - pd.Timedelta(days=days - 1)
    return df[(df["news_date"] >= start_date) & (df["news_date"] <= end_date)].copy()


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    if values.empty or weights.empty or float(weights.sum()) == 0.0:
        return 0.0
    return float(np.average(values, weights=weights))


def _decay_weight(age_days: int) -> float:
    if age_days <= 1:
        return 1.0
    if age_days == 2:
        return 0.7
    if age_days == 3:
        return 0.5
    return 0.3


def _build_symbol_features_for_date(news_window: pd.DataFrame, end_date: pd.Timestamp) -> dict:
    if news_window.empty:
        return {
            "news_headline_count": 0,
            "news_avg_sentiment": 0.0,
            "news_positive_ratio": 0.0,
            "news_negative_ratio": 0.0,
            "news_latest_sentiment": 0.0,
            "news_has_any": 0,
            "news_sentiment_3d": 0.0,
            "news_sentiment_7d": 0.0,
            "news_count_3d": 0,
            "news_count_7d": 0,
            "news_impact_score_3d": 0.0,
            "news_decayed_sentiment_7d": 0.0,
            "news_high_impact_count_3d": 0,
            "news_macro_event_count_3d": 0,
            "news_company_event_count_3d": 0,
        }

    window_1d = _window_slice(news_window, end_date, 1)
    window_3d = _window_slice(news_window, end_date, 3)
    window_7d = _window_slice(news_window, end_date, 7)

    decayed_7d = 0.0
    if not window_7d.empty:
        temp = window_7d.copy()
        temp["age_days"] = (end_date - temp["news_date"]).dt.days.astype(int)
        temp["decay_weight"] = temp["age_days"].apply(_decay_weight).astype(float)
        temp["final_weight"] = temp["impact_weight"] * temp["decay_weight"]
        decayed_7d = _weighted_average(temp["headline_sentiment"], temp["final_weight"])

    return {
        "news_headline_count": int(len(window_1d)),
        "news_avg_sentiment": float(window_1d["headline_sentiment"].mean()) if not window_1d.empty else 0.0,
        "news_positive_ratio": float((window_1d["headline_sentiment_label"] == "positive").mean()) if not window_1d.empty else 0.0,
        "news_negative_ratio": float((window_1d["headline_sentiment_label"] == "negative").mean()) if not window_1d.empty else 0.0,
        "news_latest_sentiment": float(window_1d["headline_sentiment"].iloc[0]) if not window_1d.empty else 0.0,
        "news_has_any": int(not window_1d.empty),
        "news_sentiment_3d": float(window_3d["headline_sentiment"].mean()) if not window_3d.empty else 0.0,
        "news_sentiment_7d": float(window_7d["headline_sentiment"].mean()) if not window_7d.empty else 0.0,
        "news_count_3d": int(len(window_3d)),
        "news_count_7d": int(len(window_7d)),
        "news_impact_score_3d": _weighted_average(window_3d["headline_sentiment"], window_3d["impact_weight"]) if not window_3d.empty else 0.0,
        "news_decayed_sentiment_7d": decayed_7d,
        "news_high_impact_count_3d": int((window_3d["impact_bucket"] == "high").sum()) if not window_3d.empty else 0,
        "news_macro_event_count_3d": int((window_3d["event_scope"] == "market_wide").sum()) if not window_3d.empty else 0,
        "news_company_event_count_3d": int((window_3d["event_scope"] == "company_specific").sum()) if not window_3d.empty else 0,
    }


def _build_market_features_for_date(market_news_df: pd.DataFrame, end_date: pd.Timestamp) -> dict:
    if market_news_df.empty:
        return {
            "market_news_sentiment_3d": 0.0,
            "market_news_sentiment_7d": 0.0,
            "market_news_impact_score_3d": 0.0,
            "market_news_count_3d": 0,
            "market_macro_event_count_3d": 0,
        }

    window_3d = _window_slice(market_news_df, end_date, 3)
    window_7d = _window_slice(market_news_df, end_date, 7)

    return {
        "market_news_sentiment_3d": float(window_3d["headline_sentiment"].mean()) if not window_3d.empty else 0.0,
        "market_news_sentiment_7d": float(window_7d["headline_sentiment"].mean()) if not window_7d.empty else 0.0,
        "market_news_impact_score_3d": _weighted_average(window_3d["headline_sentiment"], window_3d["impact_weight"]) if not window_3d.empty else 0.0,
        "market_news_count_3d": int(len(window_3d)),
        "market_macro_event_count_3d": int((window_3d["event_type"] == "macro").sum()) if not window_3d.empty else 0,
    }


def build_daily_news_features(
    market_df: pd.DataFrame,
    symbol_news_df: pd.DataFrame,
    market_news_df: pd.DataFrame,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> pd.DataFrame:
    base = market_df[["symbol", "date"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    base = base.sort_values(["symbol", "date"]).reset_index(drop=True)

    if not symbol_news_df.empty:
        symbol_news_df = symbol_news_df.copy()
        symbol_news_df["news_date"] = pd.to_datetime(symbol_news_df["news_date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    if not market_news_df.empty:
        market_news_df = market_news_df.copy()
        market_news_df["news_date"] = pd.to_datetime(market_news_df["news_date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    rows = []

    for symbol, group in base.groupby("symbol"):
        symbol_news = symbol_news_df[symbol_news_df["symbol"] == symbol].copy() if not symbol_news_df.empty else pd.DataFrame()

        for _, row in group.iterrows():
            current_date = row["date"]
            start_date = current_date - pd.Timedelta(days=lookback_days - 1)

            symbol_window = (
                symbol_news[(symbol_news["news_date"] >= start_date) & (symbol_news["news_date"] <= current_date)].copy()
                if not symbol_news.empty
                else pd.DataFrame()
            )

            market_window = (
                market_news_df[(market_news_df["news_date"] >= start_date) & (market_news_df["news_date"] <= current_date)].copy()
                if not market_news_df.empty
                else pd.DataFrame()
            )

            symbol_features = _build_symbol_features_for_date(symbol_window, current_date)
            market_features = _build_market_features_for_date(market_window, current_date)

            rows.append(
                {
                    "symbol": symbol,
                    "date": current_date,
                    **symbol_features,
                    **market_features,
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def merge_news_features_into_market(clean_df: pd.DataFrame, news_daily_df: pd.DataFrame) -> pd.DataFrame:
    out = clean_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    fill_cols = {
        "news_headline_count": 0,
        "news_avg_sentiment": 0.0,
        "news_positive_ratio": 0.0,
        "news_negative_ratio": 0.0,
        "news_latest_sentiment": 0.0,
        "news_has_any": 0,
        "news_sentiment_3d": 0.0,
        "news_sentiment_7d": 0.0,
        "news_count_3d": 0,
        "news_count_7d": 0,
        "news_impact_score_3d": 0.0,
        "news_decayed_sentiment_7d": 0.0,
        "news_high_impact_count_3d": 0,
        "news_macro_event_count_3d": 0,
        "news_company_event_count_3d": 0,
        "market_news_sentiment_3d": 0.0,
        "market_news_sentiment_7d": 0.0,
        "market_news_impact_score_3d": 0.0,
        "market_news_count_3d": 0,
        "market_macro_event_count_3d": 0,
    }

    if news_daily_df.empty:
        for col, val in fill_cols.items():
            out[col] = val
        return out

    news_daily_df = news_daily_df.copy()
    news_daily_df["date"] = pd.to_datetime(news_daily_df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    out = out.merge(news_daily_df, on=["symbol", "date"], how="left")

    for col, val in fill_cols.items():
        if col not in out.columns:
            out[col] = val
        out[col] = out[col].fillna(val)

    return out