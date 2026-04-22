WATCHLIST_PATH = "watchlist.csv"
DB_PATH = "db/market_research.duckdb"
TABLE_NAME = "prices_daily"

PERIOD = "2y"
INTERVAL = "1d"
WINDOW_DAYS = 730
TRADING_DAYS_PER_YEAR = 252

OUTPUT_CSV_PATH = "data/market_clean.csv"
RAW_CACHE_PATH = "data/raw_cache.csv"
NEWS_HEADLINES_PATH = "data/news_headlines.csv"
NEWS_SUMMARY_PATH = "data/news_summary.csv"
NEWS_DAILY_FEATURES_PATH = "data/news_daily_features.csv"

BENCHMARK_SYMBOL = "SPY"
REFRESH_AFTER_HOURS = 8
MAX_COMPARE_DEFAULT = 3
APP_TITLE = "AI-Assisted Market Analyst Engine"

# -----------------------------
# Asset class mapping
# -----------------------------
ASSET_CLASS_MAP = {
    "AAPL": "equity",
    "MSFT": "equity",
    "TSLA": "equity",
    "SPY": "fund",
    "BTC-USD": "crypto",
    "GC=F": "commodity",
    "SI=F": "commodity",
}

ASSET_CLASS_TARGET_FLOOR = {
    "equity": 0.0075,
    "fund": 0.0050,
    "commodity": 0.0075,
    "crypto": 0.0150,
    "unknown": 0.0100,
}

VOL_TARGET_MULTIPLIER = 0.60
PREDICTION_HORIZON_DAYS = 5

# -----------------------------
# V2 feature set
# -----------------------------
MODEL_FEATURES = [
    # price / momentum
    "daily_return_pct",
    "return_3d_pct",
    "return_5d_pct",
    "return_10d_pct",
    "return_20d_pct",
    "ma_spread_pct",
    "trend_strength_20_50",
    "rsi_14",
    "drawdown_pct",
    "volume_z_30",

    # volatility / regime
    "volatility_10d_annualized",
    "volatility_20d_annualized",
    "volatility_ratio_10_20",
    "market_regime_bull",
    "market_regime_bear",
    "market_regime_sideways",
    "vol_regime_high",
    "vol_regime_normal",

    # benchmark / market context
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "asset_vs_spy_1d",
    "asset_vs_spy_5d",
    "asset_vs_spy_20d",
    "beta_like_20d",
    "corr_to_spy_20d",

    # symbol / asset class helpers
    "is_equity",
    "is_fund",
    "is_crypto",
    "is_commodity",

    # news / context
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

    # event flags
    "macro_event_flag_3d",
    "earnings_event_flag_3d",
    "company_event_flag_3d",
    "high_impact_flag_3d",
    "market_macro_flag_3d",
]

MIN_MODEL_ROWS = 90
RECENT_MODEL_WINDOW = 260

NEWS_MAX_HEADLINES_PER_SYMBOL = 35
MARKET_NEWS_MAX_HEADLINES = 60
NEWS_LOOKBACK_DAYS = 14

WALK_FORWARD_TRAIN_WINDOW = 100
WALK_FORWARD_TEST_WINDOW = 20
DEFAULT_PROB_THRESHOLD = 0.55

PREDICTIONS_EXPORT_PATH = "data/walkforward_predictions.csv"
MODEL_BACKTEST_EXPORT_PATH = "data/model_backtest_metrics.csv"

DEFAULT_TRANSACTION_COST_BPS = 10
DEFAULT_SLIPPAGE_BPS = 5
DEFAULT_REBALANCE_FREQUENCY = "monthly"

ROLLING_OPT_TRAIN_WINDOW = 126
ROLLING_OPT_HOLD_WINDOW = 21
MAX_WEIGHT_PER_ASSET = 0.60