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

MODEL_FEATURES = [
    "ma_spread_pct",
    "rel_return_30d_pct",
    "rel_return_90d_pct",
    "return_7d_pct",
    "return_30d_pct",
    "return_90d_pct",
    "volatility_30d_annualized",
    "rsi_14",
    "daily_return_pct",
    "volume_z_30",
    "drawdown_pct",
    "rolling_risk_adjusted_30",
    "news_headline_count",
    "news_avg_sentiment",
    "news_positive_ratio",
    "news_negative_ratio",
    "news_sentiment_3d",
    "news_sentiment_7d",
    "news_count_3d",
    "news_count_7d",
    "news_impact_score_3d",
    "news_decayed_sentiment_7d",
    "news_high_impact_count_3d",
    "news_macro_event_count_3d",
    "news_company_event_count_3d",
    "market_news_sentiment_3d",
    "market_news_sentiment_7d",
    "market_news_impact_score_3d",
    "market_news_count_3d",
    "market_macro_event_count_3d",
]

MIN_MODEL_ROWS = 90
RECENT_MODEL_WINDOW = 260

NEWS_MAX_HEADLINES_PER_SYMBOL = 35
MARKET_NEWS_MAX_HEADLINES = 60
NEWS_LOOKBACK_DAYS = 14

PREDICTION_HORIZON_DAYS = 3
TARGET_RETURN_THRESHOLD = 0.01

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