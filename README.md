
# Stock Market Pipeline Dashboard Elite

## Setup

Create virtual environment:

python -m venv .venv

Activate on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install requirements:

pip install -r requirements.txt

Run pipeline:

python src\pipeline.py

Run app:

streamlit run app.py

## Main upgrades

- Batch Yahoo Finance download instead of one-symbol-at-a-time
- Cache-aware raw download refresh
- Stronger feature set with MA spread and 90d return
- Score-based signals instead of only basic labels
- Ranked watchlist table in the app
- Backtesting tab for signal strategy vs buy-and-hold
- Equal-weight portfolio simulator
- Logistic regression + neural net (MLP) modeling tab
- News headline ingestion with sentiment scoring
- News summary and headline tables inside the app
