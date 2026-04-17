# Stock Pipelines

## Overview

This project is a multi-asset market analytics and decision-support system. It collects market data, builds technical features, processes financial news, applies machine learning models, and presents results in a Streamlit dashboard.

The goal is to provide a structured **analyst-style recommendation**, not random predictions.

---

## What This Project Does

- Downloads market data for multiple assets
- Creates technical indicators (returns, volatility, RSI, etc.)
- Generates rule-based signals
- Processes news with sentiment and event logic
- Trains machine learning models
- Runs walk-forward validation
- Produces recommendations and confidence scores
- Displays everything in an interactive dashboard

---

## Project Structure


## Project Structure

Main folder:
- `App.py`
- `visuals.py`
- `requirements.txt`
- `watchlist.csv`
- `README.md`

`src/` folder:
- `__init__.py`
- `pipeline.py`
- `config.py`
- `io_utils.py`
- `features.py`
- `signals.py`
- `news_features.py`
- `modeling.py`
- `backtest.py`
- `portfolio.py`

---

## Requirements

- Windows + PowerShell (or any terminal)
- Python 3.10+
- Git installed
- Internet connection

---

## Setup

Clone the repository:

```powershell
git clone https://github.com/gyzsombor/Stock_pipelines.git
cd Stock_pipelines
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the Project

Run the data pipeline:

```powershell
python src/pipeline.py
```

Launch the dashboard:

```powershell
streamlit run App.py
```

---

## How It Works

Running the pipeline will:

- load symbols from `watchlist.csv`
- download market data
- generate technical features
- create rule-based signals
- collect news and compute sentiment
- store results locally

The Streamlit app then reads this data and displays insights, signals, and performance.

---

## Watchlist

To change the assets, edit `watchlist.csv` and run the pipeline again:

```powershell
python src/pipeline.py
```

---

## Notes

- The `data/` and `db/` folders are created automatically
- No pre-generated files are required
- Run the pipeline before opening the dashboard

---

## Troubleshooting

If virtual environment activation fails:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If dependencies are missing:

```powershell
pip install -r requirements.txt
```

If Streamlit does not start:

```powershell
python -m streamlit run App.py
```

---

## Author

Zsombor Gyemant