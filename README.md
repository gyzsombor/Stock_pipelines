# Stock Pipelines

## Purpose

This project is a multi-asset market analytics system. It downloads market data, engineers technical features, generates rule-based signals, applies machine learning models, integrates news sentiment, performs backtesting, and displays results in a Streamlit dashboard.

This README is written so a grader can run the project on a new machine without asking questions.

---

## Files Needed

Main folder:
- App.py
- visuals.py
- requirements.txt
- watchlist.csv
- README.md

src/ folder:
- __init__.py
- pipeline.py
- config.py
- io_utils.py
- features.py
- signals.py
- news_features.py
- modeling.py
- backtest.py
- portfolio.py

---

## System Requirements

- Windows + PowerShell
- Python 3.10 or newer
- Git installed
- Internet connection

---

## Setup

Clone the repository:

```powershell
git clone https://github.com/gyzsombor/Stock_pipelines.git
cd Stock_pipelines
```

Create virtual environment:

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

## Run the Project

Run pipeline:

```powershell
python src/pipeline.py
```

Run dashboard:

```powershell
streamlit run App.py
```

---

## First Run Notes

- `data/` and `db/` folders are created automatically
- No pre-generated files are required
- Always run the pipeline before opening the dashboard

---

## What the Pipeline Does

Running:

```powershell
python src/pipeline.py
```

Will:
- load symbols from `watchlist.csv`
- download market data
- generate features
- create signals
- collect news and compute sentiment
- save outputs locally

---

## Watchlist

Edit `watchlist.csv` to change assets, then rerun:

```powershell
python src/pipeline.py
```

---

## Grading Order

1. Run:
```powershell
python src/pipeline.py
```

2. Then:
```powershell
streamlit run App.py
```

---

## Troubleshooting

If activation fails:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If packages are missing:

```powershell
pip install -r requirements.txt
```

If Streamlit fails:

```powershell
python -m streamlit run App.py
```

---

## Author

Zsombor Gyemant  
MSBA 265 – Business Analytics Topics