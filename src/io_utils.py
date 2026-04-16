
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import yfinance as yf


def ensure_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("db", exist_ok=True)


def load_watchlist(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Watchlist file not found: {path}")

    wl = pd.read_csv(path)
    wl.columns = wl.columns.astype(str).str.strip().str.lower()

    if "symbol" in wl.columns:
        result = wl[["symbol"]].copy()
    elif "ticker" in wl.columns:
        result = wl.rename(columns={"ticker": "symbol"})[["symbol"]].copy()
    elif len(wl.columns) == 1:
        result = pd.read_csv(path, header=None, names=["symbol"])
    else:
        raise ValueError(
            f"{path} must have a 'symbol' or 'ticker' column, or be a one-column CSV. "
            f"Found columns: {wl.columns.tolist()}"
        )

    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    result["symbol"] = result["symbol"].replace("", np.nan)
    result = (
        result.dropna(subset=["symbol"])
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError(f"{path} contains no valid symbols.")

    return result


def _is_cache_fresh(path: str, refresh_after_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    max_age_seconds = refresh_after_hours * 60 * 60
    return age_seconds <= max_age_seconds


def _normalize_single_symbol_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.reset_index().copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    if "date" not in out.columns and "datetime" in out.columns:
        out = out.rename(columns={"datetime": "date"})

    out["symbol"] = symbol

    needed = ["symbol", "date", "open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in out.columns:
            out[col] = np.nan

    out = out[needed].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "close"]).reset_index(drop=True)
    return out


def fetch_ohlcv_batch(
    symbols: list[str],
    period: str,
    interval: str,
    cache_path: str,
    refresh_after_hours: int,
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("No symbols provided to fetch_ohlcv_batch().")

    if _is_cache_fresh(cache_path, refresh_after_hours):
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        cached = cached.dropna(subset=["date", "close"]).copy()
        cached["symbol"] = cached["symbol"].astype(str).str.upper()
        return cached.sort_values(["symbol", "date"]).reset_index(drop=True)

    tickers = " ".join(symbols)
    print(f"Downloading {len(symbols)} symbols in batch...")
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=True,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("No data downloaded from Yahoo Finance.")

    frames = []

    if isinstance(data.columns, pd.MultiIndex):
        available_symbols = set(str(x) for x in data.columns.get_level_values(0))
        for symbol in symbols:
            if symbol not in available_symbols:
                print(f"No data returned for: {symbol}")
                continue
            part = data[symbol].copy()
            part = _normalize_single_symbol_frame(part, symbol)
            if not part.empty:
                frames.append(part)
    else:
        if len(symbols) != 1:
            raise RuntimeError(
                "Expected multi-symbol download to return MultiIndex columns, but it did not."
            )
        symbol = symbols[0]
        part = _normalize_single_symbol_frame(data.copy(), symbol)
        if not part.empty:
            frames.append(part)

    if not frames:
        raise RuntimeError("No valid symbol data was returned after normalization.")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    raw.to_csv(cache_path, index=False)
    return raw
