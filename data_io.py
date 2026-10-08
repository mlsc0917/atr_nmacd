"""Load daily prices from the Nasdaq-style CSV, a Yahoo-style CSV, or Yahoo Finance."""
from __future__ import annotations

import io

import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
                         errors="coerce")


def read_csv(source) -> pd.DataFrame:
    """Accepts Nasdaq exports (Date, Close/Last, Volume, Open, High, Low with $ signs) and
    Yahoo/TradingView exports (Date, Open, High, Low, Close[, Adj Close, Volume])."""
    raw = pd.read_csv(source if not isinstance(source, bytes) else io.BytesIO(source))
    cols = {c.strip().lower(): c for c in raw.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return raw[cols[n]]
        raise ValueError(f"The file needs a column named one of {names}; it has {list(raw.columns)}.")
    df = pd.DataFrame({
        "Date": pd.to_datetime(pick("date", "time", "datetime"), errors="coerce"),
        "Open": _num(pick("open")), "High": _num(pick("high")), "Low": _num(pick("low")),
        "Close": _num(pick("close/last", "close", "last")),
    })
    df = df.dropna().drop_duplicates("Date").sort_values("Date").set_index("Date")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz is not None else df.index
    if len(df) < 120:
        raise ValueError(f"Only {len(df)} usable rows; the indicators need at least 120 bars.")
    return df


def from_yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False, actions=False)
    if df is None or df.empty:
        raise ValueError(f"Yahoo Finance returned no prices for {ticker}.")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close"]].dropna()
