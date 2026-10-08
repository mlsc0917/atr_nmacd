"""Long-only backtest: buy a bullish Normalized MACD turn near the lower ATR Trend Band, sell at NMACD +1.

Entry, checked at each daily close (all on the same bar):
  1. ATR Trend Bands trend is down.
  2. The close is near the lower band: within `near_band` of the corridor, measured from the lower
     band (0 = on the lower band, 1 = on the upper band).
  3. The NMACD line crosses above its trigger line on this bar.
  4. The NMACD line came from near -1: its lowest value over the last `lookback` bars is at or
     below `near_low`.
Exit: the NMACD line reaches `exit_level` (it can't print exactly +1; see nmacd.py), or an optional
stop loss / time limit. Orders fill at the next day's open (TradingView's default) or the signal close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import atr_bands
import nmacd


@dataclass(frozen=True)
class Settings:
    atr_length: int = 5
    atr_mult: float = 7.6
    atr_source: str = "close"
    fast: int = 13
    slow: int = 21
    trigger: int = 9
    normalize: int = 50
    ma_type: int = 1
    near_band: float = 0.25
    near_low: float = -0.8
    lookback: int = 5
    exit_level: float = 0.99
    stop_loss: float = 0.0          # fraction below entry, 0 = off
    max_days: int = 0               # trading days, 0 = off
    fill: str = "open"              # "open" = next day's open, "close" = the signal bar's close
    cost_bps: float = 5.0           # per side
    capital: float = 100_000.0


def indicators(prices: pd.DataFrame, s: Settings) -> pd.DataFrame:
    bands = atr_bands.compute(prices, s.atr_source, s.atr_length, s.atr_mult)
    nm = nmacd.compute(prices["Close"], s.fast, s.slow, s.trigger, s.normalize, s.ma_type)
    d = pd.concat([prices[["Open", "High", "Low", "Close"]],
                   bands[["Upper", "Lower", "Width", "Trend", "Signal", "ATR"]].rename(columns={"Signal": "ATR signal"}),
                   nm[["NMACD", "Trigger", "Hist"]]], axis=1)
    d["Band position"] = (d["Close"] - d["Lower"]) / d["Width"]
    prev_n, prev_t = d["NMACD"].shift(1), d["Trigger"].shift(1)
    d["Cross up"] = (d["NMACD"] > d["Trigger"]) & (prev_n <= prev_t)
    d["Recent NMACD low"] = d["NMACD"].rolling(s.lookback, min_periods=1).min()
    d["Downtrend"] = d["Trend"] == "Down"
    d["Near lower band"] = d["Band position"] <= s.near_band
    d["From near -1"] = d["Recent NMACD low"] <= s.near_low
    d["Entry signal"] = d["Downtrend"] & d["Near lower band"] & d["Cross up"] & d["From near -1"]
    d["Exit signal"] = d["NMACD"] >= s.exit_level
    return d


def run(prices: pd.DataFrame, s: Settings):
    """Bar-by-bar simulation. One position at a time, 100% of equity, compounding."""
    d = indicators(prices, s)
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    entry_sig, exit_sig = d["Entry signal"].to_numpy(), d["Exit signal"].to_numpy()
    dates, n, cost = d.index, len(d), s.cost_bps / 1e4
    equity = np.empty(n)
    cash, shares, pos, trades = s.capital, 0.0, None, []
    pending_entry = pending_exit = None

    def open_pos(i_signal, i_fill, px):
        nonlocal cash, shares, pos
        shares = cash * (1 - cost) / px
        pos = {"signal_i": i_signal, "entry_i": i_fill, "entry_px": px, "equity_in": cash, "low": px}
        cash = 0.0

    def close_pos(i_fill, px, reason, status="Closed"):
        nonlocal cash, shares, pos
        value = shares * px * (1 - cost)
        sig = pos["signal_i"]
        trades.append({
            "Signal date": dates[sig], "Entry date": dates[pos["entry_i"]], "Entry price": pos["entry_px"],
            "Exit date": dates[i_fill], "Exit price": px, "Exit reason": reason,
            "Days held": i_fill - pos["entry_i"], "Return": value / pos["equity_in"] - 1,
            "P&L ($)": value - pos["equity_in"], "Worst drawdown in trade": pos["low"] / pos["entry_px"] - 1,
            "Band position at signal": d["Band position"].iloc[sig], "NMACD at signal": d["NMACD"].iloc[sig],
            "Recent NMACD low": d["Recent NMACD low"].iloc[sig], "Status": status})
        if status == "Closed":
            cash, shares, pos = value, 0.0, None

    for i in range(n):
        # Orders decided at yesterday's close fill at today's open.
        if pending_exit and pos is not None:
            close_pos(i, o[i], pending_exit)
            pending_exit = None
        if pending_entry is not None and pos is None:
            open_pos(pending_entry, i, o[i])
            pending_entry = None
        if pos is not None:
            stop_px = pos["entry_px"] * (1 - s.stop_loss) if s.stop_loss > 0 else -np.inf
            if i > pos["signal_i"] and l[i] <= stop_px:
                pos["low"] = min(pos["low"], l[i])
                close_pos(i, min(o[i], stop_px) if i > pos["entry_i"] else stop_px, "Stop loss")
            else:
                pos["low"] = min(pos["low"], l[i]) if i >= pos["entry_i"] and not (
                    s.fill == "close" and i == pos["entry_i"]) else pos["low"]
        # Decisions at today's close.
        if pos is not None and i > pos["signal_i"]:
            reason = ("NMACD hit +1" if exit_sig[i] else
                      "Time limit" if s.max_days > 0 and i - pos["entry_i"] >= s.max_days else None)
            if reason:
                if s.fill == "close":
                    close_pos(i, c[i], reason)
                elif i + 1 < n:
                    pending_exit = reason
        if pos is None and pending_entry is None and entry_sig[i]:
            if s.fill == "close":
                open_pos(i, i, c[i])
            elif i + 1 < n:
                pending_entry = i
        equity[i] = cash + shares * c[i]
    if pos is not None:                            # still open: mark at the last close
        close_pos(n - 1, c[-1], "Still open", status="Open")
    return d, pd.Series(equity, index=dates), pd.DataFrame(trades)


def stats(eq: pd.Series, trades: pd.DataFrame, prices: pd.DataFrame, capital: float) -> dict:
    r = eq.pct_change().fillna(0.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    bh = prices["Close"] / prices["Close"].iloc[0] * capital
    closed = trades[trades["Status"] == "Closed"] if len(trades) else trades
    wins = closed[closed["Return"] > 0] if len(closed) else closed
    losses = closed[closed["Return"] <= 0] if len(closed) else closed
    in_mkt = 0
    for t in trades.itertuples(index=False) if len(trades) else []:
        in_mkt += t[trades.columns.get_loc("Days held")]
    out = {
        "Trades": len(trades), "Closed trades": len(closed),
        "Win rate": len(wins) / len(closed) if len(closed) else np.nan,
        "Average win": wins["Return"].mean() if len(wins) else np.nan,
        "Average loss": losses["Return"].mean() if len(losses) else np.nan,
        "Average trade": closed["Return"].mean() if len(closed) else np.nan,
        "Best trade": closed["Return"].max() if len(closed) else np.nan,
        "Worst trade": closed["Return"].min() if len(closed) else np.nan,
        "Profit factor": (wins["P&L ($)"].sum() / -losses["P&L ($)"].sum()) if len(losses) and losses["P&L ($)"].sum() < 0 else np.nan,
        "Total P&L ($)": eq.iloc[-1] - capital,
        "Total return": eq.iloc[-1] / capital - 1,
        "CAGR": (eq.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "Average days held": trades["Days held"].mean() if len(trades) else np.nan,
        "Time in market": in_mkt / max(len(eq) - 1, 1),
        "Worst drawdown inside a trade": trades["Worst drawdown in trade"].min() if len(trades) else np.nan,
        "Buy & hold return": bh.iloc[-1] / capital - 1,
        "Buy & hold max drawdown": (bh / bh.cummax() - 1).min(),
        "Years": years,
    }
    return out


def funnel(d: pd.DataFrame) -> pd.DataFrame:
    """How many days pass each entry condition, cumulatively."""
    steps = [("All days with both indicators", d["NMACD"].notna() & d["Trend"].ne("")),
             ("ATR trend is down", d["Downtrend"]),
             ("... and close near the lower band", d["Downtrend"] & d["Near lower band"]),
             ("... and NMACD crosses above trigger", d["Downtrend"] & d["Near lower band"] & d["Cross up"]),
             ("... and NMACD came from near -1", d["Entry signal"])]
    return pd.DataFrame({"Condition": [s[0] for s in steps], "Days": [int(s[1].sum()) for s in steps]})


def sensitivity(prices: pd.DataFrame, s: Settings, bands=(0.10, 0.25, 0.50, 1.00), lows=(-0.9, -0.8, -0.6, -0.4)):
    from dataclasses import replace
    rows = []
    for b in bands:
        for lo in lows:
            q = replace(s, near_band=b, near_low=lo)
            _, eq, tr = run(prices, q)
            st = stats(eq, tr, prices, s.capital)
            rows.append({"Near lower band (within)": b, "Near -1 (at or below)": lo, "Trades": st["Trades"],
                         "Win rate": st["Win rate"], "Average trade": st["Average trade"],
                         "Total return": st["Total return"], "Max drawdown": st["Max drawdown"]})
    return pd.DataFrame(rows)
