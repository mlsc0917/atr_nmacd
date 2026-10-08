"""
ATR Trend Bands + Normalized MACD: long-only backtest with a full trade log.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest as B
import data_io

st.set_page_config(page_title="ATR Bands + NMACD backtest", page_icon="📈", layout="wide")

FONT = "IBM Plex Sans"
INK, MUTED, GRID = "#1A2530", "#6B7780", "#E3E8EB"
UP, DOWN, NEUTRAL, ACCENT, TRIG = "#2E8B57", "#B23A48", "#8C969E", "#2C5F8A", "#C08A2E"
BUNDLED = Path(__file__).resolve().parent / "data" / "Netflix_Historical_Data.csv"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, p, li, label, input, h1, h2, h3, h4, h5, .stMarkdown,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ font-family: '{FONT}', sans-serif; }}
    h1 {{ font-weight: 600; letter-spacing: -0.02em; }}
    [data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 500; }}
    .rule-label {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 0.2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def spct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def money(x):
    return "–" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), font=dict(family=FONT, color=INK, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None))
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def segments(x, y, state, want):
    xs, ys = [], []
    for i in range(1, len(y)):
        if state[i] == want and np.isfinite(y[i - 1]) and np.isfinite(y[i]):
            xs += [x[i - 1], x[i], None]
            ys += [y[i - 1], y[i], None]
    return xs, ys


@st.cache_data(show_spinner="Reading prices…")
def load_file(content: bytes | None) -> pd.DataFrame:
    return data_io.read_csv(content if content is not None else BUNDLED)


@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance…")
def load_yahoo(ticker: str) -> pd.DataFrame:
    return data_io.from_yahoo(ticker)


@st.cache_data(show_spinner="Running the backtest…")
def run_cached(prices: pd.DataFrame, settings: dict):
    s = B.Settings(**settings)
    d, eq, tr = B.run(prices, s)
    return d, eq, tr, B.stats(eq, tr, prices, s.capital), B.funnel(d)


@st.cache_data(show_spinner="Testing other thresholds…")
def sensitivity_cached(prices: pd.DataFrame, settings: dict):
    return B.sensitivity(prices, B.Settings(**settings))


# ----------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Prices", ["Netflix file you sent", "Upload a CSV", "Yahoo Finance ticker"])
    upload, ticker = None, ""
    if source == "Upload a CSV":
        upload = st.file_uploader("Daily prices (Nasdaq, Yahoo or TradingView export)", type=["csv"])
    elif source == "Yahoo Finance ticker":
        ticker = st.text_input("Ticker", "NFLX").strip().upper()

    st.header("Entry rules")
    near_band = st.slider("Close to the lower band: within this share of the corridor", 0.05, 1.0, 0.25, 0.05,
                          help="0 = on the lower band, 1 = on the upper band. 0.25 means the bottom quarter.")
    near_low = st.slider("NMACD came from near -1: its recent low is at or below", -1.0, 0.0, -0.8, 0.05)
    lookback = st.slider("Recent = the last this many bars", 1, 20, 5)

    st.header("Exit rules")
    exit_level = st.slider("NMACD 'hits +1': at or above", 0.80, 1.0, 0.99, 0.01,
                           help="The line can't print exactly +1 because of a tiny constant in its formula.")
    stop_loss = st.slider("Optional stop loss (% below entry, 0 = off)", 0, 50, 0, 1)
    max_days = st.slider("Optional time limit (trading days, 0 = off)", 0, 250, 0, 5)

    st.header("Execution")
    fill_label = st.radio("Orders fill at", ["Next day's open (TradingView default)", "The signal day's close"])
    cost_bps = st.number_input("Commission and slippage per side (basis points)", 0.0, 100.0, 5.0, 0.5)
    capital = st.number_input("Starting capital ($)", 1_000, 100_000_000, 100_000, 1_000)
    log_scale = st.toggle("Log scale on the equity chart", True)

    with st.expander("Indicator settings"):
        st.markdown("**ATR Trend Bands**")
        atr_source = st.selectbox("Source", ("close", "open", "high", "low", "hl2", "hlc3", "ohlc4"))
        atr_length = st.number_input("Length ATR", 1, 100, 5)
        atr_mult = st.number_input("Multiplier ATR", 0.1, 50.0, 7.6, 0.1, format="%.1f")
        st.markdown("**Normalized MACD**")
        fast = st.number_input("Fast MA", 1, 200, 13)
        slow = st.number_input("Slow MA", 2, 400, 21)
        trigger = st.number_input("Trigger", 1, 100, 9)
        normalize = st.number_input("Normalize", 1, 500, 50)
        ma_type = st.number_input("1=Ema, 2=Wma, 3=Sma", 1, 3, 1)

# ----------------------------------------------------------------------------------------
# Data and backtest
# ----------------------------------------------------------------------------------------
if source == "Upload a CSV" and upload is None:
    st.info("Upload a CSV of daily prices in the sidebar.")
    st.stop()
if source == "Yahoo Finance ticker" and not ticker:
    st.info("Type a ticker in the sidebar.")
    st.stop()
prices = None
try:
    if source == "Netflix file you sent":
        prices, label = load_file(None), "Netflix (NFLX), from the file you sent"
    elif source == "Upload a CSV":
        prices, label = load_file(upload.getvalue()), upload.name
    else:
        prices, label = load_yahoo(ticker), f"{ticker} from Yahoo Finance"
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't load prices: {e}")
if prices is None:
    st.stop()

settings = B.Settings(atr_length=int(atr_length), atr_mult=float(atr_mult), atr_source=atr_source, fast=int(fast),
                      slow=int(slow), trigger=int(trigger), normalize=int(normalize), ma_type=int(ma_type),
                      near_band=near_band, near_low=near_low, lookback=int(lookback), exit_level=exit_level,
                      stop_loss=stop_loss / 100, max_days=int(max_days),
                      fill="open" if fill_label.startswith("Next") else "close",
                      cost_bps=float(cost_bps), capital=float(capital))
d, eq, trades, stats, funnel = run_cached(prices, asdict(settings))

st.title("ATR Trend Bands + Normalized MACD backtest")
st.caption(f"{label}: {len(prices):,} daily bars, {prices.index[0]:%b %d, %Y} to {prices.index[-1]:%b %d, %Y}. "
           "Long only, one position at a time, all capital in each trade.")

b, s_ = st.columns(2)
with b.container(border=True):
    st.markdown('<div class="rule-label">Buy when, all on the same day\'s close</div>', unsafe_allow_html=True)
    st.markdown(f"- The ATR Trend Bands trend is **down**.\n"
                f"- The close is **near the lower band**: in the bottom {near_band:.0%} of the corridor.\n"
                f"- The NMACD line **crosses above its trigger line**.\n"
                f"- The NMACD line **came from near -1**: it was at or below {near_low:g} within the last "
                f"{lookback} bars.")
with s_.container(border=True):
    st.markdown('<div class="rule-label">Sell when</div>', unsafe_allow_html=True)
    lines = [f"- The NMACD line **hits +1** (closes at or above {exit_level:g})."]
    if stop_loss:
        lines.append(f"- Or the price falls {stop_loss}% below the entry price (stop loss).")
    if max_days:
        lines.append(f"- Or {max_days} trading days have passed.")
    st.markdown("\n".join(lines))
st.caption(f"Orders fill at {'the next day open' if settings.fill == 'open' else 'the signal day close'}, "
           f"with {cost_bps:g} basis points of cost on each side.")

last = d.iloc[-1]
if len(trades) and trades.iloc[-1]["Status"] == "Open":
    t = trades.iloc[-1]
    st.info(f"At the last close ({d.index[-1]:%b %d, %Y}) the backtest is holding a position bought on "
            f"{t['Entry date']:%b %d, %Y} at {t['Entry price']:.2f}, currently {spct(t['Return'])}.")
elif bool(last["Entry signal"]):
    st.info(f"The rules fired a buy signal on the last close ({d.index[-1]:%b %d, %Y}). In the backtest it would "
            "fill at the next session's open, which isn't in the data yet. This is the rules' output, not advice.")

# ----------------------------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------------------------
c = st.columns(6)
c[0].metric("Total P&L", money(stats["Total P&L ($)"]), spct(stats["Total return"]), delta_color="normal")
c[1].metric("Win rate", "–" if pd.isna(stats["Win rate"]) else f"{stats['Win rate']:.0%}",
            f"{int(round(stats['Win rate'] * stats['Closed trades'])) if stats['Closed trades'] else 0} of "
            f"{stats['Closed trades']} closed trades", delta_color="off")
c[2].metric("Average win / loss", f"{spct(stats['Average win'])} / {spct(stats['Average loss'])}")
c[3].metric("Profit factor", "–" if pd.isna(stats["Profit factor"]) else f"{stats['Profit factor']:.2f}")
c[4].metric("Max drawdown", spct(stats["Max drawdown"]))
c[5].metric("Buy & hold instead", spct(stats["Buy & hold return"]),
            f"max drawdown {stats['Buy & hold max drawdown'] * 100:.0f}%", delta_color="off")
c = st.columns(6)
c[0].metric("Trades", f"{stats['Trades']}")
c[1].metric("Average trade", spct(stats["Average trade"]))
c[2].metric("Best / worst trade", f"{spct(stats['Best trade'])} / {spct(stats['Worst trade'])}")
c[3].metric("Average days held", "–" if pd.isna(stats["Average days held"]) else f"{stats['Average days held']:.0f}")
c[4].metric("Time in the market", f"{stats['Time in market']:.0%}")
c[5].metric("Worst dip inside a trade", spct(stats["Worst drawdown inside a trade"]),
            help="Lowest low while holding, against the entry price.")
if stats["Closed trades"] < 30:
    st.warning(f"Only {stats['Closed trades']} closed trades. With so few, the win rate and P&L can swing a lot "
               "from one trade, so treat them as anecdotes rather than evidence.")

# ----------------------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------------------
st.markdown("##### Equity")
bh = prices["Close"] / prices["Close"].iloc[0] * settings.capital
fig = go.Figure()
fig.add_trace(go.Scatter(x=bh.index, y=bh, name="Buy & hold", line=dict(color=NEUTRAL, width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy", line=dict(color=ACCENT, width=2.2)))
fig = style_fig(fig, 360)
fig.update_yaxes(type="log" if log_scale else "linear", tickprefix="$", tickformat=",.0f")
st.plotly_chart(fig, width="stretch")

st.markdown("##### Price with ATR Trend Bands, and the Normalized MACD")
x = list(d.index)
state = d["Trend"].map({"Up": 1, "Down": -1}).fillna(0).to_numpy()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=x, y=d["Close"], name="Close", line=dict(color=INK, width=1.1)), row=1, col=1)
for band in ("Upper", "Lower"):
    y = d[band].to_numpy()
    for want, color in ((1, UP), (-1, DOWN)):
        xs, ys = segments(x, y, state, want)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.6),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Lower"], mode="lines", line=dict(width=0), showlegend=False,
                         hovertemplate="Lower band %{y:,.2f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Upper"], mode="lines", line=dict(width=0), showlegend=False,
                         hovertemplate="Upper band %{y:,.2f}<extra></extra>"), row=1, col=1)
if len(trades):
    fig.add_trace(go.Scatter(x=trades["Entry date"], y=trades["Entry price"], mode="markers", name="Buy",
                             marker=dict(symbol="triangle-up", size=12, color=UP)), row=1, col=1)
    closed = trades[trades["Status"] == "Closed"]
    fig.add_trace(go.Scatter(x=closed["Exit date"], y=closed["Exit price"], mode="markers", name="Sell",
                             marker=dict(symbol="triangle-down", size=12, color=DOWN)), row=1, col=1)
fig.add_trace(go.Bar(x=x, y=d["Hist"], name="Histogram", marker_color="#B8C0C6", opacity=0.6), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=d["NMACD"], name="NMACD", line=dict(color=DOWN, width=1.6)), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Trigger"], name="Trigger", line=dict(color=TRIG, width=1.4)), row=2, col=1)
sig = d[d["Entry signal"]]
fig.add_trace(go.Scatter(x=sig.index, y=sig["NMACD"], mode="markers", name="Buy signal",
                         marker=dict(symbol="circle", size=8, color=UP)), row=2, col=1)
for yv, dash in ((1, "dot"), (-1, "dot"), (0, "solid")):
    fig.add_hline(y=yv, line=dict(color=MUTED, width=0.8, dash=dash), row=2, col=1)
fig = style_fig(fig, 640)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_yaxes(range=[-1.15, 1.15], row=2, col=1)
st.plotly_chart(fig, width="stretch")
st.caption("Bands are green in an uptrend and red in a downtrend. Green dots on the NMACD panel are the days "
           "all four buy conditions held.")

# ----------------------------------------------------------------------------------------
# Trades and diagnostics
# ----------------------------------------------------------------------------------------
st.markdown("##### Trade log")
if len(trades):
    show = trades.iloc[::-1].copy()
    for col in ("Signal date", "Entry date", "Exit date"):
        show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d")
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "Entry price": st.column_config.NumberColumn(format="%.2f"),
        "Exit price": st.column_config.NumberColumn(format="%.2f"),
        "Return": st.column_config.NumberColumn(format="percent"),
        "P&L ($)": st.column_config.NumberColumn(format="dollar"),
        "Worst drawdown in trade": st.column_config.NumberColumn(format="percent"),
        "Band position at signal": st.column_config.NumberColumn(format="%.2f"),
        "NMACD at signal": st.column_config.NumberColumn(format="%.2f"),
        "Recent NMACD low": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download the trade log (CSV)", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
else:
    st.write("No trades with these settings.")

st.markdown("##### Why there are so few trades")
st.dataframe(funnel, hide_index=True, width="stretch")
st.caption("Each row keeps only the days that also pass the rows above it. Signal days that come while a "
           "position is already open are skipped, so trades can be fewer than signal days.")

st.markdown("##### How the result changes with the two vague thresholds")
sens = sensitivity_cached(prices, asdict(settings))
st.dataframe(sens, hide_index=True, width="stretch", column_config={
    "Win rate": st.column_config.NumberColumn(format="percent"),
    "Average trade": st.column_config.NumberColumn(format="percent"),
    "Total return": st.column_config.NumberColumn(format="percent"),
    "Max drawdown": st.column_config.NumberColumn(format="percent")})
st.caption("'Close to the lower band' and 'near -1' have no exact definition, so this reruns the backtest "
           "across several readings of each. Everything else stays as set in the sidebar.")

with st.expander("How the indicators and rules are defined"):
    st.markdown(
        f"""
**Normalized MACD** (TradingView's "N MACD"). Fast and slow moving averages of the close
({fast} and {slow} bars; type {ma_type}: 1 = EMA, 2 = WMA, 3 = SMA). Their ratio is turned into a
momentum value, which is rescaled to between -1 and +1 against its own range over the last {normalize} bars.
The trigger line is a {trigger}-bar weighted average of the NMACD line.

**What +1 and -1 mean.** +1 means the fast average is further above the slow one than at any point in the
last {normalize} bars, and -1 means the reverse. Because the scale is relative to the last {normalize} bars,
the line can reach +1 after a modest bounce in a falling market. So "NMACD hit +1" doesn't mean the price is
above where you bought, which is why some exits here are losses.

**ATR Trend Bands** use the same replica as the ATR Trend Bands app, with source {atr_source},
length {atr_length} and multiplier {atr_mult:g}.

**Turning the words into numbers.**
- "Close to the lower band": the close sits in the bottom {near_band:.0%} of the corridor between the bands.
- "Crosses the trigger from near -1": the NMACD line closes above the trigger after closing at or below it
  the bar before, and was at or below {near_low:g} within the last {lookback} bars.
- "Hits +1": the NMACD line closes at or above {exit_level:g}.

**What the test leaves out:** taxes, borrowing costs, dividends (Netflix pays none) and gaps through a stop,
beyond the fill rules above.
        """
    )
