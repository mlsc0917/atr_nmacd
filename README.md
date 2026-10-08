# ATR Trend Bands + Normalized MACD backtest

A Streamlit website that backtests a long-only rule on Netflix (or any other stock):

- **Buy** when the ATR Trend Bands trend is down, the close is near the lower band, and the
  Normalized MACD line crosses above its trigger line after being near -1.
- **Sell** when the Normalized MACD line hits +1. A stop loss and a time limit are optional.

It shows P&L, win rate, profit factor, drawdowns, an equity curve against buy & hold, the price
chart with the bands and every buy and sell, the NMACD panel, a downloadable trade log, a count of
how many days pass each condition, and how the result changes with the vague thresholds.

Prices: the Netflix file in `data/` (Nasdaq export, Oct 2016 to Oct 2026), any uploaded CSV of daily
prices (Nasdaq, Yahoo or TradingView export), or any Yahoo Finance ticker.

## Put it online

Upload everything in this folder, including `data/` and `.streamlit/`, to a GitHub repository, then
create an app on https://share.streamlit.io with the main file `app.py`.

## Run it on a computer with Python

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `backtest.py`: the rules, the trade simulation and the statistics
- `nmacd.py`: the Normalized MACD, line by line from the TradingView script
- `atr_bands.py`: the ATR Trend Bands replica (same as the ATR Trend Bands app)
- `data_io.py`: reads the CSV formats and Yahoo Finance
- `app.py`: the web page
