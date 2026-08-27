import re
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from config.settings import VIETNAM_TICKERS

def normalize_symbol(s):
    s = (s or "").strip().upper().replace(" ", "")
    if not s:
        return "HPG.VN"
    if re.fullmatch(r"[A-Z0-9]{2,5}", s) and s in VIETNAM_TICKERS:
        return s + ".VN"
    return s

def display_symbol(s):
    return s.upper().replace(".VN","")

def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def add_indicators(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột {c}")
    df = df.dropna(subset=["Close"])
    df["Return"] = df["Close"].pct_change()
    df["RSI"] = rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Volatility20"] = df["Return"].rolling(20).std() * np.sqrt(252) * 100
    return df.dropna()

@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(symbol, period="1y"):
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")
    return add_indicators(df)
