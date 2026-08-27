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

    if re.fullmatch(r"[A-Z0-9]{2,5}", s):
        if s in VIETNAM_TICKERS or s.isalpha():
            return s + ".VN"

    return s


def display_symbol(s):
    return s.upper().replace(".VN", "")


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Thiếu cột {col}. Các cột nhận được: {list(df.columns)}"
            )

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    if len(df) < 60:
        raise ValueError(
            f"Dữ liệu quá ít: chỉ có {len(df)} phiên giao dịch."
        )

    df["Return"] = df["Close"].pct_change()

    df["RSI"] = rsi(df["Close"])

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    df["Volatility20"] = (
        df["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    df = df.dropna(
        subset=[
            "Return",
            "RSI",
            "MACD",
            "MACD_Signal",
            "SMA20",
            "SMA50",
            "Volatility20",
        ]
    )

    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(symbol, period="1y"):

    symbol = normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)

        df = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

    except Exception as e:
        raise ValueError(
            f"Lỗi kết nối Yahoo Finance khi tải {symbol}: {e}"
        )

    if df is None or df.empty:
        try:
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False
            )

        except Exception as e:
            raise ValueError(
                f"Không thể tải dữ liệu {symbol}: {e}"
            )

    if df is None or df.empty:
        raise ValueError(
            f"Không có dữ liệu thị trường cho {symbol}."
        )

    return add_indicators(df)
