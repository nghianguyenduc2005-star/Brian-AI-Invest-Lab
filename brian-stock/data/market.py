```python
import re
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from config.settings import VIETNAM_TICKERS


# ============================================================
# SYMBOL
# ============================================================

def normalize_symbol(s):
    s = (s or "").strip().upper().replace(" ", "")

    if not s:
        return "HPG.VN"

    # Người dùng nhập HPG -> HPG.VN
    if re.fullmatch(r"[A-Z0-9]{2,5}", s):
        if s in VIETNAM_TICKERS or s.isalpha():
            return s + ".VN"

    return s


def display_symbol(s):
    return s.upper().replace(".VN", "")


# ============================================================
# RSI
# ============================================================

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


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    # yfinance đôi khi trả MultiIndex
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

    # Ép kiểu số
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

    # Return
    df["Return"] = df["Close"].pct_change()

    # RSI
    df["RSI"] = rsi(df["Close"])

    # EMA / MACD
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

    # Moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # Volatility
    df["Volatility20"] = (
        df["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    # Chỉ bỏ những dòng chưa đủ indicator
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


# ============================================================
# YAHOO DATA LOADER
# ============================================================

def _download_yahoo(symbol, period="1y"):
    """
    Tải dữ liệu Yahoo Finance.

    Thử Ticker.history trước.
    Nếu thất bại / rỗng thì fallback sang yf.download.
    """

    # Cách 1
    try:
        ticker = yf.Ticker(symbol)

        df = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

        if df is not None and not df.empty:
            return df

    except Exception:
        pass

    # Cách 2
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is not None and not df.empty:
            return df

    except Exception as e:
        raise ValueError(
            f"Không thể tải dữ liệu {symbol} từ Yahoo Finance: {e}"
        )

    raise ValueError(
        f"Yahoo Finance không trả về dữ liệu cho {symbol}."
    )


# ============================================================
# STOCK DATA
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_market_data(symbol, period="1y"):

    symbol = normalize_symbol(symbol)

    df = _download_yahoo(
        symbol,
        period
    )

    return add_indicators(df)


# ============================================================
# VN-INDEX
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_vnindex_data(period="1y"):
    """
    Tải dữ liệu VN-INDEX từ Yahoo Finance.

    Yahoo Finance sử dụng mã:
        ^VNINDEX
    """

    symbol = "^VNINDEX"

    df = _download_yahoo(
        symbol,
        period
    )

    return add_indicators(df)


# ============================================================
# VN-INDEX SUMMARY
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_vnindex_summary():
    """
    Trả về số liệu phiên gần nhất của VN-INDEX.

    Kết quả:
        index       : điểm VN-INDEX
        change      : thay đổi điểm so với phiên trước
        change_pct  : % thay đổi
        volume      : thanh khoản
        date        : ngày dữ liệu
    """

    df = load_vnindex_data("1y")

    if df is None or df.empty:
        raise ValueError(
            "Không có dữ liệu VN-INDEX."
        )

    last = df.iloc[-1]

    close = float(last["Close"])
    volume = float(last["Volume"])

    # Return là % thay đổi so với phiên trước
    change_pct = float(last["Return"]) if pd.notna(last["Return"]) else 0.0

    # Tính thay đổi điểm
    if len(df) >= 2:
        previous_close = float(df.iloc[-2]["Close"])
        change = close - previous_close
    else:
        change = 0.0

    # Ngày dữ liệu
    try:
        date_value = df.index[-1]
        date = str(date_value.date())
    except Exception:
        date = ""

    return {
        "index": close,
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "date": date,
    }
```
