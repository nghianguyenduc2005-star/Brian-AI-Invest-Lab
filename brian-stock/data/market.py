```python
import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config.settings import VIETNAM_TICKERS


# ============================================================
# DNSE CONFIG
# ============================================================

DNSE_BASE_URL = "https://api.dnse.com.vn/market-data/v1"


# ============================================================
# SYMBOL
# ============================================================

def normalize_symbol(s):
    s = (s or "").strip().upper().replace(" ", "")

    if not s:
        return "HPG"

    # VN-INDEX
    if s in ["VNINDEX", "VN-INDEX", "VNINDEX.VN"]:
        return "VNINDEX"

    # Nếu người dùng nhập HPG.VN
    if s.endswith(".VN"):
        s = s[:-3]

    # HPG -> HPG
    if re.fullmatch(r"[A-Z0-9]{2,5}", s):
        return s

    return s


def display_symbol(s):
    s = str(s).upper()
    return s.replace(".VN", "").replace("-INDEX", "")


# ============================================================
# TECHNICAL INDICATORS
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


def add_indicators(df):
    df = df.copy()

    # --------------------------------------------------------
    # Chuẩn hóa columns
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(c).strip().title()
        for c in df.columns
    ]

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
                f"DNSE thiếu cột {col}. "
                f"Các cột nhận được: {list(df.columns)}"
            )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    if df.empty:
        raise ValueError(
            "DNSE trả về dữ liệu rỗng."
        )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df["Return"] = df["Close"].pct_change()

    df["RSI"] = rsi(
        df["Close"],
        period=14
    )

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

    # Không drop toàn bộ dữ liệu nếu indicator
    # chưa đủ số phiên.
    #
    # Việc này đặc biệt quan trọng với VNINDEX
    # khi API chỉ trả về một số phiên gần nhất.

    return df


# ============================================================
# DNSE OHLC
# ============================================================

def _period_to_days(period):
    mapping = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 731,
        "5y": 1826,
    }

    return mapping.get(
        str(period).lower(),
        366
    )


def _unix_timestamp(dt):
    return int(dt.timestamp())


def _extract_ohlc_rows(payload):
    """
    DNSE có thể trả response dưới dạng:

    {
        "data": [...]
    }

    hoặc trực tiếp:

    [...]
    """

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    # Các key thường gặp
    for key in [
        "data",
        "content",
        "items",
        "bars",
        "ohlc",
        "results",
    ]:
        value = payload.get(key)

        if isinstance(value, list):
            return value

    return []


def _row_to_ohlcv(row):
    """
    Chuẩn hóa một candle DNSE thành:

    Date
    Open
    High
    Low
    Close
    Volume
    """

    if not isinstance(row, dict):
        return None

    def get_value(*keys):
        for key in keys:
            if key in row:
                return row[key]
        return None

    timestamp = get_value(
        "time",
        "timestamp",
        "t",
        "startTime",
        "start_time",
        "date",
    )

    open_price = get_value(
        "open",
        "Open",
        "o",
    )

    high_price = get_value(
        "high",
        "High",
        "h",
    )

    low_price = get_value(
        "low",
        "Low",
        "l",
    )

    close_price = get_value(
        "close",
        "Close",
        "c",
    )

    volume = get_value(
        "volume",
        "Volume",
        "v",
        "qtty",
        "quantity",
    )

    if (
        open_price is None
        or high_price is None
        or low_price is None
        or close_price is None
    ):
        return None

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    try:
        if timestamp is None:
            date_value = pd.NaT

        elif isinstance(timestamp, (int, float)):
            # DNSE dùng Unix timestamp.
            # Nếu milliseconds thì chia 1000.
            if timestamp > 10_000_000_000:
                timestamp = timestamp / 1000

            date_value = pd.to_datetime(
                timestamp,
                unit="s",
                utc=True,
            )

        else:
            date_value = pd.to_datetime(
                timestamp,
                utc=True,
                errors="coerce",
            )

    except Exception:
        date_value = pd.NaT

    return {
        "Date": date_value,
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price,
        "Volume": volume if volume is not None else 0,
    }


# ============================================================
# LOAD FROM DNSE
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_market_data(symbol, period="1y"):

    symbol = normalize_symbol(symbol)

    # --------------------------------------------------------
    # Khoảng thời gian
    # --------------------------------------------------------

    now = datetime.now(timezone.utc)

    days = _period_to_days(period)

    start = now - timedelta(
        days=days
    )

    # DNSE OHLC
    params = {
        "symbol": symbol,
        "resolution": "1D",
        "from": _unix_timestamp(start),
        "to": _unix_timestamp(now),
    }

    url = f"{DNSE_BASE_URL}/price/ohlc"

    headers = {
        "Accept": "application/json",
        "User-Agent": "BrianStock/1.0",
    }

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20,
        )

    except requests.RequestException as e:
        raise ValueError(
            f"Không kết nối được DNSE: {e}"
        )

    # --------------------------------------------------------
    # HTTP errors
    # --------------------------------------------------------

    if response.status_code == 401:
        raise ValueError(
            "DNSE yêu cầu xác thực API. "
            "Kiểm tra API Key/Secret trong cấu hình DNSE."
        )

    if response.status_code == 429:
        raise ValueError(
            "DNSE đang giới hạn request (HTTP 429). "
            "Chờ một chút rồi tải lại."
        )

    if response.status_code >= 400:
        raise ValueError(
            f"DNSE API lỗi HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:
        payload = response.json()

    except Exception:
        raise ValueError(
            "DNSE trả về dữ liệu không phải JSON."
        )

    rows = _extract_ohlc_rows(
        payload
    )

    if not rows:
        raise ValueError(
            f"DNSE không trả về dữ liệu OHLC cho {symbol}."
        )

    # --------------------------------------------------------
    # Normalize rows
    # --------------------------------------------------------

    normalized = []

    for row in rows:
        item = _row_to_ohlcv(row)

        if item is not None:
            normalized.append(item)

    if not normalized:
        raise ValueError(
            f"Không thể chuẩn hóa dữ liệu OHLC của DNSE cho {symbol}."
        )

    df = pd.DataFrame(
        normalized
    )

    # --------------------------------------------------------
    # Date index
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["Date"]
    )

    # Việt Nam UTC+7
    df["Date"] = (
        df["Date"]
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.tz_localize(None)
    )

    df = df.set_index(
        "Date"
    )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df = add_indicators(df)

    if df.empty:
        raise ValueError(
            f"DNSE có dữ liệu nhưng sau khi xử lý "
            f"không còn phiên hợp lệ cho {symbol}."
        )

    return df
```
