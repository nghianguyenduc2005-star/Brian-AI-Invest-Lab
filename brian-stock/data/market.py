import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config.settings import VIETNAM_TICKERS


# ============================================================
# DNSE
# ============================================================

DNSE_BASE_URL = "https://api.dnse.com.vn/market-data/v1"


# ============================================================
# SYMBOL
# ============================================================

def normalize_symbol(symbol):
    """
    Chuẩn hóa mã:

    HPG       -> HPG
    HPG.VN    -> HPG
    VNINDEX   -> VNINDEX
    VN-INDEX  -> VNINDEX
    """

    symbol = (symbol or "").strip().upper().replace(" ", "")

    if not symbol:
        return "HPG"

    if symbol in (
        "VNINDEX",
        "VN-INDEX",
        "VNINDEX.VN",
    ):
        return "VNINDEX"

    if symbol.endswith(".VN"):
        symbol = symbol[:-3]

    if re.fullmatch(r"[A-Z0-9]{2,10}", symbol):
        return symbol

    return symbol


def display_symbol(symbol):
    symbol = str(symbol).upper()

    return (
        symbol
        .replace(".VN", "")
        .replace("-INDEX", "")
    )


# ============================================================
# RSI
# ============================================================

def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    # --------------------------------------------------------
    # MultiIndex
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(column).strip().title()
        for column in df.columns
    ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Thiếu cột {column}. "
                f"Các cột hiện có: {list(df.columns)}"
            )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
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
            "Không còn dữ liệu hợp lệ sau khi xử lý."
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    df["Return"] = (
        df["Close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = rsi(
        df["Close"],
        14,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    df["SMA20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["Volatility20"] = (
        df["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    return df


# ============================================================
# PERIOD
# ============================================================

def period_to_days(period):
    period = str(period).lower()

    periods = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 731,
        "5y": 1826,
    }

    return periods.get(
        period,
        366,
    )


# ============================================================
# DNSE AUTH
# ============================================================

def get_dnse_token():
    """
    Lấy token từ Streamlit Secrets hoặc environment.

    Ưu tiên:

    st.secrets["DNSE_API_TOKEN"]

    hoặc:

    os.environ["DNSE_API_TOKEN"]
    """

    token = None

    try:
        token = st.secrets.get(
            "DNSE_API_TOKEN"
        )
    except Exception:
        pass

    if not token:
        token = os.getenv(
            "DNSE_API_TOKEN"
        )

    if token:
        return str(token).strip()

    return None


# ============================================================
# DNSE REQUEST
# ============================================================

def dnse_get(
    endpoint,
    params=None,
):
    url = (
        f"{DNSE_BASE_URL}"
        f"/{endpoint.lstrip('/')}"
    )

    token = get_dnse_token()

    headers = {
        "Accept": "application/json",
        "User-Agent": "BrianStock/1.0",
    }

    if token:
        headers[
            "Authorization"
        ] = f"Bearer {token}"

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20,
        )

    except requests.RequestException as exc:
        raise ValueError(
            f"Không kết nối được DNSE: {exc}"
        )

    if response.status_code == 401:
        raise ValueError(
            "DNSE trả HTTP 401. "
            "Kiểm tra DNSE_API_TOKEN."
        )

    if response.status_code == 403:
        raise ValueError(
            "DNSE trả HTTP 403. "
            "Token không có quyền truy cập API."
        )

    if response.status_code == 404:
        raise ValueError(
            f"DNSE không tìm thấy endpoint: {url}"
        )

    if response.status_code == 429:
        raise ValueError(
            "DNSE đang rate-limit request "
            "(HTTP 429). Hãy thử lại sau."
        )

    if response.status_code >= 500:
        raise ValueError(
            f"DNSE server error "
            f"(HTTP {response.status_code})."
        )

    if response.status_code >= 400:
        raise ValueError(
            f"DNSE API lỗi HTTP "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    try:
        return response.json()

    except ValueError:
        raise ValueError(
            "DNSE trả về response không phải JSON."
        )


# ============================================================
# EXTRACT DATA
# ============================================================

def extract_rows(payload):
    """
    DNSE có thể trả dữ liệu nằm trong data/content/items
    hoặc trả list trực tiếp.
    """

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in (
        "data",
        "content",
        "items",
        "results",
        "bars",
        "ohlc",
    ):
        value = payload.get(key)

        if isinstance(value, list):
            return value

    return []


# ============================================================
# PARSE DATE
# ============================================================

def parse_datetime(value):
    if value is None:
        return pd.NaT

    try:
        if isinstance(
            value,
            (int, float),
        ):
            # milliseconds
            if value > 10_000_000_000:
                return pd.to_datetime(
                    value,
                    unit="ms",
                    utc=True,
                )

            # seconds
            return pd.to_datetime(
                value,
                unit="s",
                utc=True,
            )

        return pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )

    except Exception:
        return pd.NaT


# ============================================================
# PARSE OHLC ROW
# ============================================================

def parse_ohlc_row(row):
    if not isinstance(row, dict):
        return None

    def value(*keys):
        for key in keys:
            if key in row:
                return row[key]

        return None

    timestamp = value(
        "time",
        "timestamp",
        "t",
        "startTime",
        "start_time",
        "date",
        "datetime",
    )

    open_price = value(
        "open",
        "Open",
        "o",
    )

    high_price = value(
        "high",
        "High",
        "h",
    )

    low_price = value(
        "low",
        "Low",
        "l",
    )

    close_price = value(
        "close",
        "Close",
        "c",
    )

    volume = value(
        "volume",
        "Volume",
        "v",
        "quantity",
        "qtty",
    )

    if (
        open_price is None
        or high_price is None
        or low_price is None
        or close_price is None
    ):
        return None

    return {
        "Date": parse_datetime(
            timestamp
        ),
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price,
        "Volume": (
            volume
            if volume is not None
            else 0
        ),
    }


# ============================================================
# LOAD MARKET DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_market_data(
    symbol,
    period="1y",
):
    """
    Lấy dữ liệu OHLCV từ DNSE.

    Không sử dụng:
        yfinance
        vnstock
    """

    symbol = normalize_symbol(
        symbol
    )

    days = period_to_days(
        period
    )

    now = datetime.now(
        timezone.utc
    )

    start = (
        now
        - timedelta(days=days)
    )

    # --------------------------------------------------------
    # DNSE OHLC
    # --------------------------------------------------------

    params = {
        "symbol": symbol,
        "resolution": "1D",
        "from": int(
            start.timestamp()
        ),
        "to": int(
            now.timestamp()
        ),
    }

    payload = dnse_get(
        "price/ohlc",
        params=params,
    )

    rows = extract_rows(
        payload
    )

    if not rows:
        raise ValueError(
            f"DNSE không trả dữ liệu "
            f"OHLC cho {symbol}."
        )

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    parsed = []

    for row in rows:
        item = parse_ohlc_row(
            row
        )

        if item is not None:
            parsed.append(item)

    if not parsed:
        raise ValueError(
            f"Không thể đọc dữ liệu OHLC "
            f"DNSE của {symbol}."
        )

    df = pd.DataFrame(
        parsed
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["Date"]
    )

    if df.empty:
        raise ValueError(
            "DNSE trả dữ liệu nhưng "
            "không có ngày hợp lệ."
        )

    # UTC -> Vietnam
    df["Date"] = (
        df["Date"]
        .dt.tz_convert(
            "Asia/Ho_Chi_Minh"
        )
        .dt.tz_localize(None)
    )

    df = df.set_index(
        "Date"
    )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for column in (
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    # --------------------------------------------------------
    # Sort + duplicate
    # --------------------------------------------------------

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    if df.empty:
        raise ValueError(
            f"Không có phiên hợp lệ "
            f"cho {symbol}."
        )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df = add_indicators(
        df
    )

    return df
