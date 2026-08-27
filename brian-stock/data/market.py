import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config.settings import VIETNAM_TICKERS


# ============================================================
# CONFIG
# ============================================================

DNSE_BASE_URL = "https://api.dnse.com.vn"

DNSE_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://banggia.dnse.com.vn",
    "Referer": "https://banggia.dnse.com.vn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
}

TIMEOUT = 20


# ============================================================
# SYMBOL
# ============================================================

def normalize_symbol(s):
    s = (s or "").strip().upper().replace(" ", "")

    if not s:
        return "HPG"

    # Không dùng .VN cho DNSE
    s = s.replace(".VN", "")

    if re.fullmatch(r"[A-Z0-9]{2,10}", s):
        return s

    return "HPG"


def display_symbol(s):
    return str(s).upper().replace(".VN", "")


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

    result = 100 - (100 / (1 + rs))

    return result


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame rỗng.")

    # MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            str(c[0]) if isinstance(c, tuple) else str(c)
            for c in df.columns
        ]

    # Chuẩn hóa tên cột
    rename_map = {}

    for col in df.columns:
        c = str(col).strip().lower()

        if c in ["open", "openprice", "open_price"]:
            rename_map[col] = "Open"

        elif c in ["high", "highprice", "high_price"]:
            rename_map[col] = "High"

        elif c in ["low", "lowprice", "low_price"]:
            rename_map[col] = "Low"

        elif c in ["close", "closeprice", "close_price", "lastprice"]:
            rename_map[col] = "Close"

        elif c in ["volume", "totalvolume", "total_volume"]:
            rename_map[col] = "Volume"

        elif c in ["date", "datetime", "time", "tradingdate"]:
            rename_map[col] = "Date"

    df = df.rename(columns=rename_map)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "DNSE trả về thiếu cột: "
            + ", ".join(missing)
            + f". Cột hiện có: {list(df.columns)}"
        )

    # Numeric
    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.set_index("Date")

    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )

    df = df.sort_index()

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
            "DNSE trả về dữ liệu nhưng không có OHLC hợp lệ."
        )

    # ========================================================
    # RETURN
    # ========================================================

    df["Return"] = df["Close"].pct_change()

    # ========================================================
    # RSI
    # ========================================================

    df["RSI"] = rsi(df["Close"])

    # ========================================================
    # MACD
    # ========================================================

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

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

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

    # ========================================================
    # VOLATILITY
    # ========================================================

    df["Volatility20"] = (
        df["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    # Không bắt buộc drop hết indicator
    # để tránh mất dữ liệu nếu DNSE trả ít phiên.
    df = df.dropna(
        subset=["Close"]
    )

    return df


# ============================================================
# DNSE RESPONSE PARSER
# ============================================================

def _find_records(obj):
    """
    Tìm list record trong response DNSE.
    DNSE có thể thay đổi wrapper JSON,
    nên không hard-code một key duy nhất.
    """

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):

        preferred_keys = [
            "data",
            "items",
            "results",
            "content",
            "rows",
            "candles",
            "ohlc",
            "dataList",
        ]

        for key in preferred_keys:
            value = obj.get(key)

            if isinstance(value, list):
                return value

        # Tìm sâu hơn
        for value in obj.values():
            result = _find_records(value)

            if result:
                return result

    return []


def _records_to_dataframe(records):
    if not records:
        return pd.DataFrame()

    # Record dạng dict
    if isinstance(records[0], dict):
        return pd.DataFrame(records)

    # Record dạng array
    if isinstance(records[0], (list, tuple)):

        if len(records[0]) >= 6:

            columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            return pd.DataFrame(
                records,
                columns=columns[:len(records[0])]
            )

    return pd.DataFrame()


# ============================================================
# DNSE HISTORY
# ============================================================

def _request_dnse_history(
    symbol,
    start_date,
    end_date,
):
    """
    Thử các endpoint market-data hiện có.

    Không dùng vnstock.
    Không dùng yfinance.
    """

    symbol = normalize_symbol(symbol)

    # Endpoint REST cũ của bảng giá DNSE.
    # Một số hệ thống DNSE dùng POST /price-api/query.
    url = (
        f"{DNSE_BASE_URL}"
        "/price-api/query"
    )

    payloads = [

        # Query OHLC
        {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
            "resolution": "1D",
        },

        {
            "symbol": symbol,
            "fromDate": start_date,
            "toDate": end_date,
            "resolution": "1D",
        },

        {
            "symbols": [symbol],
            "from": start_date,
            "to": end_date,
            "resolution": "1D",
        },

    ]

    last_error = None

    for payload in payloads:

        try:

            response = requests.post(
                url,
                headers=DNSE_HEADERS,
                json=payload,
                timeout=TIMEOUT,
            )

            if response.status_code == 404:
                last_error = (
                    f"DNSE 404: {url}"
                )
                continue

            if response.status_code >= 400:
                last_error = (
                    f"DNSE HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
                continue

            body = response.json()

            records = _find_records(body)

            if records:

                df = _records_to_dataframe(
                    records
                )

                if not df.empty:
                    return df

        except requests.RequestException as e:
            last_error = str(e)

        except ValueError as e:
            last_error = (
                f"JSON không hợp lệ: {e}"
            )

    raise ValueError(
        "DNSE không trả dữ liệu OHLC cho "
        f"{symbol}. "
        f"Endpoint hiện tại: {url}. "
        f"Lỗi cuối: {last_error}"
    )


# ============================================================
# PUBLIC LOAD FUNCTION
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_market_data(
    symbol,
    period="1y",
):

    symbol = normalize_symbol(symbol)

    # ========================================================
    # PERIOD
    # ========================================================

    period_days = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 732,
        "5y": 1825,
    }

    days = period_days.get(
        period,
        366
    )

    end_date = datetime.now()

    start_date = (
        end_date
        - timedelta(days=days)
    )

    start_str = start_date.strftime(
        "%Y-%m-%d"
    )

    end_str = end_date.strftime(
        "%Y-%m-%d"
    )

    # ========================================================
    # DNSE
    # ========================================================

    df = _request_dnse_history(
        symbol,
        start_str,
        end_str,
    )

    if df is None or df.empty:
        raise ValueError(
            f"DNSE không có dữ liệu cho {symbol}."
        )

    return add_indicators(df)
