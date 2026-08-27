import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config.settings import VIETNAM_TICKERS


# ============================================================
# SYMBOL
# ============================================================

def normalize_symbol(s):
    s = (s or "").strip().upper().replace(" ", "")

    if not s:
        return "HPG"

    # Cho phép người dùng nhập HPG hoặc HPG.VN
    if s.endswith(".VN"):
        s = s[:-3]

    if re.fullmatch(r"[A-Z0-9]{2,5}", s):
        return s

    return s


def display_symbol(s):
    return normalize_symbol(s)


# ============================================================
# DNSE
# ============================================================

DNSE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    "Origin": "https://banggia.dnse.com.vn",
    "Referer": "https://banggia.dnse.com.vn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
}


# Endpoint chart data của DNSE/Entrade.
# Không dùng /price-api/query vì endpoint đó đang trả 422
# với payload hiện tại của project.
DNSE_OHLC_URLS = [
    "https://api.dnse.com.vn/chart-api/v2/ohlcs/stock",
    "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock",
]


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

    if df.empty:
        raise ValueError("DataFrame rỗng.")

    # Chuẩn hóa tên cột
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

            # Một số response có volume viết khác
            if col == "Volume":

                alternatives = [
                    "Vol",
                    "Totalvolume",
                    "Total Volume",
                ]

                found = next(
                    (
                        x
                        for x in alternatives
                        if x in df.columns
                    ),
                    None,
                )

                if found:
                    df["Volume"] = df[found]
                    continue

            raise ValueError(
                f"DNSE thiếu cột {col}. "
                f"Các cột nhận được: {list(df.columns)}"
            )

    # Numeric
    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Date
    if not isinstance(df.index, pd.DatetimeIndex):

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df = df.set_index("Date")

        elif "Time" in df.columns:

            df["Time"] = pd.to_datetime(
                df["Time"],
                errors="coerce"
            )

            df = df.set_index("Time")

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    df = df.sort_index()

    if len(df) < 20:

        raise ValueError(
            f"DNSE chỉ trả {len(df)} phiên, "
            "không đủ dữ liệu để tính chỉ báo."
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    df["Return"] = df["Close"].pct_change()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = rsi(
        df["Close"],
        period=14
    )

    # --------------------------------------------------------
    # EMA / MACD
    # --------------------------------------------------------

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
    # VOLATILITY
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
# DNSE RESPONSE PARSER
# ============================================================

def _parse_dnse_response(payload):

    if payload is None:
        return None

    # --------------------------------------------------------
    # Case 1:
    # {
    #   "t": [...],
    #   "o": [...],
    #   "h": [...],
    #   "l": [...],
    #   "c": [...],
    #   "v": [...]
    # }
    # --------------------------------------------------------

    if isinstance(payload, dict):

        keys = {
            str(k).lower(): k
            for k in payload.keys()
        }

        if all(
            x in keys
            for x in ["t", "o", "h", "l", "c"]
        ):

            t = payload[keys["t"]]
            o = payload[keys["o"]]
            h = payload[keys["h"]]
            l = payload[keys["l"]]
            c = payload[keys["c"]]

            v = payload.get(
                keys.get("v"),
                [0] * len(c)
            )

            df = pd.DataFrame(
                {
                    "Timestamp": t,
                    "Open": o,
                    "High": h,
                    "Low": l,
                    "Close": c,
                    "Volume": v,
                }
            )

            return df

        # ----------------------------------------------------
        # Một số response bọc trong data
        # ----------------------------------------------------

        for key in [
            "data",
            "result",
            "results",
            "candles",
            "items",
        ]:

            if key in payload:

                result = _parse_dnse_response(
                    payload[key]
                )

                if result is not None:
                    return result

    # --------------------------------------------------------
    # Case 2:
    # list of candles
    # --------------------------------------------------------

    if isinstance(payload, list):

        if not payload:
            return None

        # list dict
        if isinstance(payload[0], dict):

            rows = []

            for item in payload:

                lower = {
                    str(k).lower(): v
                    for k, v in item.items()
                }

                rows.append(
                    {
                        "Timestamp": (
                            lower.get("t")
                            or lower.get("time")
                            or lower.get("timestamp")
                            or lower.get("date")
                        ),
                        "Open": (
                            lower.get("o")
                            or lower.get("open")
                        ),
                        "High": (
                            lower.get("h")
                            or lower.get("high")
                        ),
                        "Low": (
                            lower.get("l")
                            or lower.get("low")
                        ),
                        "Close": (
                            lower.get("c")
                            or lower.get("close")
                        ),
                        "Volume": (
                            lower.get("v")
                            or lower.get("volume")
                            or 0
                        ),
                    }
                )

            return pd.DataFrame(rows)

        # list arrays
        if isinstance(payload[0], (list, tuple)):

            rows = []

            for item in payload:

                if len(item) >= 5:

                    rows.append(
                        {
                            "Timestamp": item[0],
                            "Open": item[1],
                            "High": item[2],
                            "Low": item[3],
                            "Close": item[4],
                            "Volume": (
                                item[5]
                                if len(item) > 5
                                else 0
                            ),
                        }
                    )

            if rows:
                return pd.DataFrame(rows)

    return None


# ============================================================
# DNSE HISTORY
# ============================================================

def _fetch_dnse_history(
    symbol,
    period="1y",
):

    symbol = normalize_symbol(symbol)

    now = int(time.time())

    period_seconds = {
        "1mo": 35 * 86400,
        "3mo": 100 * 86400,
        "6mo": 190 * 86400,
        "1y": 380 * 86400,
        "2y": 760 * 86400,
        "5y": 1900 * 86400,
    }

    seconds = period_seconds.get(
        period,
        period_seconds["1y"]
    )

    start = now - seconds

    params = {
        "from": start,
        "to": now,
        "symbol": symbol,
        "resolution": "1D",
    }

    last_error = None

    for url in DNSE_OHLC_URLS:

        try:

            response = requests.get(
                url,
                params=params,
                headers=DNSE_HEADERS,
                timeout=20,
            )

            if response.status_code != 200:

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

                continue

            payload = response.json()

            df = _parse_dnse_response(
                payload
            )

            if df is None or df.empty:

                last_error = (
                    "DNSE trả response nhưng "
                    "không tìm thấy OHLC."
                )

                continue

            return df

        except Exception as e:

            last_error = str(e)

    raise ValueError(
        f"Không lấy được dữ liệu DNSE cho {symbol}. "
        f"Lỗi cuối: {last_error}"
    )


# ============================================================
# NORMALIZE HISTORY
# ============================================================

def _normalize_history(df):

    df = df.copy()

    if "Timestamp" in df.columns:

        ts = pd.to_numeric(
            df["Timestamp"],
            errors="coerce"
        )

        # milliseconds
        if ts.dropna().max() > 10_000_000_000:

            df["Timestamp"] = pd.to_datetime(
                ts,
                unit="ms",
                errors="coerce"
            )

        else:

            df["Timestamp"] = pd.to_datetime(
                ts,
                unit="s",
                errors="coerce"
            )

        df = df.set_index(
            "Timestamp"
        )

    elif "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.set_index("Date")

    df.index = pd.to_datetime(
        df.index,
        errors="coerce"
    )

    df = df[
        ~df.index.isna()
    ]

    # Remove timezone
    try:

        if df.index.tz is not None:

            df.index = (
                df.index
                .tz_convert("Asia/Ho_Chi_Minh")
                .tz_localize(None)
            )

    except Exception:
        pass

    return df.sort_index()


# ============================================================
# MAIN MARKET DATA
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_market_data(
    symbol,
    period="1y",
):

    symbol = normalize_symbol(symbol)

    df = _fetch_dnse_history(
        symbol,
        period
    )

    df = _normalize_history(df)

    df = add_indicators(df)

    if df.empty:

        raise ValueError(
            f"Không có dữ liệu sau khi xử lý {symbol}."
        )

    return df


# ============================================================
# LATEST QUOTE
# ============================================================

def get_latest_price(symbol):

    df = load_market_data(
        symbol,
        "1mo"
    )

    if df.empty:
        return None

    last = df.iloc[-1]

    close = float(last["Close"])

    previous = (
        float(df.iloc[-2]["Close"])
        if len(df) >= 2
        else close
    )

    change = close - previous

    change_pct = (
        change / previous * 100
        if previous
        else 0
    )

    volume = float(
        last.get("Volume", 0)
    )

    return {
        "symbol": normalize_symbol(symbol),
        "price": close,
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "date": df.index[-1],
    }


# ============================================================
# VN-INDEX
# ============================================================

def get_vn_index():

    candidates = [
        "VNINDEX",
        "VN-INDEX",
        "VNINDEX.VN",
    ]

    errors = []

    for symbol in candidates:

        try:

            df = _fetch_dnse_history(
                symbol,
                "1mo"
            )

            df = _normalize_history(df)

            if df.empty:
                continue

            df = add_indicators(df)

            last = df.iloc[-1]

            close = float(
                last["Close"]
            )

            previous = (
                float(
                    df.iloc[-2]["Close"]
                )
                if len(df) >= 2
                else close
            )

            change = close - previous

            change_pct = (
                change / previous * 100
                if previous
                else 0
            )

            volume = float(
                last.get("Volume", 0)
            )

            return {
                "symbol": "VN-INDEX",
                "price": close,
                "change": change,
                "change_pct": change_pct,
                "volume": volume,
                "date": df.index[-1],
            }

        except Exception as e:

            errors.append(
                f"{symbol}: {e}"
            )

    raise ValueError(
        "DNSE chưa trả được VN-INDEX. "
        + " | ".join(errors)
    )
def get_vn_index():
    """
    Lấy VN-INDEX từ DNSE market index.
    Không dùng vnstock.
    """

    url = "https://api.dnse.com.vn/market-data/v1/index"

    headers = {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/151.0 Safari/537.36"
        ),
    }

    params = {
        "indexCode": "VNINDEX",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        # DNSE có thể trả object hoặc bọc trong data/result
        if isinstance(data, dict):

            if isinstance(data.get("data"), dict):
                data = data["data"]

            elif isinstance(data.get("result"), dict):
                data = data["result"]

        if not isinstance(data, dict):
            raise ValueError(
                f"Response VN-INDEX không hợp lệ: {data}"
            )

        def pick(*keys, default=0):
            for key in keys:
                if key in data and data[key] is not None:
                    return data[key]
            return default

        price = float(
            pick(
                "indexValue",
                "value",
                "close",
                "last",
                "price",
            )
        )

        change = float(
            pick(
                "change",
                "indexChange",
                "changeValue",
            )
        )

        change_pct = float(
            pick(
                "changePercent",
                "changePct",
                "percentChange",
                "changePercentage",
            )
        )

        volume = float(
            pick(
                "volume",
                "totalVolume",
                "totalVol",
                default=0,
            )
        )

        return {
            "symbol": "VN-INDEX",
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
        }

    except Exception as e:
        raise ValueError(
            f"Không lấy được VN-INDEX từ DNSE: {e}"
        )
