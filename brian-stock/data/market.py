from __future__ import annotations

import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# VNSTOCK API KEY / AUTH
# ============================================================

def _configure_vnstock_api_key() -> bool:
    """Ưu tiên Streamlit Secrets, sau đó tới biến môi trường."""
    key = os.environ.get("VNSTOCK_API_KEY", "").strip()

    if not key:
        try:
            key = str(st.secrets.get("VNSTOCK_API_KEY", "")).strip()
        except Exception:
            key = ""

    if key:
        os.environ["VNSTOCK_API_KEY"] = key
        return True

    return False


VNSTOCK_API_KEY_AVAILABLE = _configure_vnstock_api_key()


# ============================================================
# VNSTOCK IMPORT
# ============================================================

try:
    from vnstock import Market
    VNSTOCK_AVAILABLE = True
except Exception:
    try:
        from vnstock.ui import Market
        VNSTOCK_AVAILABLE = True
    except Exception:
        Market = None
        VNSTOCK_AVAILABLE = False


try:
    from vnstock import Listing
    LISTING_AVAILABLE = True
except Exception:
    Listing = None
    LISTING_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

CACHE_TTL_PRICE = 300
CACHE_TTL_INDEX = 300
CACHE_TTL_LATEST = 30
CACHE_TTL_RESEARCH = 900
CACHE_TTL_LISTING = 6 * 60 * 60

# Guest: giữ dưới 20 request/phút.
# API key: giữ dưới 60 request/phút.
REQUEST_WINDOW_SECONDS = 60.0
GUEST_MAX_REQUESTS_PER_MINUTE = 18
API_KEY_MAX_REQUESTS_PER_MINUTE = 55
GUEST_MIN_REQUEST_GAP = 3.5
API_KEY_MIN_REQUEST_GAP = 1.25
_REQUEST_TIMESTAMPS: list[float] = []
_LAST_REQUEST_TIME: float | None = None

# 120 ngày lịch thường < 95 phiên giao dịch.
# Nếu response chạm giới hạn, mới chia nhỏ tiếp.
RESEARCH_CHUNK_DAYS = 120
MAX_ROWS_PER_OHLCV_REQUEST = 95
MIN_CHUNK_DAYS = 15
MAX_API_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3.0


# ============================================================
# ERROR / RATE LIMIT HELPERS
# ============================================================

def _extract_rate_limit_wait_seconds(error: Any) -> int | None:
    text = str(error or "")
    patterns = [
        r"Chờ\s*(\d+)\s*giây",
        r"Wait[^0-9]{0,30}(\d+)\s*(?:s|sec|secs|seconds|giây)",
        r"retry[^0-9]{0,30}(\d+)\s*(?:s|sec|secs|seconds|giây)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return max(1, int(match.group(1)))
            except Exception:
                pass
    return None


def _is_rate_limit_error(error: Any) -> bool:
    text = str(error or "").lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "rate_limit",
            "too many requests",
            "maximum api request",
            "requests/minute",
            "requests/phút",
        )
    )


def _reset_request_window() -> None:
    global _REQUEST_TIMESTAMPS
    _REQUEST_TIMESTAMPS = []


def _sleep_after_rate_limit(error: Any) -> None:
    """Chờ đủ một cửa sổ mới trước khi thử lại cùng request."""
    suggested = _extract_rate_limit_wait_seconds(error)
    wait_seconds = max(61, (suggested or 60) + 2)

    print(
        f"[VNSTOCK] Rate limit: chờ {wait_seconds} giây "
        "trước khi thử lại đúng request."
    )
    time.sleep(wait_seconds)
    _reset_request_window()


def _throttle_request() -> None:
    """Điều tiết request tập trung tại một chỗ, không sleep trùng."""
    global _LAST_REQUEST_TIME, _REQUEST_TIMESTAMPS

    now = time.monotonic()
    max_requests = (
        API_KEY_MAX_REQUESTS_PER_MINUTE
        if VNSTOCK_API_KEY_AVAILABLE
        else GUEST_MAX_REQUESTS_PER_MINUTE
    )
    min_gap = (
        API_KEY_MIN_REQUEST_GAP
        if VNSTOCK_API_KEY_AVAILABLE
        else GUEST_MIN_REQUEST_GAP
    )

    _REQUEST_TIMESTAMPS = [
        t for t in _REQUEST_TIMESTAMPS
        if now - t < REQUEST_WINDOW_SECONDS
    ]

    if _LAST_REQUEST_TIME is not None:
        gap = now - _LAST_REQUEST_TIME
        if gap < min_gap:
            time.sleep(min_gap - gap)
            now = time.monotonic()

    _REQUEST_TIMESTAMPS = [
        t for t in _REQUEST_TIMESTAMPS
        if now - t < REQUEST_WINDOW_SECONDS
    ]

    if len(_REQUEST_TIMESTAMPS) >= max_requests:
        oldest = min(_REQUEST_TIMESTAMPS)
        wait_seconds = max(
            1.0,
            REQUEST_WINDOW_SECONDS - (now - oldest) + 1.0,
        )
        print(
            f"[VNSTOCK] Chủ động giữ dưới giới hạn: "
            f"{len(_REQUEST_TIMESTAMPS)}/{max_requests}. "
            f"Chờ {wait_seconds:.0f} giây."
        )
        time.sleep(wait_seconds)
        now = time.monotonic()
        _REQUEST_TIMESTAMPS = [
            t for t in _REQUEST_TIMESTAMPS
            if now - t < REQUEST_WINDOW_SECONDS
        ]

    _REQUEST_TIMESTAMPS.append(now)
    _LAST_REQUEST_TIME = now


# ============================================================
# CACHE RESOURCE
# ============================================================

@st.cache_resource(show_spinner=False)
def _create_market():
    global VNSTOCK_API_KEY_AVAILABLE
    VNSTOCK_API_KEY_AVAILABLE = _configure_vnstock_api_key()

    if not VNSTOCK_AVAILABLE or Market is None:
        raise RuntimeError("Không tìm thấy vnstock Market.")

    if not VNSTOCK_API_KEY_AVAILABLE:
        print(
            "[VNSTOCK] Không tìm thấy VNSTOCK_API_KEY. "
            "Ứng dụng đang chạy với hạn mức Guest."
        )

    try:
        return Market()
    except Exception as error:
        raise RuntimeError(
            f"Không khởi tạo được Market(): {error}"
        ) from error


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_symbol(symbol: Any) -> str:
    value = str(symbol or "HPG").strip().upper()
    value = re.sub(r"\s+", "", value)
    if value.endswith(".VN"):
        value = value[:-3]
    return value or "HPG"


def display_symbol(symbol: Any) -> str:
    return normalize_symbol(symbol)


def _to_number(value: Any, default=np.nan) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _normalize_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[-/]+", "_", text)
    return text


def _find_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    mapping = {
        _normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in mapping:
            return mapping[key]

    for column in df.columns:
        column_key = _normalize_column_name(column)
        for candidate in candidates:
            candidate_key = _normalize_column_name(candidate)
            if candidate_key in column_key or column_key in candidate_key:
                return column

    return None


# ============================================================
# DATETIME / OHLCV NORMALIZATION
# ============================================================

def _normalize_datetime_index(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    time_column = _find_column(
        work,
        ["time", "date", "datetime", "timestamp", "trading_date", "tradingdate"],
    )

    if time_column is not None:
        work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
        work = work.set_index(time_column)
    else:
        work.index = pd.to_datetime(work.index, errors="coerce")

    work = work[~work.index.isna()].copy()
    work = work.sort_index()
    work = work[~work.index.duplicated(keep="last")].copy()
    return work


def _normalize_ohlcv(df: Any, stock: bool = True) -> pd.DataFrame:
    if df is None:
        raise ValueError("Nguồn dữ liệu trả về None.")

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty:
        raise ValueError("Nguồn dữ liệu trả về DataFrame rỗng.")

    work = df.copy()

    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [
            str(column[-1]) if isinstance(column, tuple) else str(column)
            for column in work.columns
        ]

    rename_map: dict[Any, str] = {}
    for column in work.columns:
        key = _normalize_column_name(column)
        if key in {"time", "date", "datetime", "timestamp", "trading_date", "tradingdate"}:
            rename_map[column] = "Time"
        elif key in {"open", "open_price", "openprice"}:
            rename_map[column] = "Open"
        elif key in {"high", "high_price", "highprice"}:
            rename_map[column] = "High"
        elif key in {"low", "low_price", "lowprice"}:
            rename_map[column] = "Low"
        elif key in {"close", "close_price", "closeprice", "last", "last_price", "lastprice"}:
            rename_map[column] = "Close"
        elif key in {"volume", "vol", "total_volume", "totalvolume", "matched_volume", "match_volume", "matchvolume"}:
            rename_map[column] = "Volume"
        elif key in {"value", "trading_value", "trade_value", "value_traded", "match_value", "matched_value", "turnover", "total_value"}:
            rename_map[column] = "Value"

    work = work.rename(columns=rename_map)

    if "Time" in work.columns:
        work["Time"] = pd.to_datetime(work["Time"], errors="coerce")
        work = work.set_index("Time")
    else:
        work.index = pd.to_datetime(work.index, errors="coerce")

    work = work[~work.index.isna()].copy()
    work = work.sort_index()
    work = work[~work.index.duplicated(keep="last")].copy()

    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise ValueError("Thiếu cột OHLC: " + ", ".join(missing))

    if "Volume" not in work.columns:
        work["Volume"] = 0.0

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if "Value" in work.columns:
        work["Value"] = pd.to_numeric(work["Value"], errors="coerce")

    # Một số nguồn trả giá cổ phiếu theo nghìn đồng.
    if stock:
        median_price = work["Close"].dropna().median()
        if pd.notna(median_price) and 0 < median_price < 1000:
            for column in ["Open", "High", "Low", "Close"]:
                work[column] *= 1000

    for column in ["Open", "High", "Low", "Close"]:
        work.loc[work[column] <= 0, column] = np.nan
    work.loc[work["Volume"] < 0, "Volume"] = np.nan

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["Open", "High", "Low", "Close"])
    if work.empty:
        raise ValueError("Không còn OHLC hợp lệ.")

    return work


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    prices = pd.to_numeric(prices, errors="coerce")
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative))
    result = result.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    result = result.where(~((avg_loss == 0) & (avg_gain == 0)), 50)
    return result


def add_indicators(df: pd.DataFrame, la_co_phieu: bool = False) -> pd.DataFrame:
    work = _normalize_ohlcv(df, stock=la_co_phieu).copy()
    close = work["Close"]
    high = work["High"]
    low = work["Low"]
    volume = work["Volume"]

    work["Return"] = close.pct_change()
    work["ReturnPct"] = work["Return"] * 100
    work["LogReturn"] = np.log(close / close.shift(1))
    work["RSI"] = rsi(close, 14)

    for period in [9, 12, 20, 26, 50, 100, 200]:
        work[f"EMA{period}"] = close.ewm(span=period, adjust=False).mean()

    for period in [5, 10, 20, 50, 100, 200]:
        work[f"SMA{period}"] = close.rolling(period).mean()

    for period in [20, 50, 100, 200]:
        work[f"Price_vs_SMA{period}"] = (close / work[f"SMA{period}"] - 1) * 100
        work[f"Price_vs_EMA{period}"] = (close / work[f"EMA{period}"] - 1) * 100

    work["MACD"] = work["EMA12"] - work["EMA26"]
    work["MACD_Signal"] = work["MACD"].ewm(span=9, adjust=False).mean()
    work["MACD_Hist"] = work["MACD"] - work["MACD_Signal"]

    rolling_std = close.rolling(20).std()
    work["Bollinger_Mid"] = work["SMA20"]
    work["Bollinger_Upper"] = work["SMA20"] + 2 * rolling_std
    work["Bollinger_Lower"] = work["SMA20"] - 2 * rolling_std
    work["Bollinger_Width"] = (work["Bollinger_Upper"] - work["Bollinger_Lower"]) / work["Bollinger_Mid"] * 100
    work["Bollinger_Position"] = (close - work["Bollinger_Lower"]) / (work["Bollinger_Upper"] - work["Bollinger_Lower"])

    for period in [5, 20, 60]:
        work[f"Volatility{period}"] = work["Return"].rolling(period).std() * np.sqrt(252) * 100
    work["Volatility_20D"] = work["Volatility20"] / 100

    for period in [5, 20, 50]:
        work[f"Volume_SMA{period}"] = volume.rolling(period).mean()
    work["Volume_Change"] = volume.pct_change()
    work["Relative_Volume"] = volume / work["Volume_SMA20"]

    if "Value" not in work.columns:
        work["Value"] = close * volume
    else:
        work["Value"] = pd.to_numeric(work["Value"], errors="coerce")
    work["Trading_Value"] = work["Value"]
    work["Trading_Value_Change"] = work["Trading_Value"].pct_change()
    work["Trading_Value_SMA20"] = work["Trading_Value"].rolling(20).mean()

    work["Range"] = high - low
    work["Range_Percent"] = work["Range"] / close * 100
    work["Body"] = close - work["Open"]
    work["Body_Percent"] = work["Body"] / work["Open"] * 100

    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    work["TrueRange"] = true_range
    work["ATR14"] = true_range.rolling(14).mean()
    work["ATR_Percent"] = work["ATR14"] / close * 100

    for period in [1, 5, 10, 20, 60]:
        work[f"Momentum{period}"] = close / close.shift(period) - 1
        work[f"Momentum{period}Pct"] = work[f"Momentum{period}"] * 100

    for period in [20, 50, 252]:
        work[f"High{period}"] = high.rolling(period).max()
        work[f"Low{period}"] = low.rolling(period).min()
        work[f"Distance_From_High{period}"] = (close / work[f"High{period}"] - 1) * 100
        work[f"Distance_From_Low{period}"] = (close / work[f"Low{period}"] - 1) * 100

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    spread = (high14 - low14).replace(0, np.nan)
    work["Stochastic_K"] = (close - low14) / spread * 100
    work["Stochastic_D"] = work["Stochastic_K"].rolling(3).mean()

    for period in [5, 10, 20]:
        work[f"ROC{period}"] = (close / close.shift(period) - 1) * 100

    work["OBV_Proxy"] = (np.sign(close.diff()) * volume).cumsum()
    work["Dollar_Volume"] = close * volume

    return work.replace([np.inf, -np.inf], np.nan)


# ============================================================
# PERIOD HELPERS
# ============================================================

def _period_days(period: str) -> int:
    mapping = {
        "1d": 5,
        "5d": 12,
        "1w": 14,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 365,
        "2y": 730,
        "3y": 1095,
        "5y": 1825,
        "10y": 3652,
        "max": 5000,
    }
    return mapping.get(str(period or "1y").strip().lower(), 365)


# ============================================================
# GENERIC VNSTOCK CALL
# ============================================================

def _call_vnstock(callable_obj, *, label: str):
    """Một điểm duy nhất để throttle + retry; không sleep trùng."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        _throttle_request()
        try:
            return callable_obj()
        except Exception as error:
            last_error = error
            print(f"[VNSTOCK] {label}: lỗi lần {attempt}: {error}")

            if _is_rate_limit_error(error):
                if attempt < MAX_API_RETRIES:
                    _sleep_after_rate_limit(error)
                    continue
                raise

            if attempt < MAX_API_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise

    if last_error is not None:
        raise last_error
    return pd.DataFrame()


# ============================================================
# OHLCV REQUESTS
# ============================================================

def _request_equity_ohlcv(market, symbol, start_date, end_date):
    equity = market.equity(normalize_symbol(symbol))
    return _call_vnstock(
        lambda: equity.ohlcv(
            start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1D",
        ),
        label=f"OHLCV {normalize_symbol(symbol)} {pd.Timestamp(start_date).date()}→{pd.Timestamp(end_date).date()}",
    )


def _request_index_ohlcv(market, index_symbol, start_date, end_date):
    index_obj = market.index(index_symbol)
    return _call_vnstock(
        lambda: index_obj.ohlcv(
            start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1D",
        ),
        label=f"INDEX {index_symbol} {pd.Timestamp(start_date).date()}→{pd.Timestamp(end_date).date()}",
    )


def _load_equity_range_complete(market, symbol, start_date, end_date) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start > end:
        return pd.DataFrame()

    raw = _request_equity_ohlcv(market, symbol, start, end)
    try:
        normalized = _normalize_ohlcv(raw, stock=True)
    except Exception as error:
        print(f"[VNSTOCK] {symbol}: không chuẩn hóa được chunk {start.date()}→{end.date()}: {error}")
        return pd.DataFrame()

    if normalized.empty:
        return pd.DataFrame()

    span_days = (end - start).days + 1

    # Nếu ít hơn ngưỡng, coi như đủ trong chunk.
    if len(normalized) < MAX_ROWS_PER_OHLCV_REQUEST:
        return normalized.loc[start:end]

    if span_days <= MIN_CHUNK_DAYS:
        print(
            f"[VNSTOCK] {symbol}: chunk nhỏ nhưng trả {len(normalized)} dòng; "
            "giữ nguyên dữ liệu nhận được."
        )
        return normalized.loc[start:end]

    midpoint = start + pd.Timedelta(days=(span_days // 2) - 1)
    if midpoint < start or midpoint >= end:
        return normalized.loc[start:end]

    print(
        f"[VNSTOCK] {symbol}: response {len(normalized)} dòng, "
        f"chia {start.date()}→{end.date()} thành 2 phần."
    )

    left = _load_equity_range_complete(market, symbol, start, midpoint)
    right = _load_equity_range_complete(
        market,
        symbol,
        midpoint + pd.Timedelta(days=1),
        end,
    )

    pieces = [part for part in (left, right) if not part.empty]
    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, axis=0).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result.loc[start:end]


def _load_index_range_complete(market, index_symbol, start_date, end_date) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start > end:
        return pd.DataFrame()

    try:
        raw = _request_index_ohlcv(market, index_symbol, start, end)
    except Exception as error:
        print(
            f"[VNSTOCK] {index_symbol}: lỗi {start.date()}→{end.date()}: {error}"
        )
        return pd.DataFrame()

    try:
        normalized = _normalize_ohlcv(raw, stock=False)
    except Exception as error:
        print(f"[VNSTOCK] {index_symbol}: không chuẩn hóa được: {error}")
        return pd.DataFrame()

    if normalized.empty:
        return pd.DataFrame()

    span_days = (end - start).days + 1
    if len(normalized) < MAX_ROWS_PER_OHLCV_REQUEST:
        return normalized.loc[start:end]

    if span_days <= MIN_CHUNK_DAYS:
        return normalized.loc[start:end]

    midpoint = start + pd.Timedelta(days=(span_days // 2) - 1)
    if midpoint < start or midpoint >= end:
        return normalized.loc[start:end]

    left = _load_index_range_complete(market, index_symbol, start, midpoint)
    right = _load_index_range_complete(
        market,
        index_symbol,
        midpoint + pd.Timedelta(days=1),
        end,
    )

    pieces = [part for part in (left, right) if not part.empty]
    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, axis=0).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result.loc[start:end]


# ============================================================
# STOCK LOAD
# ============================================================

def _load_stock_raw(symbol, start_date, end_date, progress_callback=None) -> pd.DataFrame:
    market = _create_market()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    symbol_norm = normalize_symbol(symbol)

    if start > end:
        raise ValueError("Khoảng ngày không hợp lệ.")

    if progress_callback:
        progress_callback(
            f"Đang tải {symbol_norm}: {start.date()} → {end.date()}..."
        )

    # FAST PATH: đúng khoảng người dùng yêu cầu, một request duy nhất.
    try:
        direct = _request_equity_ohlcv(market, symbol_norm, start, end)
        direct = _normalize_ohlcv(direct, stock=True)
    except Exception as error:
        print(f"[VNSTOCK] FAST {symbol_norm} lỗi: {error}")
        direct = pd.DataFrame()

    if not direct.empty:
        first_date = direct.index.min().normalize()
        last_date = direct.index.max().normalize()
        covers_start = first_date <= start + pd.Timedelta(days=5)
        covers_end = last_date >= end - pd.Timedelta(days=5)

        # Không còn lỗi cũ: tuyệt đối KHÔNG coi "99 dòng" là complete nếu
        # response thực tế chỉ phủ vài tháng cuối của một khoảng 3 năm.
        if covers_start and covers_end:
            result = direct.loc[start:end].copy()
            result = result[~result.index.duplicated(keep="last")]
            print(
                f"[VNSTOCK] FAST {symbol_norm}: {len(result)} phiên | "
                f"{result.index.min().date()} → {result.index.max().date()} | "
                f"yêu cầu {start.date()} → {end.date()}"
            )
            return result

        print(
            f"[VNSTOCK] FAST {symbol_norm}: {len(direct)} phiên | "
            f"{first_date.date()} → {last_date.date()} | "
            "response không phủ đủ khoảng, chuyển sang chunk."
        )

    # FALLBACK: 120 ngày lịch/chunk.
    chunks: list[pd.DataFrame] = []
    cursor = start
    chunk_number = 0
    estimated_chunks = int(np.ceil(((end - start).days + 1) / RESEARCH_CHUNK_DAYS))

    while cursor <= end:
        chunk_number += 1
        chunk_end = min(
            cursor + pd.Timedelta(days=RESEARCH_CHUNK_DAYS - 1),
            end,
        )

        if progress_callback:
            progress_callback(
                f"Đang tải {symbol_norm}: đoạn {chunk_number}/{estimated_chunks} "
                f"({cursor.date()} → {chunk_end.date()})..."
            )

        try:
            chunk = _load_equity_range_complete(
                market,
                symbol_norm,
                cursor,
                chunk_end,
            )
        except Exception as error:
            print(
                f"[VNSTOCK] {symbol_norm}: chunk {cursor.date()}→{chunk_end.date()} lỗi: {error}"
            )
            chunk = pd.DataFrame()

        if not chunk.empty:
            chunks.append(chunk)

        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise ValueError(f"Không có dữ liệu giá {symbol_norm}.")

    result = pd.concat(chunks, axis=0).sort_index()
    result = _normalize_ohlcv(result, stock=True)
    result = result.loc[start:end]
    result = result[~result.index.duplicated(keep="last")]

    if result.empty:
        raise ValueError(
            f"Không có dữ liệu hợp lệ cho {symbol_norm} "
            f"{start.date()} → {end.date()}."
        )

    print(
        f"[VNSTOCK] CHUNKED {symbol_norm}: {len(result)} phiên | "
        f"{result.index.min().date()} → {result.index.max().date()} | "
        f"yêu cầu {start.date()} → {end.date()}"
    )
    return result


# ============================================================
# PUBLIC STOCK DATA
# ============================================================

@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def load_market_data(symbol, period="1y"):
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=_period_days(period))
    raw = _load_stock_raw(symbol, start, end)
    data = add_indicators(raw, la_co_phieu=True)
    data.attrs["symbol"] = normalize_symbol(symbol)
    data.attrs["source"] = "Vnstock"
    return data


@st.cache_data(ttl=CACHE_TTL_INDEX, show_spinner=False)
def load_vnindex_data():
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=450)
    market = _create_market()
    df = _load_index_range_complete(market, "VNINDEX", start, end)
    if df.empty:
        raise ValueError("Không lấy được dữ liệu VNINDEX.")
    data = add_indicators(df, la_co_phieu=False)
    data.attrs["symbol"] = "VNINDEX"
    data.attrs["source"] = "Vnstock"
    return data


@st.cache_data(ttl=CACHE_TTL_LATEST, show_spinner=False)
def load_latest_price(symbol):
    data = load_market_data(symbol, "5d")
    if data.empty:
        raise ValueError("Không có dữ liệu mới nhất.")

    last = data.iloc[-1]
    price = _to_number(last.get("Close"))
    previous = _to_number(data["Close"].iloc[-2]) if len(data) >= 2 else price
    change = (price / previous - 1) * 100 if previous not in (None, 0) else np.nan

    return {
        "ma": display_symbol(symbol),
        "gia": price,
        "thay_doi": change,
        "khoi_luong": _to_number(last.get("Volume"), 0.0),
        "thoi_gian": data.index[-1],
    }


# ============================================================
# SNAPSHOT
# ============================================================

def market_snapshot(data):
    if data is None or data.empty:
        return {}

    last = data.iloc[-1]
    previous = data.iloc[-2] if len(data) >= 2 else last
    price = _to_number(last.get("Close"))
    previous_price = _to_number(previous.get("Close"))
    change = (
        (price / previous_price - 1) * 100
        if previous_price not in (None, 0)
        else np.nan
    )

    def get_column(name):
        return _to_number(last.get(name))

    return {
        "price": price,
        "change_1d": change,
        "return_1d": get_column("ReturnPct"),
        "rsi": get_column("RSI"),
        "macd": get_column("MACD"),
        "sma20": get_column("SMA20"),
        "sma50": get_column("SMA50"),
        "volatility20": get_column("Volatility20"),
        "volume": get_column("Volume"),
        "trading_value": get_column("Trading_Value"),
        "atr14": get_column("ATR14"),
        "relative_volume": get_column("Relative_Volume"),
    }


# ============================================================
# LEGACY FLOW COMPATIBILITY — KHÔNG GỌI API
# ============================================================

def load_stock_flow_history(symbol, start_date, end_date):
    """
    Giữ hàm để các module cũ không lỗi import.
    KHÔNG gọi trade_history / foreign_flow / proprietary_flow.
    """
    print(
        f"[RESEARCH] Bỏ qua flow API cho {normalize_symbol(symbol)}: "
        "Guest không dùng trade_history/foreign_flow/proprietary_flow."
    )
    return pd.DataFrame()


# ============================================================
# LISTING / METADATA
# ============================================================

@st.cache_data(ttl=CACHE_TTL_LISTING, show_spinner=False)
def _load_listing():
    if not LISTING_AVAILABLE or Listing is None:
        return pd.DataFrame()

    try:
        listing = Listing(source="VCI")
        df = listing.all_symbols(to_df=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as error:
        print(f"[VNSTOCK] Listing lỗi: {error}")
    return pd.DataFrame()


def _get_stock_metadata(symbol):
    listing = _load_listing()
    target = normalize_symbol(symbol)
    empty = {
        "symbol": target,
        "sector": "",
        "icb_code": "",
        "icb_code4": "",
    }
    if listing.empty:
        return empty

    symbol_column = _find_column(listing, ["symbol", "ticker", "code"])
    if symbol_column is None:
        return empty

    values = (
        listing[symbol_column]
        .astype(str)
        .str.upper()
        .str.replace(".VN", "", regex=False)
    )
    rows = listing[values == target]
    if rows.empty:
        return empty

    row = rows.iloc[0]
    sector_column = _find_column(listing, ["icb_name", "industry_name", "industry", "sector"])
    icb_column = _find_column(listing, ["icb_code", "industry_code"])
    icb4_column = _find_column(listing, ["icb_code4", "industry_code4"])

    def text_value(column):
        value = row.get(column) if column is not None else ""
        return str(value).strip() if pd.notna(value) else ""

    return {
        "symbol": target,
        "sector": text_value(sector_column),
        "icb_code": text_value(icb_column),
        "icb_code4": text_value(icb4_column),
    }


def _get_sector_peers(symbol, max_peers=5):
    listing = _load_listing()
    if listing.empty:
        return []

    symbol_column = _find_column(listing, ["symbol", "ticker", "code"])
    industry_column = _find_column(listing, ["icb_code4", "icb_code", "industry_code"])
    if symbol_column is None or industry_column is None:
        return []

    target = normalize_symbol(symbol)
    values = (
        listing[symbol_column]
        .astype(str)
        .str.upper()
        .str.replace(".VN", "", regex=False)
    )
    target_rows = listing[values == target]
    if target_rows.empty:
        return []

    target_code = str(target_rows.iloc[0][industry_column]).strip()
    if not target_code or target_code.lower() == "nan":
        return []

    same = listing[
        listing[industry_column].astype(str).str.strip().eq(target_code)
    ][symbol_column]

    peers = (
        same.astype(str)
        .str.upper()
        .str.replace(".VN", "", regex=False)
        .tolist()
    )
    return [peer for peer in peers if peer != target][:max_peers]


# ============================================================
# MARKET FACTORS
# ============================================================

@st.cache_data(ttl=CACHE_TTL_RESEARCH, show_spinner=False)
def load_market_factor_history(start_date, end_date):
    market = _create_market()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    indices = {
        "VNINDEX": "market_vnindex",
        "VN30": "market_vn30",
        "HNXINDEX": "market_hnx",
    }

    result = None

    for index_symbol, prefix in indices.items():
        df = _load_index_range_complete(market, index_symbol, start, end)
        if df.empty:
            print(
                f"[RESEARCH] {index_symbol}: không có dữ liệu, bỏ qua."
            )
            continue

        print(
            f"[RESEARCH] {index_symbol}: {len(df)} phiên | "
            f"{df.index.min().date()} → {df.index.max().date()}"
        )

        close = df["Close"]
        temp = pd.DataFrame(index=df.index)
        temp[f"{prefix}_close"] = close
        temp[f"{prefix}_return"] = close.pct_change()
        temp[f"{prefix}_return_pct"] = temp[f"{prefix}_return"] * 100
        temp[f"{prefix}_momentum20"] = close / close.shift(20) - 1
        temp[f"{prefix}_volatility20"] = temp[f"{prefix}_return"].rolling(20).std() * np.sqrt(252)
        temp[f"{prefix}_volume"] = df["Volume"]
        temp[f"{prefix}_value"] = df["Value"] if "Value" in df.columns else df["Close"] * df["Volume"]

        result = temp if result is None else result.join(temp, how="outer")

    if result is None:
        return pd.DataFrame()

    return result.sort_index().loc[start:end].replace([np.inf, -np.inf], np.nan)


# ============================================================
# SECTOR FACTORS
# ============================================================

@st.cache_data(ttl=CACHE_TTL_RESEARCH, show_spinner=False)
def load_sector_factor_history(symbol, start_date, end_date):
    peers = _get_sector_peers(symbol, max_peers=5)
    if not peers:
        print(f"[RESEARCH] {normalize_symbol(symbol)}: không tìm được peer cùng ngành.")
        return pd.DataFrame()

    market = _create_market()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    peer_returns = []
    successful_peers = 0

    for peer in peers:
        try:
            df = _load_equity_range_complete(market, peer, start, end)
        except Exception as error:
            print(f"[RESEARCH] Sector peer {peer}: lỗi {error}")
            df = pd.DataFrame()

        if df.empty:
            print(f"[RESEARCH] Sector peer {peer}: không có dữ liệu, bỏ qua.")
            continue

        print(
            f"[RESEARCH] Sector peer {peer}: {len(df)} phiên | "
            f"{df.index.min().date()} → {df.index.max().date()}"
        )
        peer_returns.append(df["Close"].pct_change().rename(peer))
        successful_peers += 1

    if not peer_returns:
        return pd.DataFrame()

    peers_df = pd.concat(peer_returns, axis=1).sort_index()
    result = pd.DataFrame(index=peers_df.index)
    result["sector_return"] = peers_df.mean(axis=1, skipna=True)
    result["sector_return_pct"] = result["sector_return"] * 100
    result["sector_momentum20"] = (1 + result["sector_return"]).rolling(20).apply(np.prod, raw=True) - 1
    result["sector_momentum60"] = (1 + result["sector_return"]).rolling(60).apply(np.prod, raw=True) - 1
    result["sector_volatility20"] = result["sector_return"].rolling(20).std() * np.sqrt(252)
    denominator = peers_df.notna().sum(axis=1).replace(0, np.nan)
    result["sector_positive_ratio"] = peers_df.gt(0).sum(axis=1) / denominator
    result["sector_negative_ratio"] = peers_df.lt(0).sum(axis=1) / denominator
    result["sector_peer_count"] = peers_df.notna().sum(axis=1)

    result.attrs["peers_requested"] = peers
    result.attrs["peers_loaded"] = successful_peers
    return result.replace([np.inf, -np.inf], np.nan)


# ============================================================
# RESEARCH PROGRESS
# ============================================================

def _research_progress_ui():
    try:
        progress = st.progress(0, text="1/4 Đang chuẩn bị nghiên cứu...")
        status = st.empty()
        return progress, status
    except Exception:
        return None, None


def _update_research_progress(progress, status, step, message):
    text_value = f"{step}/4 {message}"
    print(f"[RESEARCH] {text_value}")

    if progress is not None:
        try:
            progress.progress(int(step / 4 * 100), text=text_value)
        except Exception:
            pass
    if status is not None:
        try:
            status.info(text_value)
        except Exception:
            pass


# ============================================================
# COMPLETENESS — KHÔNG CÒN FLOW BẮT BUỘC
# ============================================================

def _date_index(index) -> pd.DatetimeIndex:
    values = pd.to_datetime(index, errors="coerce")
    try:
        if getattr(values, "tz", None) is not None:
            values = values.tz_localize(None)
    except Exception:
        pass
    values = pd.DatetimeIndex(values).normalize()
    values = values[~values.isna()]
    return values[~values.duplicated()].sort_values()


def _coverage_info(df, columns=None) -> pd.DatetimeIndex:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DatetimeIndex([])
    if columns is None:
        return _date_index(df.index)

    valid_columns = [column for column in columns if column in df.columns]
    if not valid_columns:
        return pd.DatetimeIndex([])

    mask = df[valid_columns].notna().any(axis=1)
    return _date_index(df.index[mask])


def _missing_dates(expected_dates, actual_dates) -> pd.DatetimeIndex:
    return _date_index(expected_dates).difference(_date_index(actual_dates))


def _format_missing_dates(dates, limit=12) -> str:
    if len(dates) == 0:
        return ""
    sample = [str(pd.Timestamp(value).date()) for value in dates[:limit]]
    suffix = f" ... +{len(dates) - limit} ngày" if len(dates) > limit else ""
    return ", ".join(sample) + suffix


def _validate_research_completeness(symbol, start, end, stock, market, sector):
    if stock is None or stock.empty:
        raise ValueError(f"{symbol}: không có dữ liệu cổ phiếu.")

    report: dict[str, Any] = {
        "requested_start": str(start.date()),
        "requested_end": str(end.date()),
        "stock_days": len(stock),
        "market": {},
        "sector": {},
    }

    # Chỉ dùng VNINDEX làm lịch chuẩn nếu có. Không bắt VN30/HNX/sector phải đủ.
    expected = pd.DatetimeIndex([])
    if market is not None and not market.empty and "market_vnindex_close" in market.columns:
        expected = _coverage_info(market, ["market_vnindex_close"])
        expected = expected[(expected >= start) & (expected <= end)]

    if not expected.empty:
        stock_dates = _coverage_info(stock)
        stock_first = stock_dates.min()
        effective = expected[expected >= stock_first]
        missing_stock = _missing_dates(effective, stock_dates)
        report["expected_trading_days"] = len(expected)
        report["stock_first"] = str(stock_first.date())
        report["stock_last"] = str(stock_dates.max().date())
        report["stock_missing_days"] = len(missing_stock)

        if len(missing_stock) > 0:
            print(
                f"[RESEARCH] Cảnh báo {symbol}: thiếu {len(missing_stock)} phiên theo lịch VNINDEX: "
                f"{_format_missing_dates(missing_stock)}"
            )
    else:
        report["expected_trading_days"] = len(stock)
        print("[RESEARCH] Không có VNINDEX để làm lịch chuẩn; vẫn cho phép chạy dataset cổ phiếu.")

    for label, column in {
        "VNINDEX": "market_vnindex_close",
        "VN30": "market_vn30_close",
        "HNXINDEX": "market_hnx_close",
    }.items():
        if market is not None and column in market.columns:
            actual = _coverage_info(market, [column])
            missing = _missing_dates(expected, actual) if not expected.empty else pd.DatetimeIndex([])
            report["market"][label] = {
                "actual": len(actual),
                "missing": len(missing),
            }
            if len(missing) > 0:
                print(
                    f"[RESEARCH] Cảnh báo {label}: thiếu {len(missing)} ngày, bỏ qua phần ngày thiếu."
                )

    if sector is not None and not sector.empty and "sector_peer_count" in sector.columns:
        report["sector"] = {
            "peers_requested": sector.attrs.get("peers_requested", []),
            "peers_loaded": sector.attrs.get("peers_loaded", 0),
            "days": len(sector),
        }
    else:
        print("[RESEARCH] Không có dữ liệu ngành/peer; tiếp tục không có sector factor.")

    return report


# ============================================================
# MULTIFACTOR RESEARCH DATASET
# ============================================================

def load_multifactor_research_history(symbol, start_date, end_date):
    """
    Pipeline nghiên cứu mới:
      1) OHLCV cổ phiếu + technical factors
      2) VNINDEX/VN30/HNX (nếu có)
      3) Peer ngành (nếu lấy được)
      4) Merge + target + kiểm tra

    QUAN TRỌNG:
    - Không gọi trade_history.
    - Không gọi foreign_flow.
    - Không gọi proprietary_flow.
    - Không dùng flow để chặn dataset.
    """
    symbol = normalize_symbol(symbol)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start > end:
        raise ValueError("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")

    progress, status = _research_progress_ui()

    # --------------------------------------------------------
    # 1. STOCK
    # --------------------------------------------------------
    _update_research_progress(
        progress,
        status,
        1,
        f"Đang tải giá cổ phiếu {symbol} ({start.date()} → {end.date()})... "
        f"({'API key' if VNSTOCK_API_KEY_AVAILABLE else 'Guest'})",
    )

    stock_raw = _load_stock_raw(
        symbol,
        start,
        end,
        progress_callback=lambda message: _update_research_progress(
            progress,
            status,
            1,
            message,
        ),
    )
    stock = add_indicators(stock_raw, la_co_phieu=True)
    print(f"[RESEARCH] Giá cổ phiếu hoàn tất: {len(stock)} phiên.")

    # --------------------------------------------------------
    # 2. MARKET
    # --------------------------------------------------------
    _update_research_progress(
        progress,
        status,
        2,
        "Đang tải VN-Index, VN30 và HNX...",
    )
    market = load_market_factor_history(start, end)

    # --------------------------------------------------------
    # 3. SECTOR
    # --------------------------------------------------------
    _update_research_progress(
        progress,
        status,
        3,
        "Đang tải dữ liệu nhóm ngành và cổ phiếu cùng ngành...",
    )
    sector = load_sector_factor_history(symbol, start, end)

    # --------------------------------------------------------
    # 4. MERGE / TARGET
    # --------------------------------------------------------
    _update_research_progress(
        progress,
        status,
        4,
        "Ghép dataset, tạo biến mục tiêu và kiểm tra dữ liệu...",
    )

    metadata = _get_stock_metadata(symbol)
    completeness = _validate_research_completeness(
        symbol,
        start,
        end,
        stock,
        market,
        sector,
    )

    result = stock.copy()

    if isinstance(market, pd.DataFrame) and not market.empty:
        market = market[~market.index.duplicated(keep="last")]
        result = result.join(market, how="left")

    if isinstance(sector, pd.DataFrame) and not sector.empty:
        sector_cols = [column for column in sector.columns if str(column).startswith("sector_")]
        if sector_cols:
            sector = sector[~sector.index.duplicated(keep="last")]
            result = result.join(sector[sector_cols], how="left")

    # Relative performance vs market.
    if "market_vnindex_return" in result.columns:
        result["stock_minus_market_1d"] = result["Return"] - result["market_vnindex_return"]
    if "market_vnindex_momentum20" in result.columns:
        result["stock_minus_market_momentum20"] = result["Momentum20"] - result["market_vnindex_momentum20"]
    if "sector_return" in result.columns:
        result["stock_minus_sector_1d"] = result["Return"] - result["sector_return"]
    if "sector_momentum20" in result.columns:
        result["stock_minus_sector_momentum20"] = result["Momentum20"] - result["sector_momentum20"]

    # Market breadth.
    market_return_columns = [
        column
        for column in result.columns
        if str(column).startswith("market_") and str(column).endswith("_return")
    ]
    if market_return_columns:
        result["market_positive_index_count"] = result[market_return_columns].gt(0).sum(axis=1)
        result["market_negative_index_count"] = result[market_return_columns].lt(0).sum(axis=1)
        result["market_average_return"] = result[market_return_columns].mean(axis=1)

    # Targets: giữ tương thích với ML hiện tại.
    close = result["Close"]
    for horizon, periods in {"1D": 1, "5D": 5, "20D": 20}.items():
        result[f"Target_{horizon}"] = close.shift(-periods) / close - 1
        result[f"Target_{horizon}_Pct"] = result[f"Target_{horizon}"] * 100

    result.attrs["symbol"] = symbol
    result.attrs["display_symbol"] = display_symbol(symbol)
    result.attrs["source"] = "Vnstock multifactor (không flow API)"
    result.attrs["research_start"] = str(start.date())
    result.attrs["research_end"] = str(end.date())
    result.attrs["sector"] = metadata.get("sector", "")
    result.attrs["icb_code"] = metadata.get("icb_code", "")
    result.attrs["icb_code4"] = metadata.get("icb_code4", "")
    result.attrs["sector_peers"] = sector.attrs.get("peers_requested", []) if not sector.empty else []
    result.attrs["sector_peers_loaded"] = sector.attrs.get("peers_loaded", 0) if not sector.empty else 0
    result.attrs["completeness"] = completeness

    result = result.replace([np.inf, -np.inf], np.nan)

    if not result.empty:
        start_label = result.index.min().date()
        end_label = result.index.max().date()
    else:
        start_label = "n/a"
        end_label = "n/a"

    print(
        f"[RESEARCH] HOÀN TẤT {symbol}: {len(result)} phiên | "
        f"{start_label} → {end_label}"
    )

    if progress is not None:
        try:
            progress.progress(
                100,
                text=f"4/4 Hoàn tất: {len(result)} phiên | {start_label} → {end_label}",
            )
        except Exception:
            pass
    if status is not None:
        try:
            status.success(
                f"Đã hoàn tất nghiên cứu {symbol}: {len(result)} phiên, {start_label} → {end_label}."
            )
        except Exception:
            pass

    return result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def load_research_history(symbol, start_date, end_date):
    return load_multifactor_research_history(symbol, start_date, end_date)


def market_data(symbol, period="1y"):
    return load_market_data(symbol, period)


def get_research_data(symbol, start_date, end_date):
    return load_multifactor_research_history(symbol, start_date, end_date)


def load_full_research_data(symbol, start_date, end_date):
    return load_multifactor_research_history(symbol, start_date, end_date)


# ============================================================
# NEWS SENTIMENT
# ============================================================

def classify_news(title: str) -> str:
    text = str(title or "").lower()

    positive_words = [
        "tăng", "tích cực", "lợi nhuận", "kỷ lục", "tăng trưởng",
        "bứt phá", "vượt kỳ vọng", "vượt kế hoạch", "hưởng lợi", "cải thiện",
    ]
    negative_words = [
        "giảm", "tiêu cực", "thua lỗ", "rủi ro", "sụt giảm", "áp lực",
        "bán tháo", "khó khăn", "nợ xấu", "cảnh báo", "điều tra", "vi phạm",
    ]

    positive = sum(word in text for word in positive_words)
    negative = sum(word in text for word in negative_words)

    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


# ============================================================
# OLS / RANDOM FOREST / QUANT
# ============================================================

def run_ols(data):
    if data is None or data.empty:
        return None
    try:
        import statsmodels.api as sm
    except Exception:
        return None

    candidate = [
        "RSI", "MACD", "MACD_Hist", "Volatility20", "Volume_Change",
        "Momentum20", "Range_Percent", "Relative_Volume",
        "market_vnindex_return", "market_vnindex_momentum20",
        "sector_return", "sector_momentum20",
    ]
    features = [column for column in candidate if column in data.columns]
    if not features:
        return None

    dataset = data[features + ["Return"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dataset) < 50:
        return None

    try:
        X = sm.add_constant(dataset[features], has_constant="add")
        return sm.OLS(dataset["Return"], X).fit(cov_type="HC3")
    except Exception:
        return None


def run_random_forest(data):
    if data is None or data.empty:
        return None
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return None

    excluded = {
        "Target_1D", "Target_5D", "Target_20D",
        "Target_1D_Pct", "Target_5D_Pct", "Target_20D_Pct",
    }
    features = [
        column
        for column in data.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not features or "Target_1D" not in data.columns:
        return None

    work = data[features + ["Target_1D"]].replace([np.inf, -np.inf], np.nan)
    X = work[features].copy()
    y = work["Target_1D"].copy()

    for column in X.columns:
        median = X[column].median()
        X[column] = X[column].fillna(0.0 if pd.isna(median) else median)

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if len(X) < 80:
        return None

    split = int(len(X) * 0.8)
    if split < 40:
        return None

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    try:
        model.fit(X.iloc[:split], y.iloc[:split])
        prediction = float(model.predict(X.iloc[[-1]])[0])
        return {
            "model": model,
            "prediction": prediction,
            "importance": dict(zip(features, model.feature_importances_)),
        }
    except Exception:
        return None


def build_quant(data):
    if data is None or data.empty:
        return None

    try:
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    excluded = {
        "Target_1D", "Target_5D", "Target_20D",
        "Target_1D_Pct", "Target_5D_Pct", "Target_20D_Pct",
    }
    features = [
        column
        for column in data.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not features:
        return None

    results = {}
    for horizon in ["1D", "5D", "20D"]:
        target_column = f"Target_{horizon}"
        if target_column not in data.columns:
            continue

        work = data[features + [target_column]].replace([np.inf, -np.inf], np.nan).copy()
        X = work[features].copy()
        y = work[target_column].copy()

        for column in X.columns:
            median = X[column].median()
            X[column] = X[column].fillna(0.0 if pd.isna(median) else median)

        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < 60:
            continue

        split = int(len(X) * 0.8)
        if split < 30 or len(X) - split < 10:
            continue

        models = [
            ("Ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
            ("Lasso", Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.0001, max_iter=50000))])),
            ("Elastic Net", Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=50000))])),
            ("Random Forest", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ("Extra Trees", ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=3, min_samples_leaf=3, random_state=42)),
        ]

        rows = []
        fitted = {}
        predictions = {}

        for name, model in models:
            try:
                model.fit(X.iloc[:split], y.iloc[:split])
                pred = np.asarray(model.predict(X.iloc[split:]), dtype=float)
                actual = np.asarray(y.iloc[split:], dtype=float)
                mse = float(mean_squared_error(actual, pred))
                rows.append({
                    "Mô hình": name,
                    "MAE": float(mean_absolute_error(actual, pred)),
                    "MSE": mse,
                    "RMSE": float(np.sqrt(mse)),
                    "R²": float(r2_score(actual, pred)),
                })
                fitted[name] = model
                predictions[name] = pred
            except Exception as error:
                print(f"[QUANT] {name}/{horizon}: lỗi {error}")

        models_df = pd.DataFrame(rows)
        if not models_df.empty:
            models_df = models_df.sort_values("RMSE", ascending=True).reset_index(drop=True)

        importance_df = pd.DataFrame()
        if not models_df.empty:
            best_name = str(models_df.iloc[0]["Mô hình"])
            best_model = fitted.get(best_name)
            raw_model = best_model
            if best_model is not None and hasattr(best_model, "named_steps"):
                raw_model = best_model.named_steps.get("model", best_model)
            if raw_model is not None and hasattr(raw_model, "feature_importances_"):
                importance_df = pd.DataFrame({
                    "Biến": X.columns,
                    "Importance": raw_model.feature_importances_,
                }).sort_values("Importance", ascending=False).reset_index(drop=True)

        results[horizon] = {
            "observations": len(X),
            "train": len(X.iloc[:split]),
            "test": len(X.iloc[split:]),
            "features": features,
            "models": models_df,
            "fitted": fitted,
            "predictions": predictions,
            "tree_importance": importance_df,
        }

    return results or None


# ============================================================
# FACTOR GROUPS / CORRELATION
# ============================================================

def research_factor_groups(data):
    if data is None or data.empty:
        return {"technical": [], "flow": [], "foreign": [], "proprietary": [], "market": [], "sector": []}

    columns = [str(column) for column in data.columns]
    return {
        "technical": [
            column for column in columns
            if not any(column.startswith(prefix) for prefix in ("market_", "sector_", "foreign_", "proprietary_", "flow_", "Target_"))
        ],
        "flow": [column for column in columns if "Trading_Value" in column or column.startswith("flow_")],
        "foreign": [],
        "proprietary": [],
        "market": [column for column in columns if column.startswith("market_")],
        "sector": [column for column in columns if column.startswith("sector_")],
    }


def factor_correlation_table(data, target_column):
    if data is None or data.empty or target_column not in data.columns:
        return pd.DataFrame()

    excluded = {"Target_1D", "Target_5D", "Target_20D", "Target_1D_Pct", "Target_5D_Pct", "Target_20D_Pct"}
    rows = []
    y = pd.to_numeric(data[target_column], errors="coerce")

    for column in data.columns:
        if column == target_column or column in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(data[column]):
            continue

        x = pd.to_numeric(data[column], errors="coerce")
        valid = x.notna() & y.notna()
        x_valid = x.loc[valid]
        y_valid = y.loc[valid]
        if len(x_valid) < 20:
            continue

        pearson = x_valid.corr(y_valid, method="pearson")
        spearman = x_valid.corr(y_valid, method="spearman")
        rows.append({
            "Biến": column,
            "Pearson": float(pearson) if pd.notna(pearson) else np.nan,
            "Spearman": float(spearman) if pd.notna(spearman) else np.nan,
            "|Spearman|": abs(float(spearman)) if pd.notna(spearman) else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("|Spearman|", ascending=False, na_position="last").reset_index(drop=True)


def build_research_factor_dataset(data, target_column="Target_20D"):
    return {
        "data": data,
        "correlation": factor_correlation_table(data, target_column),
        "groups": research_factor_groups(data),
    }
