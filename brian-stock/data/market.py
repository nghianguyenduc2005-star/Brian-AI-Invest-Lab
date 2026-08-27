# data/market.py

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import requests
import pandas as pd


# ============================================================
# DNSE CONFIG
# ============================================================

DNSE_BASE_URL = "https://api.dnse.com.vn"

OHLC_ENDPOINT = f"{DNSE_BASE_URL}/market-data/v1/price/ohlc"

REQUEST_TIMEOUT = 20


# ============================================================
# SYMBOL
# ============================================================

def normalize_symbol(symbol: str) -> str:
    """
    Chuẩn hóa mã cổ phiếu.

    Ví dụ:
        HPG       -> HPG.VN
        hpg       -> HPG.VN
        HPG.VN    -> HPG.VN
        FPT       -> FPT.VN
    """

    if not symbol:
        return ""

    symbol = str(symbol).strip().upper()

    # Loại bỏ các hậu tố thường gặp
    for suffix in [".VN", ".HOSE", ".HNX", ".UPCOM"]:
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]

    symbol = symbol.strip()

    if not symbol:
        return ""

    return f"{symbol}.VN"


def dnse_symbol(symbol: str) -> str:
    """
    Chuyển symbol nội bộ về symbol DNSE.

    HPG.VN -> HPG
    HPG    -> HPG
    """

    symbol = normalize_symbol(symbol)

    if not symbol:
        return ""

    return symbol.replace(".VN", "")


# ============================================================
# HTTP
# ============================================================

def _request_dnse(
    url: str,
    params: dict[str, Any],
) -> Any:
    """
    Gọi DNSE API.

    Không dùng vnstock.
    Không dùng Yahoo Finance.
    Không dùng dữ liệu random.
    """

    headers = {
        "Accept": "application/json",
        "User-Agent": (
            "BrianStock/1.0 "
            "(Investment Research Dashboard)"
        ),
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "DNSE đang giới hạn request (HTTP 429). "
            "Hãy thử lại sau vài giây."
        )

    if response.status_code == 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"DNSE từ chối request (HTTP 400): {detail}"
        )

    if response.status_code == 401:
        raise RuntimeError(
            "DNSE yêu cầu xác thực cho endpoint này."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "DNSE từ chối quyền truy cập endpoint."
        )

    if response.status_code >= 500:
        raise RuntimeError(
            f"DNSE server error: HTTP {response.status_code}"
        )

    response.raise_for_status()

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            "DNSE trả về dữ liệu không phải JSON."
        ) from exc


# ============================================================
# RESPONSE PARSER
# ============================================================

def _find_records(data: Any) -> list[dict[str, Any]]:
    """
    DNSE có thể trả JSON với nhiều cấu trúc khác nhau.

    Hàm này cố gắng tìm danh sách OHLC
    mà không phụ thuộc cứng vào một key duy nhất.
    """

    if isinstance(data, list):
        return [
            x for x in data
            if isinstance(x, dict)
        ]

    if not isinstance(data, dict):
        return []

    # Các key thường gặp
    possible_keys = [
        "data",
        "items",
        "content",
        "results",
        "candles",
        "ohlc",
        "rows",
    ]

    for key in possible_keys:
        value = data.get(key)

        if isinstance(value, list):
            records = [
                x for x in value
                if isinstance(x, dict)
            ]

            if records:
                return records

        if isinstance(value, dict):
            nested = _find_records(value)

            if nested:
                return nested

    # Tìm sâu hơn trong dictionary
    for value in data.values():

        if isinstance(value, dict):
            nested = _find_records(value)

            if nested:
                return nested

        elif isinstance(value, list):

            records = [
                x for x in value
                if isinstance(x, dict)
            ]

            if records:
                return records

    return []


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu OHLCV thành:

        Date
        Open
        High
        Low
        Close
        Volume
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    rename_map = {}

    for col in df.columns:

        raw = str(col).strip()

        key = (
            raw
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        mapping = {
            "time": "Date",
            "timestamp": "Date",
            "datetime": "Date",
            "date": "Date",
            "tradingdate": "Date",
            "tradingtime": "Date",

            "open": "Open",
            "openprice": "Open",

            "high": "High",
            "highprice": "High",

            "low": "Low",
            "lowprice": "Low",

            "close": "Close",
            "closeprice": "Close",
            "lastprice": "Close",

            "volume": "Volume",
            "totalvolume": "Volume",
            "matchvolume": "Volume",
        }

        if key in mapping:
            rename_map[col] = mapping[key]

    df = df.rename(columns=rename_map)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if "Date" not in df.columns:

        # Một số API trả time dưới dạng index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

            if "index" in df.columns:
                df = df.rename(
                    columns={"index": "Date"}
                )

    if "Date" in df.columns:

        # Unix timestamp
        if pd.api.types.is_numeric_dtype(df["Date"]):

            sample = df["Date"].dropna()

            if not sample.empty:

                median_value = float(
                    sample.abs().median()
                )

                if median_value > 10**12:
                    unit = "ms"

                elif median_value > 10**9:
                    unit = "s"

                else:
                    unit = None

                if unit:
                    df["Date"] = pd.to_datetime(
                        df["Date"],
                        unit=unit,
                        errors="coerce",
                    )

        else:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    existing_required = [
        col
        for col in required
        if col in df.columns
    ]

    if existing_required:

        df = df.dropna(
            subset=existing_required
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    if "Date" in df.columns:

        df = df.sort_values(
            "Date"
        )

        df = df.drop_duplicates(
            subset=["Date"],
            keep="last",
        )

    else:
        df = df.reset_index(drop=True)

    return df


# ============================================================
# DNSE PRICE UNIT
# ============================================================

def _normalize_price_unit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    DNSE có thể trả giá cổ phiếu ở đơn vị nghìn VND
    trong một số dữ liệu.

    Ví dụ:

        22.2 -> 22,200 VND

    Nhưng nếu API đã trả:

        22,200

    thì KHÔNG nhân thêm.

    Dùng median để tránh bị một dòng lỗi làm sai toàn bộ dữ liệu.
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    existing = [
        col
        for col in price_columns
        if col in df.columns
    ]

    if not existing:
        return df

    # Lấy median Close trước
    reference_col = (
        "Close"
        if "Close" in df.columns
        else existing[0]
    )

    median_price = pd.to_numeric(
        df[reference_col],
        errors="coerce",
    ).median()

    if pd.isna(median_price):
        return df

    # --------------------------------------------------------
    # Giá cổ phiếu Việt Nam:
    #
    # 22.2       => khả năng cao là 22,200 VND
    # 22,200     => đã là VND
    #
    # Không dùng < 1000 một cách mù quáng cho mọi loại asset.
    # Chỉ xử lý khoảng giá hợp lý của cổ phiếu Việt Nam.
    # --------------------------------------------------------

    if 0 < median_price < 1000:

        for col in existing:

            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                * 1000
            )

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Thêm:

        Return
        SMA20
        SMA50
        RSI
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    if "Close" not in df.columns:
        return df

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Daily return
    # --------------------------------------------------------

    df["Return"] = close.pct_change()

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    df["SMA20"] = (
        close
        .rolling(
            window=20,
            min_periods=1,
        )
        .mean()
    )

    df["SMA50"] = (
        close
        .rolling(
            window=50,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        pd.NA,
    )

    df["RSI"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # Trường hợp loss = 0
    df.loc[
        (avg_loss == 0)
        & (avg_gain > 0),
        "RSI",
    ] = 100

    # Trường hợp cả gain và loss = 0
    df.loc[
        (avg_gain == 0)
        & (avg_loss == 0),
        "RSI",
    ] = 50

    return df


# ============================================================
# FETCH HISTORICAL OHLC
# ============================================================

def load_market_data(
    symbol: str,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Lấy dữ liệu OHLCV lịch sử từ DNSE.

    Ví dụ:

        load_market_data("HPG.VN", "1y")

    Trả về DataFrame gồm:

        Date
        Open
        High
        Low
        Close
        Volume
        Return
        SMA20
        SMA50
        RSI
    """

    clean_symbol = dnse_symbol(symbol)

    if not clean_symbol:
        raise ValueError(
            "Mã cổ phiếu không hợp lệ."
        )

    # --------------------------------------------------------
    # Period
    # --------------------------------------------------------

    period_days = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 731,
        "3y": 1096,
        "5y": 1826,
    }

    days = period_days.get(
        period,
        366,
    )

    end_date = datetime.now()

    start_date = (
        end_date
        - timedelta(days=days)
    )

    # --------------------------------------------------------
    # DNSE API
    # --------------------------------------------------------

    params = {
        "symbol": clean_symbol,
        "resolution": "1D",
        "from": int(
            start_date.timestamp()
        ),
        "to": int(
            end_date.timestamp()
        ),
    }

    data = _request_dnse(
        OHLC_ENDPOINT,
        params,
    )

    records = _find_records(data)

    if not records:
        raise ValueError(
            f"DNSE không trả dữ liệu OHLC "
            f"cho {clean_symbol}."
        )

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df = _normalize_history(df)

    if df.empty:
        raise ValueError(
            f"Không có dữ liệu sau khi xử lý "
            f"{clean_symbol}."
        )

    # --------------------------------------------------------
    # QUAN TRỌNG:
    #
    # Chuẩn hóa giá trước khi tính indicators.
    # --------------------------------------------------------

    df = _normalize_price_unit(df)

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df = add_indicators(df)

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# LATEST QUOTE
# ============================================================

def get_latest_quote(
    symbol: str,
) -> dict[str, Any]:
    """
    Lấy giá gần nhất.

    Hàm này dùng endpoint quote riêng.
    Nếu endpoint không khả dụng thì dashboard
    vẫn có thể lấy giá cuối từ OHLC.
    """

    clean_symbol = dnse_symbol(symbol)

    if not clean_symbol:
        raise ValueError(
            "Mã cổ phiếu không hợp lệ."
        )

    url = (
        f"{DNSE_BASE_URL}"
        f"/market-data/v1/stocks/"
        f"{clean_symbol}/quotes"
    )

    data = _request_dnse(
        url,
        {},
    )

    if isinstance(data, dict):

        # Một số response có data bọc ngoài
        if isinstance(
            data.get("data"),
            dict,
        ):
            data = data["data"]

        return data

    return {}


# ============================================================
# MARKET SUMMARY
# ============================================================

def get_market_summary() -> dict[str, Any]:
    """
    Khung dữ liệu cho VN-INDEX.

    Chưa hard-code dữ liệu.
    Nếu DNSE endpoint index thay đổi,
    dashboard sẽ trả None thay vì hiển thị dữ liệu giả.
    """

    return {
        "symbol": "VNINDEX",
        "point": None,
        "change": None,
        "change_percent": None,
        "volume": None,
    }
