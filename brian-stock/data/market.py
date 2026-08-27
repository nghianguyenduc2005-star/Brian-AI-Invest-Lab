from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


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
CACHE_TTL_FLOW = 900
CACHE_TTL_LISTING = 6 * 60 * 60

# Giữ khoảng cách giữa request để tránh rate limit.
#
# Guest có thể bị giới hạn request/phút thấp hơn.
# Dùng 2 giây/request để an toàn.
REQUEST_SLEEP_SECONDS = 2.0

# Lịch sử chính có thể lấy theo chunk.
# 120 ngày lịch ~ khoảng 80-85 phiên giao dịch.
RESEARCH_CHUNK_DAYS = 120


# ============================================================
# CACHE RESOURCE
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def _create_market():
    if not VNSTOCK_AVAILABLE or Market is None:
        raise RuntimeError(
            "Không tìm thấy vnstock Market."
        )

    try:
        return Market()

    except Exception as error:
        raise RuntimeError(
            f"Không khởi tạo được Market(): {error}"
        ) from error


# ============================================================
# BASIC
# ============================================================

def normalize_symbol(
    symbol,
):
    if symbol is None:
        return "HPG"

    value = (
        str(symbol)
        .strip()
        .upper()
    )

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    if value.endswith(".VN"):
        value = value[:-3]

    return value or "HPG"


def display_symbol(
    symbol,
):
    if symbol is None:
        return ""

    value = (
        str(symbol)
        .strip()
        .upper()
    )

    if value.endswith(".VN"):
        value = value[:-3]

    return value


def _to_number(
    value,
    default=np.nan,
):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def _normalize_column_name(
    value,
):
    text = (
        str(value)
        .strip()
        .lower()
    )

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    text = re.sub(
        r"[-/]+",
        "_",
        text,
    )

    return text


# ============================================================
# GENERIC COLUMN FINDER
# ============================================================

def _find_column(
    df,
    candidates,
):
    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):
        return None

    mapping = {
        _normalize_column_name(column): column
        for column in df.columns
    }

    # Exact
    for candidate in candidates:

        key = _normalize_column_name(
            candidate
        )

        if key in mapping:
            return mapping[key]

    # Fuzzy
    for column in df.columns:

        column_key = _normalize_column_name(
            column
        )

        for candidate in candidates:

            candidate_key = _normalize_column_name(
                candidate
            )

            if (
                candidate_key in column_key
                or column_key in candidate_key
            ):
                return column

    return None


# ============================================================
# DATETIME NORMALIZATION
# ============================================================

def _normalize_datetime_index(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    work = df.copy()

    time_column = _find_column(
        work,
        [
            "time",
            "date",
            "datetime",
            "timestamp",
            "trading_date",
            "tradingdate",
        ],
    )

    if time_column is not None:

        work[
            time_column
        ] = pd.to_datetime(
            work[
                time_column
            ],
            errors="coerce",
        )

        work = (
            work
            .set_index(
                time_column
            )
        )

    else:

        work.index = pd.to_datetime(
            work.index,
            errors="coerce",
        )

    work = work[
        ~work.index.isna()
    ].copy()

    work = (
        work
        .sort_index()
    )

    work = work[
        ~work.index.duplicated(
            keep="last"
        )
    ].copy()

    return work


# ============================================================
# OHLCV NORMALIZATION
# ============================================================

def _normalize_ohlcv(
    df,
    stock=True,
):
    if df is None:
        raise ValueError(
            "Nguồn dữ liệu trả về None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        df = pd.DataFrame(
            df
        )

    if df.empty:
        raise ValueError(
            "Nguồn dữ liệu trả về DataFrame rỗng."
        )

    work = df.copy()

    # MultiIndex
    if isinstance(
        work.columns,
        pd.MultiIndex,
    ):

        columns = []

        for column in work.columns:

            if isinstance(
                column,
                tuple,
            ):
                columns.append(
                    str(
                        column[-1]
                    )
                )
            else:
                columns.append(
                    str(
                        column
                    )
                )

        work.columns = columns

    rename_map = {}

    for column in work.columns:

        key = _normalize_column_name(
            column
        )

        if key in {
            "time",
            "date",
            "datetime",
            "timestamp",
            "trading_date",
            "tradingdate",
        }:

            rename_map[
                column
            ] = "Time"

        elif key in {
            "open",
            "open_price",
            "openprice",
        }:

            rename_map[
                column
            ] = "Open"

        elif key in {
            "high",
            "high_price",
            "highprice",
        }:

            rename_map[
                column
            ] = "High"

        elif key in {
            "low",
            "low_price",
            "lowprice",
        }:

            rename_map[
                column
            ] = "Low"

        elif key in {
            "close",
            "close_price",
            "closeprice",
            "last",
            "last_price",
            "lastprice",
        }:

            rename_map[
                column
            ] = "Close"

        elif key in {
            "volume",
            "vol",
            "total_volume",
            "totalvolume",
            "matched_volume",
            "match_volume",
            "matchvolume",
        }:

            rename_map[
                column
            ] = "Volume"

        elif key in {
            "value",
            "trading_value",
            "trade_value",
            "value_traded",
            "match_value",
            "matched_value",
            "turnover",
            "total_value",
        }:

            rename_map[
                column
            ] = "Value"

    work = work.rename(
        columns=rename_map
    )

    # Datetime
    if "Time" in work.columns:

        work[
            "Time"
        ] = pd.to_datetime(
            work[
                "Time"
            ],
            errors="coerce",
        )

        work = (
            work
            .set_index(
                "Time"
            )
        )

    else:

        work.index = pd.to_datetime(
            work.index,
            errors="coerce",
        )

    work = work[
        ~work.index.isna()
    ].copy()

    work = (
        work
        .sort_index()
    )

    work = work[
        ~work.index.duplicated(
            keep="last"
        )
    ].copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in work.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu cột OHLC: "
            + ", ".join(
                missing
            )
        )

    if "Volume" not in work.columns:

        work[
            "Volume"
        ] = 0.0

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

        work[
            column
        ] = pd.to_numeric(
            work[
                column
            ],
            errors="coerce",
        )

    if "Value" in work.columns:

        work[
            "Value"
        ] = pd.to_numeric(
            work[
                "Value"
            ],
            errors="coerce",
        )

    # Giá cổ phiếu có thể về đơn vị nghìn.
    if stock:

        median_price = (
            work[
                "Close"
            ]
            .dropna()
            .median()
        )

        if (
            pd.notna(
                median_price
            )
            and median_price > 0
            and median_price < 1000
        ):

            for column in [
                "Open",
                "High",
                "Low",
                "Close",
            ]:

                work[
                    column
                ] *= 1000

            if "Value" in work.columns:
                # Value thường đã là VND,
                # không tự nhân nếu chưa chắc.
                pass

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        work.loc[
            work[
                column
            ] <= 0,
            column,
        ] = np.nan

    work.loc[
        work[
            "Volume"
        ] < 0,
        "Volume",
    ] = np.nan

    work = work.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    work = work.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    if work.empty:
        raise ValueError(
            "Không còn OHLC hợp lệ."
        )

    return work


# ============================================================
# RSI
# ============================================================

def rsi(
    prices,
    period=14,
):

    prices = pd.to_numeric(
        prices,
        errors="coerce",
    )

    delta = prices.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    relative = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    result = (
        100
        - (
            100
            / (
                1
                + relative
            )
        )
    )

    result = result.where(
        ~(
            (
                avg_loss == 0
            )
            & (
                avg_gain > 0
            )
        ),
        100,
    )

    result = result.where(
        ~(
            (
                avg_loss == 0
            )
            & (
                avg_gain == 0
            )
        ),
        50,
    )

    return result


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df,
    la_co_phieu=False,
):

    work = _normalize_ohlcv(
        df,
        stock=la_co_phieu,
    ).copy()

    close = work[
        "Close"
    ]

    high = work[
        "High"
    ]

    low = work[
        "Low"
    ]

    volume = work[
        "Volume"
    ]

    # ========================================================
    # PRICE RETURN
    # ========================================================

    work[
        "Return"
    ] = close.pct_change()

    work[
        "ReturnPct"
    ] = (
        work[
            "Return"
        ]
        * 100
    )

    # ========================================================
    # LOG RETURN
    # ========================================================

    work[
        "LogReturn"
    ] = np.log(
        close
        / close.shift(1)
    )

    # ========================================================
    # RSI
    # ========================================================

    work[
        "RSI"
    ] = rsi(
        close,
        14,
    )

    # ========================================================
    # EMA
    # ========================================================

    for period in [
        9,
        12,
        20,
        26,
        50,
        100,
        200,
    ]:

        work[
            f"EMA{period}"
        ] = (
            close
            .ewm(
                span=period,
                adjust=False,
            )
            .mean()
        )

    # ========================================================
    # SMA
    # ========================================================

    for period in [
        5,
        10,
        20,
        50,
        100,
        200,
    ]:

        work[
            f"SMA{period}"
        ] = (
            close
            .rolling(
                period
            )
            .mean()
        )

    # ========================================================
    # DISTANCE FROM MA
    # ========================================================

    for period in [
        20,
        50,
        100,
        200,
    ]:

        sma_col = (
            f"SMA{period}"
        )

        ema_col = (
            f"EMA{period}"
        )

        if sma_col in work.columns:

            work[
                f"Price_vs_SMA{period}"
            ] = (
                (
                    close
                    / work[
                        sma_col
                    ]
                )
                - 1
            ) * 100

        if ema_col in work.columns:

            work[
                f"Price_vs_EMA{period}"
            ] = (
                (
                    close
                    / work[
                        ema_col
                    ]
                )
                - 1
            ) * 100

    # ========================================================
    # MACD
    # ========================================================

    work[
        "MACD"
    ] = (
        work[
            "EMA12"
        ]
        - work[
            "EMA26"
        ]
    )

    work[
        "MACD_Signal"
    ] = (
        work[
            "MACD"
        ]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    work[
        "MACD_Hist"
    ] = (
        work[
            "MACD"
        ]
        - work[
            "MACD_Signal"
        ]
    )

    # ========================================================
    # BOLLINGER
    # ========================================================

    rolling_std = (
        close
        .rolling(
            20
        )
        .std()
    )

    work[
        "Bollinger_Mid"
    ] = work[
        "SMA20"
    ]

    work[
        "Bollinger_Upper"
    ] = (
        work[
            "SMA20"
        ]
        + 2 * rolling_std
    )

    work[
        "Bollinger_Lower"
    ] = (
        work[
            "SMA20"
        ]
        - 2 * rolling_std
    )

    work[
        "Bollinger_Width"
    ] = (
        (
            work[
                "Bollinger_Upper"
            ]
            - work[
                "Bollinger_Lower"
            ]
        )
        / work[
            "Bollinger_Mid"
        ]
        * 100
    )

    work[
        "Bollinger_Position"
    ] = (
        (
            close
            - work[
                "Bollinger_Lower"
            ]
        )
        / (
            work[
                "Bollinger_Upper"
            ]
            - work[
                "Bollinger_Lower"
            ]
        )
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    for period in [
        5,
        20,
        60,
    ]:

        work[
            f"Volatility{period}"
        ] = (
            work[
                "Return"
            ]
            .rolling(
                period
            )
            .std()
            * np.sqrt(
                252
            )
            * 100
        )

    work[
        "Volatility_20D"
    ] = (
        work[
            "Volatility20"
        ]
        / 100
    )

    # ========================================================
    # VOLUME
    # ========================================================

    for period in [
        5,
        20,
        50,
    ]:

        work[
            f"Volume_SMA{period}"
        ] = (
            volume
            .rolling(
                period
            )
            .mean()
        )

    work[
        "Volume_Change"
    ] = volume.pct_change()

    work[
        "Relative_Volume"
    ] = (
        volume
        / work[
            "Volume_SMA20"
        ]
    )

    # ========================================================
    # VALUE
    # ========================================================

    if "Value" not in work.columns:

        work[
            "Value"
        ] = (
            close
            * volume
        )

    else:

        work[
            "Value"
        ] = pd.to_numeric(
            work[
                "Value"
            ],
            errors="coerce",
        )

    work[
        "Trading_Value"
    ] = work[
        "Value"
    ]

    work[
        "Trading_Value_Change"
    ] = (
        work[
            "Trading_Value"
        ]
        .pct_change()
    )

    work[
        "Trading_Value_SMA20"
    ] = (
        work[
            "Trading_Value"
        ]
        .rolling(
            20
        )
        .mean()
    )

    # ========================================================
    # RANGE
    # ========================================================

    work[
        "Range"
    ] = (
        high
        - low
    )

    work[
        "Range_Percent"
    ] = (
        work[
            "Range"
        ]
        / close
        * 100
    )

    work[
        "Body"
    ] = (
        close
        - work[
            "Open"
        ]
    )

    work[
        "Body_Percent"
    ] = (
        work[
            "Body"
        ]
        / work[
            "Open"
        ]
        * 100
    )

    # ========================================================
    # ATR14
    # ========================================================

    true_range_1 = (
        high
        - low
    )

    true_range_2 = (
        high
        - close.shift(1)
    ).abs()

    true_range_3 = (
        low
        - close.shift(1)
    ).abs()

    true_range = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3,
        ],
        axis=1,
    ).max(
        axis=1
    )

    work[
        "TrueRange"
    ] = true_range

    work[
        "ATR14"
    ] = (
        true_range
        .rolling(
            14
        )
        .mean()
    )

    work[
        "ATR_Percent"
    ] = (
        work[
            "ATR14"
        ]
        / close
        * 100
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    for period in [
        1,
        5,
        10,
        20,
        60,
    ]:

        work[
            f"Momentum{period}"
        ] = (
            close
            / close.shift(
                period
            )
            - 1
        )

        work[
            f"Momentum{period}Pct"
        ] = (
            work[
                f"Momentum{period}"
            ]
            * 100
        )

    # ========================================================
    # HIGH / LOW
    # ========================================================

    for period in [
        20,
        50,
        252,
    ]:

        work[
            f"High{period}"
        ] = (
            high
            .rolling(
                period
            )
            .max()
        )

        work[
            f"Low{period}"
        ] = (
            low
            .rolling(
                period
            )
            .min()
        )

        work[
            f"Distance_From_High{period}"
        ] = (
            close
            / work[
                f"High{period}"
            ]
            - 1
        ) * 100

        work[
            f"Distance_From_Low{period}"
        ] = (
            close
            / work[
                f"Low{period}"
            ]
            - 1
        ) * 100

    # ========================================================
    # STOCHASTIC
    # ========================================================

    low14 = (
        low
        .rolling(
            14
        )
        .min()
    )

    high14 = (
        high
        .rolling(
            14
        )
        .max()
    )

    work[
        "Stochastic_K"
    ] = (
        (
            close
            - low14
        )
        / (
            high14
            - low14
        )
        * 100
    )

    work[
        "Stochastic_D"
    ] = (
        work[
            "Stochastic_K"
        ]
        .rolling(
            3
        )
        .mean()
    )

    # ========================================================
    # ROC
    # ========================================================

    for period in [
        5,
        10,
        20,
    ]:

        work[
            f"ROC{period}"
        ] = (
            (
                close
                / close.shift(
                    period
                )
            )
            - 1
        ) * 100

    # ========================================================
    # PRICE / VOLUME TREND
    # ========================================================

    work[
        "OBV_Proxy"
    ] = (
        np.sign(
            close.diff()
        )
        * volume
    ).cumsum()

    work[
        "Dollar_Volume"
    ] = (
        close
        * volume
    )

    # ========================================================
    # CLEAN
    # ========================================================

    work = work.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return work


# ============================================================
# PERIOD -> DAYS
# ============================================================

def _period_days(
    period,
):
    key = str(
        period or "1y"
    ).strip().lower()

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

    return mapping.get(
        key,
        365,
    )


# ============================================================
# REQUEST RATE LIMIT
# ============================================================

def _sleep_between_requests(
    previous_request_time,
):
    if previous_request_time is None:
        return

    elapsed = (
        time.monotonic()
        - previous_request_time
    )

    remaining = (
        REQUEST_SLEEP_SECONDS
        - elapsed
    )

    if remaining > 0:
        time.sleep(
            remaining
        )


# ============================================================
# EQUITY OHLCV REQUEST
# ============================================================

def _request_equity_ohlcv(
    market,
    symbol,
    start_date,
    end_date,
):
    return (
        market
        .equity(
            normalize_symbol(
                symbol
            )
        )
        .ohlcv(
            start=pd.Timestamp(
                start_date
            ).strftime(
                "%Y-%m-%d"
            ),
            end=(
                pd.Timestamp(
                    end_date
                )
                + pd.Timedelta(
                    days=1
                )
            ).strftime(
                "%Y-%m-%d"
            ),
            interval="1D",
        )
    )


# ============================================================
# INDEX OHLCV REQUEST
# ============================================================

def _request_index_ohlcv(
    market,
    index_symbol,
    start_date,
    end_date,
):
    return (
        market
        .index(
            index_symbol
        )
        .ohlcv(
            start=pd.Timestamp(
                start_date
            ).strftime(
                "%Y-%m-%d"
            ),
            end=(
                pd.Timestamp(
                    end_date
                )
                + pd.Timedelta(
                    days=1
                )
            ).strftime(
                "%Y-%m-%d"
            ),
            interval="1D",
        )
    )


# ============================================================
# LOAD STOCK DATA
# ============================================================

def _load_stock_raw(
    symbol,
    start_date,
    end_date,
):

    market = _create_market()

    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    chunks = []

    cursor = start

    last_request = None

    while cursor <= end:

        chunk_end = min(
            cursor
            + pd.Timedelta(
                days=(
                    RESEARCH_CHUNK_DAYS
                    - 1
                )
            ),
            end,
        )

        _sleep_between_requests(
            last_request
        )

        request_start = (
            time.monotonic()
        )

        df_chunk = _request_equity_ohlcv(
            market,
            symbol,
            cursor,
            chunk_end,
        )

        last_request = request_start

        if (
            df_chunk is not None
            and not df_chunk.empty
        ):

            chunks.append(
                df_chunk
            )

        cursor = (
            chunk_end
            + pd.Timedelta(
                days=1
            )
        )

    if not chunks:

        raise ValueError(
            f"Không có dữ liệu giá {symbol}."
        )

    raw = pd.concat(
        chunks,
        axis=0,
    )

    raw = _normalize_ohlcv(
        raw,
        stock=True,
    )

    return raw


# ============================================================
# OLD PUBLIC STOCK API
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_PRICE,
    show_spinner=False,
)
def load_market_data(
    symbol,
    period="1y",
):

    start = (
        pd.Timestamp.today()
        - pd.Timedelta(
            days=_period_days(
                period
            )
        )
    ).date()

    end = (
        pd.Timestamp.today()
        .date()
    )

    raw = _load_stock_raw(
        symbol,
        start,
        end,
    )

    data = add_indicators(
        raw,
        la_co_phieu=True,
    )

    data.attrs[
        "symbol"
    ] = normalize_symbol(
        symbol
    )

    data.attrs[
        "source"
    ] = "Vnstock"

    return data


# ============================================================
# LOAD VNINDEX
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_INDEX,
    show_spinner=False,
)
def load_vnindex_data():

    market = _create_market()

    end = (
        pd.Timestamp.today()
        .date()
    )

    start = (
        pd.Timestamp(
            end
        )
        - pd.Timedelta(
            days=450
        )
    ).date()

    df = _request_index_ohlcv(
        market,
        "VNINDEX",
        start,
        end,
    )

    data = _normalize_ohlcv(
        df,
        stock=False,
    )

    data = add_indicators(
        data,
        la_co_phieu=False,
    )

    data.attrs[
        "symbol"
    ] = "VNINDEX"

    data.attrs[
        "source"
    ] = "Vnstock"

    return data


# ============================================================
# LATEST PRICE
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_LATEST,
    show_spinner=False,
)
def load_latest_price(
    symbol,
):

    data = load_market_data(
        symbol,
        "5d",
    )

    if data.empty:

        raise ValueError(
            "Không có dữ liệu mới nhất."
        )

    last = data.iloc[
        -1
    ]

    price = _to_number(
        last.get(
            "Close"
        )
    )

    if len(data) >= 2:

        previous = _to_number(
            data[
                "Close"
            ].iloc[
                -2
            ]
        )

    else:

        previous = price

    if (
        previous is not None
        and previous != 0
    ):

        change = (
            price
            / previous
            - 1
        ) * 100

    else:

        change = np.nan

    return {
        "ma": display_symbol(
            symbol
        ),
        "gia": price,
        "thay_doi": change,
        "khoi_luong": _to_number(
            last.get(
                "Volume"
            ),
            0.0,
        ),
        "thoi_gian": data.index[
            -1
        ],
    }


# ============================================================
# SNAPSHOT
# ============================================================

def market_snapshot(
    data,
):

    if (
        data is None
        or data.empty
    ):
        return {}

    last = data.iloc[
        -1
    ]

    previous = (
        data.iloc[
            -2
        ]
        if len(data) >= 2
        else last
    )

    price = _to_number(
        last.get(
            "Close"
        )
    )

    previous_price = _to_number(
        previous.get(
            "Close"
        )
    )

    if (
        price is not None
        and previous_price is not None
        and previous_price != 0
    ):

        change = (
            price
            / previous_price
            - 1
        ) * 100

    else:

        change = np.nan

    def get_column(
        name,
    ):
        return _to_number(
            last.get(
                name
            )
        )

    return {
        "price": price,
        "change_1d": change,
        "return_1d": get_column(
            "ReturnPct"
        ),
        "rsi": get_column(
            "RSI"
        ),
        "macd": get_column(
            "MACD"
        ),
        "sma20": get_column(
            "SMA20"
        ),
        "sma50": get_column(
            "SMA50"
        ),
        "volatility20": get_column(
            "Volatility20"
        ),
        "volume": get_column(
            "Volume"
        ),
        "trading_value": get_column(
            "Trading_Value"
        ),
        "atr14": get_column(
            "ATR14"
        ),
        "relative_volume": get_column(
            "Relative_Volume"
        ),
    }


# ============================================================
# TRADING / FLOW COLUMN EXTRACTION
# ============================================================

def _extract_date_column(
    df,
):
    return _find_column(
        df,
        [
            "time",
            "date",
            "datetime",
            "trading_date",
            "tradingdate",
        ],
    )


def _extract_volume_column(
    df,
):
    return _find_column(
        df,
        [
            "matched_volume",
            "match_volume",
            "matchvolume",
            "total_match_volume",
            "volume",
            "vol",
        ],
    )


def _extract_value_column(
    df,
):
    return _find_column(
        df,
        [
            "matched_value",
            "match_value",
            "matchvalue",
            "total_value",
            "trading_value",
            "value_traded",
            "value",
            "turnover",
        ],
    )


def _rename_flow_columns(
    df,
    prefix,
):

    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    work = df.copy()

    date_column = _extract_date_column(
        work
    )

    if date_column is not None:

        work[
            date_column
        ] = pd.to_datetime(
            work[
                date_column
            ],
            errors="coerce",
        )

        work = work[
            work[
                date_column
            ].notna()
        ].copy()

        work = (
            work
            .set_index(
                date_column
            )
            .sort_index()
        )

    else:

        work.index = pd.to_datetime(
            work.index,
            errors="coerce",
        )

        work = work[
            ~work.index.isna()
        ].copy()

    rename = {}

    groups = {
        "buy_vol": [
            "buy_vol",
            "buy_volume",
            "total_buy_volume",
            "foreign_buy_volume",
            "proprietary_buy_volume",
        ],
        "sell_vol": [
            "sell_vol",
            "sell_volume",
            "total_sell_volume",
            "foreign_sell_volume",
            "proprietary_sell_volume",
        ],
        "net_vol": [
            "net_vol",
            "net_volume",
            "net_buy_volume",
            "foreign_net_volume",
            "proprietary_net_volume",
        ],
        "buy_val": [
            "buy_val",
            "buy_value",
            "foreign_buy_value",
            "proprietary_buy_value",
        ],
        "sell_val": [
            "sell_val",
            "sell_value",
            "foreign_sell_value",
            "proprietary_sell_value",
        ],
        "net_val": [
            "net_val",
            "net_value",
            "net_buy_value",
            "foreign_net_value",
            "proprietary_net_value",
        ],
        "active_buy_volume": [
            "active_buy_volume",
            "total_buy_trade_volume",
            "buy_trade_volume",
        ],
        "active_sell_volume": [
            "active_sell_volume",
            "total_sell_trade_volume",
            "sell_trade_volume",
        ],
        "active_buy_value": [
            "active_buy_value",
            "total_buy_trade_value",
            "buy_trade_value",
        ],
        "active_sell_value": [
            "active_sell_value",
            "total_sell_trade_value",
            "sell_trade_value",
        ],
    }

    for normalized_name, candidates in groups.items():

        column = _find_column(
            work,
            candidates,
        )

        if column is not None:

            rename[
                column
            ] = (
                f"{prefix}_{normalized_name}"
            )

    work = work.rename(
        columns=rename
    )

    for column in work.columns:

        work[
            column
        ] = pd.to_numeric(
            work[
                column
            ],
            errors="coerce",
        )

    result_columns = [
        column
        for column in work.columns
        if str(column).startswith(
            f"{prefix}_"
        )
    ]

    if not result_columns:

        return pd.DataFrame()

    result = work[
        result_columns
    ].copy()

    return (
        result
        .sort_index()
        .groupby(
            result.index
        )
        .last()
    )


# ============================================================
# CALL FLOW METHOD WITH COMPATIBILITY
# ============================================================

def _call_historical_method(
    obj,
    method_name,
    start_date,
    end_date,
):
    method = getattr(
        obj,
        method_name,
        None,
    )

    if method is None:
        return None

    # Ưu tiên start/end.
    try:

        return method(
            start=pd.Timestamp(
                start_date
            ).strftime(
                "%Y-%m-%d"
            ),
            end=pd.Timestamp(
                end_date
            ).strftime(
                "%Y-%m-%d"
            ),
        )

    except TypeError:
        pass

    except Exception:
        pass

    # Fallback method không tham số.
    try:

        return method()

    except Exception:
        return None


# ============================================================
# FLOW HISTORY
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_FLOW,
    show_spinner=False,
)
def load_stock_flow_history(
    symbol,
    start_date,
    end_date,
):

    market = _create_market()

    obj = market.equity(
        normalize_symbol(
            symbol
        )
    )

    result_parts = []

    # ========================================================
    # TRADE HISTORY
    # ========================================================

    trade = _call_historical_method(
        obj,
        "trade_history",
        start_date,
        end_date,
    )

    if (
        isinstance(
            trade,
            pd.DataFrame,
        )
        and not trade.empty
    ):

        trade = _rename_flow_columns(
            trade,
            "flow",
        )

        if not trade.empty:

            result_parts.append(
                trade
            )

        # Tổng hợp khối lượng/value
        trade_raw = _normalize_datetime_index(
            trade
        )

    # ========================================================
    # FOREIGN
    # ========================================================

    foreign = _call_historical_method(
        obj,
        "foreign_flow",
        start_date,
        end_date,
    )

    foreign = _rename_flow_columns(
        foreign,
        "foreign",
    )

    if not foreign.empty:

        result_parts.append(
            foreign
        )

    # ========================================================
    # PROPRIETARY
    # ========================================================

    proprietary = _call_historical_method(
        obj,
        "proprietary_flow",
        start_date,
        end_date,
    )

    proprietary = _rename_flow_columns(
        proprietary,
        "proprietary",
    )

    if not proprietary.empty:

        result_parts.append(
            proprietary
        )

    if not result_parts:

        return pd.DataFrame()

    # ========================================================
    # MERGE
    # ========================================================

    merged = result_parts[
        0
    ].copy()

    for part in result_parts[
        1:
    ]:

        merged = merged.join(
            part,
            how="outer",
        )

    merged = (
        merged
        .sort_index()
        .loc[
            pd.Timestamp(
                start_date
            ):
            pd.Timestamp(
                end_date
            )
        ]
    )

    # ========================================================
    # DERIVED FLOW FACTORS
    # ========================================================

    volume_buy_cols = [
        column
        for column in merged.columns
        if column.endswith(
            "_buy_vol"
        )
    ]

    volume_sell_cols = [
        column
        for column in merged.columns
        if column.endswith(
            "_sell_vol"
        )
    ]

    value_buy_cols = [
        column
        for column in merged.columns
        if column.endswith(
            "_buy_val"
        )
    ]

    value_sell_cols = [
        column
        for column in merged.columns
        if column.endswith(
            "_sell_val"
        )
    ]

    # ========================================================
    # FOREIGN NET
    # ========================================================

    if (
        "foreign_buy_val" in merged.columns
        and "foreign_sell_val" in merged.columns
    ):

        merged[
            "foreign_net_val_calc"
        ] = (
            merged[
                "foreign_buy_val"
            ]
            - merged[
                "foreign_sell_val"
            ]
        )

    if (
        "foreign_buy_vol" in merged.columns
        and "foreign_sell_vol" in merged.columns
    ):

        merged[
            "foreign_net_vol_calc"
        ] = (
            merged[
                "foreign_buy_vol"
            ]
            - merged[
                "foreign_sell_vol"
            ]
        )

    # ========================================================
    # PROPRIETARY NET
    # ========================================================

    if (
        "proprietary_buy_val" in merged.columns
        and "proprietary_sell_val" in merged.columns
    ):

        merged[
            "proprietary_net_val_calc"
        ] = (
            merged[
                "proprietary_buy_val"
            ]
            - merged[
                "proprietary_sell_val"
            ]
        )

    if (
        "proprietary_buy_vol" in merged.columns
        and "proprietary_sell_vol" in merged.columns
    ):

        merged[
            "proprietary_net_vol_calc"
        ] = (
            merged[
                "proprietary_buy_vol"
            ]
            - merged[
                "proprietary_sell_vol"
            ]
        )

    # ========================================================
    # FLOW MOMENTUM
    # ========================================================

    for column in list(
        merged.columns
    ):

        if any(
            token in str(column)
            for token in [
                "net_val",
                "net_vol",
                "buy_val",
                "sell_val",
            ]
        ):

            merged[
                f"{column}_chg"
            ] = (
                merged[
                    column
                ]
                .pct_change()
            )

            merged[
                f"{column}_sma20"
            ] = (
                merged[
                    column
                ]
                .rolling(
                    20
                )
                .mean()
            )

    return merged.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# LISTING
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_LISTING,
    show_spinner=False,
)
def _load_listing():

    if not LISTING_AVAILABLE or Listing is None:

        return pd.DataFrame()

    try:

        listing = Listing(
            source="VCI"
        )

        df = listing.all_symbols(
            to_df=True
        )

        if (
            isinstance(
                df,
                pd.DataFrame,
            )
            and not df.empty
        ):

            return df

    except Exception:
        pass

    return pd.DataFrame()


def _get_stock_metadata(
    symbol,
):

    listing = _load_listing()

    if listing.empty:

        return {
            "symbol": normalize_symbol(symbol),
            "sector": "",
            "icb_code": "",
            "icb_code4": "",
        }

    symbol_column = _find_column(
        listing,
        [
            "symbol",
            "ticker",
            "code",
        ],
    )

    if symbol_column is None:

        return {
            "symbol": normalize_symbol(symbol),
            "sector": "",
            "icb_code": "",
            "icb_code4": "",
        }

    target = normalize_symbol(
        symbol
    )

    values = (
        listing[
            symbol_column
        ]
        .astype(str)
        .str.upper()
        .str.replace(
            ".VN",
            "",
            regex=False,
        )
    )

    rows = listing[
        values == target
    ]

    if rows.empty:

        return {
            "symbol": target,
            "sector": "",
            "icb_code": "",
            "icb_code4": "",
        }

    row = rows.iloc[
        0
    ]

    sector_column = _find_column(
        listing,
        [
            "icb_name",
            "industry_name",
            "industry",
            "sector",
        ],
    )

    icb_column = _find_column(
        listing,
        [
            "icb_code",
            "industry_code",
        ],
    )

    icb4_column = _find_column(
        listing,
        [
            "icb_code4",
            "industry_code4",
        ],
    )

    sector = ""

    if sector_column is not None:

        value = row.get(
            sector_column
        )

        if pd.notna(
            value
        ):

            sector = str(
                value
            ).strip()

    icb_code = ""

    if icb_column is not None:

        value = row.get(
            icb_column
        )

        if pd.notna(
            value
        ):

            icb_code = str(
                value
            ).strip()

    icb_code4 = ""

    if icb4_column is not None:

        value = row.get(
            icb4_column
        )

        if pd.notna(
            value
        ):

            icb_code4 = str(
                value
            ).strip()

    return {
        "symbol": target,
        "sector": sector,
        "icb_code": icb_code,
        "icb_code4": icb_code4,
    }


# ============================================================
# SECTOR PEERS
# ============================================================

def _get_sector_peers(
    symbol,
    max_peers=5,
):

    listing = _load_listing()

    if listing.empty:

        return []

    symbol_column = _find_column(
        listing,
        [
            "symbol",
            "ticker",
            "code",
        ],
    )

    industry_column = _find_column(
        listing,
        [
            "icb_code4",
            "icb_code",
            "industry_code",
        ],
    )

    if (
        symbol_column is None
        or industry_column is None
    ):

        return []

    target = normalize_symbol(
        symbol
    )

    target_rows = listing[
        listing[
            symbol_column
        ]
        .astype(str)
        .str.upper()
        .str.replace(
            ".VN",
            "",
            regex=False,
        )
        .eq(
            target
        )
    ]

    if target_rows.empty:

        return []

    target_code = str(
        target_rows.iloc[
            0
        ][
            industry_column
        ]
    ).strip()

    if not target_code:

        return []

    peer_symbols = (
        listing[
            listing[
                industry_column
            ]
            .astype(str)
            .str.strip()
            .eq(
                target_code
            )
        ][
            symbol_column
        ]
        .astype(str)
        .str.upper()
        .str.replace(
            ".VN",
            "",
            regex=False,
        )
        .tolist()
    )

    peer_symbols = [
        peer
        for peer in peer_symbols
        if peer != target
    ]

    return peer_symbols[
        :max_peers
    ]


# ============================================================
# MARKET FACTOR HISTORY
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_RESEARCH,
    show_spinner=False,
)
def load_market_factor_history(
    start_date,
    end_date,
):

    market = _create_market()

    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    indices = {
        "VNINDEX": "market_vnindex",
        "VN30": "market_vn30",
        "HNXINDEX": "market_hnx",
    }

    result = None

    last_request = None

    for index_symbol, prefix in indices.items():

        _sleep_between_requests(
            last_request
        )

        request_start = time.monotonic()

        try:

            df = _request_index_ohlcv(
                market,
                index_symbol,
                start,
                end,
            )

        except Exception:

            last_request = request_start

            continue

        last_request = request_start

        if (
            df is None
            or df.empty
        ):
            continue

        try:

            df = _normalize_ohlcv(
                df,
                stock=False,
            )

        except Exception:

            continue

        close = df[
            "Close"
        ]

        temp = pd.DataFrame(
            index=df.index
        )

        temp[
            f"{prefix}_close"
        ] = close

        temp[
            f"{prefix}_return"
        ] = close.pct_change()

        temp[
            f"{prefix}_return_pct"
        ] = (
            temp[
                f"{prefix}_return"
            ]
            * 100
        )

        temp[
            f"{prefix}_momentum20"
        ] = (
            close
            / close.shift(
                20
            )
            - 1
        )

        temp[
            f"{prefix}_volatility20"
        ] = (
            temp[
                f"{prefix}_return"
            ]
            .rolling(
                20
            )
            .std()
            * np.sqrt(
                252
            )
        )

        temp[
            f"{prefix}_volume"
        ] = df[
            "Volume"
        ]

        if "Value" in df.columns:

            temp[
                f"{prefix}_value"
            ] = df[
                "Value"
            ]

        else:

            temp[
                f"{prefix}_value"
            ] = (
                df[
                    "Close"
                ]
                * df[
                    "Volume"
                ]
            )

        if result is None:

            result = temp

        else:

            result = result.join(
                temp,
                how="outer",
            )

    if result is None:

        return pd.DataFrame()

    result = (
        result
        .sort_index()
        .loc[
            start:end
        ]
    )

    # ========================================================
    # MARKET BREADTH PROXY
    #
    # Dùng số index return dương/âm từ các index có sẵn.
    # Đây là proxy, không phải breadth toàn sàn.
    # ========================================================

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# SECTOR FACTOR VIA PEERS
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_RESEARCH,
    show_spinner=False,
)
def load_sector_factor_history(
    symbol,
    start_date,
    end_date,
):

    peers = _get_sector_peers(
        symbol,
        max_peers=5,
    )

    if not peers:

        return pd.DataFrame()

    market = _create_market()

    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    peer_returns = []

    last_request = None

    successful_peers = 0

    for peer in peers:

        _sleep_between_requests(
            last_request
        )

        request_start = time.monotonic()

        try:

            df = _request_equity_ohlcv(
                market,
                peer,
                start,
                end,
            )

        except Exception:

            last_request = request_start

            continue

        last_request = request_start

        if (
            df is None
            or df.empty
        ):
            continue

        try:

            df = _normalize_ohlcv(
                df,
                stock=True,
            )

        except Exception:

            continue

        close = df[
            "Close"
        ]

        ret = close.pct_change()

        peer_returns.append(
            ret.rename(
                peer
            )
        )

        successful_peers += 1

    if not peer_returns:

        return pd.DataFrame()

    peers_df = pd.concat(
        peer_returns,
        axis=1,
    ).sort_index()

    result = pd.DataFrame(
        index=peers_df.index
    )

    # ========================================================
    # SECTOR DAILY GROWTH
    # ========================================================

    result[
        "sector_return"
    ] = peers_df.mean(
        axis=1,
        skipna=True,
    )

    result[
        "sector_return_pct"
    ] = (
        result[
            "sector_return"
        ]
        * 100
    )

    # ========================================================
    # SECTOR MOMENTUM
    # ========================================================

    result[
        "sector_momentum20"
    ] = (
        (
            1
            + result[
                "sector_return"
            ]
        )
        .rolling(
            20
        )
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    result[
        "sector_momentum60"
    ] = (
        (
            1
            + result[
                "sector_return"
            ]
        )
        .rolling(
            60
        )
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    # ========================================================
    # SECTOR VOLATILITY
    # ========================================================

    result[
        "sector_volatility20"
    ] = (
        result[
            "sector_return"
        ]
        .rolling(
            20
        )
        .std()
        * np.sqrt(
            252
        )
    )

    # ========================================================
    # SECTOR BREADTH
    # ========================================================

    result[
        "sector_positive_ratio"
    ] = (
        peers_df
        .gt(0)
        .sum(
            axis=1
        )
        / peers_df.notna()
        .sum(
            axis=1
        )
    )

    result[
        "sector_negative_ratio"
    ] = (
        peers_df
        .lt(0)
        .sum(
            axis=1
        )
        / peers_df.notna()
        .sum(
            axis=1
        )
    )

    result[
        "sector_peer_count"
    ] = (
        peers_df
        .notna()
        .sum(
            axis=1
        )
    )

    result.attrs[
        "peers_requested"
    ] = peers

    result.attrs[
        "peers_loaded"
    ] = successful_peers

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# MULTIFACTOR RESEARCH DATASET
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_RESEARCH,
    show_spinner=False,
)
def load_multifactor_research_history(
    symbol,
    start_date,
    end_date,
):
    """
    DATASET NGHIÊN CỨU ĐẦY ĐỦ:

    1. Technical factors của cổ phiếu.
    2. Trading flow.
    3. Foreign flow.
    4. Proprietary flow.
    5. Market factors.
    6. Sector factors.

    Tất cả được align theo ngày.
    """

    symbol = normalize_symbol(
        symbol
    )

    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    if start > end:

        raise ValueError(
            "Ngày bắt đầu phải nhỏ hơn ngày kết thúc."
        )

    # ========================================================
    # STOCK
    # ========================================================

    stock_raw = _load_stock_raw(
        symbol,
        start,
        end,
    )

    stock = add_indicators(
        stock_raw,
        la_co_phieu=True,
    )

    # ========================================================
    # STOCK META
    # ========================================================

    metadata = _get_stock_metadata(
        symbol
    )

    # ========================================================
    # FLOWS
    # ========================================================

    flow = load_stock_flow_history(
        symbol,
        start,
        end,
    )

    # ========================================================
    # MARKET
    # ========================================================

    market = load_market_factor_history(
        start,
        end,
    )

    # ========================================================
    # SECTOR
    # ========================================================

    sector = load_sector_factor_history(
        symbol,
        start,
        end,
    )

    result = stock.copy()

    # ========================================================
    # MERGE FLOW
    # ========================================================

    if (
        isinstance(
            flow,
            pd.DataFrame,
        )
        and not flow.empty
    ):

        flow = flow[
            ~flow.index.duplicated(
                keep="last"
            )
        ]

        result = result.join(
            flow,
            how="left",
        )

    # ========================================================
    # MERGE MARKET
    # ========================================================

    if (
        isinstance(
            market,
            pd.DataFrame,
        )
        and not market.empty
    ):

        market = market[
            ~market.index.duplicated(
                keep="last"
            )
        ]

        result = result.join(
            market,
            how="left",
        )

    # ========================================================
    # MERGE SECTOR
    # ========================================================

    if (
        isinstance(
            sector,
            pd.DataFrame,
        )
        and not sector.empty
    ):

        sector = sector[
            ~sector.index.duplicated(
                keep="last"
            )
        ]

        # Không merge peer raw.
        result = result.join(
            sector[
                [
                    column
                    for column in sector.columns
                    if column.startswith(
                        "sector_"
                    )
                ]
            ],
            how="left",
        )

    # ========================================================
    # CROSS MARKET / STOCK FACTORS
    # ========================================================

    if (
        "market_vnindex_return"
        in result.columns
    ):

        result[
            "stock_minus_market_1d"
        ] = (
            result[
                "Return"
            ]
            - result[
                "market_vnindex_return"
            ]
        )

    if (
        "market_vnindex_momentum20"
        in result.columns
    ):

        result[
            "stock_minus_market_momentum20"
        ] = (
            result[
                "Momentum20"
            ]
            - result[
                "market_vnindex_momentum20"
            ]
        )

    if (
        "sector_return"
        in result.columns
    ):

        result[
            "stock_minus_sector_1d"
        ] = (
            result[
                "Return"
            ]
            - result[
                "sector_return"
            ]
        )

    if (
        "sector_momentum20"
        in result.columns
    ):

        result[
            "stock_minus_sector_momentum20"
        ] = (
            result[
                "Momentum20"
            ]
            - result[
                "sector_momentum20"
            ]
        )

    # ========================================================
    # FLOW RATIOS
    # ========================================================

    ratio_pairs = [
        (
            "foreign_buy_val",
            "foreign_sell_val",
            "foreign_buy_sell_value_ratio",
        ),
        (
            "proprietary_buy_val",
            "proprietary_sell_val",
            "proprietary_buy_sell_value_ratio",
        ),
        (
            "foreign_buy_vol",
            "foreign_sell_vol",
            "foreign_buy_sell_volume_ratio",
        ),
        (
            "proprietary_buy_vol",
            "proprietary_sell_vol",
            "proprietary_buy_sell_volume_ratio",
        ),
    ]

    for buy_column, sell_column, output_column in ratio_pairs:

        if (
            buy_column in result.columns
            and sell_column in result.columns
        ):

            denominator = (
                result[
                    sell_column
                ]
                .abs()
                .replace(
                    0,
                    np.nan,
                )
            )

            result[
                output_column
            ] = (
                result[
                    buy_column
                ]
                / denominator
            )

    # ========================================================
    # FLOW NET / TRADING VALUE RATIOS
    # ========================================================

    flow_ratio_pairs = [
        (
            "foreign_net_val_calc",
            "Trading_Value",
            "foreign_net_value_to_trading_value",
        ),
        (
            "proprietary_net_val_calc",
            "Trading_Value",
            "proprietary_net_value_to_trading_value",
        ),
    ]

    for numerator, denominator, output in flow_ratio_pairs:

        if (
            numerator in result.columns
            and denominator in result.columns
        ):

            result[
                output
            ] = (
                result[
                    numerator
                ]
                / result[
                    denominator
                ].replace(
                    0,
                    np.nan,
                )
            )

    # ========================================================
    # MARKET BREADTH ACROSS AVAILABLE INDICES
    # ========================================================

    market_return_columns = [
        column
        for column in result.columns
        if (
            column.endswith(
                "_return"
            )
            and column.startswith(
                "market_"
            )
        )
    ]

    if market_return_columns:

        result[
            "market_positive_index_count"
        ] = (
            result[
                market_return_columns
            ]
            .gt(0)
            .sum(
                axis=1
            )
        )

        result[
            "market_negative_index_count"
        ] = (
            result[
                market_return_columns
            ]
            .lt(0)
            .sum(
                axis=1
            )
        )

        result[
            "market_average_return"
        ] = (
            result[
                market_return_columns
            ]
            .mean(
                axis=1
            )
        )

    # ========================================================
    # TARGETS
    #
    # Đây là biến phụ thuộc cho nghiên cứu.
    # ========================================================

    close = result[
        "Close"
    ]

    for horizon, periods in {
        "1D": 1,
        "5D": 5,
        "20D": 20,
    }.items():

        result[
            f"Target_{horizon}"
        ] = (
            close.shift(
                -periods
            )
            / close
            - 1
        )

        result[
            f"Target_{horizon}_Pct"
        ] = (
            result[
                f"Target_{horizon}"
            ]
            * 100
        )

    # ========================================================
    # ATTRS
    # ========================================================

    result.attrs[
        "symbol"
    ] = symbol

    result.attrs[
        "display_symbol"
    ] = display_symbol(
        symbol
    )

    result.attrs[
        "source"
    ] = "Vnstock multifactor"

    result.attrs[
        "research_start"
    ] = str(
        start.date()
    )

    result.attrs[
        "research_end"
    ] = str(
        end.date()
    )

    result.attrs[
        "sector"
    ] = metadata.get(
        "sector",
        "",
    )

    result.attrs[
        "icb_code"
    ] = metadata.get(
        "icb_code",
        "",
    )

    result.attrs[
        "icb_code4"
    ] = metadata.get(
        "icb_code4",
        "",
    )

    result.attrs[
        "sector_peers"
    ] = sector.attrs.get(
        "peers_requested",
        [],
    )

    result.attrs[
        "sector_peers_loaded"
    ] = sector.attrs.get(
        "peers_loaded",
        0,
    )

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# RESEARCH HISTORY COMPATIBILITY
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_RESEARCH,
    show_spinner=False,
)
def load_research_history(
    symbol,
    start_date,
    end_date,
):

    data = load_multifactor_research_history(
        symbol,
        start_date,
        end_date,
    )

    return data


# ============================================================
# MARKET DATA COMPATIBILITY
# ============================================================

def market_data(
    symbol,
    period="1y",
):

    return load_market_data(
        symbol,
        period,
    )


# ============================================================
# NEWS SENTIMENT
# ============================================================

def classify_news(
    title,
):

    text = str(
        title or ""
    ).lower()

    positive_words = [
        "tăng",
        "tích cực",
        "lợi nhuận",
        "kỷ lục",
        "tăng trưởng",
        "bứt phá",
        "vượt kỳ vọng",
        "vượt kế hoạch",
        "hưởng lợi",
        "cải thiện",
    ]

    negative_words = [
        "giảm",
        "tiêu cực",
        "thua lỗ",
        "rủi ro",
        "sụt giảm",
        "áp lực",
        "bán tháo",
        "khó khăn",
        "nợ xấu",
        "cảnh báo",
        "điều tra",
        "vi phạm",
    ]

    positive = sum(
        word in text
        for word in positive_words
    )

    negative = sum(
        word in text
        for word in negative_words
    )

    if positive > negative:
        return "positive"

    if negative > positive:
        return "negative"

    return "neutral"


# ============================================================
# LEGACY OLS
# ============================================================

def run_ols(
    data,
):

    if (
        data is None
        or data.empty
    ):
        return None

    try:

        import statsmodels.api as sm

    except Exception:
        return None

    candidate = [
        "RSI",
        "MACD",
        "MACD_Hist",
        "Volatility20",
        "Volume_Change",
        "Momentum20",
        "Range_Percent",
        "Relative_Volume",
        "market_vnindex_return",
        "market_vnindex_momentum20",
        "sector_return",
        "sector_momentum20",
        "foreign_net_val_calc",
        "proprietary_net_val_calc",
    ]

    features = [
        column
        for column in candidate
        if column in data.columns
    ]

    if not features:
        return None

    dataset = data[
        features
        + ["Return"]
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if len(dataset) < 50:
        return None

    try:

        X = sm.add_constant(
            dataset[
                features
            ],
            has_constant="add",
        )

        y = dataset[
            "Return"
        ]

        return (
            sm.OLS(
                y,
                X,
            )
            .fit(
                cov_type="HC3"
            )
        )

    except Exception:

        return None


# ============================================================
# RANDOM FOREST
# ============================================================

def run_random_forest(
    data,
):

    if (
        data is None
        or data.empty
    ):
        return None

    try:

        from sklearn.ensemble import (
            RandomForestRegressor,
        )

    except Exception:
        return None

    excluded = {
        "Target_1D",
        "Target_5D",
        "Target_20D",
        "Target_1D_Pct",
        "Target_5D_Pct",
        "Target_20D_Pct",
    }

    features = [
        column
        for column in data.columns
        if (
            column not in excluded
            and pd.api.types.is_numeric_dtype(
                data[column]
            )
        )
    ]

    if not features:
        return None

    work = data[
        features
        + ["Target_1D"]
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    X = work[
        features
    ].copy()

    y = work[
        "Target_1D"
    ].copy()

    for column in X.columns:

        median = X[
            column
        ].median()

        if pd.isna(
            median
        ):
            median = 0.0

        X[
            column
        ] = X[
            column
        ].fillna(
            median
        )

    valid = y.notna()

    X = X.loc[
        valid
    ]

    y = y.loc[
        valid
    ]

    if len(X) < 80:
        return None

    split = int(
        len(X)
        * 0.8
    )

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

        model.fit(
            X.iloc[
                :split
            ],
            y.iloc[
                :split
            ],
        )

        prediction = float(
            model.predict(
                X.iloc[
                    [
                        -1
                    ]
                ]
            )[0]
        )

        importance = dict(
            zip(
                features,
                model.feature_importances_,
            )
        )

        return {
            "model": model,
            "prediction": prediction,
            "importance": importance,
        }

    except Exception:

        return None


# ============================================================
# BUILD QUANT — FULL FACTOR VERSION
# ============================================================

def build_quant(
    data,
):

    if (
        data is None
        or data.empty
    ):
        return None

    try:

        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )

        from sklearn.linear_model import (
            ElasticNet,
            Lasso,
            Ridge,
        )

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        from sklearn.pipeline import (
            Pipeline,
        )

        from sklearn.preprocessing import (
            StandardScaler,
        )

    except Exception:
        return None

    # ========================================================
    # ALL NUMERIC FACTORS
    # ========================================================

    excluded = {
        "Target_1D",
        "Target_5D",
        "Target_20D",
        "Target_1D_Pct",
        "Target_5D_Pct",
        "Target_20D_Pct",
    }

    features = [
        column
        for column in data.columns
        if (
            column not in excluded
            and pd.api.types.is_numeric_dtype(
                data[column]
            )
        )
    ]

    # Không dùng target/current future leakage.
    leakage_names = {
        column
        for column in features
        if str(
            column
        ).startswith(
            "Target_"
        )
    }

    features = [
        column
        for column in features
        if column not in leakage_names
    ]

    if not features:

        return None

    results = {}

    # ========================================================
    # THREE HORIZONS
    # ========================================================

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        target_column = (
            f"Target_{horizon}"
        )

        if target_column not in data.columns:
            continue

        work = data[
            features
            + [
                target_column
            ]
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).copy()

        X = work[
            features
        ].copy()

        y = work[
            target_column
        ].copy()

        # Median fill.
        for column in X.columns:

            median = X[
                column
            ].median()

            if pd.isna(
                median
            ):
                median = 0.0

            X[
                column
            ] = X[
                column
            ].fillna(
                median
            )

        valid = y.notna()

        X = X.loc[
            valid
        ]

        y = y.loc[
            valid
        ]

        if len(X) < 60:
            continue

        split = int(
            len(X)
            * 0.8
        )

        if (
            split < 30
            or len(X) - split < 10
        ):
            continue

        X_train = X.iloc[
            :split
        ]

        X_test = X.iloc[
            split:
        ]

        y_train = y.iloc[
            :split
        ]

        y_test = y.iloc[
            split:
        ]

        models = [
            (
                "Ridge",
                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                        (
                            "model",
                            Ridge(
                                alpha=1.0
                            ),
                        ),
                    ]
                ),
            ),
            (
                "Lasso",
                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                        (
                            "model",
                            Lasso(
                                alpha=0.0001,
                                max_iter=50000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "Elastic Net",
                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                        (
                            "model",
                            ElasticNet(
                                alpha=0.0001,
                                l1_ratio=0.5,
                                max_iter=50000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "Extra Trees",
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "Gradient Boosting",
                GradientBoostingRegressor(
                    n_estimators=250,
                    learning_rate=0.03,
                    max_depth=3,
                    min_samples_leaf=3,
                    random_state=42,
                ),
            ),
        ]

        rows = []

        fitted = {}

        predictions = {}

        for name, model in models:

            try:

                model.fit(
                    X_train,
                    y_train,
                )

                pred = np.asarray(
                    model.predict(
                        X_test
                    ),
                    dtype=float,
                )

                actual = np.asarray(
                    y_test,
                    dtype=float,
                )

                mae = float(
                    mean_absolute_error(
                        actual,
                        pred,
                    )
                )

                mse = float(
                    mean_squared_error(
                        actual,
                        pred,
                    )
                )

                rmse = float(
                    np.sqrt(
                        mse
                    )
                )

                try:

                    r2 = float(
                        r2_score(
                            actual,
                            pred,
                        )
                    )

                except Exception:

                    r2 = np.nan

                rows.append(
                    {
                        "Mô hình": name,
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R²": r2,
                    }
                )

                fitted[
                    name
                ] = model

                predictions[
                    name
                ] = pred

            except Exception:
                continue

        models_df = (
            pd.DataFrame(
                rows
            )
            .sort_values(
                "RMSE",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

        # ====================================================
        # FEATURE IMPORTANCE BEST TREE
        # ====================================================

        tree_importance = pd.DataFrame()

        if (
            not models_df.empty
        ):

            best_name = str(
                models_df.iloc[
                    0
                ][
                    "Mô hình"
                ]
            )

            best_model = fitted.get(
                best_name
            )

            if best_model is not None:

                raw_model = best_model

                if hasattr(
                    best_model,
                    "named_steps",
                ):

                    raw_model = (
                        best_model
                        .named_steps
                        .get(
                            "model",
                            best_model,
                        )
                    )

                if hasattr(
                    raw_model,
                    "feature_importances_",
                ):

                    tree_importance = (
                        pd.DataFrame(
                            {
                                "Biến": X.columns,
                                "Importance": (
                                    raw_model
                                    .feature_importances_
                                ),
                            }
                        )
                        .sort_values(
                            "Importance",
                            ascending=False,
                        )
                        .reset_index(
                            drop=True
                        )
                    )

        results[
            horizon
        ] = {
            "observations": len(X),
            "train": len(X_train),
            "test": len(X_test),
            "features": features,
            "models": models_df,
            "fitted": fitted,
            "predictions": predictions,
            "tree_importance": tree_importance,
        }

    if not results:
        return None

    return results


# ============================================================
# SNAPSHOT FACTOR SUMMARY
# ============================================================

def research_factor_groups(
    data,
):

    if (
        data is None
        or data.empty
    ):

        return {
            "technical": [],
            "flow": [],
            "foreign": [],
            "proprietary": [],
            "market": [],
            "sector": [],
        }

    columns = [
        str(column)
        for column in data.columns
    ]

    return {
        "technical": [
            column
            for column in columns
            if not (
                column.startswith(
                    "market_"
                )
                or column.startswith(
                    "sector_"
                )
                or column.startswith(
                    "foreign_"
                )
                or column.startswith(
                    "proprietary_"
                )
                or column.startswith(
                    "flow_"
                )
                or column.startswith(
                    "Target_"
                )
            )
        ],
        "flow": [
            column
            for column in columns
            if (
                column.startswith(
                    "flow_"
                )
                or "Trading_Value" in column
            )
        ],
        "foreign": [
            column
            for column in columns
            if column.startswith(
                "foreign_"
            )
        ],
        "proprietary": [
            column
            for column in columns
            if column.startswith(
                "proprietary_"
            )
        ],
        "market": [
            column
            for column in columns
            if column.startswith(
                "market_"
            )
        ],
        "sector": [
            column
            for column in columns
            if column.startswith(
                "sector_"
            )
        ],
    }


# ============================================================
# FACTOR IMPORTANCE — CORRELATION
# ============================================================

def factor_correlation_table(
    data,
    target_column,
):

    if (
        data is None
        or data.empty
        or target_column not in data.columns
    ):

        return pd.DataFrame()

    excluded = {
        "Target_1D",
        "Target_5D",
        "Target_20D",
        "Target_1D_Pct",
        "Target_5D_Pct",
        "Target_20D_Pct",
    }

    rows = []

    y = pd.to_numeric(
        data[
            target_column
        ],
        errors="coerce",
    )

    for column in data.columns:

        if column == target_column:
            continue

        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            data[
                column
            ]
        ):
            continue

        x = pd.to_numeric(
            data[
                column
            ],
            errors="coerce",
        )

        valid = (
            x.notna()
            & y.notna()
        )

        x_valid = x.loc[
            valid
        ]

        y_valid = y.loc[
            valid
        ]

        if len(
            x_valid
        ) < 20:
            continue

        try:

            pearson = float(
                x_valid.corr(
                    y_valid,
                    method="pearson",
                )
            )

        except Exception:

            pearson = np.nan

        try:

            spearman = float(
                x_valid.corr(
                    y_valid,
                    method="spearman",
                )
            )

        except Exception:

            spearman = np.nan

        rows.append(
            {
                "Biến": column,
                "Pearson": pearson,
                "Spearman": spearman,
                "|Spearman|": (
                    abs(
                        spearman
                    )
                    if np.isfinite(
                        spearman
                    )
                    else np.nan
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "|Spearman|",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FINAL FACTOR DATASET
# ============================================================

def build_research_factor_dataset(
    data,
    target_column="Target_20D",
):

    correlation = factor_correlation_table(
        data,
        target_column,
    )

    return {
        "data": data,
        "correlation": correlation,
        "groups": research_factor_groups(
            data
        ),
    }


# ============================================================
# ALIASES FOR FUTURE PAGES
# ============================================================

def get_research_data(
    symbol,
    start_date,
    end_date,
):

    return load_multifactor_research_history(
        symbol,
        start_date,
        end_date,
    )


def load_full_research_data(
    symbol,
    start_date,
    end_date,
):

    return load_multifactor_research_history(
        symbol,
        start_date,
        end_date,
    )
