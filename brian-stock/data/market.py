from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# VNSTOCK
# ============================================================

try:
    from vnstock import Market

    NGUON_VNSTOCK_CO_SAN = True

except Exception:
    Market = None
    NGUON_VNSTOCK_CO_SAN = False


# ============================================================
# CACHE
# ============================================================

THOI_GIAN_LUU_CO_PHIEU = 300
THOI_GIAN_LUU_VNINDEX = 300
THOI_GIAN_LUU_GIA_MOI = 30
THOI_GIAN_LUU_TONG_QUAN_VNINDEX = 30
THOI_GIAN_LUU_NGHIEN_CUU = 900


# ============================================================
# RATE LIMIT
# ============================================================

# Không dùng 100 ngày nữa.
# 60 ngày lịch thường tương đương khoảng 40-45 phiên.
SO_NGAY_MOI_CHUNK_NGHIEN_CUU = 60

# 1.6 giây/request ≈ 37.5 request/phút.
# Có khoảng đệm so với giới hạn 40 request/phút.
THOI_GIAN_CHO_GIUA_REQUEST = 1.6

# Retry tối đa khi nguồn trả lỗi tạm thời.
SO_LAN_RETRY_RATE_LIMIT = 3

# Thời gian chờ khi retry.
THOI_GIAN_RETRY_CO_BAN = 5.0


# ============================================================
# CACHE RESOURCE - MARKET
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def _tao_nguon_thi_truong():

    if not NGUON_VNSTOCK_CO_SAN:

        raise RuntimeError(
            "Chưa cài vnstock. "
            "Hãy thêm vnstock vào requirements.txt."
        )

    try:

        return Market()

    except Exception as loi:

        raise RuntimeError(
            f"Không khởi tạo được nguồn dữ liệu: {loi}"
        ) from loi


# ============================================================
# CHUẨN HÓA MÃ
# ============================================================

def normalize_symbol(
    symbol,
):

    if symbol is None:
        return "HPG"

    ma = (
        str(symbol)
        .strip()
        .upper()
    )

    ma = re.sub(
        r"\s+",
        "",
        ma,
    )

    if ma.endswith(".VN"):

        ma = ma[:-3]

    if not ma:

        return "HPG"

    return ma


def display_symbol(
    symbol,
):

    if symbol is None:
        return ""

    ma = (
        str(symbol)
        .strip()
        .upper()
    )

    if ma.endswith(".VN"):

        ma = ma[:-3]

    return ma


# ============================================================
# TIỆN ÍCH SỐ
# ============================================================

def _so(
    value: Any,
    mac_dinh=np.nan,
):

    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):

            return mac_dinh

        return value

    except Exception:

        return mac_dinh


# ============================================================
# CHUẨN HÓA TÊN CỘT
# ============================================================

def _ten_cot_chuan(
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
# CHUẨN HÓA DATAFRAME OHLCV
# ============================================================

def _chuan_hoa_bang_gia(
    du_lieu,
    la_co_phieu=False,
):

    if du_lieu is None:

        raise ValueError(
            "Nguồn dữ liệu trả về rỗng."
        )

    if not isinstance(
        du_lieu,
        pd.DataFrame,
    ):

        du_lieu = pd.DataFrame(
            du_lieu
        )

    if du_lieu.empty:

        raise ValueError(
            "Nguồn dữ liệu không có dữ liệu."
        )

    du_lieu = du_lieu.copy()

    # ========================================================
    # MULTIINDEX
    # ========================================================

    if isinstance(
        du_lieu.columns,
        pd.MultiIndex,
    ):

        columns = []

        for column in du_lieu.columns:

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
                    str(column)
                )

        du_lieu.columns = columns

    # ========================================================
    # ÁNH XẠ CỘT
    # ========================================================

    mapping = {}

    for column in du_lieu.columns:

        name = _ten_cot_chuan(
            column
        )

        if name in {
            "time",
            "date",
            "datetime",
            "timestamp",
            "tradingdate",
            "trading_date",
        }:

            mapping[
                column
            ] = "Time"

        elif name in {
            "open",
            "open_price",
            "openprice",
        }:

            mapping[
                column
            ] = "Open"

        elif name in {
            "high",
            "high_price",
            "highprice",
        }:

            mapping[
                column
            ] = "High"

        elif name in {
            "low",
            "low_price",
            "lowprice",
        }:

            mapping[
                column
            ] = "Low"

        elif name in {
            "close",
            "close_price",
            "closeprice",
            "last",
            "lastprice",
            "last_price",
        }:

            mapping[
                column
            ] = "Close"

        elif name in {
            "volume",
            "vol",
            "total_volume",
            "totalvolume",
            "matchvolume",
            "match_volume",
        }:

            mapping[
                column
            ] = "Volume"

        elif name in {
            "value",
            "trading_value",
            "value_traded",
            "matchvalue",
            "match_value",
            "turnover",
            "total_value",
            "totalvalue",
        }:

            mapping[
                column
            ] = "Value"

    du_lieu = du_lieu.rename(
        columns=mapping
    )

    # ========================================================
    # TIME INDEX
    # ========================================================

    if "Time" in du_lieu.columns:

        du_lieu[
            "Time"
        ] = pd.to_datetime(
            du_lieu[
                "Time"
            ],
            errors="coerce",
        )

        du_lieu = (
            du_lieu
            .set_index(
                "Time"
            )
        )

    else:

        du_lieu.index = pd.to_datetime(
            du_lieu.index,
            errors="coerce",
        )

    du_lieu = du_lieu[
        ~du_lieu.index.isna()
    ].copy()

    du_lieu = (
        du_lieu
        .sort_index()
    )

    du_lieu = (
        du_lieu[
            ~du_lieu.index.duplicated(
                keep="last"
            )
        ]
        .copy()
    )

    # ========================================================
    # OHLC BẮT BUỘC
    # ========================================================

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in du_lieu.columns
    ]

    if missing:

        raise ValueError(
            "Thiếu cột dữ liệu: "
            + ", ".join(
                missing
            )
        )

    # ========================================================
    # VOLUME
    # ========================================================

    if "Volume" not in du_lieu.columns:

        du_lieu[
            "Volume"
        ] = 0.0

    # ========================================================
    # NUMERIC
    # ========================================================

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

        du_lieu[
            column
        ] = pd.to_numeric(
            du_lieu[
                column
            ],
            errors="coerce",
        )

    if "Value" in du_lieu.columns:

        du_lieu[
            "Value"
        ] = pd.to_numeric(
            du_lieu[
                "Value"
            ],
            errors="coerce",
        )

    # ========================================================
    # CHUẨN HÓA GIÁ CỔ PHIẾU
    # ========================================================

    if la_co_phieu:

        median_price = (
            du_lieu[
                "Close"
            ]
            .dropna()
            .median()
        )

        # vnstock có thể trả giá theo nghìn đồng.
        # Ví dụ 22.2 -> 22,200.
        #
        # VNINDEX tuyệt đối không đi qua nhánh này.
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

                du_lieu[
                    column
                ] = (
                    du_lieu[
                        column
                    ]
                    * 1000
                )

    # ========================================================
    # LOẠI GIÁ LỖI
    # ========================================================

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        du_lieu.loc[
            du_lieu[
                column
            ] <= 0,
            column,
        ] = np.nan

    du_lieu.loc[
        du_lieu[
            "Volume"
        ] < 0,
        "Volume",
    ] = np.nan

    # ========================================================
    # INF
    # ========================================================

    du_lieu = du_lieu.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # ========================================================
    # OHLC HỢP LỆ
    # ========================================================

    du_lieu = du_lieu.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    if du_lieu.empty:

        raise ValueError(
            "Không còn dữ liệu OHLC hợp lệ."
        )

    return du_lieu


# ============================================================
# RSI
# ============================================================

def rsi(
    chuoi_gia,
    chu_ky=14,
):

    chuoi_gia = pd.to_numeric(
        chuoi_gia,
        errors="coerce",
    )

    thay_doi = (
        chuoi_gia.diff()
    )

    tang = (
        thay_doi.clip(
            lower=0
        )
    )

    giam = (
        -thay_doi.clip(
            upper=0
        )
    )

    avg_gain = (
        tang
        .ewm(
            alpha=1 / chu_ky,
            adjust=False,
            min_periods=chu_ky,
        )
        .mean()
    )

    avg_loss = (
        giam
        .ewm(
            alpha=1 / chu_ky,
            adjust=False,
            min_periods=chu_ky,
        )
        .mean()
    )

    rs = (
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
                + rs
            )
        )
    )

    result = result.where(
        ~(
            (
                avg_loss
                == 0
            )
            & (
                avg_gain
                > 0
            )
        ),
        100,
    )

    result = result.where(
        ~(
            (
                avg_loss
                == 0
            )
            & (
                avg_gain
                == 0
            )
        ),
        50,
    )

    return result


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    du_lieu,
    la_co_phieu=False,
):

    du_lieu = (
        _chuan_hoa_bang_gia(
            du_lieu,
            la_co_phieu=la_co_phieu,
        )
        .copy()
    )

    gia = du_lieu[
        "Close"
    ]

    # ========================================================
    # RETURN
    # ========================================================

    du_lieu[
        "Return"
    ] = gia.pct_change()

    du_lieu[
        "ReturnPct"
    ] = (
        du_lieu[
            "Return"
        ]
        * 100
    )

    # ========================================================
    # RSI
    # ========================================================

    du_lieu[
        "RSI"
    ] = rsi(
        gia,
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

        du_lieu[
            f"EMA{period}"
        ] = (
            gia
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

        du_lieu[
            f"SMA{period}"
        ] = (
            gia
            .rolling(
                period
            )
            .mean()
        )

    # ========================================================
    # MACD
    # ========================================================

    du_lieu[
        "MACD"
    ] = (
        du_lieu[
            "EMA12"
        ]
        - du_lieu[
            "EMA26"
        ]
    )

    du_lieu[
        "MACD_Signal"
    ] = (
        du_lieu[
            "MACD"
        ]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    du_lieu[
        "MACD_Hist"
    ] = (
        du_lieu[
            "MACD"
        ]
        - du_lieu[
            "MACD_Signal"
        ]
    )

    # ========================================================
    # BOLLINGER
    # ========================================================

    std20 = (
        gia
        .rolling(
            20
        )
        .std()
    )

    du_lieu[
        "Bollinger_Mid"
    ] = (
        du_lieu[
            "SMA20"
        ]
    )

    du_lieu[
        "Bollinger_Upper"
    ] = (
        du_lieu[
            "SMA20"
        ]
        + 2 * std20
    )

    du_lieu[
        "Bollinger_Lower"
    ] = (
        du_lieu[
            "SMA20"
        ]
        - 2 * std20
    )

    du_lieu[
        "Bollinger_Width"
    ] = (
        (
            du_lieu[
                "Bollinger_Upper"
            ]
            - du_lieu[
                "Bollinger_Lower"
            ]
        )
        / du_lieu[
            "Bollinger_Mid"
        ]
        * 100
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    for period in [
        5,
        20,
        60,
    ]:

        du_lieu[
            f"Volatility{period}"
        ] = (
            du_lieu[
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

    du_lieu[
        "Volatility_20D"
    ] = (
        du_lieu[
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

        du_lieu[
            f"Volume_SMA{period}"
        ] = (
            du_lieu[
                "Volume"
            ]
            .rolling(
                period
            )
            .mean()
        )

    du_lieu[
        "Volume_Change"
    ] = (
        du_lieu[
            "Volume"
        ]
        .pct_change()
    )

    du_lieu[
        "Relative_Volume"
    ] = (
        du_lieu[
            "Volume"
        ]
        / du_lieu[
            "Volume_SMA20"
        ]
    )

    # ========================================================
    # RANGE
    # ========================================================

    du_lieu[
        "Range"
    ] = (
        du_lieu[
            "High"
        ]
        - du_lieu[
            "Low"
        ]
    )

    du_lieu[
        "Range_Percent"
    ] = (
        du_lieu[
            "Range"
        ]
        / du_lieu[
            "Close"
        ]
        * 100
    )

    # ========================================================
    # ATR14
    # ========================================================

    tr1 = (
        du_lieu[
            "High"
        ]
        - du_lieu[
            "Low"
        ]
    )

    tr2 = (
        du_lieu[
            "High"
        ]
        - du_lieu[
            "Close"
        ].shift(
            1
        )
    ).abs()

    tr3 = (
        du_lieu[
            "Low"
        ]
        - du_lieu[
            "Close"
        ].shift(
            1
        )
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(
        axis=1
    )

    du_lieu[
        "ATR14"
    ] = (
        true_range
        .rolling(
            14
        )
        .mean()
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    for period in [
        5,
        10,
        20,
    ]:

        du_lieu[
            f"Momentum{period}"
        ] = (
            gia
            / gia.shift(
                period
            )
            - 1
        )

    # ========================================================
    # HIGH / LOW
    # ========================================================

    for period in [
        20,
        50,
        252,
    ]:

        du_lieu[
            f"High{period}"
        ] = (
            du_lieu[
                "High"
            ]
            .rolling(
                period
            )
            .max()
        )

        du_lieu[
            f"Low{period}"
        ] = (
            du_lieu[
                "Low"
            ]
            .rolling(
                period
            )
            .min()
        )

        du_lieu[
            f"Distance_From_High{period}"
        ] = (
            (
                gia
                / du_lieu[
                    f"High{period}"
                ]
                - 1
            )
            * 100
        )

        du_lieu[
            f"Distance_From_Low{period}"
        ] = (
            (
                gia
                / du_lieu[
                    f"Low{period}"
                ]
                - 1
            )
            * 100
        )

    return du_lieu.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# SỐ NGÀY THEO KỲ
# ============================================================

def _so_ngay_theo_ky(
    ky_hieu,
):

    ky_hieu = str(
        ky_hieu or "1y"
    ).strip().lower()

    bang_ngay = {
        "1d": 3,
        "5d": 10,
        "1w": 14,
        "1mo": 45,
        "3mo": 120,
        "6mo": 240,
        "1y": 450,
        "2y": 850,
        "3y": 1250,
        "5y": 1950,
        "10y": 3900,
        "max": 5000,
    }

    return bang_ngay.get(
        ky_hieu,
        450,
    )


# ============================================================
# REQUEST OHLCV
# ============================================================

def _request_equity_ohlcv(
    nguon,
    ma,
    start_date,
    end_date,
):

    start = pd.Timestamp(
        start_date
    )

    end = pd.Timestamp(
        end_date
    )

    return (
        nguon
        .equity(
            ma
        )
        .ohlcv(
            start=start.strftime(
                "%Y-%m-%d"
            ),
            end=(
                end
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
# REQUEST CÓ RETRY RATE LIMIT
# ============================================================

def _request_equity_ohlcv_safe(
    nguon,
    ma,
    start_date,
    end_date,
):

    last_error = None

    for attempt in range(
        SO_LAN_RETRY_RATE_LIMIT + 1
    ):

        try:

            return _request_equity_ohlcv(
                nguon,
                ma,
                start_date,
                end_date,
            )

        except Exception as error:

            last_error = error

            error_text = str(
                error
            ).lower()

            is_rate_limit = any(
                phrase in error_text
                for phrase in [
                    "rate limit",
                    "too many requests",
                    "429",
                    "request limit",
                    "limit exceeded",
                ]
            )

            if (
                not is_rate_limit
                or attempt
                >= SO_LAN_RETRY_RATE_LIMIT
            ):

                raise

            wait_seconds = (
                THOI_GIAN_RETRY_CO_BAN
                * (
                    attempt
                    + 1
                )
            )

            time.sleep(
                wait_seconds
            )

    if last_error is not None:

        raise last_error

    raise RuntimeError(
        "Không thể gọi API."
    )


# ============================================================
# DỮ LIỆU CỔ PHIẾU NGẮN
# ============================================================

def _lay_co_phieu_vnstock(
    ma,
    ky_hieu="1y",
):

    nguon = (
        _tao_nguon_thi_truong()
    )

    ma = normalize_symbol(
        ma
    )

    days = _so_ngay_theo_ky(
        ky_hieu
    )

    ngay_cuoi = datetime.now()

    ngay_dau = (
        ngay_cuoi
        - timedelta(
            days=int(
                days
            )
        )
    )

    du_lieu = (
        _request_equity_ohlcv_safe(
            nguon,
            ma,
            ngay_dau,
            ngay_cuoi,
        )
    )

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        raise ValueError(
            f"Không có dữ liệu OHLCV cho {ma}."
        )

    return _chuan_hoa_bang_gia(
        du_lieu,
        la_co_phieu=True,
    )


# ============================================================
# LỊCH SỬ NGHIÊN CỨU DÀI
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_NGHIEN_CUU,
    show_spinner=False,
)
def load_research_history(
    symbol,
    start_date,
    end_date,
):
    """
    ĐÂY LÀ HÀM DÀNH RIÊNG CHO NGHIÊN CỨU.

    Không sử dụng load_market_data().

    Ví dụ 1Y:

        27/08/2025
             ↓
        chunk 60 ngày
             ↓
        chunk 60 ngày
             ↓
        chunk 60 ngày
             ↓
        ...
             ↓
        27/08/2026

    Sau đó ghép toàn bộ phiên giao dịch.

    Vì mỗi chunk chỉ 60 ngày lịch nên dữ liệu
    không bị endpoint giới hạn ở ~100 rows.
    """

    if not NGUON_VNSTOCK_CO_SAN:

        raise RuntimeError(
            "Chưa có vnstock."
        )

    ma = normalize_symbol(
        symbol
    )

    if not ma:

        raise ValueError(
            "Mã cổ phiếu không hợp lệ."
        )

    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    if start > end:

        raise ValueError(
            "Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc."
        )

    nguon = (
        _tao_nguon_thi_truong()
    )

    chunks = []

    cursor = start

    so_request = 0

    tong_so_request_uoc_tinh = int(
        np.ceil(
            (
                end
                - start
            ).days
            / SO_NGAY_MOI_CHUNK_NGHIEN_CUU
        )
    )

    # ========================================================
    # LOOP CHUNK
    # ========================================================

    while cursor <= end:

        chunk_end = min(
            cursor
            + pd.Timedelta(
                days=(
                    SO_NGAY_MOI_CHUNK_NGHIEN_CUU
                    - 1
                )
            ),
            end,
        )

        so_request += 1

        # Progress trên Streamlit nếu đang được gọi
        # trực tiếp từ page.
        try:

            st.session_state[
                "research_api_progress"
            ] = (
                so_request,
                tong_so_request_uoc_tinh,
            )

        except Exception:
            pass

        du_lieu_chunk = (
            _request_equity_ohlcv_safe(
                nguon,
                ma,
                cursor,
                chunk_end,
            )
        )

        if (
            isinstance(
                du_lieu_chunk,
                pd.DataFrame,
            )
            and not du_lieu_chunk.empty
        ):

            chunks.append(
                du_lieu_chunk.copy()
            )

        next_cursor = (
            chunk_end
            + pd.Timedelta(
                days=1
            )
        )

        if next_cursor <= end:

            time.sleep(
                THOI_GIAN_CHO_GIUA_REQUEST
            )

        cursor = next_cursor

    # ========================================================
    # KHÔNG CÓ DATA
    # ========================================================

    if not chunks:

        raise ValueError(
            f"Không lấy được dữ liệu lịch sử "
            f"cho {ma} trong khoảng "
            f"{start.strftime('%d/%m/%Y')} "
            f"→ "
            f"{end.strftime('%d/%m/%Y')}."
        )

    # ========================================================
    # CONCAT
    # ========================================================

    raw = pd.concat(
        chunks,
        axis=0,
    )

    # ========================================================
    # CHUẨN HÓA
    # ========================================================

    raw = _chuan_hoa_bang_gia(
        raw,
        la_co_phieu=True,
    )

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    raw.index = pd.to_datetime(
        raw.index,
        errors="coerce",
    )

    raw = raw[
        ~raw.index.isna()
    ].copy()

    raw = (
        raw
        .sort_index()
    )

    raw = (
        raw[
            ~raw.index.duplicated(
                keep="last"
            )
        ]
        .copy()
    )

    # ========================================================
    # CẮT CHÍNH XÁC
    # ========================================================

    raw = raw.loc[
        (
            raw.index
            >= start
        )
        & (
            raw.index
            <= end
        )
    ].copy()

    if raw.empty:

        raise ValueError(
            "Không có phiên giao dịch nào trong khoảng đã chọn."
        )

    # ========================================================
    # QUAN TRỌNG:
    # TÍNH INDICATOR SAU KHI GHÉP
    # ========================================================

    result = add_indicators(
        raw,
        la_co_phieu=True,
    )

    # ========================================================
    # ATTRS
    # ========================================================

    result.attrs[
        "symbol"
    ] = ma

    result.attrs[
        "display_symbol"
    ] = display_symbol(
        ma
    )

    result.attrs[
        "source"
    ] = "Vnstock"

    result.attrs[
        "la_co_phieu"
    ] = True

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
        "research_observations"
    ] = len(
        result
    )

    result.attrs[
        "research_requests"
    ] = so_request

    return result


# ============================================================
# HELPER THEO PRESET
# ============================================================

def load_research_history_period(
    symbol,
    period="1y",
    end_date=None,
):
    """
    Dùng trực tiếp cho page:

        load_research_history_period(
            "HPG",
            "1y",
        )

    Hỗ trợ:

        1mo
        3mo
        6mo
        1y
        3y
        5y
        10y
    """

    if end_date is None:

        end = pd.Timestamp.today().normalize()

    else:

        end = pd.Timestamp(
            end_date
        ).normalize()

    days = _so_ngay_theo_ky(
        period
    )

    start = (
        end
        - pd.Timedelta(
            days=days
        )
    )

    return load_research_history(
        symbol,
        start.date(),
        end.date(),
    )


# ============================================================
# VNINDEX
# ============================================================

def _lay_vnindex_vnstock():

    nguon = (
        _tao_nguon_thi_truong()
    )

    end = datetime.now()

    start = (
        end
        - timedelta(
            days=450
        )
    )

    du_lieu = (
        nguon
        .index(
            "VNINDEX"
        )
        .ohlcv(
            start=start.strftime(
                "%Y-%m-%d"
            ),
            end=(
                end
                + timedelta(
                    days=1
                )
            ).strftime(
                "%Y-%m-%d"
            ),
            interval="1D",
        )
    )

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        raise ValueError(
            "Không có dữ liệu VN-INDEX."
        )

    du_lieu = _chuan_hoa_bang_gia(
        du_lieu,
        la_co_phieu=False,
    )

    du_lieu = add_indicators(
        du_lieu,
        la_co_phieu=False,
    )

    return du_lieu


# ============================================================
# CỘT SUMMARY
# ============================================================

def _tim_cot_tong_quat(
    df,
    cac_ten,
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

    mapping = {}

    for column in df.columns:

        mapping[
            _ten_cot_chuan(
                column
            )
        ] = column

    for name in cac_ten:

        key = _ten_cot_chuan(
            name
        )

        if key in mapping:

            return mapping[
                key
            ]

    for column in df.columns:

        key = _ten_cot_chuan(
            column
        )

        for name in cac_ten:

            target = _ten_cot_chuan(
                name
            )

            if (
                target in key
                or key in target
            ):

                return column

    return None


def _tim_cot_khoi_luong(
    df,
):

    return _tim_cot_tong_quat(
        df,
        [
            "total_match_volume",
            "total_volume",
            "trading_volume",
            "matched_volume",
            "match_volume",
            "volume",
            "vol",
            "khoi_luong",
            "khối lượng",
        ],
    )


def _tim_cot_gia_tri(
    df,
):

    return _tim_cot_tong_quat(
        df,
        [
            "total_value",
            "total_trading_value",
            "trading_value",
            "trade_value",
            "value_traded",
            "match_value",
            "matched_value",
            "value",
            "gia_tri",
            "giá trị",
            "turnover",
        ],
    )


def _lay_dong_cuoi(
    df,
):

    if (
        df is None
        or df.empty
    ):

        return None

    return df.iloc[
        -1
    ]


# ============================================================
# SUMMARY VNINDEX
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_TONG_QUAN_VNINDEX,
    show_spinner=False,
)
def load_vnindex_market_summary():

    nguon = (
        _tao_nguon_thi_truong()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    try:

        summary = (
            nguon
            .index(
                "VNINDEX"
            )
            .summary()
        )

    except Exception:

        summary = None

    if (
        isinstance(
            summary,
            pd.DataFrame,
        )
        and not summary.empty
    ):

        volume_column = (
            _tim_cot_khoi_luong(
                summary
            )
        )

        value_column = (
            _tim_cot_gia_tri(
                summary
            )
        )

        last_row = (
            _lay_dong_cuoi(
                summary
            )
        )

        if last_row is not None:

            volume = None
            value = None

            if volume_column is not None:

                volume = _so(
                    last_row[
                        volume_column
                    ],
                    None,
                )

            if value_column is not None:

                value = _so(
                    last_row[
                        value_column
                    ],
                    None,
                )

            if (
                volume is not None
                or value is not None
            ):

                return {
                    "khoi_luong": volume,
                    "gia_tri": value,
                    "nguon": "index.summary",
                }

    # ========================================================
    # TRADE HISTORY
    # ========================================================

    try:

        trade = (
            nguon
            .index(
                "VNINDEX"
            )
            .trade_history()
        )

    except Exception:

        trade = None

    if (
        isinstance(
            trade,
            pd.DataFrame,
        )
        and not trade.empty
    ):

        volume_column = (
            _tim_cot_khoi_luong(
                trade
            )
        )

        value_column = (
            _tim_cot_gia_tri(
                trade
            )
        )

        if (
            volume_column is not None
            or value_column is not None
        ):

            trade = trade.copy()

            date_column = (
                _tim_cot_tong_quat(
                    trade,
                    [
                        "trading_date",
                        "date",
                        "time",
                        "datetime",
                    ],
                )
            )

            if date_column is not None:

                trade[
                    date_column
                ] = pd.to_datetime(
                    trade[
                        date_column
                    ],
                    errors="coerce",
                )

                trade = (
                    trade
                    .sort_values(
                        date_column
                    )
                )

            last_row = (
                _lay_dong_cuoi(
                    trade
                )
            )

            if last_row is not None:

                volume = None
                value = None

                if volume_column is not None:

                    volume = _so(
                        last_row[
                            volume_column
                        ],
                        None,
                    )

                if value_column is not None:

                    value = _so(
                        last_row[
                            value_column
                        ],
                        None,
                    )

                if (
                    volume is not None
                    or value is not None
                ):

                    return {
                        "khoi_luong": volume,
                        "gia_tri": value,
                        "nguon": (
                            "index.trade_history"
                        ),
                    }

    return {
        "khoi_luong": None,
        "gia_tri": None,
        "nguon": None,
    }


# ============================================================
# LOAD MARKET DATA
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_CO_PHIEU,
    show_spinner=False,
)
def load_market_data(
    symbol,
    period="1y",
):

    ma = normalize_symbol(
        symbol
    )

    if not ma:

        raise ValueError(
            "Mã cổ phiếu không hợp lệ."
        )

    if not NGUON_VNSTOCK_CO_SAN:

        raise RuntimeError(
            "Chưa có vnstock trong môi trường."
        )

    raw = (
        _lay_co_phieu_vnstock(
            ma,
            period,
        )
    )

    data = add_indicators(
        raw,
        la_co_phieu=True,
    )

    if data.empty:

        raise ValueError(
            f"Không có dữ liệu hợp lệ cho {ma}."
        )

    data.attrs[
        "symbol"
    ] = ma

    data.attrs[
        "display_symbol"
    ] = display_symbol(
        ma
    )

    data.attrs[
        "source"
    ] = "Vnstock"

    data.attrs[
        "la_co_phieu"
    ] = True

    return data


# ============================================================
# LOAD VNINDEX
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_VNINDEX,
    show_spinner=False,
)
def load_vnindex_data():

    if not NGUON_VNSTOCK_CO_SAN:

        raise RuntimeError(
            "Chưa có vnstock trong môi trường."
        )

    data = (
        _lay_vnindex_vnstock()
    )

    if data.empty:

        raise ValueError(
            "VN-INDEX không có dữ liệu."
        )

    data.attrs[
        "symbol"
    ] = "VNINDEX"

    data.attrs[
        "display_symbol"
    ] = "VN-INDEX"

    data.attrs[
        "source"
    ] = "Vnstock"

    data.attrs[
        "la_co_phieu"
    ] = False

    return data


# ============================================================
# LATEST PRICE
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_GIA_MOI,
    show_spinner=False,
)
def load_latest_price(
    symbol,
):

    ma = normalize_symbol(
        symbol
    )

    data = load_market_data(
        ma,
        "5d",
    )

    if (
        data is None
        or data.empty
    ):

        raise ValueError(
            f"Không có dữ liệu mới nhất cho {ma}."
        )

    last = data.iloc[
        -1
    ]

    price = float(
        last[
            "Close"
        ]
    )

    if len(data) >= 2:

        previous = float(
            data[
                "Close"
            ].iloc[
                -2
            ]
        )

    else:

        previous = price

    if previous != 0:

        change = (
            price
            / previous
            - 1
        ) * 100

    else:

        change = 0.0

    return {
        "ma": display_symbol(
            ma
        ),
        "gia": price,
        "thay_doi": change,
        "khoi_luong": float(
            last[
                "Volume"
            ]
        ),
        "thoi_gian": (
            data.index[
                -1
            ]
        ),
    }


# ============================================================
# SNAPSHOT
# ============================================================

def market_snapshot(
    du_lieu,
):

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        return {}

    last = du_lieu.iloc[
        -1
    ]

    if len(du_lieu) >= 2:

        previous = (
            du_lieu.iloc[
                -2
            ]
        )

    else:

        previous = last

    price = _so(
        last.get(
            "Close"
        ),
        np.nan,
    )

    previous_price = _so(
        previous.get(
            "Close"
        ),
        np.nan,
    )

    if (
        np.isfinite(
            previous_price
        )
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

        try:

            value = float(
                last.get(
                    name
                )
            )

            if np.isfinite(
                value
            ):

                return value

        except Exception:
            pass

        return np.nan

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
        "atr14": get_column(
            "ATR14"
        ),
        "volume_sma20": get_column(
            "Volume_SMA20"
        ),
        "relative_volume": get_column(
            "Relative_Volume"
        ),
        "ema20": get_column(
            "EMA20"
        ),
        "ema50": get_column(
            "EMA50"
        ),
    }


# ============================================================
# COMPATIBILITY
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

    positive_score = sum(
        word in text
        for word in positive_words
    )

    negative_score = sum(
        word in text
        for word in negative_words
    )

    if positive_score > negative_score:
        return "positive"

    if negative_score > positive_score:
        return "negative"

    return "neutral"


# ============================================================
# OLS CŨ
# ============================================================

def run_ols(
    du_lieu,
):

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        return None

    try:

        import statsmodels.api as sm

    except Exception:

        return None

    columns = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility20",
    ]

    if any(
        column not in du_lieu.columns
        for column in columns
    ):

        return None

    data = (
        du_lieu[
            columns
            + ["Return"]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if len(data) < 50:
        return None

    try:

        X = sm.add_constant(
            data[
                columns
            ],
            has_constant="add",
        )

        y = data[
            "Return"
        ]

        return (
            sm.OLS(
                y,
                X,
            )
            .fit()
        )

    except Exception:

        return None


# ============================================================
# RANDOM FOREST CŨ
# ============================================================

def run_random_forest(
    du_lieu,
):

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        return None

    try:

        from sklearn.ensemble import (
            RandomForestRegressor,
        )

    except Exception:

        return None

    columns = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility20",
    ]

    if any(
        column not in du_lieu.columns
        for column in columns
    ):

        return None

    data = (
        du_lieu[
            columns
            + ["Return"]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .copy()
    )

    data[
        "Target"
    ] = (
        data[
            "Return"
        ]
        .shift(
            -1
        )
    )

    data = data.dropna()

    if len(data) < 80:
        return None

    try:

        X = data[
            columns
        ].astype(
            float
        )

        y = data[
            "Target"
        ].astype(
            float
        )

        split = int(
            len(data)
            * 0.8
        )

        if split < 40:
            return None

        model = (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        )

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
                columns,
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
# BUILD QUANT CŨ
# ============================================================

def build_quant(
    du_lieu,
):

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        return None

    try:

        from sklearn.ensemble import (
            RandomForestRegressor,
        )

        from sklearn.metrics import (
            mean_absolute_error,
            r2_score,
        )

        import statsmodels.api as sm

    except Exception:

        return None

    columns = [
        "RSI",
        "MACD",
        "MACD_Hist",
        "Volatility20",
        "Volume_Change",
        "Return",
    ]

    if any(
        column not in du_lieu.columns
        for column in columns
    ):

        return None

    data = (
        du_lieu.copy()
    )

    data[
        "Target"
    ] = (
        data[
            "Return"
        ]
        .shift(
            -1
        )
    )

    data = (
        data
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=(
                columns
                + ["Target"]
            )
        )
    )

    if len(data) < 60:
        return None

    X = data[
        columns
    ].astype(
        float
    )

    y = data[
        "Target"
    ].astype(
        float
    )

    split = int(
        len(data)
        * 0.8
    )

    if (
        split < 30
        or split >= len(data)
    ):

        return None

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

    try:

        ols_model = (
            sm.OLS(
                y_train,
                sm.add_constant(
                    X_train,
                    has_constant="add",
                ),
            )
            .fit(
                cov_type="HC3"
            )
        )

        rf_model = (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        )

        rf_model.fit(
            X_train,
            y_train,
        )

        prediction_test = (
            rf_model.predict(
                X_test
            )
        )

        mae = float(
            mean_absolute_error(
                y_test,
                prediction_test,
            )
        )

        try:

            r2 = float(
                r2_score(
                    y_test,
                    prediction_test,
                )
            )

        except Exception:

            r2 = np.nan

        next_prediction = float(
            rf_model.predict(
                X.iloc[
                    [
                        -1
                    ]
                ]
            )[0]
        )

        importance = (
            pd.Series(
                rf_model.feature_importances_,
                index=columns,
            )
            .sort_values(
                ascending=False
            )
        )

        return (
            ols_model,
            rf_model,
            {
                "MAE": mae,
                "R2": r2,
            },
            next_prediction,
            importance,
        )

    except Exception:

        return None
