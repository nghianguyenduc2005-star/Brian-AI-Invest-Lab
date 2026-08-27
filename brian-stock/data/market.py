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
# CẤU HÌNH CACHE
# ============================================================

THOI_GIAN_LUU_CO_PHIEU = 300
THOI_GIAN_LUU_VNINDEX = 300
THOI_GIAN_LUU_GIA_MOI = 30
THOI_GIAN_LUU_TONG_QUAN_VNINDEX = 30

# Cache lịch sử nghiên cứu dài.
THOI_GIAN_LUU_NGHIEN_CUU = 900


# ============================================================
# CẤU HÌNH RATE LIMIT
# ============================================================

# Không lấy chunk quá lớn vì nguồn có thể giới hạn
# số dòng trả về mỗi request.
#
# 100 ngày lịch thường chỉ khoảng 65-75 phiên giao dịch,
# an toàn hơn giới hạn khoảng 100 dòng/request.
SO_NGAY_MOI_CHUNK_NGHIEN_CUU = 100

# 40 request/phút.
# 1.6 giây/request => tối đa khoảng 37 request/phút.
#
# Điều này giúp tránh trường hợp chạy 10Y bị:
# Rate Limit Exceeded
THOI_GIAN_CHO_GIUA_REQUEST = 1.6


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
# CHUYỂN TÊN CỘT
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
# CHUẨN HÓA BẢNG GIÁ
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

        cot_moi = []

        for cot in du_lieu.columns:

            if isinstance(
                cot,
                tuple,
            ):

                cot_moi.append(
                    str(
                        cot[-1]
                    )
                )

            else:

                cot_moi.append(
                    str(cot)
                )

        du_lieu.columns = cot_moi

    # ========================================================
    # CHUẨN HÓA TÊN CỘT
    # ========================================================

    anh_xa = {}

    for cot in du_lieu.columns:

        ten = _ten_cot_chuan(
            cot
        )

        if ten in {
            "time",
            "date",
            "datetime",
            "timestamp",
            "tradingdate",
            "trading_date",
        }:

            anh_xa[
                cot
            ] = "Time"

        elif ten in {
            "open",
            "open_price",
            "openprice",
        }:

            anh_xa[
                cot
            ] = "Open"

        elif ten in {
            "high",
            "high_price",
            "highprice",
        }:

            anh_xa[
                cot
            ] = "High"

        elif ten in {
            "low",
            "low_price",
            "lowprice",
        }:

            anh_xa[
                cot
            ] = "Low"

        elif ten in {
            "close",
            "close_price",
            "closeprice",
            "last",
            "lastprice",
            "last_price",
        }:

            anh_xa[
                cot
            ] = "Close"

        elif ten in {
            "volume",
            "vol",
            "total_volume",
            "totalvolume",
            "matchvolume",
            "match_volume",
        }:

            anh_xa[
                cot
            ] = "Volume"

        elif ten in {
            "value",
            "trading_value",
            "value_traded",
            "matchvalue",
            "match_value",
            "turnover",
            "total_value",
        }:

            anh_xa[
                cot
            ] = "Value"

    du_lieu = du_lieu.rename(
        columns=anh_xa
    )

    # ========================================================
    # INDEX THỜI GIAN
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
    # CỘT BẮT BUỘC
    # ========================================================

    cac_cot_bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    thieu = [
        cot
        for cot in cac_cot_bat_buoc
        if cot not in du_lieu.columns
    ]

    if thieu:

        raise ValueError(
            "Thiếu cột dữ liệu: "
            + ", ".join(
                thieu
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
    # ÉP KIỂU SỐ
    # ========================================================

    for cot in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

        du_lieu[
            cot
        ] = pd.to_numeric(
            du_lieu[
                cot
            ],
            errors="coerce",
        )

    # Value nếu tồn tại.
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
    # CHUẨN HÓA ĐƠN VỊ GIÁ CỔ PHIẾU
    # ========================================================

    if la_co_phieu:

        gia_trung_vi = (
            du_lieu[
                "Close"
            ]
            .dropna()
            .median()
        )

        # Ví dụ:
        # 22.2 -> 22,200 VND
        #
        # Không áp dụng cho VNINDEX.
        if (
            pd.notna(
                gia_trung_vi
            )
            and gia_trung_vi > 0
            and gia_trung_vi < 1000
        ):

            for cot in [
                "Open",
                "High",
                "Low",
                "Close",
            ]:

                du_lieu[
                    cot
                ] = (
                    du_lieu[
                        cot
                    ]
                    * 1000
                )

    # ========================================================
    # LOẠI GIÁ LỖI
    # ========================================================

    for cot in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        du_lieu.loc[
            du_lieu[
                cot
            ] <= 0,
            cot,
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
    # CHỈ GIỮ OHLC HỢP LỆ
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

    trung_binh_tang = (
        tang.ewm(
            alpha=1 / chu_ky,
            adjust=False,
            min_periods=chu_ky,
        )
        .mean()
    )

    trung_binh_giam = (
        giam.ewm(
            alpha=1 / chu_ky,
            adjust=False,
            min_periods=chu_ky,
        )
        .mean()
    )

    ti_so = (
        trung_binh_tang
        / trung_binh_giam.replace(
            0,
            np.nan,
        )
    )

    ket_qua = (
        100
        - (
            100
            / (
                1
                + ti_so
            )
        )
    )

    ket_qua = ket_qua.where(
        ~(
            (
                trung_binh_giam
                == 0
            )
            & (
                trung_binh_tang
                > 0
            )
        ),
        100,
    )

    ket_qua = ket_qua.where(
        ~(
            (
                trung_binh_giam
                == 0
            )
            & (
                trung_binh_tang
                == 0
            )
        ),
        50,
    )

    return ket_qua


# ============================================================
# TOÀN BỘ CHỈ BÁO
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

    for so_phien in [
        9,
        12,
        20,
        26,
        50,
        100,
        200,
    ]:

        du_lieu[
            f"EMA{so_phien}"
        ] = (
            gia.ewm(
                span=so_phien,
                adjust=False,
            )
            .mean()
        )

    # ========================================================
    # SMA
    # ========================================================

    for so_phien in [
        5,
        10,
        20,
        50,
        100,
        200,
    ]:

        du_lieu[
            f"SMA{so_phien}"
        ] = (
            gia
            .rolling(
                so_phien
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

    do_lech = (
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
        + 2 * do_lech
    )

    du_lieu[
        "Bollinger_Lower"
    ] = (
        du_lieu[
            "SMA20"
        ]
        - 2 * do_lech
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

    for so_phien in [
        5,
        20,
        60,
    ]:

        du_lieu[
            f"Volatility{so_phien}"
        ] = (
            du_lieu[
                "Return"
            ]
            .rolling(
                so_phien
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

    for so_phien in [
        5,
        20,
        50,
    ]:

        du_lieu[
            f"Volume_SMA{so_phien}"
        ] = (
            du_lieu[
                "Volume"
            ]
            .rolling(
                so_phien
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

    bien_1 = (
        du_lieu[
            "High"
        ]
        - du_lieu[
            "Low"
        ]
    )

    bien_2 = (
        du_lieu[
            "High"
        ]
        - du_lieu[
            "Close"
        ].shift(
            1
        )
    ).abs()

    bien_3 = (
        du_lieu[
            "Low"
        ]
        - du_lieu[
            "Close"
        ].shift(
            1
        )
    ).abs()

    bien_do_that = pd.concat(
        [
            bien_1,
            bien_2,
            bien_3,
        ],
        axis=1,
    ).max(
        axis=1
    )

    du_lieu[
        "ATR14"
    ] = (
        bien_do_that
        .rolling(
            14
        )
        .mean()
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    for so_phien in [
        5,
        10,
        20,
    ]:

        du_lieu[
            f"Momentum{so_phien}"
        ] = (
            gia
            / gia.shift(
                so_phien
            )
            - 1
        )

    # ========================================================
    # HIGH / LOW
    # ========================================================

    for so_phien in [
        20,
        50,
        252,
    ]:

        du_lieu[
            f"High{so_phien}"
        ] = (
            du_lieu[
                "High"
            ]
            .rolling(
                so_phien
            )
            .max()
        )

        du_lieu[
            f"Low{so_phien}"
        ] = (
            du_lieu[
                "Low"
            ]
            .rolling(
                so_phien
            )
            .min()
        )

        du_lieu[
            f"Distance_From_High{so_phien}"
        ] = (
            (
                gia
                / du_lieu[
                    f"High{so_phien}"
                ]
                - 1
            )
            * 100
        )

        du_lieu[
            f"Distance_From_Low{so_phien}"
        ] = (
            (
                gia
                / du_lieu[
                    f"Low{so_phien}"
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
# LẤY OHLCV MỘT KHOẢNG
# ============================================================

def _request_equity_ohlcv(
    nguon,
    ma,
    start_date,
    end_date,
):

    return (
        nguon
        .equity(
            ma
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
# LẤY DỮ LIỆU CỔ PHIẾU
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

    so_ngay = _so_ngay_theo_ky(
        ky_hieu
    )

    ngay_cuoi = datetime.now()

    ngay_dau = (
        ngay_cuoi
        - timedelta(
            days=int(
                so_ngay
            )
        )
    )

    du_lieu = (
        _request_equity_ohlcv(
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
# LẤY LỊCH SỬ NGHIÊN CỨU DÀI
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
    Lấy lịch sử chính xác theo khoảng yêu cầu.

    Thiết kế:

        start
           ↓
        chunk 100 ngày
           ↓
        API request tuần tự
           ↓
        nghỉ 1.6 giây
           ↓
        chunk kế tiếp
           ↓
        concat
           ↓
        deduplicate
           ↓
        add_indicators một lần trên toàn bộ lịch sử

    Ưu điểm:

    - 1Y không còn bị kẹt ở 99 observations.
    - 3Y / 5Y / 10Y có thể lấy lịch sử dài.
    - Không tạo hàng chục request song song.
    - Có cache.
    - Không tạo dữ liệu giả.
    """

    if not NGUON_VNSTOCK_CO_SAN:

        raise RuntimeError(
            "Chưa có vnstock trong môi trường."
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

    cac_chunk = []

    cursor = start

    so_request = 0

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

        try:

            du_lieu_chunk = (
                _request_equity_ohlcv(
                    nguon,
                    ma,
                    cursor,
                    chunk_end,
                )
            )

        except Exception as loi:

            raise RuntimeError(
                f"Lỗi API ở chunk "
                f"{cursor.strftime('%d/%m/%Y')} "
                f"→ "
                f"{chunk_end.strftime('%d/%m/%Y')} "
                f"cho {ma}: "
                f"{loi}"
            ) from loi

        if (
            du_lieu_chunk is not None
            and not du_lieu_chunk.empty
        ):

            if isinstance(
                du_lieu_chunk,
                pd.DataFrame,
            ):

                cac_chunk.append(
                    du_lieu_chunk.copy()
                )

        # ----------------------------------------------------
        # Rate limit
        #
        # Không sleep sau request cuối.
        # ----------------------------------------------------

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
    # KHÔNG CÓ CHUNK
    # ========================================================

    if not cac_chunk:

        raise ValueError(
            f"Không lấy được dữ liệu lịch sử "
            f"cho {ma} trong khoảng "
            f"{start.strftime('%Y-%m-%d')} "
            f"→ "
            f"{end.strftime('%Y-%m-%d')}."
        )

    # ========================================================
    # GHÉP TẤT CẢ
    # ========================================================

    try:

        raw = pd.concat(
            cac_chunk,
            axis=0,
        )

    except Exception as loi:

        raise RuntimeError(
            f"Không thể ghép dữ liệu lịch sử {ma}: {loi}"
        ) from loi

    # ========================================================
    # CHUẨN HÓA MỘT LẦN
    #
    # Không tính indicator từng chunk.
    #
    # Điều này rất quan trọng:
    # SMA200 / EMA200 / Bollinger / ATR / Momentum...
    # phải nhìn thấy toàn bộ lịch sử.
    # ========================================================

    raw = _chuan_hoa_bang_gia(
        raw,
        la_co_phieu=True,
    )

    # ========================================================
    # CẮT CHÍNH XÁC LẠI
    # ========================================================

    raw.index = pd.to_datetime(
        raw.index,
        errors="coerce",
    )

    raw = raw[
        ~raw.index.isna()
    ].copy()

    end_inclusive = (
        end
        + pd.Timedelta(
            days=1
        )
        - pd.Timedelta(
            microseconds=1
        )
    )

    raw = raw.loc[
        (
            raw.index >= start
        )
        & (
            raw.index <= end_inclusive
        )
    ].copy()

    raw = (
        raw
        .sort_index()
        .loc[
            ~raw.index.duplicated(
                keep="last"
            )
        ]
    )

    if raw.empty:

        raise ValueError(
            f"Không có phiên giao dịch nào "
            f"trong khoảng đã chọn cho {ma}."
        )

    # ========================================================
    # TÍNH INDICATORS TOÀN BỘ SAMPLE
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
        "research_requests"
    ] = so_request

    return result


# ============================================================
# LẤY VN-INDEX
# ============================================================

def _lay_vnindex_vnstock():

    nguon = (
        _tao_nguon_thi_truong()
    )

    ngay_cuoi = datetime.now()

    ngay_dau = (
        ngay_cuoi
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
            start=ngay_dau.strftime(
                "%Y-%m-%d"
            ),
            end=(
                ngay_cuoi
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

    du_lieu = (
        _chuan_hoa_bang_gia(
            du_lieu,
            la_co_phieu=False,
        )
    )

    du_lieu = add_indicators(
        du_lieu,
        la_co_phieu=False,
    )

    return du_lieu


# ============================================================
# TÌM CỘT TỔNG QUÁT
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

    ban_do = {}

    for cot in df.columns:

        khoa = _ten_cot_chuan(
            cot
        )

        ban_do[
            khoa
        ] = cot

    # Exact match.
    for ten in cac_ten:

        khoa = _ten_cot_chuan(
            ten
        )

        if khoa in ban_do:

            return ban_do[
                khoa
            ]

    # Fuzzy match.
    for cot in df.columns:

        khoa = _ten_cot_chuan(
            cot
        )

        for ten in cac_ten:

            mau = _ten_cot_chuan(
                ten
            )

            if (
                mau in khoa
                or khoa in mau
            ):

                return cot

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


# ============================================================
# DÒNG CUỐI
# ============================================================

def _lay_dong_cuoi(
    df,
):

    if (
        df is None
        or df.empty
    ):

        return None

    try:

        return df.iloc[
            -1
        ]

    except Exception:

        return None


# ============================================================
# TÓM TẮT VNINDEX MARKET
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
    # 1. SUMMARY
    # ========================================================

    try:

        df_summary = (
            nguon
            .index(
                "VNINDEX"
            )
            .summary()
        )

    except Exception:

        df_summary = None

    if (
        isinstance(
            df_summary,
            pd.DataFrame,
        )
        and not df_summary.empty
    ):

        cot_volume = (
            _tim_cot_khoi_luong(
                df_summary
            )
        )

        cot_value = (
            _tim_cot_gia_tri(
                df_summary
            )
        )

        dong = _lay_dong_cuoi(
            df_summary
        )

        if dong is not None:

            volume = None
            value = None

            if cot_volume is not None:

                volume = _so(
                    dong[
                        cot_volume
                    ],
                    None,
                )

            if cot_value is not None:

                value = _so(
                    dong[
                        cot_value
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
    # 2. TRADE HISTORY
    # ========================================================

    try:

        df_trade = (
            nguon
            .index(
                "VNINDEX"
            )
            .trade_history()
        )

    except Exception:

        df_trade = None

    if (
        isinstance(
            df_trade,
            pd.DataFrame,
        )
        and not df_trade.empty
    ):

        cot_volume = (
            _tim_cot_khoi_luong(
                df_trade
            )
        )

        cot_value = (
            _tim_cot_gia_tri(
                df_trade
            )
        )

        if (
            cot_volume is not None
            or cot_value is not None
        ):

            df_trade = (
                df_trade.copy()
            )

            cot_date = (
                _tim_cot_tong_quat(
                    df_trade,
                    [
                        "trading_date",
                        "date",
                        "time",
                        "datetime",
                    ],
                )
            )

            if cot_date is not None:

                df_trade[
                    cot_date
                ] = pd.to_datetime(
                    df_trade[
                        cot_date
                    ],
                    errors="coerce",
                )

                df_trade = (
                    df_trade
                    .sort_values(
                        cot_date
                    )
                )

            dong = _lay_dong_cuoi(
                df_trade
            )

            if dong is not None:

                volume = None
                value = None

                if cot_volume is not None:

                    volume = _so(
                        dong[
                            cot_volume
                        ],
                        None,
                    )

                if cot_value is not None:

                    value = _so(
                        dong[
                            cot_value
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
                        "nguon": "index.trade_history",
                    }

    # ========================================================
    # 3. KHÔNG CÓ DỮ LIỆU
    # ========================================================

    return {
        "khoi_luong": None,
        "gia_tri": None,
        "nguon": None,
    }


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
# HÀM CHÍNH — CỔ PHIẾU
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

    du_lieu_goc = (
        _lay_co_phieu_vnstock(
            ma,
            period,
        )
    )

    du_lieu = add_indicators(
        du_lieu_goc,
        la_co_phieu=True,
    )

    if du_lieu.empty:

        raise ValueError(
            f"Không có dữ liệu hợp lệ cho {ma}."
        )

    du_lieu.attrs[
        "symbol"
    ] = ma

    du_lieu.attrs[
        "display_symbol"
    ] = display_symbol(
        ma
    )

    du_lieu.attrs[
        "source"
    ] = "Vnstock"

    du_lieu.attrs[
        "la_co_phieu"
    ] = True

    return du_lieu


# ============================================================
# HÀM CHÍNH — VNINDEX
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

    du_lieu = (
        _lay_vnindex_vnstock()
    )

    if du_lieu.empty:

        raise ValueError(
            "VN-INDEX không có dữ liệu."
        )

    du_lieu.attrs[
        "symbol"
    ] = "VNINDEX"

    du_lieu.attrs[
        "display_symbol"
    ] = "VN-INDEX"

    du_lieu.attrs[
        "source"
    ] = "Vnstock"

    du_lieu.attrs[
        "la_co_phieu"
    ] = False

    return du_lieu


# ============================================================
# GIÁ MỚI NHẤT
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

    du_lieu = load_market_data(
        ma,
        "5d",
    )

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        raise ValueError(
            f"Không có dữ liệu mới nhất cho {ma}."
        )

    dong_cuoi = (
        du_lieu.iloc[
            -1
        ]
    )

    gia = float(
        dong_cuoi[
            "Close"
        ]
    )

    if len(
        du_lieu
    ) >= 2:

        gia_truoc = float(
            du_lieu[
                "Close"
            ].iloc[
                -2
            ]
        )

    else:

        gia_truoc = gia

    if gia_truoc != 0:

        thay_doi = (
            gia
            / gia_truoc
            - 1
        ) * 100

    else:

        thay_doi = 0.0

    return {
        "ma": display_symbol(
            ma
        ),
        "gia": gia,
        "thay_doi": thay_doi,
        "khoi_luong": float(
            dong_cuoi[
                "Volume"
            ]
        ),
        "thoi_gian": (
            du_lieu.index[
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

    dong_cuoi = (
        du_lieu.iloc[
            -1
        ]
    )

    if len(
        du_lieu
    ) >= 2:

        dong_truoc = (
            du_lieu.iloc[
                -2
            ]
        )

    else:

        dong_truoc = dong_cuoi

    gia = float(
        dong_cuoi[
            "Close"
        ]
    )

    gia_truoc = float(
        dong_truoc[
            "Close"
        ]
    )

    if gia_truoc != 0:

        thay_doi = (
            gia
            / gia_truoc
            - 1
        ) * 100

    else:

        thay_doi = np.nan

    def _so_cot(
        ten_cot,
    ):

        try:

            value = float(
                dong_cuoi[
                    ten_cot
                ]
            )

            if pd.notna(
                value
            ):

                return value

        except Exception:
            pass

        return np.nan

    return {
        "price": gia,

        "change_1d": thay_doi,

        "return_1d": _so_cot(
            "ReturnPct"
        ),

        "rsi": _so_cot(
            "RSI"
        ),

        "macd": _so_cot(
            "MACD"
        ),

        "sma20": _so_cot(
            "SMA20"
        ),

        "sma50": _so_cot(
            "SMA50"
        ),

        "volatility20": _so_cot(
            "Volatility20"
        ),

        "volume": _so_cot(
            "Volume"
        ),
    }


# ============================================================
# TƯƠNG THÍCH CODE CŨ
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
# SENTIMENT
# ============================================================

def classify_news(
    title,
):

    van_ban = str(
        title or ""
    ).lower()

    tu_tich_cuc = [
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

    tu_tieu_cuc = [
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

    diem_tich_cuc = sum(
        tu in van_ban
        for tu in tu_tich_cuc
    )

    diem_tieu_cuc = sum(
        tu in van_ban
        for tu in tu_tieu_cuc
    )

    if (
        diem_tich_cuc
        > diem_tieu_cuc
    ):

        return "positive"

    if (
        diem_tieu_cuc
        > diem_tich_cuc
    ):

        return "negative"

    return "neutral"


# ============================================================
# OLS CŨ - TƯƠNG THÍCH
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

    cac_cot = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility20",
    ]

    if any(
        cot not in du_lieu.columns
        for cot in cac_cot
    ):

        return None

    sach = (
        du_lieu[
            cac_cot
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

    if len(
        sach
    ) < 50:

        return None

    try:

        X = sm.add_constant(
            sach[
                cac_cot
            ],
            has_constant="add",
        )

        y = sach[
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
# RANDOM FOREST CŨ - TƯƠNG THÍCH
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

    cac_cot = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility20",
    ]

    if any(
        cot not in du_lieu.columns
        for cot in cac_cot
    ):

        return None

    sach = (
        du_lieu[
            cac_cot
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

    sach[
        "Target"
    ] = (
        sach[
            "Return"
        ]
        .shift(
            -1
        )
    )

    sach = (
        sach.dropna()
    )

    if len(
        sach
    ) < 80:

        return None

    try:

        X = (
            sach[
                cac_cot
            ]
            .astype(float)
        )

        y = (
            sach[
                "Target"
            ]
            .astype(float)
        )

        vi_tri = int(
            len(
                sach
            )
            * 0.8
        )

        if vi_tri < 40:

            return None

        mo_hinh = (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        )

        mo_hinh.fit(
            X.iloc[
                :vi_tri
            ],
            y.iloc[
                :vi_tri
            ],
        )

        du_bao = float(
            mo_hinh.predict(
                X.iloc[
                    [
                        -1
                    ]
                ]
            )[0]
        )

        tam_quan_trong = dict(
            zip(
                cac_cot,
                mo_hinh.feature_importances_,
            )
        )

        return {
            "model": mo_hinh,
            "prediction": du_bao,
            "importance": tam_quan_trong,
        }

    except Exception:

        return None


# ============================================================
# BUILD QUANT CŨ - TƯƠNG THÍCH
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

    cac_cot = [
        "RSI",
        "MACD",
        "MACD_Hist",
        "Volatility20",
        "Volume_Change",
        "Return",
    ]

    if any(
        cot not in du_lieu.columns
        for cot in cac_cot
    ):

        return None

    sach = (
        du_lieu.copy()
    )

    sach[
        "Target"
    ] = (
        sach[
            "Return"
        ]
        .shift(
            -1
        )
    )

    sach = (
        sach.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    sach = sach.dropna(
        subset=(
            cac_cot
            + [
                "Target"
            ]
        )
    )

    if len(
        sach
    ) < 60:

        return None

    X = (
        sach[
            cac_cot
        ]
        .astype(float)
    )

    y = (
        sach[
            "Target"
        ]
        .astype(float)
    )

    vi_tri = int(
        len(
            sach
        )
        * 0.8
    )

    if vi_tri < 30:

        return None

    if vi_tri >= len(
        sach
    ):

        return None

    X_train = X.iloc[
        :vi_tri
    ]

    X_test = X.iloc[
        vi_tri:
    ]

    y_train = y.iloc[
        :vi_tri
    ]

    y_test = y.iloc[
        vi_tri:
    ]

    try:

        mo_hinh_ols = (
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

        mo_hinh_rung = (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        )

        mo_hinh_rung.fit(
            X_train,
            y_train,
        )

        du_bao_kiem_tra = (
            mo_hinh_rung.predict(
                X_test
            )
        )

        sai_so = float(
            mean_absolute_error(
                y_test,
                du_bao_kiem_tra,
            )
        )

        try:

            r2 = float(
                r2_score(
                    y_test,
                    du_bao_kiem_tra,
                )
            )

        except Exception:

            r2 = float(
                "nan"
            )

        du_bao_tiep = float(
            mo_hinh_rung.predict(
                X.iloc[
                    [
                        -1
                    ]
                ]
            )[0]
        )

        tam_quan_trong = (
            pd.Series(
                mo_hinh_rung.feature_importances_,
                index=cac_cot,
            )
            .sort_values(
                ascending=False
            )
        )

        return (
            mo_hinh_ols,
            mo_hinh_rung,
            {
                "MAE": sai_so,
                "R2": r2,
            },
            du_bao_tiep,
            tam_quan_trong,
        )

    except Exception:

        return None
