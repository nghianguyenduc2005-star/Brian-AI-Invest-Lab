from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from data.market import (
    load_market_data,
    normalize_symbol,
)


# ============================================================
# CẤU HÌNH
# ============================================================

# ------------------------------------------------------------
# API Vnstock Guest thực tế có thể bị giới hạn 20 request/phút.
#
# Không chạy sát quota.
# Ta chủ động giới hạn khoảng 10 request/phút.
# ------------------------------------------------------------

KHOANG_CACH_GIUA_REQUEST = 6.0

# Chỉ lấy 8 mã trong một lần refresh.
# 8 request x 6 giây ~= 42 giây.
SO_MA_THI_TRUONG = 8

# Cache từng mã.
THOI_GIAN_CACHE_MA = 300

# Cache toàn bộ bảng.
THOI_GIAN_CACHE_TONG = 600

# Cache universe.
THOI_GIAN_CACHE_DANH_SACH = 24 * 60 * 60


# ============================================================
# UNIVERSE CỐ ĐỊNH
# ============================================================
#
# KHÔNG GỌI Listing API.
#
# Mục đích:
# - không ăn quota thêm
# - không phải request metadata
# - luôn có danh sách mã để chạy
#
# Metadata sàn/ngành ở đây là metadata tĩnh phục vụ filter.
# Giá, volume, change vẫn lấy trực tiếp từ market API.
# ============================================================

UNIVERSE = [
    {
        "ma": "VIC",
        "ten_doanh_nghiep": "Vingroup",
        "san": "HOSE",
        "ma_nganh": "Bất động sản",
        "ten_nganh": "Bất động sản",
    },
    {
        "ma": "VHM",
        "ten_doanh_nghiep": "Vinhomes",
        "san": "HOSE",
        "ma_nganh": "Bất động sản",
        "ten_nganh": "Bất động sản",
    },
    {
        "ma": "VCB",
        "ten_doanh_nghiep": "Vietcombank",
        "san": "HOSE",
        "ma_nganh": "Ngân hàng",
        "ten_nganh": "Ngân hàng",
    },
    {
        "ma": "TCB",
        "ten_doanh_nghiep": "Techcombank",
        "san": "HOSE",
        "ma_nganh": "Ngân hàng",
        "ten_nganh": "Ngân hàng",
    },
    {
        "ma": "HPG",
        "ten_doanh_nghiep": "Hoa Phat Group",
        "san": "HOSE",
        "ma_nganh": "Thép",
        "ten_nganh": "Thép",
    },
    {
        "ma": "FPT",
        "ten_doanh_nghiep": "FPT",
        "san": "HOSE",
        "ma_nganh": "Công nghệ",
        "ten_nganh": "Công nghệ",
    },
    {
        "ma": "VNM",
        "ten_doanh_nghiep": "Vinamilk",
        "san": "HOSE",
        "ma_nganh": "Thực phẩm",
        "ten_nganh": "Thực phẩm",
    },
    {
        "ma": "MWG",
        "ten_doanh_nghiep": "Mobile World",
        "san": "HOSE",
        "ma_nganh": "Bán lẻ",
        "ten_nganh": "Bán lẻ",
    },
]


# ============================================================
# TIỆN ÍCH
# ============================================================

def _so(
    gia_tri: Any,
    mac_dinh=np.nan,
):
    try:

        gia_tri = float(
            gia_tri
        )

        if not np.isfinite(
            gia_tri
        ):
            return mac_dinh

        return gia_tri

    except Exception:

        return mac_dinh


def _tim_cot(
    du_lieu: pd.DataFrame,
    *ten_cot,
):
    if (
        du_lieu is None
        or not isinstance(
            du_lieu,
            pd.DataFrame,
        )
        or du_lieu.empty
    ):
        return None

    anh_xa = {
        str(cot).strip().lower(): cot
        for cot in du_lieu.columns
    }

    for ten in ten_cot:

        khoa = (
            str(ten)
            .strip()
            .lower()
        )

        if khoa in anh_xa:
            return anh_xa[khoa]

    return None


def _chuan_ma(
    gia_tri,
):
    if gia_tri is None:
        return ""

    try:

        return normalize_symbol(
            gia_tri
        )

    except Exception:

        return (
            str(gia_tri)
            .strip()
            .upper()
            .replace(
                ".VN",
                "",
            )
            .replace(
                " ",
                "",
            )
        )


# ============================================================
# DANH SÁCH MÃ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_DANH_SACH,
    show_spinner=False,
)
def lay_danh_sach_ma():

    return pd.DataFrame(
        UNIVERSE
    ).copy()


# ============================================================
# RATE LIMITER
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def _bo_dieu_tiet_request():

    return {
        "lan_cuoi": 0.0,
    }


def _cho_request():

    state = _bo_dieu_tiet_request()

    now = time.monotonic()

    lan_cuoi = float(
        state.get(
            "lan_cuoi",
            0.0,
        )
    )

    can_cho = (
        KHOANG_CACH_GIUA_REQUEST
        - (
            now
            - lan_cuoi
        )
    )

    if can_cho > 0:

        time.sleep(
            can_cho
        )

    state["lan_cuoi"] = (
        time.monotonic()
    )


# ============================================================
# LẤY MỘT MÃ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_MA,
    show_spinner=False,
)
def _lay_mot_ma(
    ma,
):

    ma = _chuan_ma(
        ma
    )

    if not ma:
        return None

    # --------------------------------------------------------
    # Điều tiết request.
    #
    # QUAN TRỌNG:
    # mỗi mã chỉ được gọi tuần tự.
    # --------------------------------------------------------

    _cho_request()

    try:

        df = load_market_data(
            ma,
            "5d",
        )

    except Exception:

        return None

    if (
        df is None
        or df.empty
    ):
        return None

    try:

        dong = df.iloc[-1]

    except Exception:

        return None

    if len(df) >= 2:

        dong_truoc = (
            df.iloc[-2]
        )

    else:

        dong_truoc = dong

    gia = _so(
        dong.get(
            "Close"
        )
    )

    if (
        not np.isfinite(gia)
        or gia <= 0
    ):
        return None

    gia_truoc = _so(
        dong_truoc.get(
            "Close"
        )
    )

    if (
        np.isfinite(
            gia_truoc
        )
        and gia_truoc != 0
    ):

        thay_doi = (
            gia
            - gia_truoc
        )

        thay_doi_pct = (
            gia
            / gia_truoc
            - 1
        ) * 100

    else:

        thay_doi = np.nan
        thay_doi_pct = np.nan

    gia_mo = _so(
        dong.get(
            "Open"
        )
    )

    gia_cao = _so(
        dong.get(
            "High"
        )
    )

    gia_thap = _so(
        dong.get(
            "Low"
        )
    )

    volume = _so(
        dong.get(
            "Volume"
        )
    )

    # --------------------------------------------------------
    # VALUE
    # --------------------------------------------------------

    cot_value = _tim_cot(
        df,
        "Value",
        "value",
        "value_traded",
        "trading_value",
        "traded_value",
        "match_value",
        "matchvalue",
        "turnover",
    )

    value = np.nan

    if cot_value is not None:

        value = _so(
            dong.get(
                cot_value
            )
        )

    # --------------------------------------------------------
    # Nếu API OHLCV không trả Value,
    # dùng Close x Volume làm giá trị ước tính.
    #
    # Không coi đây là giá trị trực tiếp từ API.
    # --------------------------------------------------------

    value_estimated = False

    if (
        not np.isfinite(
            value
        )
        and np.isfinite(
            volume
        )
    ):

        value = (
            gia
            * volume
        )

        value_estimated = True

    return {
        "ma": ma,

        "gia": gia,

        "gia_tham_chieu": (
            gia_truoc
            if np.isfinite(
                gia_truoc
            )
            else np.nan
        ),

        "gia_mo_cua": gia_mo,

        "gia_cao_nhat": gia_cao,

        "gia_thap_nhat": gia_thap,

        "thay_doi": thay_doi,

        "thay_doi_pct": thay_doi_pct,

        "khoi_luong": volume,

        "gia_tri_giao_dich": value,

        "gia_tri_uoc_tinh": value_estimated,

        "RSI": _so(
            dong.get(
                "RSI"
            )
        ),

        "MACD": _so(
            dong.get(
                "MACD"
            )
        ),

        "SMA20": _so(
            dong.get(
                "SMA20"
            )
        ),

        "SMA50": _so(
            dong.get(
                "SMA50"
            )
        ),

        "EMA20": _so(
            dong.get(
                "EMA20"
            )
        ),

        "EMA50": _so(
            dong.get(
                "EMA50"
            )
        ),

        "Volatility20": _so(
            dong.get(
                "Volatility20"
            )
        ),

        "ATR14": _so(
            dong.get(
                "ATR14"
            )
        ),

        "Relative_Volume": _so(
            dong.get(
                "Relative_Volume"
            )
        ),
    }


# ============================================================
# TẠO BẢNG GIÁ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_TONG,
    show_spinner=False,
)
def lay_bang_gia_toan_thi_truong(
    force_reload=False,
):

    # Dùng tham số để tạo cache key khác khi force reload.
    _ = force_reload

    metadata = lay_danh_sach_ma()

    if (
        metadata is None
        or metadata.empty
    ):

        raise RuntimeError(
            "Danh sách mã thị trường đang rỗng."
        )

    danh_sach = (
        metadata[
            "ma"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    danh_sach = [
        _chuan_ma(ma)
        for ma in danh_sach
        if _chuan_ma(ma)
    ]

    danh_sach = danh_sach[
        :SO_MA_THI_TRUONG
    ]

    if not danh_sach:

        raise RuntimeError(
            "Không có mã cổ phiếu để tải."
        )

    ket_qua = []

    # ========================================================
    # TUẦN TỰ
    # ========================================================

    for ma in danh_sach:

        item = _lay_mot_ma(
            ma
        )

        if item is not None:

            ket_qua.append(
                item
            )

    if not ket_qua:

        raise RuntimeError(
            "Không lấy được dữ liệu giá."
        )

    bang_gia = pd.DataFrame(
        ket_qua
    )

    # ========================================================
    # GHÉP METADATA
    # ========================================================

    bang_gia = bang_gia.merge(
        metadata[
            [
                "ma",
                "ten_doanh_nghiep",
                "san",
                "ma_nganh",
                "ten_nganh",
            ]
        ],
        on="ma",
        how="left",
    )

    # ========================================================
    # TEXT
    # ========================================================

    for cot in [
        "ten_doanh_nghiep",
        "san",
        "ma_nganh",
        "ten_nganh",
    ]:

        if cot not in bang_gia.columns:

            bang_gia[
                cot
            ] = ""

        bang_gia[
            cot
        ] = (
            bang_gia[
                cot
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # ========================================================
    # NUMBER
    # ========================================================

    for cot in [
        "gia",
        "gia_tham_chieu",
        "gia_mo_cua",
        "gia_cao_nhat",
        "gia_thap_nhat",
        "thay_doi",
        "thay_doi_pct",
        "khoi_luong",
        "gia_tri_giao_dich",
        "RSI",
        "MACD",
        "SMA20",
        "SMA50",
        "EMA20",
        "EMA50",
        "Volatility20",
        "ATR14",
        "Relative_Volume",
    ]:

        if cot in bang_gia.columns:

            bang_gia[
                cot
            ] = pd.to_numeric(
                bang_gia[
                    cot
                ],
                errors="coerce",
            )

    # ========================================================
    # TRẠNG THÁI
    # ========================================================

    thay_doi = bang_gia[
        "thay_doi_pct"
    ]

    bang_gia[
        "trang_thai"
    ] = np.select(
        [
            thay_doi > 0.05,
            thay_doi < -0.05,
        ],
        [
            "Tăng",
            "Giảm",
        ],
        default="Đứng giá",
    )

    # ========================================================
    # CHUẨN HÓA
    # ========================================================

    bang_gia = (
        bang_gia
        .drop_duplicates(
            "ma",
            keep="last",
        )
        .sort_values(
            "thay_doi_pct",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return bang_gia


# ============================================================
# LỌC
# ============================================================

def loc_bang_gia(
    bang_gia,
    san="Tất cả",
    tu_khoa="",
    nganh="Tất cả",
    huong="Tất cả",
):

    if (
        bang_gia is None
        or bang_gia.empty
    ):

        return pd.DataFrame()

    df = bang_gia.copy()

    # --------------------------------------------------------
    # Từ khóa
    # --------------------------------------------------------

    keyword = str(
        tu_khoa or ""
    ).strip()

    if keyword:

        mask_ma = (
            df[
                "ma"
            ]
            .fillna("")
            .astype(str)
            .str.contains(
                keyword,
                case=False,
                regex=False,
                na=False,
            )
        )

        mask_ten = pd.Series(
            False,
            index=df.index,
        )

        if (
            "ten_doanh_nghiep"
            in df.columns
        ):

            mask_ten = (
                df[
                    "ten_doanh_nghiep"
                ]
                .fillna("")
                .astype(str)
                .str.contains(
                    keyword,
                    case=False,
                    regex=False,
                    na=False,
                )
            )

        df = df[
            mask_ma
            | mask_ten
        ].copy()

    # --------------------------------------------------------
    # Sàn
    # --------------------------------------------------------

    if (
        san != "Tất cả"
        and "san" in df.columns
    ):

        df = df[
            df[
                "san"
            ]
            .astype(str)
            .str.upper()
            .eq(
                str(
                    san
                ).upper()
            )
        ].copy()

    # --------------------------------------------------------
    # Ngành
    # --------------------------------------------------------

    if (
        nganh != "Tất cả"
        and "ten_nganh"
        in df.columns
    ):

        df = df[
            df[
                "ten_nganh"
            ]
            .astype(str)
            .eq(
                str(
                    nganh
                )
            )
        ].copy()

    # --------------------------------------------------------
    # Trạng thái
    # --------------------------------------------------------

    if (
        huong != "Tất cả"
        and "trang_thai"
        in df.columns
    ):

        df = df[
            df[
                "trang_thai"
            ].eq(
                huong
            )
        ].copy()

    return df.reset_index(
        drop=True
    )


# ============================================================
# THỐNG KÊ
# ============================================================

def thong_ke_thi_truong(
    bang_gia,
):

    if (
        bang_gia is None
        or bang_gia.empty
    ):

        return {
            "tong_ma": 0,
            "co_du_lieu_gia": 0,
            "tang": 0,
            "dung_gia": 0,
            "giam": 0,
            "tong_khoi_luong": 0.0,
            "tong_gia_tri": 0.0,
            "phan_tram_tang": 0.0,
            "phan_tram_giam": 0.0,
        }

    change = pd.to_numeric(
        bang_gia[
            "thay_doi_pct"
        ],
        errors="coerce",
    )

    volume = pd.to_numeric(
        bang_gia[
            "khoi_luong"
        ],
        errors="coerce",
    )

    value = pd.to_numeric(
        bang_gia[
            "gia_tri_giao_dich"
        ],
        errors="coerce",
    )

    total_valid = int(
        change.notna().sum()
    )

    up = int(
        (
            change > 0.05
        ).sum()
    )

    down = int(
        (
            change < -0.05
        ).sum()
    )

    flat = int(
        change.between(
            -0.05,
            0.05,
        ).sum()
    )

    return {
        "tong_ma": len(
            bang_gia
        ),

        "co_du_lieu_gia": total_valid,

        "tang": up,

        "dung_gia": flat,

        "giam": down,

        "tong_khoi_luong": (
            volume.sum(
                min_count=1
            )
        ),

        "tong_gia_tri": (
            value.sum(
                min_count=1
            )
        ),

        "phan_tram_tang": (
            up
            / total_valid
            * 100
            if total_valid
            else 0.0
        ),

        "phan_tram_giam": (
            down
            / total_valid
            * 100
            if total_valid
            else 0.0
        ),
    }


# ============================================================
# TÂM LÝ
# ============================================================

def tinh_diem_tam_ly(
    bang_gia,
):

    stats = thong_ke_thi_truong(
        bang_gia
    )

    total = stats[
        "co_du_lieu_gia"
    ]

    if total <= 0:
        return 50.0

    score = (
        50
        + (
            stats["tang"]
            - stats["giam"]
        )
        / total
        * 50
    )

    return float(
        max(
            0,
            min(
                100,
                score,
            ),
        )
    )


def nhan_diem_tam_ly(
    diem,
):

    diem = _so(
        diem,
        50,
    )

    if diem >= 80:
        return "Rất tích cực"

    if diem >= 65:
        return "Tích cực"

    if diem >= 55:
        return "Nghiêng tích cực"

    if diem >= 45:
        return "Trung tính"

    if diem >= 35:
        return "Nghiêng tiêu cực"

    if diem >= 20:
        return "Tiêu cực"

    return "Rất tiêu cực"


# ============================================================
# TOP TĂNG
# ============================================================

def top_tang(
    bang_gia,
    so_luong=10,
):

    if (
        bang_gia is None
        or bang_gia.empty
    ):

        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "thay_doi_pct",
            ascending=False,
            na_position="last",
        )
        .head(
            so_luong
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TOP GIẢM
# ============================================================

def top_giam(
    bang_gia,
    so_luong=10,
):

    if (
        bang_gia is None
        or bang_gia.empty
    ):

        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "thay_doi_pct",
            ascending=True,
            na_position="last",
        )
        .head(
            so_luong
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TOP KHỐI LƯỢNG
# ============================================================

def top_khoi_luong(
    bang_gia,
    so_luong=10,
):

    if (
        bang_gia is None
        or bang_gia.empty
    ):

        return pd.DataFrame()

    df = bang_gia.copy()

    df[
        "khoi_luong"
    ] = pd.to_numeric(
        df[
            "khoi_luong"
        ],
        errors="coerce",
    )

    df = df[
        df[
            "khoi_luong"
        ].notna()
        & (
            df[
                "khoi_luong"
            ]
            > 0
        )
    ].copy()

    return (
        df
        .sort_values(
            "khoi_luong",
            ascending=False,
        )
        .head(
            so_luong
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TOP GIÁ TRỊ
# ============================================================

def top_gia_tri_giao_dich(
    bang_gia,
    so_luong=10,
):

    if (
        bang_gia is None
        or bang_gia.empty
    ):

        return pd.DataFrame()

    df = bang_gia.copy()

    df[
        "gia_tri_giao_dich"
    ] = pd.to_numeric(
        df[
            "gia_tri_giao_dich"
        ],
        errors="coerce",
    )

    df = df[
        df[
            "gia_tri_giao_dich"
        ].notna()
        & (
            df[
                "gia_tri_giao_dich"
            ]
            > 0
        )
    ].copy()

    return (
        df
        .sort_values(
            "gia_tri_giao_dich",
            ascending=False,
        )
        .head(
            so_luong
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# THEO SÀN
# ============================================================

def thong_ke_theo_san(
    bang_gia,
):

    if (
        bang_gia is None
        or bang_gia.empty
        or "san" not in bang_gia.columns
    ):

        return pd.DataFrame()

    return (
        bang_gia
        .groupby(
            "san",
            dropna=False,
        )
        .agg(
            so_ma=(
                "ma",
                "count",
            ),

            tang=(
                "thay_doi_pct",
                lambda x: int(
                    (
                        x > 0.05
                    ).sum()
                ),
            ),

            dung_gia=(
                "thay_doi_pct",
                lambda x: int(
                    x.between(
                        -0.05,
                        0.05,
                    ).sum()
                ),
            ),

            giam=(
                "thay_doi_pct",
                lambda x: int(
                    (
                        x < -0.05
                    ).sum()
                ),
            ),

            bien_dong_binh_quan=(
                "thay_doi_pct",
                "mean",
            ),

            tong_khoi_luong=(
                "khoi_luong",
                "sum",
            ),

            tong_gia_tri=(
                "gia_tri_giao_dich",
                "sum",
            ),
        )
        .reset_index()
    )


# ============================================================
# THEO NGÀNH
# ============================================================

def thong_ke_theo_nganh(
    bang_gia,
):

    if (
        bang_gia is None
        or bang_gia.empty
        or "ten_nganh"
        not in bang_gia.columns
    ):

        return pd.DataFrame()

    df = bang_gia.copy()

    df[
        "ten_nganh"
    ] = (
        df[
            "ten_nganh"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df[
            "ten_nganh"
        ] != ""
    ].copy()

    if df.empty:

        return pd.DataFrame()

    return (
        df
        .groupby(
            "ten_nganh"
        )
        .agg(
            so_ma=(
                "ma",
                "count",
            ),

            tang=(
                "thay_doi_pct",
                lambda x: int(
                    (
                        x > 0.05
                    ).sum()
                ),
            ),

            dung_gia=(
                "thay_doi_pct",
                lambda x: int(
                    x.between(
                        -0.05,
                        0.05,
                    ).sum()
                ),
            ),

            giam=(
                "thay_doi_pct",
                lambda x: int(
                    (
                        x < -0.05
                    ).sum()
                ),
            ),

            bien_dong_binh_quan=(
                "thay_doi_pct",
                "mean",
            ),

            tong_khoi_luong=(
                "khoi_luong",
                "sum",
            ),

            tong_gia_tri=(
                "gia_tri_giao_dich",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "bien_dong_binh_quan",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# NƯỚC NGOÀI
# ============================================================

def thong_ke_nuoc_ngoai(
    bang_gia,
):

    # Chưa có endpoint foreign flow riêng.
    # Không tạo số giả.

    return {
        "co_du_lieu": False,
        "mua": np.nan,
        "ban": np.nan,
        "rong": np.nan,
    }


# ============================================================
# SOURCE
# ============================================================

def thong_tin_nguon(
    bang_gia,
):

    so_ma = 0

    if (
        isinstance(
            bang_gia,
            pd.DataFrame,
        )
        and not bang_gia.empty
        and "ma" in bang_gia.columns
    ):

        so_ma = int(
            bang_gia[
                "ma"
            ].nunique()
        )

    return {
        "so_ma": so_ma,

        "cap_nhat": pd.Timestamp.now(),

        "nguon": (
            "Vnstock · OHLCV"
        ),
    }


# ============================================================
# API CHÍNH
# ============================================================

def lay_market_overview(
    force_reload=False,
):

    bang_gia = (
        lay_bang_gia_toan_thi_truong(
            force_reload=force_reload
        )
    )

    tam_ly = (
        tinh_diem_tam_ly(
            bang_gia
        )
    )

    return {
        "bang_gia": bang_gia,

        "universe": (
            lay_danh_sach_ma()
        ),

        "thong_ke": (
            thong_ke_thi_truong(
                bang_gia
            )
        ),

        "tam_ly": tam_ly,

        "nhan_tam_ly": (
            nhan_diem_tam_ly(
                tam_ly
            )
        ),

        "theo_san": (
            thong_ke_theo_san(
                bang_gia
            )
        ),

        "theo_nganh": (
            thong_ke_theo_nganh(
                bang_gia
            )
        ),

        "top_tang": (
            top_tang(
                bang_gia
            )
        ),

        "top_giam": (
            top_giam(
                bang_gia
            )
        ),

        "top_khoi_luong": (
            top_khoi_luong(
                bang_gia
            )
        ),

        "top_gia_tri": (
            top_gia_tri_giao_dich(
                bang_gia
            )
        ),

        "nuoc_ngoai": (
            thong_ke_nuoc_ngoai(
                bang_gia
            )
        ),

        "nguon": (
            thong_tin_nguon(
                bang_gia
            )
        ),
    }
