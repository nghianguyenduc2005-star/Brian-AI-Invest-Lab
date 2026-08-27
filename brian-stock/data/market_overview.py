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

# Cache dữ liệu từng mã.
# load_market_data() bên data.market cũng đã có cache riêng,
# nhưng giữ cache ở lớp này để tránh xử lý lại nhiều lần.
THOI_GIAN_CACHE_MA = 300

# Cache bảng tổng hợp.
THOI_GIAN_CACHE_TONG = 600

# Cache danh sách mã.
THOI_GIAN_CACHE_DANH_SACH = 6 * 60 * 60

# ============================================================
# GIỚI HẠN API
# ============================================================

# API chỉ cho khoảng 40 request/phút.
# Giữ mức an toàn dưới giới hạn.
MAX_REQUESTS_PER_MINUTE = 35

# 60 / 35 ~= 1.71 giây/request.
KHOANG_CACH_GIUA_REQUEST = 1.75

# Không tải 60 mã cùng lúc nữa.
# 30 mã ~ 52 giây trong trường hợp toàn bộ đều phải gọi API.
# Lần sau các mã được cache nên nhanh hơn nhiều.
SO_MA_THI_TRUONG = 30


# ============================================================
# DANH SÁCH ƯU TIÊN
# ============================================================

DANH_SACH_MA_UU_TIEN = [
    "VIC",
    "VHM",
    "VRE",
    "VCB",
    "BID",
    "CTG",
    "MBB",
    "TCB",
    "VPB",
    "ACB",
    "HDB",
    "STB",
    "SSB",
    "TPB",
    "LPB",
    "VIB",
    "MSB",
    "SHB",
    "HPG",
    "HSG",
    "NKG",
    "GAS",
    "PLX",
    "POW",
    "PVD",
    "FPT",
    "MWG",
    "MSN",
    "VNM",
    "SSI",
]


# ============================================================
# TIỆN ÍCH
# ============================================================

def _so(
    gia_tri: Any,
    mac_dinh=np.nan,
):
    try:
        gia_tri = float(gia_tri)

        if not np.isfinite(gia_tri):
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


def _chuan_san(
    gia_tri,
):
    text = str(
        gia_tri or ""
    ).strip().upper()

    if text in {
        "HSX",
        "HOSE",
        "HO CHI MINH",
        "HOCHIMINH",
    }:
        return "HOSE"

    if text in {
        "HNX",
        "HA NOI",
        "HANOI",
    }:
        return "HNX"

    if text == "UPCOM":
        return "UPCOM"

    return text


# ============================================================
# LISTING
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_DANH_SACH,
    show_spinner=False,
)
def _lay_listing():

    try:

        from vnstock import Listing

        listing = Listing(
            source="VCI"
        )

        bang = listing.all_symbols(
            to_df=True
        )

        if (
            isinstance(
                bang,
                pd.DataFrame,
            )
            and not bang.empty
        ):

            return bang.copy()

    except Exception:
        pass

    return pd.DataFrame()


# ============================================================
# DANH SÁCH MÃ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_DANH_SACH,
    show_spinner=False,
)
def lay_danh_sach_ma():

    bang_listing = _lay_listing()

    rows = []

    # --------------------------------------------------------
    # Lấy metadata từ Listing
    # --------------------------------------------------------

    if not bang_listing.empty:

        cot_ma = _tim_cot(
            bang_listing,
            "symbol",
            "ticker",
            "code",
        )

        cot_ten = _tim_cot(
            bang_listing,
            "organ_name",
            "company_name",
            "name",
        )

        cot_san = _tim_cot(
            bang_listing,
            "exchange",
            "floor",
        )

        cot_nganh = _tim_cot(
            bang_listing,
            "icb_name",
            "industry_name",
            "industry",
        )

        cot_ma_nganh = _tim_cot(
            bang_listing,
            "icb_code",
            "industry_code",
        )

        cot_von_hoa = _tim_cot(
            bang_listing,
            "market_cap",
        )

        if cot_ma is not None:

            for _, dong in (
                bang_listing.iterrows()
            ):

                ma = _chuan_ma(
                    dong.get(
                        cot_ma
                    )
                )

                if not ma:
                    continue

                ten_doanh_nghiep = ""

                if (
                    cot_ten is not None
                    and pd.notna(
                        dong.get(
                            cot_ten
                        )
                    )
                ):

                    ten_doanh_nghiep = (
                        str(
                            dong.get(
                                cot_ten
                            )
                        )
                        .strip()
                    )

                san = ""

                if (
                    cot_san is not None
                    and pd.notna(
                        dong.get(
                            cot_san
                        )
                    )
                ):

                    san = _chuan_san(
                        dong.get(
                            cot_san
                        )
                    )

                ten_nganh = ""

                if (
                    cot_nganh is not None
                    and pd.notna(
                        dong.get(
                            cot_nganh
                        )
                    )
                ):

                    ten_nganh = (
                        str(
                            dong.get(
                                cot_nganh
                            )
                        )
                        .strip()
                    )

                ma_nganh = ""

                if (
                    cot_ma_nganh is not None
                    and pd.notna(
                        dong.get(
                            cot_ma_nganh
                        )
                    )
                ):

                    ma_nganh = (
                        str(
                            dong.get(
                                cot_ma_nganh
                            )
                        )
                        .strip()
                    )

                von_hoa = np.nan

                if cot_von_hoa is not None:

                    von_hoa = _so(
                        dong.get(
                            cot_von_hoa
                        )
                    )

                rows.append(
                    {
                        "ma": ma,
                        "ten_doanh_nghiep": (
                            ten_doanh_nghiep
                        ),
                        "san": san,
                        "ma_nganh": ma_nganh,
                        "ten_nganh": ten_nganh,
                        "von_hoa": von_hoa,
                    }
                )

    metadata = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Fallback nếu Listing không hoạt động
    # --------------------------------------------------------

    if metadata.empty:

        metadata = pd.DataFrame(
            {
                "ma": [
                    _chuan_ma(
                        ma
                    )
                    for ma
                    in DANH_SACH_MA_UU_TIEN
                ],

                "ten_doanh_nghiep": "",

                "san": "",

                "ma_nganh": "",

                "ten_nganh": "",

                "von_hoa": np.nan,
            }
        )

    metadata = (
        metadata
        .drop_duplicates(
            "ma",
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    ma_co_san = set(
        metadata[
            "ma"
        ].tolist()
    )

    danh_sach = []

    # --------------------------------------------------------
    # Ưu tiên các mã lớn
    # --------------------------------------------------------

    for ma in DANH_SACH_MA_UU_TIEN:

        ma = _chuan_ma(
            ma
        )

        if (
            ma
            and ma in ma_co_san
            and ma not in danh_sach
        ):

            danh_sach.append(
                ma
            )

    # --------------------------------------------------------
    # Bổ sung từ Listing
    # --------------------------------------------------------

    for ma in metadata[
        "ma"
    ].tolist():

        if (
            ma
            and ma not in danh_sach
        ):

            danh_sach.append(
                ma
            )

        if len(danh_sach) >= SO_MA_THI_TRUONG:
            break

    danh_sach = danh_sach[
        :SO_MA_THI_TRUONG
    ]

    metadata = metadata[
        metadata[
            "ma"
        ].isin(
            danh_sach
        )
    ].copy()

    # Giữ thứ tự ưu tiên
    thu_tu = {
        ma: index
        for index, ma
        in enumerate(
            danh_sach
        )
    }

    metadata[
        "_thu_tu"
    ] = metadata[
        "ma"
    ].map(
        thu_tu
    )

    metadata = (
        metadata
        .sort_values(
            "_thu_tu"
        )
        .drop(
            columns=[
                "_thu_tu"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return metadata


# ============================================================
# HẠN CHẾ TỐC ĐỘ REQUEST
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def _bo_dieu_tiet_request():
    return {
        "lan_cuoi": 0.0
    }


def _cho_request():

    state = _bo_dieu_tiet_request()

    now = time.monotonic()

    lan_cuoi = state.get(
        "lan_cuoi",
        0.0,
    )

    thoi_gian_can_cho = (
        KHOANG_CACH_GIUA_REQUEST
        - (
            now
            - lan_cuoi
        )
    )

    if thoi_gian_can_cho > 0:

        time.sleep(
            thoi_gian_can_cho
        )

    state["lan_cuoi"] = (
        time.monotonic()
    )


# ============================================================
# LẤY DỮ LIỆU MỘT MÃ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_MA,
    show_spinner=False,
)
def _lay_mot_ma_cached(
    ma,
):

    _cho_request()

    try:

        du_lieu = load_market_data(
            ma,
            "5d",
        )

    except Exception:

        return None

    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return None

    try:

        dong = du_lieu.iloc[-1]

    except Exception:

        return None

    if len(du_lieu) >= 2:

        dong_truoc = (
            du_lieu.iloc[-2]
        )

    else:

        dong_truoc = dong

    gia = _so(
        dong.get("Close")
    )

    if (
        not np.isfinite(gia)
        or gia <= 0
    ):
        return None

    gia_truoc = _so(
        dong_truoc.get("Close")
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

    volume = _so(
        dong.get("Volume")
    )

    # --------------------------------------------------------
    # Value nếu nguồn có
    # --------------------------------------------------------

    value_column = _tim_cot(
        du_lieu,
        "Value",
        "value",
        "ValueTraded",
        "value_traded",
        "trading_value",
        "traded_value",
        "turnover",
    )

    value = np.nan

    if value_column is not None:

        try:

            value = _so(
                dong.get(
                    value_column
                )
            )

        except Exception:

            value = np.nan

    # --------------------------------------------------------
    # Fallback:
    # giá * khối lượng
    #
    # Đây là giá trị ước tính, không phải số liệu trực tiếp
    # nếu nguồn không có cột Value.
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
        "ma": _chuan_ma(ma),

        "gia": gia,

        "gia_tham_chieu": (
            gia_truoc
            if np.isfinite(
                gia_truoc
            )
            else np.nan
        ),

        "gia_mo_cua": _so(
            dong.get("Open")
        ),

        "gia_cao_nhat": _so(
            dong.get("High")
        ),

        "gia_thap_nhat": _so(
            dong.get("Low")
        ),

        "thay_doi": thay_doi,

        "thay_doi_pct": thay_doi_pct,

        "khoi_luong": volume,

        "gia_tri_giao_dich": value,

        "gia_tri_uoc_tinh": (
            value_estimated
        ),

        "rsi": _so(
            dong.get("RSI")
        ),

        "macd": _so(
            dong.get("MACD")
        ),

        "sma20": _so(
            dong.get("SMA20")
        ),

        "sma50": _so(
            dong.get("SMA50")
        ),

        "ema20": _so(
            dong.get("EMA20")
        ),

        "ema50": _so(
            dong.get("EMA50")
        ),

        "volatility20": _so(
            dong.get("Volatility20")
        ),

        "atr14": _so(
            dong.get("ATR14")
        ),

        "relative_volume": _so(
            dong.get(
                "Relative_Volume"
            )
        ),
    }


# ============================================================
# BẢNG GIÁ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_TONG,
    show_spinner=False,
)
def lay_bang_gia_toan_thi_truong(
    force_reload=False,
):
    """
    Lấy dữ liệu thị trường theo kiểu an toàn với API.

    QUAN TRỌNG:
    - Không dùng ThreadPoolExecutor.
    - Không request song song.
    - Tối đa 30 mã trong một lần tải.
    - Mỗi request được điều tiết khoảng 1.75 giây.
    """

    # force_reload chỉ dùng để thay đổi cache key.
    _ = force_reload

    metadata = lay_danh_sach_ma()

    if (
        metadata is None
        or metadata.empty
    ):

        raise RuntimeError(
            "Không có danh sách mã cổ phiếu."
        )

    danh_sach_ma = (
        metadata[
            "ma"
        ]
        .dropna()
        .astype(str)
        .str.upper()
        .tolist()
    )

    danh_sach_ma = danh_sach_ma[
        :SO_MA_THI_TRUONG
    ]

    if not danh_sach_ma:

        raise RuntimeError(
            "Danh sách mã cổ phiếu đang rỗng."
        )

    ket_qua = []

    # ========================================================
    # TUẦN TỰ
    # ========================================================

    for index, ma in enumerate(
        danh_sach_ma,
        start=1,
    ):

        try:

            item = _lay_mot_ma_cached(
                ma
            )

        except Exception:

            item = None

        if item is not None:

            ket_qua.append(
                item
            )

    if not ket_qua:

        raise RuntimeError(
            "Không lấy được dữ liệu giá hợp lệ."
        )

    bang_gia = pd.DataFrame(
        ket_qua
    )

    # ========================================================
    # GHÉP METADATA
    # ========================================================

    bang_ghep = metadata[
        [
            "ma",
            "ten_doanh_nghiep",
            "san",
            "ma_nganh",
            "ten_nganh",
            "von_hoa",
        ]
    ].copy()

    bang_gia = bang_gia.merge(
        bang_ghep,
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
            bang_gia[cot] = ""

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
        "von_hoa",
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
    # SẮP XẾP
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
# LỌC BẢNG
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
    # TỪ KHÓA
    # --------------------------------------------------------

    tu_khoa = str(
        tu_khoa or ""
    ).strip()

    if tu_khoa:

        mask_ma = (
            df[
                "ma"
            ]
            .fillna("")
            .astype(str)
            .str.contains(
                tu_khoa,
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
                    tu_khoa,
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
    # SÀN
    # --------------------------------------------------------

    if (
        san != "Tất cả"
        and "san" in df.columns
    ):

        df = df[
            df[
                "san"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq(
                str(
                    san
                ).upper()
            )
        ].copy()

    # --------------------------------------------------------
    # NGÀNH
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
            .fillna("")
            .astype(str)
            .eq(
                str(nganh)
            )
        ].copy()

    # --------------------------------------------------------
    # TRẠNG THÁI
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
# THỐNG KÊ THỊ TRƯỜNG
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

    thay_doi = pd.to_numeric(
        bang_gia[
            "thay_doi_pct"
        ],
        errors="coerce",
    )

    khoi_luong = pd.to_numeric(
        bang_gia[
            "khoi_luong"
        ],
        errors="coerce",
    )

    gia_tri = pd.to_numeric(
        bang_gia[
            "gia_tri_giao_dich"
        ],
        errors="coerce",
    )

    tong = int(
        thay_doi.notna().sum()
    )

    tang = int(
        (
            thay_doi > 0.05
        ).sum()
    )

    giam = int(
        (
            thay_doi < -0.05
        ).sum()
    )

    dung = int(
        thay_doi.between(
            -0.05,
            0.05,
        ).sum()
    )

    return {
        "tong_ma": len(
            bang_gia
        ),

        "co_du_lieu_gia": tong,

        "tang": tang,

        "dung_gia": dung,

        "giam": giam,

        "tong_khoi_luong": (
            khoi_luong.sum(
                min_count=1
            )
        ),

        "tong_gia_tri": (
            gia_tri.sum(
                min_count=1
            )
        ),

        "phan_tram_tang": (
            tang
            / tong
            * 100
            if tong
            else 0.0
        ),

        "phan_tram_giam": (
            giam
            / tong
            * 100
            if tong
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
        50.0
        + (
            stats["tang"]
            - stats["giam"]
        )
        / total
        * 50.0
    )

    return float(
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        )
    )


def nhan_diem_tam_ly(
    diem,
):

    diem = _so(
        diem,
        50.0,
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
# TOP
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

    # Repo hiện tại chưa có endpoint riêng cho foreign flow.
    # Không sinh dữ liệu giả.
    return {
        "co_du_lieu": False,
        "mua": np.nan,
        "ban": np.nan,
        "rong": np.nan,
    }


# ============================================================
# THÔNG TIN NGUỒN
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
            "Vnstock · dữ liệu OHLCV"
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

    tam_ly = tinh_diem_tam_ly(
        bang_gia
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
