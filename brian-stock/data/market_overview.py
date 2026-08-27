from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
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

THOI_GIAN_CACHE = 60

SO_MA_THI_TRUONG = 60

SO_LUONG_LUONG =
    12

SO_WORKER = 10


# ============================================================
# DANH SÁCH MÃ MẶC ĐỊNH
#
# Đây là tập 60 mã đại diện để page mở nhanh.
# Sau này có thể thay bằng nguồn danh sách mã toàn thị trường.
# ============================================================

DANH_SACH_MA_MAC_DINH = [
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
    "TLG",
    "DGC",
    "GAS",
    "PLX",
    "POW",
    "PVD",
    "PVT",
    "FPT",
    "CMG",
    "MWG",
    "DGW",
    "MSN",
    "VNM",
    "SAB",
    "VHC",
    "MCH",
    "SSI",
    "VND",
    "HCM",
    "VCI",
    "FTS",
    "BSI",
    "CTS",
    "VIX",
    "DSE",
    "PDR",
    "DIG",
    "DXG",
    "KDH",
    "NLG",
    "VPI",
    "VRE",
    "NVL",
    "CEO",
    "HUT",
    "CTD",
    "HHV",
    "GEX",
    "REE",
    "BMP",
    "PHR",
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
    du_lieu,
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
        str(cot)
        .strip()
        .lower(): cot
        for cot in du_lieu.columns
    }

    for ten in ten_cot:

        cot = anh_xa.get(
            str(ten)
            .strip()
            .lower()
        )

        if cot is not None:
            return cot

    return None


def _chuan_ma(
    ma,
):
    return normalize_symbol(
        ma
    )


# ============================================================
# LẤY DANH SÁCH MÃ
# ============================================================

@st.cache_data(
    ttl=6 * 60 * 60,
    show_spinner=False,
)
def lay_danh_sach_ma():
    """
    Cố lấy danh sách mã từ Listing nếu repo đang có.
    Nếu không được thì dùng danh sách 60 mã mặc định.
    """

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

            cot_ma = _tim_cot(
                bang,
                "symbol",
                "ticker",
                "code",
            )

            if cot_ma is not None:

                danh_sach = (
                    bang[
                        cot_ma
                    ]
                    .map(
                        _chuan_ma
                    )
                    .drop_duplicates()
                    .tolist()
                )

                danh_sach = [
                    ma
                    for ma
                    in danh_sach
                    if ma
                ]

                if danh_sach:

                    return danh_sach

    except Exception:
        pass

    return [
        _chuan_ma(ma)
        for ma
        in DANH_SACH_MA_MAC_DINH
    ]


# ============================================================
# LẤY MỘT MÃ
# ============================================================

def _lay_mot_ma(
    ma,
):
    try:

        du_lieu = load_market_data(
            ma,
            "5d",
        )

        if (
            du_lieu is None
            or du_lieu.empty
        ):
            return None

        dong_cuoi = (
            du_lieu.iloc[-1]
        )

        if len(
            du_lieu
        ) >= 2:

            dong_truoc = (
                du_lieu.iloc[-2]
            )

        else:

            dong_truoc = (
                dong_cuoi
            )

        gia = _so(
            dong_cuoi.get(
                "Close"
            )
        )

        if (
            not np.isfinite(
                gia
            )
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

            thay_doi_pct = (
                (
                    gia
                    / gia_truoc
                )
                - 1
            ) * 100

        else:

            thay_doi_pct = np.nan

        thay_doi = (
            gia
            - gia_truoc
            if np.isfinite(
                gia_truoc
            )
            else np.nan
        )

        volume = _so(
            dong_cuoi.get(
                "Volume"
            )
        )

        gia_tri = np.nan

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "ma": normalize_symbol(
                ma
            ),

            "gia": gia,

            "gia_tham_chieu": gia_truoc,

            "gia_mo_cua": _so(
                dong_cuoi.get(
                    "Open"
                )
            ),

            "gia_cao_nhat": _so(
                dong_cuoi.get(
                    "High"
                )
            ),

            "gia_thap_nhat": _so(
                dong_cuoi.get(
                    "Low"
                )
            ),

            "thay_doi": thay_doi,

            "thay_doi_pct": (
                thay_doi_pct
            ),

            "khoi_luong": volume,

            "gia_tri_giao_dich": (
                gia_tri
            ),

            "san": "",

            "ten_doanh_nghiep": "",

            "ma_nganh": "",

            "ten_nganh": "",

            "rsi": _so(
                dong_cuoi.get(
                    "RSI"
                )
            ),

            "macd": _so(
                dong_cuoi.get(
                    "MACD"
                )
            ),

            "sma20": _so(
                dong_cuoi.get(
                    "SMA20"
                )
            ),

            "sma50": _so(
                dong_cuoi.get(
                    "SMA50"
                )
            ),

            "ema20": _so(
                dong_cuoi.get(
                    "EMA20"
                )
            ),

            "ema50": _so(
                dong_cuoi.get(
                    "EMA50"
                )
            ),

            "volatility20": _so(
                dong_cuoi.get(
                    "Volatility20"
                )
            ),

            "atr14": _so(
                dong_cuoi.get(
                    "ATR14"
                )
            ),

            "relative_volume": _so(
                dong_cuoi.get(
                    "Relative_Volume"
                )
            ),
        }

    except Exception:
        return None


# ============================================================
# LẤY 60 MÃ SONG SONG
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE,
    show_spinner=False,
)
def lay_bang_gia_toan_thi_truong(
    force_reload=False,
):
    _ = force_reload

    danh_sach = (
        lay_danh_sach_ma()
    )

    # --------------------------------------------------------
    # Chỉ lấy 60 mã
    # --------------------------------------------------------

    danh_sach = danh_sach[
        :SO_MA_THI_TRUONG
    ]

    if not danh_sach:

        raise RuntimeError(
            "Không có danh sách mã cổ phiếu."
        )

    ket_qua = []

    # --------------------------------------------------------
    # CHẠY SONG SONG
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=SO_WORKER
    ) as bo_luong:

        cac_tac_vu = {
            bo_luong.submit(
                _lay_mot_ma,
                ma,
            ): ma
            for ma in danh_sach
        }

        for tac_vu in as_completed(
            cac_tac_vu
        ):

            try:

                ket_qua_mot_ma = (
                    tac_vu.result()
                )

                if (
                    ket_qua_mot_ma
                    is not None
                ):

                    ket_qua.append(
                        ket_qua_mot_ma
                    )

            except Exception:
                continue

    if not ket_qua:

        raise RuntimeError(
            "Không lấy được dữ liệu giá hợp lệ."
        )

    bang_gia = pd.DataFrame(
        ket_qua
    )

    # --------------------------------------------------------
    # Trạng thái
    # --------------------------------------------------------

    bien_dong = pd.to_numeric(
        bang_gia[
            "thay_doi_pct"
        ],
        errors="coerce",
    )

    bang_gia[
        "trang_thai"
    ] = np.select(
        [
            bien_dong > 0.05,
            bien_dong < -0.05,
        ],
        [
            "Tăng",
            "Giảm",
        ],
        default="Đứng giá",
    )

    # --------------------------------------------------------
    # Sắp xếp %
    # --------------------------------------------------------

    bang_gia = (
        bang_gia
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
    # Mã / doanh nghiệp
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
        ]

    # --------------------------------------------------------
    # Sàn
    # --------------------------------------------------------

    if (
        san != "Tất cả"
        and "san"
        in df.columns
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
        ]

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
            .fillna("")
            .astype(str)
            .eq(
                nganh
            )
        ]

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
        ]

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

    bien_dong = pd.to_numeric(
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

    hop_le = int(
        bien_dong.notna().sum()
    )

    tang = int(
        (
            bien_dong
            > 0.05
        ).sum()
    )

    giam = int(
        (
            bien_dong
            < -0.05
        ).sum()
    )

    dung = int(
        bien_dong.between(
            -0.05,
            0.05,
        ).sum()
    )

    gia_tri = pd.to_numeric(
        bang_gia[
            "gia_tri_giao_dich"
        ],
        errors="coerce",
    )

    return {
        "tong_ma": len(
            bang_gia
        ),

        "co_du_lieu_gia": hop_le,

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
            / hop_le
            * 100
            if hop_le
            else 0.0
        ),

        "phan_tram_giam": (
            giam
            / hop_le
            * 100
            if hop_le
            else 0.0
        ),
    }


# ============================================================
# TÂM LÝ
# ============================================================

def tinh_diem_tam_ly(
    bang_gia,
):
    thong_ke = (
        thong_ke_thi_truong(
            bang_gia
        )
    )

    tong = (
        thong_ke[
            "co_du_lieu_gia"
        ]
    )

    if tong <= 0:
        return 50.0

    diem = (
        50
        + (
            thong_ke[
                "tang"
            ]
            - thong_ke[
                "giam"
            ]
        )
        / tong
        * 50
    )

    return float(
        max(
            0,
            min(
                100,
                diem,
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

    return (
        df[
            df[
                "khoi_luong"
            ].notna()
        ]
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

    return (
        df[
            df[
                "gia_tri_giao_dich"
            ].notna()
        ]
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

    df = bang_gia[
        bang_gia[
            "ten_nganh"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ].copy()

    if df.empty:
        return pd.DataFrame()

    return (
        df
        .groupby(
            "ten_nganh",
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
        )
    )


# ============================================================
# NGOẠI
#
# Tạm thời không tự bịa số liệu.
# Khi có endpoint ngoại riêng sẽ nối vào đây.
# ============================================================

def thong_ke_nuoc_ngoai(
    bang_gia,
):
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

    diem_tam_ly = (
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

        "tam_ly": diem_tam_ly,

        "nhan_tam_ly": (
            nhan_diem_tam_ly(
                diem_tam_ly
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
            ),
        ),
    }
