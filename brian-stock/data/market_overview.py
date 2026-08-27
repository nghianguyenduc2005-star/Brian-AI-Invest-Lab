from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data.market import (
    load_market_data,
)


CACHE_SECONDS = 60
SO_MA = 60


# ============================================================
# TIỆN ÍCH
# ============================================================

def _so(value, mac_dinh=np.nan):
    try:
        value = float(value)

        if not np.isfinite(value):
            return mac_dinh

        return value

    except Exception:
        return mac_dinh


def _chuan_ma(value):
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace(".VN", "")
        .replace(" ", "")
    )


def _tim_cot(df, *ten):
    if df is None or df.empty:
        return None

    anh_xa = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for ten_cot in ten:
        if str(ten_cot).lower() in anh_xa:
            return anh_xa[
                str(ten_cot).lower()
            ]

    return None


# ============================================================
# DANH SÁCH MÃ
# ============================================================

@st.cache_data(
    ttl=6 * 60 * 60,
    show_spinner=False,
)
def lay_danh_sach_ma():
    """
    Không dùng vnstock_data.
    Dùng danh sách mã cơ bản từ nguồn hiện có.
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

            if cot_ma:

                ma = (
                    bang[
                        cot_ma
                    ]
                    .map(
                        _chuan_ma
                    )
                    .drop_duplicates()
                )

                ma = [
                    x
                    for x in ma.tolist()
                    if x
                ]

                return ma

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback tối thiểu.
    # Không tạo dữ liệu giá giả.
    # --------------------------------------------------------

    return [
        "VIC",
        "VHM",
        "VRE",
        "VCB",
        "TCB",
        "MBB",
        "HPG",
        "HSG",
        "FPT",
        "MSN",
        "VNM",
        "GAS",
        "SSI",
        "VND",
        "MWG",
        "DGC",
        "PDR",
        "DIG",
        "DXG",
        "KDH",
    ]


# ============================================================
# LẤY DỮ LIỆU
# ============================================================

def _lay_du_lieu_ma(ma):
    try:
        du_lieu = load_market_data(
            ma,
            "5d",
        )

        if (
            isinstance(
                du_lieu,
                pd.DataFrame,
            )
            and not du_lieu.empty
        ):
            return du_lieu

    except Exception:
        pass

    return pd.DataFrame()


# ============================================================
# CHUYỂN DỮ LIỆU MỘT MÃ
# ============================================================

def _tach_snapshot(
    ma,
    df,
):
    if df.empty:
        return None

    df = df.copy()

    cot_close = _tim_cot(
        df,
        "Close",
        "close",
    )

    cot_open = _tim_cot(
        df,
        "Open",
        "open",
    )

    cot_high = _tim_cot(
        df,
        "High",
        "high",
    )

    cot_low = _tim_cot(
        df,
        "Low",
        "low",
    )

    cot_volume = _tim_cot(
        df,
        "Volume",
        "volume",
    )

    if cot_close is None:
        return None

    gia = pd.to_numeric(
        df[
            cot_close
        ],
        errors="coerce",
    ).dropna()

    if gia.empty:
        return None

    gia_hien_tai = _so(
        gia.iloc[-1]
    )

    if (
        not np.isfinite(
            gia_hien_tai
        )
        or gia_hien_tai <= 0
    ):
        return None

    if len(gia) >= 2:

        gia_truoc = _so(
            gia.iloc[-2]
        )

        thay_doi = (
            gia_hien_tai
            - gia_truoc
        )

        if gia_truoc != 0:

            thay_doi_pct = (
                thay_doi
                / gia_truoc
                * 100
            )

        else:

            thay_doi_pct = np.nan

    else:

        thay_doi = np.nan
        thay_doi_pct = np.nan

    def _cuoi(
        cot,
    ):

        if cot is None:
            return np.nan

        seri = pd.to_numeric(
            df[cot],
            errors="coerce",
        ).dropna()

        if seri.empty:
            return np.nan

        return _so(
            seri.iloc[-1]
        )

    return {
        "ma": ma,
        "gia": gia_hien_tai,
        "gia_tham_chieu": np.nan,
        "gia_mo_cua": _cuoi(
            cot_open
        ),
        "gia_cao_nhat": _cuoi(
            cot_high
        ),
        "gia_thap_nhat": _cuoi(
            cot_low
        ),
        "thay_doi": thay_doi,
        "thay_doi_pct": thay_doi_pct,
        "khoi_luong": _cuoi(
            cot_volume
        ),
        "gia_tri_giao_dich": np.nan,
        "san": "",
        "ten_nganh": "",
        "ten_doanh_nghiep": "",
        "ma_nganh": "",
    }


# ============================================================
# BẢNG GIÁ 60 MÃ
# ============================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
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
    # Không gọi 1.500 mã.
    # Chỉ lấy 60 mã.
    # --------------------------------------------------------

    danh_sach = danh_sach[
        :SO_MA
    ]

    ket_qua = []

    for ma in danh_sach:

        df = _lay_du_lieu_ma(
            ma
        )

        snapshot = _tach_snapshot(
            ma,
            df,
        )

        if snapshot is not None:

            ket_qua.append(
                snapshot
            )

    if not ket_qua:

        raise RuntimeError(
            "Không lấy được dữ liệu giá hợp lệ."
        )

    bang = pd.DataFrame(
        ket_qua
    )

    # --------------------------------------------------------
    # Trạng thái
    # --------------------------------------------------------

    bang[
        "trang_thai"
    ] = np.select(
        [
            bang[
                "thay_doi_pct"
            ] > 0.05,

            bang[
                "thay_doi_pct"
            ] < -0.05,
        ],
        [
            "Tăng",
            "Giảm",
        ],
        default="Đứng giá",
    )

    return bang


# ============================================================
# LỌC BẢNG GIÁ
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

    if (
        san != "Tất cả"
        and "san" in df.columns
    ):

        df = df[
            df[
                "san"
            ]
            .fillna("")
            .eq(
                san
            )
        ]

    if (
        nganh != "Tất cả"
        and "ten_nganh" in df.columns
    ):

        df = df[
            df[
                "ten_nganh"
            ]
            .fillna("")
            .eq(
                nganh
            )
        ]

    if (
        huong != "Tất cả"
        and "trang_thai" in df.columns
    ):

        df = df[
            df[
                "trang_thai"
            ]
            .eq(
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
            "tong_khoi_luong": 0,
            "tong_gia_tri": 0,
            "phan_tram_tang": 0,
            "phan_tram_giam": 0,
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
        thay_doi
        .between(
            -0.05,
            0.05,
        )
        .sum()
    )

    hop_le = int(
        thay_doi.notna().sum()
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
            pd.to_numeric(
                bang_gia[
                    "gia_tri_giao_dich"
                ],
                errors="coerce",
            ).sum(
                min_count=1
            )
        ),

        "phan_tram_tang": (
            tang / hop_le * 100
            if hop_le
            else 0
        ),

        "phan_tram_giam": (
            giam / hop_le * 100
            if hop_le
            else 0
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

    if tong == 0:
        return 50.0

    return float(
        max(
            0,
            min(
                100,
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
                * 50,
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
# TOP
# ============================================================

def top_tang(
    bang_gia,
    so_luong=10,
):
    if bang_gia.empty:
        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "thay_doi_pct",
            ascending=False,
        )
        .head(
            so_luong
        )
    )


def top_giam(
    bang_gia,
    so_luong=10,
):
    if bang_gia.empty:
        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "thay_doi_pct",
            ascending=True,
        )
        .head(
            so_luong
        )
    )


def top_khoi_luong(
    bang_gia,
    so_luong=10,
):
    if bang_gia.empty:
        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "khoi_luong",
            ascending=False,
            na_position="last",
        )
        .head(
            so_luong
        )
    )


def top_gia_tri_giao_dich(
    bang_gia,
    so_luong=10,
):
    if bang_gia.empty:
        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "gia_tri_giao_dich",
            ascending=False,
            na_position="last",
        )
        .head(
            so_luong
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
        df.groupby(
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
    return {
        "so_ma": (
            len(bang_gia)
            if isinstance(
                bang_gia,
                pd.DataFrame,
            )
            else 0
        ),
        "cap_nhat": pd.Timestamp.now(),
        "nguon": (
            "Vnstock"
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
