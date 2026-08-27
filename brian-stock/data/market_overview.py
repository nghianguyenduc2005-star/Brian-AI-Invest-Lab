from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
THOI_GIAN_CACHE_DANH_SACH = 6 * 60 * 60

SO_MA_THI_TRUONG = 60
SO_LUONG_WORKER = 8


# ============================================================
# DANH SÁCH ƯU TIÊN
#
# Dùng để đảm bảo 60 mã đầu tiên có tính đại diện tốt.
# Nếu Listing hoạt động, các mã còn thiếu sẽ được bổ sung
# từ danh sách thực tế.
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
    "NVL",
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

    anh_xa = {}

    for cot in du_lieu.columns:

        khoa = (
            str(cot)
            .strip()
            .lower()
        )

        anh_xa[khoa] = cot

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

        ma = (
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

        return ma


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
# LẤY THÔNG TIN LISTING
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_DANH_SACH,
    show_spinner=False,
)
def _lay_listing():
    """
    Cố lấy danh sách thực tế từ vnstock.

    Nếu Listing không khả dụng thì trả DataFrame rỗng.
    Không làm app chết chỉ vì metadata listing.
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

            return bang.copy()

    except Exception:
        pass

    return pd.DataFrame()


# ============================================================
# DANH SÁCH 60 MÃ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE_DANH_SACH,
    show_spinner=False,
)
def lay_danh_sach_ma():
    """
    Trả metadata cho danh sách mã dùng ở bảng thị trường.
    """

    bang_listing = _lay_listing()

    rows = []

    # --------------------------------------------------------
    # Nếu Listing có dữ liệu
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
                    dong[
                        cot_ma
                    ]
                )

                if not ma:
                    continue

                ten_doanh_nghiep = ""

                if (
                    cot_ten is not None
                    and pd.notna(
                        dong[
                            cot_ten
                        ]
                    )
                ):

                    ten_doanh_nghiep = (
                        str(
                            dong[
                                cot_ten
                            ]
                        )
                        .strip()
                    )

                san = ""

                if (
                    cot_san is not None
                    and pd.notna(
                        dong[
                            cot_san
                        ]
                    )
                ):

                    san = _chuan_san(
                        dong[
                            cot_san
                        ]
                    )

                ten_nganh = ""

                if (
                    cot_nganh is not None
                    and pd.notna(
                        dong[
                            cot_nganh
                        ]
                    )
                ):

                    ten_nganh = (
                        str(
                            dong[
                                cot_nganh
                            ]
                        )
                        .strip()
                    )

                ma_nganh = ""

                if (
                    cot_ma_nganh is not None
                    and pd.notna(
                        dong[
                            cot_ma_nganh
                        ]
                    )
                ):

                    ma_nganh = (
                        str(
                            dong[
                                cot_ma_nganh
                            ]
                        )
                        .strip()
                    )

                von_hoa = np.nan

                if cot_von_hoa is not None:

                    von_hoa = _so(
                        dong[
                            cot_von_hoa
                        ]
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

    # --------------------------------------------------------
    # Nếu không có Listing
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        rows
    )

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

    # --------------------------------------------------------
    # Ưu tiên các mã mong muốn
    # --------------------------------------------------------

    ma_co_san = set(
        metadata[
            "ma"
        ]
        .tolist()
    )

    danh_sach = []

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
    # Bổ sung mã còn thiếu từ Listing
    # --------------------------------------------------------

    for ma in (
        metadata[
            "ma"
        ].tolist()
    ):

        if (
            ma
            and ma not in danh_sach
        ):

            danh_sach.append(
                ma
            )

        if len(danh_sach) >= (
            SO_MA_THI_TRUONG
        ):
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

    # Giữ đúng thứ tự ưu tiên
    thu_tu = {
        ma: vi_tri
        for vi_tri, ma
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
# LẤY DỮ LIỆU MỘT MÃ
# ============================================================

def _lay_mot_ma(
    ma,
):
    """
    Lấy 5 ngày dữ liệu cho một mã.

    Chạy song song ở hàm phía dưới.
    """

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

        dong = (
            du_lieu.iloc[-1]
        )

        if len(
            du_lieu
        ) >= 2:

            dong_truoc = (
                du_lieu.iloc[-2]
            )

        else:

            dong_truoc = dong

        gia = _so(
            dong.get(
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

            thay_doi = (
                gia
                - gia_truoc
            )

            thay_doi_pct = (
                (
                    gia
                    / gia_truoc
                )
                - 1
            ) * 100

        else:

            thay_doi = np.nan
            thay_doi_pct = np.nan

        khoi_luong = _so(
            dong.get(
                "Volume"
            )
        )

        # ----------------------------------------------------
        # Giá trị giao dịch
        #
        # Nếu nguồn OHLCV không có value,
        # dùng giá * volume như giá trị ước tính trong phiên.
        # ----------------------------------------------------

        gia_tri = np.nan

        if (
            np.isfinite(
                gia
            )
            and np.isfinite(
                khoi_luong
            )
        ):

            gia_tri = (
                gia
                * khoi_luong
            )

        return {
            "ma": _chuan_ma(
                ma
            ),

            "gia": gia,

            "gia_tham_chieu": (
                gia_truoc
            ),

            "gia_mo_cua": _so(
                dong.get(
                    "Open"
                )
            ),

            "gia_cao_nhat": _so(
                dong.get(
                    "High"
                )
            ),

            "gia_thap_nhat": _so(
                dong.get(
                    "Low"
                )
            ),

            "thay_doi": (
                thay_doi
            ),

            "thay_doi_pct": (
                thay_doi_pct
            ),

            "khoi_luong": (
                khoi_luong
            ),

            "gia_tri_giao_dich": (
                gia_tri
            ),

            "rsi": _so(
                dong.get(
                    "RSI"
                )
            ),

            "macd": _so(
                dong.get(
                    "MACD"
                )
            ),

            "sma20": _so(
                dong.get(
                    "SMA20"
                )
            ),

            "sma50": _so(
                dong.get(
                    "SMA50"
                )
            ),

            "ema20": _so(
                dong.get(
                    "EMA20"
                )
            ),

            "ema50": _so(
                dong.get(
                    "EMA50"
                )
            ),

            "volatility20": _so(
                dong.get(
                    "Volatility20"
                )
            ),

            "atr14": _so(
                dong.get(
                    "ATR14"
                )
            ),

            "relative_volume": _so(
                dong.get(
                    "Relative_Volume"
                )
            ),
        }

    except Exception:
        return None


# ============================================================
# BẢNG GIÁ 60 MÃ
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_CACHE,
    show_spinner=False,
)
def lay_bang_gia_toan_thi_truong(
    force_reload=False,
):
    """
    Lấy tối đa 60 mã song song.

    Quan trọng:
    - Không gọi 60 mã tuần tự.
    - Không dùng vnstock_data.
    - Không dùng giá giả.
    """

    _ = force_reload

    metadata = (
        lay_danh_sach_ma()
    )

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

    # --------------------------------------------------------
    # Chạy song song
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=SO_LUONG_WORKER
    ) as bo_luong:

        nhiem_vu = {
            bo_luong.submit(
                _lay_mot_ma,
                ma,
            ): ma
            for ma
            in danh_sach_ma
        }

        for future in as_completed(
            nhiem_vu
        ):

            try:

                ket_qua_mot_ma = (
                    future.result()
                )

            except Exception:

                ket_qua_mot_ma = None

            if (
                ket_qua_mot_ma
                is not None
            ):

                ket_qua.append(
                    ket_qua_mot_ma
                )

    if not ket_qua:

        raise RuntimeError(
            "Không lấy được dữ liệu giá hợp lệ."
        )

    bang_gia = pd.DataFrame(
        ket_qua
    )

    # --------------------------------------------------------
    # Ghép metadata
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Chuẩn hóa text
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sắp xếp theo % thay đổi
    # --------------------------------------------------------

    bang_gia[
        "thay_doi_pct"
    ] = pd.to_numeric(
        bang_gia[
            "thay_doi_pct"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Trạng thái
    # --------------------------------------------------------

    bang_gia[
        "trang_thai"
    ] = np.select(
        [
            bang_gia[
                "thay_doi_pct"
            ] > 0.05,

            bang_gia[
                "thay_doi_pct"
            ] < -0.05,
        ],
        [
            "Tăng",
            "Giảm",
        ],
        default="Đứng giá",
    )

    # --------------------------------------------------------
    # Kiểm tra giá trị giao dịch
    # --------------------------------------------------------

    bang_gia[
        "gia_tri_giao_dich"
    ] = pd.to_numeric(
        bang_gia[
            "gia_tri_giao_dich"
        ],
        errors="coerce",
    )

    bang_gia[
        "khoi_luong"
    ] = pd.to_numeric(
        bang_gia[
            "khoi_luong"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Cuối cùng
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Từ khóa
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

    tong_hop_le = int(
        thay_doi.notna().sum()
    )

    tang = int(
        (
            thay_doi
            > 0.05
        ).sum()
    )

    giam = int(
        (
            thay_doi
            < -0.05
        ).sum()
    )

    dung_gia = int(
        thay_doi.between(
            -0.05,
            0.05,
        ).sum()
    )

    return {
        "tong_ma": len(
            bang_gia
        ),

        "co_du_lieu_gia": (
            tong_hop_le
        ),

        "tang": tang,

        "dung_gia": dung_gia,

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
            / tong_hop_le
            * 100
            if tong_hop_le
            else 0.0
        ),

        "phan_tram_giam": (
            giam
            / tong_hop_le
            * 100
            if tong_hop_le
            else 0.0
        ),
    }


# ============================================================
# TÂM LÝ THỊ TRƯỜNG
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
        50.0
        + (
            thong_ke[
                "tang"
            ]
            - thong_ke[
                "giam"
            ]
        )
        / tong
        * 50.0
    )

    return float(
        max(
            0.0,
            min(
                100.0,
                diem,
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
        &
        (
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
# TOP GIÁ TRỊ GIAO DỊCH
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
        &
        (
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
# THỐNG KÊ THEO SÀN
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
                        x
                        > 0.05
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
                        x
                        < -0.05
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
# THỐNG KÊ THEO NGÀNH
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
                        x
                        > 0.05
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
                        x
                        < -0.05
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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# GIAO DỊCH NƯỚC NGOÀI
# ============================================================

def thong_ke_nuoc_ngoai(
    bang_gia,
):
    """
    data.market hiện tại của repo chưa có endpoint
    dữ liệu mua/bán nước ngoài.

    Vì vậy không hiển thị dữ liệu giả.
    """

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
            "Vnstock · OHLCV thị trường"
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
