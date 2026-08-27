from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CẤU HÌNH
# ============================================================

CACHE_SECONDS = 60
SO_MA_MAC_DINH = 60


# ============================================================
# TIỆN ÍCH CƠ BẢN
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


def _chuan_hoa_ma(
    gia_tri: Any,
):
    if gia_tri is None:
        return ""

    ma = (
        str(gia_tri)
        .strip()
        .upper()
        .replace(" ", "")
    )

    if ma.endswith(".VN"):
        ma = ma[:-3]

    return ma


def _tim_cot(
    du_lieu: pd.DataFrame,
    *ten_cot: str,
):
    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return None

    anh_xa = {
        str(cot).strip().lower(): cot
        for cot in du_lieu.columns
    }

    for ten in ten_cot:
        cot = anh_xa.get(
            str(ten).strip().lower()
        )

        if cot is not None:
            return cot

    return None


def _df(
    du_lieu,
):
    if isinstance(
        du_lieu,
        pd.DataFrame,
    ):
        return du_lieu.copy()

    return pd.DataFrame()


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
# VNSTOCK DATA
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def _tao_insights():
    try:
        from vnstock_data import Insights

        return Insights()

    except Exception as loi:
        raise RuntimeError(
            "Không thể khởi tạo vnstock_data.Insights."
        ) from loi


@st.cache_resource(
    show_spinner=False,
)
def _tao_listing():
    try:
        from vnstock_data import Listing

        return Listing(
            source="vci"
        )

    except Exception:
        return None


# ============================================================
# LẤY NGÀNH ICB
# ============================================================

@st.cache_data(
    ttl=6 * 60 * 60,
    show_spinner=False,
)
def _lay_bang_nganh():
    listing = _tao_listing()

    if listing is None:
        return pd.DataFrame()

    try:
        bang_nganh = (
            listing
            .industries_icb()
        )

    except Exception:
        return pd.DataFrame()

    if (
        not isinstance(
            bang_nganh,
            pd.DataFrame,
        )
        or bang_nganh.empty
    ):
        return pd.DataFrame()

    bang_nganh = bang_nganh.copy()

    cot_cap = _tim_cot(
        bang_nganh,
        "level",
    )

    cot_ma = _tim_cot(
        bang_nganh,
        "icb_code",
    )

    cot_ten = _tim_cot(
        bang_nganh,
        "icb_name",
    )

    if (
        cot_cap is None
        or cot_ma is None
        or cot_ten is None
    ):
        return pd.DataFrame()

    bang_nganh[
        "ma_nganh"
    ] = (
        bang_nganh[
            cot_ma
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    bang_nganh[
        "ten_nganh"
    ] = (
        bang_nganh[
            cot_ten
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cap = pd.to_numeric(
        bang_nganh[
            cot_cap
        ],
        errors="coerce",
    )

    bang_nganh = bang_nganh[
        cap == 1
    ].copy()

    return (
        bang_nganh[
            [
                "ma_nganh",
                "ten_nganh",
            ]
        ]
        .drop_duplicates(
            "ma_nganh"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# UNIVERSE NHẸ
# ============================================================

@st.cache_data(
    ttl=6 * 60 * 60,
    show_spinner=False,
)
def lay_universe(
    force_reload: bool = False,
):
    _ = force_reload

    insights = _tao_insights()

    try:

        bang = (
            insights
            .screener
            .filter(
                limit=SO_MA_MAC_DINH
            )
        )

    except Exception as loi:

        raise RuntimeError(
            "Không lấy được universe từ Screener."
        ) from loi

    bang = _df(
        bang
    )

    if bang.empty:
        raise RuntimeError(
            "Screener trả về dữ liệu rỗng."
        )

    cot_ma = _tim_cot(
        bang,
        "symbol",
        "ticker",
        "code",
    )

    if cot_ma is None:
        raise RuntimeError(
            "Không tìm thấy cột mã cổ phiếu."
        )

    ket_qua = pd.DataFrame()

    ket_qua[
        "ma"
    ] = (
        bang[
            cot_ma
        ]
        .map(
            _chuan_hoa_ma
        )
    )

    cot_san = _tim_cot(
        bang,
        "exchange",
    )

    if cot_san is not None:

        ket_qua[
            "san"
        ] = (
            bang[
                cot_san
            ]
            .map(
                _chuan_san
            )
        )

    else:

        ket_qua[
            "san"
        ] = ""

    cot_von_hoa = _tim_cot(
        bang,
        "market_cap",
    )

    if cot_von_hoa is not None:

        ket_qua[
            "von_hoa"
        ] = pd.to_numeric(
            bang[
                cot_von_hoa
            ],
            errors="coerce",
        )

    else:

        ket_qua[
            "von_hoa"
        ] = np.nan

    ket_qua = (
        ket_qua[
            ket_qua[
                "ma"
            ] != ""
        ]
        .drop_duplicates(
            "ma"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Ghép ngành nếu screener có sector_lv1
    # --------------------------------------------------------

    cot_nganh = _tim_cot(
        bang,
        "sector_lv1",
        "icb_code",
    )

    if cot_nganh is not None:

        tmp = pd.DataFrame(
            {
                "ma": (
                    bang[
                        cot_ma
                    ]
                    .map(
                        _chuan_hoa_ma
                    )
                ),
                "ma_nganh": (
                    bang[
                        cot_nganh
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                ),
            }
        )

        tmp = (
            tmp[
                tmp["ma"] != ""
            ]
            .drop_duplicates(
                "ma"
            )
        )

        ket_qua = ket_qua.merge(
            tmp,
            on="ma",
            how="left",
        )

    else:

        ket_qua[
            "ma_nganh"
        ] = ""

    bang_nganh = _lay_bang_nganh()

    if not bang_nganh.empty:

        ket_qua = ket_qua.merge(
            bang_nganh,
            on="ma_nganh",
            how="left",
        )

    if "ten_nganh" not in ket_qua.columns:

        ket_qua[
            "ten_nganh"
        ] = ""

    ket_qua[
        "ten_nganh"
    ] = (
        ket_qua[
            "ten_nganh"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    ket_qua[
        "ten_doanh_nghiep"
    ] = ""

    return ket_qua


# ============================================================
# CHUẨN HÓA SCREENER
# ============================================================

def _chuan_hoa_screener(
    bang,
):
    if bang.empty:
        return pd.DataFrame()

    df = bang.copy()

    cot_ma = _tim_cot(
        df,
        "symbol",
        "ticker",
        "code",
    )

    if cot_ma is None:
        return pd.DataFrame()

    cot_san = _tim_cot(
        df,
        "exchange",
    )

    cot_gia = _tim_cot(
        df,
        "price",
        "last_price",
        "match_price",
    )

    cot_tc = _tim_cot(
        df,
        "ref_price",
        "reference_price",
    )

    cot_mo = _tim_cot(
        df,
        "open_price",
        "open",
    )

    cot_cao = _tim_cot(
        df,
        "high",
        "high_price",
    )

    cot_thap = _tim_cot(
        df,
        "low",
        "low_price",
    )

    cot_phan_tram = _tim_cot(
        df,
        "price_change_percent",
        "price_change_percent_1d",
        "change_percent",
        "change_pct",
    )

    cot_thay_doi = _tim_cot(
        df,
        "price_change",
        "change",
    )

    cot_khoi_luong = _tim_cot(
        df,
        "accumulated_volume",
        "volume",
        "match_volume",
    )

    cot_gia_tri = _tim_cot(
        df,
        "accumulated_value",
        "total_value",
        "value",
        "trading_value",
    )

    cot_tb_khoi_luong = _tim_cot(
        df,
        "avg_volume_30d",
        "avg_volume",
    )

    cot_tb_gia_tri = _tim_cot(
        df,
        "avg_value_30d",
        "avg_value",
    )

    cot_pe = _tim_cot(
        df,
        "pe",
        "ttm_pe",
    )

    cot_pb = _tim_cot(
        df,
        "pb",
        "ttm_pb",
    )

    cot_roe = _tim_cot(
        df,
        "roe",
        "ttm_roe",
    )

    cot_rsi = _tim_cot(
        df,
        "rsi",
    )

    cot_macd = _tim_cot(
        df,
        "macd",
    )

    cot_ema20 = _tim_cot(
        df,
        "ema20",
        "price_ema_20",
    )

    cot_ema50 = _tim_cot(
        df,
        "ema50",
        "price_ema_50",
    )

    cot_von_hoa = _tim_cot(
        df,
        "market_cap",
    )

    rows = []

    for _, dong in df.iterrows():

        ma = _chuan_hoa_ma(
            dong[
                cot_ma
            ]
        )

        if not ma:
            continue

        gia = (
            _so(
                dong[
                    cot_gia
                ]
            )
            if cot_gia
            else np.nan
        )

        # Screener có thể trả một số mã chưa có giá.
        if (
            not np.isfinite(
                gia
            )
            or gia <= 0
        ):
            continue

        tham_chieu = (
            _so(
                dong[
                    cot_tc
                ]
            )
            if cot_tc
            else np.nan
        )

        thay_doi = (
            _so(
                dong[
                    cot_thay_doi
                ]
            )
            if cot_thay_doi
            else np.nan
        )

        phan_tram = (
            _so(
                dong[
                    cot_phan_tram
                ]
            )
            if cot_phan_tram
            else np.nan
        )

        if (
            not np.isfinite(
                phan_tram
            )
            and np.isfinite(
                tham_chieu
            )
            and tham_chieu != 0
        ):

            phan_tram = (
                (
                    gia
                    - tham_chieu
                )
                / tham_chieu
                * 100
            )

        if (
            not np.isfinite(
                thay_doi
            )
            and np.isfinite(
                tham_chieu
            )
        ):

            thay_doi = (
                gia
                - tham_chieu
            )

        rows.append(
            {
                "ma": ma,
                "san": (
                    _chuan_san(
                        dong[
                            cot_san
                        ]
                    )
                    if cot_san
                    else ""
                ),
                "gia": gia,
                "gia_tham_chieu": tham_chieu,
                "gia_mo_cua": (
                    _so(
                        dong[
                            cot_mo
                        ]
                    )
                    if cot_mo
                    else np.nan
                ),
                "gia_cao_nhat": (
                    _so(
                        dong[
                            cot_cao
                        ]
                    )
                    if cot_cao
                    else np.nan
                ),
                "gia_thap_nhat": (
                    _so(
                        dong[
                            cot_thap
                        ]
                    )
                    if cot_thap
                    else np.nan
                ),
                "thay_doi": thay_doi,
                "thay_doi_pct": phan_tram,
                "khoi_luong": (
                    _so(
                        dong[
                            cot_khoi_luong
                        ]
                    )
                    if cot_khoi_luong
                    else np.nan
                ),
                "gia_tri_giao_dich": (
                    _so(
                        dong[
                            cot_gia_tri
                        ]
                    )
                    if cot_gia_tri
                    else np.nan
                ),
                "khoi_luong_tb_30d": (
                    _so(
                        dong[
                            cot_tb_khoi_luong
                        ]
                    )
                    if cot_tb_khoi_luong
                    else np.nan
                ),
                "gia_tri_tb_30d": (
                    _so(
                        dong[
                            cot_tb_gia_tri
                        ]
                    )
                    if cot_tb_gia_tri
                    else np.nan
                ),
                "pe": (
                    _so(
                        dong[
                            cot_pe
                        ]
                    )
                    if cot_pe
                    else np.nan
                ),
                "pb": (
                    _so(
                        dong[
                            cot_pb
                        ]
                    )
                    if cot_pb
                    else np.nan
                ),
                "roe": (
                    _so(
                        dong[
                            cot_roe
                        ]
                    )
                    if cot_roe
                    else np.nan
                ),
                "rsi": (
                    _so(
                        dong[
                            cot_rsi
                        ]
                    )
                    if cot_rsi
                    else np.nan
                ),
                "macd": (
                    _so(
                        dong[
                            cot_macd
                        ]
                    )
                    if cot_macd
                    else np.nan
                ),
                "ema20": (
                    _so(
                        dong[
                            cot_ema20
                        ]
                    )
                    if cot_ema20
                    else np.nan
                ),
                "ema50": (
                    _so(
                        dong[
                            cot_ema50
                        ]
                    )
                    if cot_ema50
                    else np.nan
                ),
                "von_hoa": (
                    _so(
                        dong[
                            cot_von_hoa
                        ]
                    )
                    if cot_von_hoa
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
        .drop_duplicates(
            "ma",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LẤY 60 MÃ THỊ TRƯỜNG
# ============================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False,
)
def lay_bang_gia_toan_thi_truong(
    force_reload: bool = False,
):
    _ = force_reload

    insights = _tao_insights()

    try:

        bang_goc = (
            insights
            .screener
            .filter(
                limit=SO_MA_MAC_DINH
            )
        )

    except Exception as loi:

        raise RuntimeError(
            "Không lấy được bảng thị trường từ Screener."
        ) from loi

    bang_gia = (
        _chuan_hoa_screener(
            _df(
                bang_goc
            )
        )
    )

    if bang_gia.empty:

        raise RuntimeError(
            "Không có giá hợp lệ sau khi chuẩn hóa Screener."
        )

    universe = lay_universe()

    if (
        universe is not None
        and not universe.empty
    ):

        thong_tin = universe[
            [
                "ma",
                "ten_doanh_nghiep",
                "san",
                "ma_nganh",
                "ten_nganh",
            ]
        ].copy()

        bang_gia = bang_gia.merge(
            thong_tin,
            on="ma",
            how="left",
            suffixes=(
                "",
                "_ref",
            ),
        )

        # Nếu screener đã có sàn thì giữ.
        if "san_ref" in bang_gia.columns:

            bang_gia[
                "san"
            ] = (
                bang_gia[
                    "san"
                ]
                .replace(
                    "",
                    np.nan,
                )
                .fillna(
                    bang_gia[
                        "san_ref"
                    ]
                )
                .fillna("")
            )

            bang_gia = (
                bang_gia.drop(
                    columns=[
                        "san_ref"
                    ]
                )
            )

        for cot in [
            "ten_doanh_nghiep",
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

    else:

        bang_gia[
            "ten_doanh_nghiep"
        ] = ""

        bang_gia[
            "ma_nganh"
        ] = ""

        bang_gia[
            "ten_nganh"
        ] = ""

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

    return (
        bang_gia
        .drop_duplicates(
            "ma",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


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

    tu_khoa = str(
        tu_khoa or ""
    ).strip()

    if tu_khoa:

        mask_ma = (
            df["ma"]
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
            .eq(
                str(san).upper()
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
            .astype(str)
            .eq(
                str(nganh)
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
            "tong_khoi_luong": np.nan,
            "tong_gia_tri": np.nan,
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

    hop_le = thay_doi.notna()

    tong = int(
        hop_le.sum()
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

    dung_gia = int(
        thay_doi
        .between(
            -0.05,
            0.05,
        )
        .sum()
    )

    return {
        "tong_ma": len(
            bang_gia
        ),
        "co_du_lieu_gia": tong,
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
# RANKING
# ============================================================

def _lay_ranking(
    ten_ham,
    limit=20,
):
    insights = _tao_insights()

    try:

        ham = getattr(
            insights.ranking,
            ten_ham,
        )

        bang = ham(
            limit=limit
        )

        return _df(
            bang
        )

    except Exception:
        return pd.DataFrame()


def _chuan_hoa_ranking(
    bang,
):
    if bang.empty:
        return pd.DataFrame()

    df = bang.copy()

    cot_ma = _tim_cot(
        df,
        "symbol",
        "ticker",
        "code",
    )

    if cot_ma is None:
        return pd.DataFrame()

    cot_san = _tim_cot(
        df,
        "exchange",
    )

    cot_gia = _tim_cot(
        df,
        "last_price",
        "price",
        "match_price",
    )

    cot_thay_doi = _tim_cot(
        df,
        "price_change_1d",
        "change",
        "price_change",
    )

    cot_phan_tram = _tim_cot(
        df,
        "price_change_pct_1d",
        "price_change_percent",
        "change_percent",
    )

    cot_gia_tri = _tim_cot(
        df,
        "total_value",
        "accumulated_value",
    )

    cot_khoi_luong = _tim_cot(
        df,
        "accumulated_volume",
        "volume",
        "total_volume",
    )

    ket_qua = pd.DataFrame()

    ket_qua[
        "ma"
    ] = (
        df[
            cot_ma
        ]
        .map(
            _chuan_hoa_ma
        )
    )

    ket_qua[
        "san"
    ] = (
        df[
            cot_san
        ]
        .map(
            _chuan_san
        )
        if cot_san
        else ""
    )

    ket_qua[
        "gia"
    ] = (
        pd.to_numeric(
            df[
                cot_gia
            ],
            errors="coerce",
        )
        if cot_gia
        else np.nan
    )

    ket_qua[
        "thay_doi"
    ] = (
        pd.to_numeric(
            df[
                cot_thay_doi
            ],
            errors="coerce",
        )
        if cot_thay_doi
        else np.nan
    )

    ket_qua[
        "thay_doi_pct"
    ] = (
        pd.to_numeric(
            df[
                cot_phan_tram
            ],
            errors="coerce",
        )
        if cot_phan_tram
        else np.nan
    )

    ket_qua[
        "gia_tri_giao_dich"
    ] = (
        pd.to_numeric(
            df[
                cot_gia_tri
            ],
            errors="coerce",
        )
        if cot_gia_tri
        else np.nan
    )

    ket_qua[
        "khoi_luong"
    ] = (
        pd.to_numeric(
            df[
                cot_khoi_luong
            ],
            errors="coerce",
        )
        if cot_khoi_luong
        else np.nan
    )

    return (
        ket_qua[
            ket_qua[
                "ma"
            ] != ""
        ]
        .drop_duplicates(
            "ma"
        )
        .reset_index(
            drop=True
        )
    )


def top_tang(
    bang_gia=None,
    so_luong=20,
):
    bang = _lay_ranking(
        "gainer",
        so_luong,
    )

    if not bang.empty:

        return _chuan_hoa_ranking(
            bang
        )

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
        )
        .head(
            so_luong
        )
        .copy()
    )


def top_giam(
    bang_gia=None,
    so_luong=20,
):
    bang = _lay_ranking(
        "loser",
        so_luong,
    )

    if not bang.empty:

        return _chuan_hoa_ranking(
            bang
        )

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
        )
        .head(
            so_luong
        )
        .copy()
    )


def top_khoi_luong(
    bang_gia=None,
    so_luong=20,
):
    bang = _lay_ranking(
        "volume",
        so_luong,
    )

    if not bang.empty:

        return _chuan_hoa_ranking(
            bang
        )

    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "khoi_luong",
            ascending=False,
        )
        .head(
            so_luong
        )
        .copy()
    )


def top_gia_tri_giao_dich(
    bang_gia=None,
    so_luong=20,
):
    bang = _lay_ranking(
        "value",
        so_luong,
    )

    if not bang.empty:

        return _chuan_hoa_ranking(
            bang
        )

    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    return (
        bang_gia
        .sort_values(
            "gia_tri_giao_dich",
            ascending=False,
        )
        .head(
            so_luong
        )
        .copy()
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

    df = bang_gia.copy()

    return (
        df.groupby(
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
    ):
        return pd.DataFrame()

    if "ten_nganh" not in bang_gia.columns:
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
        df.groupby(
            "ten_nganh",
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
        .sort_values(
            "bien_dong_binh_quan",
            ascending=False,
        )
    )


# ============================================================
# GIAO DỊCH NƯỚC NGOÀI
# ============================================================

def thong_ke_nuoc_ngoai(
    bang_gia,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return {
            "co_du_lieu": False,
            "mua": np.nan,
            "ban": np.nan,
            "rong": np.nan,
        }

    insights = _tao_insights()

    mua = pd.DataFrame()
    ban = pd.DataFrame()

    try:

        mua = _df(
            insights.ranking.foreign_buy(
                limit=20
            )
        )

        ban = _df(
            insights.ranking.foreign_sell(
                limit=20
            )
        )

    except Exception:
        mua = pd.DataFrame()
        ban = pd.DataFrame()

    if mua.empty and ban.empty:

        return {
            "co_du_lieu": False,
            "mua": np.nan,
            "ban": np.nan,
            "rong": np.nan,
        }

    def _tong_gia_tri(
        bang,
    ):

        if bang.empty:
            return 0.0

        cot = _tim_cot(
            bang,
            "net_value",
        )

        if cot is None:
            return 0.0

        return pd.to_numeric(
            bang[
                cot
            ],
            errors="coerce",
        ).sum(
            min_count=1
        )

    tong_mua = _tong_gia_tri(
        mua
    )

    tong_ban = _tong_gia_tri(
        ban
    )

    if np.isfinite(
        _so(tong_mua)
    ) and np.isfinite(
        _so(tong_ban)
    ):

        rong = (
            tong_mua
            - tong_ban
        )

    else:

        rong = np.nan

    return {
        "co_du_lieu": True,
        "mua": tong_mua,
        "ban": tong_ban,
        "rong": rong,
    }


# ============================================================
# THÔNG TIN NGUỒN
# ============================================================

def thong_tin_nguon(
    bang_gia,
):
    return {
        "so_ma": (
            int(
                bang_gia[
                    "ma"
                ].nunique()
            )
            if (
                isinstance(
                    bang_gia,
                    pd.DataFrame,
                )
                and not bang_gia.empty
                and "ma" in bang_gia.columns
            )
            else 0
        ),
        "cap_nhat": pd.Timestamp.now(),
        "nguon": (
            "Vnstock Data · Insights Screener/Ranking"
        ),
    }


# ============================================================
# API TỔNG
# ============================================================

def lay_market_overview(
    force_reload: bool = False,
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

        "universe": lay_universe(
            force_reload=force_reload
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
