from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CẤU HÌNH
# ============================================================

CACHE_SECONDS = 60

# Nếu nguồn không chấp nhận toàn bộ symbols trong 1 request,
# fallback sẽ chia thành các lô lớn hơn nhiều so với bản cũ.
FALLBACK_BATCH_SIZE = 500


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
    *ten_cot,
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


def _chuan_hoa_dataframe(
    du_lieu,
):
    if isinstance(
        du_lieu,
        pd.DataFrame,
    ):
        return du_lieu.copy()

    return pd.DataFrame()


# ============================================================
# KHỞI TẠO VNSTOCK
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def _tao_reference():
    try:
        from vnstock_data import Reference

        return Reference()

    except Exception:
        pass

    try:
        from vnstock import Reference

        return Reference()

    except Exception as loi:
        raise RuntimeError(
            "Không thể khởi tạo Vnstock Reference."
        ) from loi


@st.cache_resource(
    show_spinner=False,
)
def _tao_market():
    try:
        from vnstock_data import Market

        return Market()

    except Exception:
        pass

    try:
        from vnstock import Market

        return Market()

    except Exception as loi:
        raise RuntimeError(
            "Không thể khởi tạo Vnstock Market."
        ) from loi


# ============================================================
# UNIVERSE TOÀN THỊ TRƯỜNG
# ============================================================

@st.cache_data(
    ttl=6 * 60 * 60,
    show_spinner=False,
)
def lay_universe(
    force_reload: bool = False,
):
    _ = force_reload

    reference = _tao_reference()

    # --------------------------------------------------------
    # TOÀN BỘ CỔ PHIẾU
    # --------------------------------------------------------

    du_lieu = pd.DataFrame()

    try:
        du_lieu = (
            reference
            .equity
            .list()
        )
    except Exception:
        du_lieu = pd.DataFrame()

    # --------------------------------------------------------
    # FALLBACK: THEO SÀN
    # --------------------------------------------------------

    if du_lieu.empty:

        try:

            du_lieu_san = (
                reference
                .equity
                .list_by_exchange()
            )

            if isinstance(
                du_lieu_san,
                pd.DataFrame,
            ):

                du_lieu = (
                    du_lieu_san.copy()
                )

            elif isinstance(
                du_lieu_san,
                dict,
            ):

                cac_bang = []

                for ten_san, bang in (
                    du_lieu_san.items()
                ):

                    if not isinstance(
                        bang,
                        pd.DataFrame,
                    ):
                        continue

                    bang = (
                        bang.copy()
                    )

                    cot_ma = _tim_cot(
                        bang,
                        "symbol",
                        "ticker",
                        "code",
                    )

                    if cot_ma is None:
                        continue

                    bang[
                        "_san_tam"
                    ] = (
                        str(
                            ten_san
                        ).upper()
                    )

                    cac_bang.append(
                        bang
                    )

                if cac_bang:

                    du_lieu = pd.concat(
                        cac_bang,
                        ignore_index=True,
                    )

        except Exception:
            du_lieu = pd.DataFrame()

    if du_lieu.empty:

        raise RuntimeError(
            "Không lấy được danh sách cổ phiếu toàn thị trường."
        )

    du_lieu = _chuan_hoa_dataframe(
        du_lieu
    )

    # --------------------------------------------------------
    # CỘT CHUẨN
    # --------------------------------------------------------

    cot_ma = _tim_cot(
        du_lieu,
        "symbol",
        "ticker",
        "code",
        "stock_code",
        "_ma",
    )

    if cot_ma is None:

        raise RuntimeError(
            "Universe không có cột mã cổ phiếu."
        )

    cot_ten = _tim_cot(
        du_lieu,
        "organ_name",
        "company_name",
        "company",
        "name",
    )

    cot_san = _tim_cot(
        du_lieu,
        "exchange",
        "exchange_name",
        "floor",
        "_san_tam",
    )

    cot_ma_nganh = _tim_cot(
        du_lieu,
        "icb_code",
        "industry_code",
    )

    cot_nganh = _tim_cot(
        du_lieu,
        "icb_name",
        "industry_name",
        "industry",
    )

    rows = []

    for _, dong in du_lieu.iterrows():

        ma = _chuan_hoa_ma(
            dong[cot_ma]
        )

        if not ma:
            continue

        ten_doanh_nghiep = ""

        if (
            cot_ten
            and pd.notna(
                dong[cot_ten]
            )
        ):

            ten_doanh_nghiep = str(
                dong[cot_ten]
            ).strip()

        san = ""

        if (
            cot_san
            and pd.notna(
                dong[cot_san]
            )
        ):

            san = str(
                dong[cot_san]
            ).strip().upper()

        ma_nganh = ""

        if (
            cot_ma_nganh
            and pd.notna(
                dong[cot_ma_nganh]
            )
        ):

            ma_nganh = str(
                dong[cot_ma_nganh]
            ).strip()

        ten_nganh = ""

        if (
            cot_nganh
            and pd.notna(
                dong[cot_nganh]
            )
        ):

            ten_nganh = str(
                dong[cot_nganh]
            ).strip()

        rows.append(
            {
                "ma": ma,
                "ten_doanh_nghiep": (
                    ten_doanh_nghiep
                ),
                "san": san,
                "ma_nganh": ma_nganh,
                "ten_nganh": ten_nganh,
            }
        )

    universe = pd.DataFrame(
        rows
    )

    if universe.empty:

        raise RuntimeError(
            "Universe sau khi chuẩn hóa đang rỗng."
        )

    # --------------------------------------------------------
    # CHUẨN HÓA SÀN
    # --------------------------------------------------------

    def _chuan_san(
        value,
    ):

        text = str(
            value or ""
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

        if text in {
            "UPCOM",
        }:
            return "UPCOM"

        return text

    universe["san"] = (
        universe["san"]
        .map(_chuan_san)
    )

    universe = (
        universe
        .drop_duplicates(
            subset=["ma"],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # BỔ SUNG SÀN TỪ list_by_exchange()
    # --------------------------------------------------------

    if (
        universe["san"]
        .eq("")
        .all()
    ):

        try:

            bang_san = (
                reference
                .equity
                .list_by_exchange()
            )

            if isinstance(
                bang_san,
                pd.DataFrame,
            ):

                cot_ma_san = _tim_cot(
                    bang_san,
                    "symbol",
                    "ticker",
                    "code",
                )

                cot_san_san = _tim_cot(
                    bang_san,
                    "exchange",
                    "exchange_name",
                    "floor",
                )

                if (
                    cot_ma_san is not None
                    and cot_san_san is not None
                ):

                    bang_san = (
                        bang_san.copy()
                    )

                    bang_san[
                        "_ma"
                    ] = (
                        bang_san[
                            cot_ma_san
                        ]
                        .map(
                            _chuan_hoa_ma
                        )
                    )

                    bang_san[
                        "_san"
                    ] = (
                        bang_san[
                            cot_san_san
                        ]
                        .map(
                            _chuan_san
                        )
                    )

                    bang_san = (
                        bang_san[
                            [
                                "_ma",
                                "_san",
                            ]
                        ]
                        .drop_duplicates(
                            "_ma"
                        )
                    )

                    universe = universe.merge(
                        bang_san,
                        left_on="ma",
                        right_on="_ma",
                        how="left",
                    )

                    universe[
                        "san"
                    ] = (
                        universe["san"]
                        .replace(
                            "",
                            np.nan,
                        )
                        .fillna(
                            universe["_san"]
                        )
                        .fillna("")
                    )

                    universe = (
                        universe.drop(
                            columns=[
                                "_ma",
                                "_san",
                            ],
                            errors="ignore",
                        )
                    )

        except Exception:
            pass

    # --------------------------------------------------------
    # BỔ SUNG NGÀNH
    # --------------------------------------------------------

    if (
        universe["ten_nganh"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .all()
    ):

        try:

            bang_nganh = (
                reference
                .equity
                .list_by_industry(
                    lang="vi"
                )
            )

            if isinstance(
                bang_nganh,
                pd.DataFrame,
            ):

                cot_ma_nganh = _tim_cot(
                    bang_nganh,
                    "symbol",
                    "ticker",
                    "code",
                )

                cot_ten_nganh = _tim_cot(
                    bang_nganh,
                    "icb_name",
                    "industry_name",
                    "industry",
                )

                if (
                    cot_ma_nganh
                    is not None
                    and cot_ten_nganh
                    is not None
                ):

                    bang_nganh = (
                        bang_nganh.copy()
                    )

                    bang_nganh[
                        "_ma"
                    ] = (
                        bang_nganh[
                            cot_ma_nganh
                        ]
                        .map(
                            _chuan_hoa_ma
                        )
                    )

                    bang_nganh[
                        "_ten_nganh"
                    ] = (
                        bang_nganh[
                            cot_ten_nganh
                        ]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )

                    bang_nganh = (
                        bang_nganh[
                            [
                                "_ma",
                                "_ten_nganh",
                            ]
                        ]
                        .drop_duplicates(
                            "_ma"
                        )
                    )

                    universe = universe.merge(
                        bang_nganh,
                        left_on="ma",
                        right_on="_ma",
                        how="left",
                    )

                    universe[
                        "ten_nganh"
                    ] = (
                        universe[
                            "ten_nganh"
                        ]
                        .replace(
                            "",
                            np.nan,
                        )
                        .fillna(
                            universe[
                                "_ten_nganh"
                            ]
                        )
                        .fillna("")
                    )

                    universe = (
                        universe.drop(
                            columns=[
                                "_ma",
                                "_ten_nganh",
                            ],
                            errors="ignore",
                        )
                    )

        except Exception:
            pass

    return universe


# ============================================================
# QUOTE BULK
# ============================================================

def _goi_quote(
    market,
    danh_sach_ma,
):
    if not danh_sach_ma:
        return pd.DataFrame()

    # --------------------------------------------------------
    # CÁCH CHÍNH:
    # Market.quote(symbols_list)
    # --------------------------------------------------------

    try:

        du_lieu = market.quote(
            danh_sach_ma
        )

        if (
            isinstance(
                du_lieu,
                pd.DataFrame,
            )
            and not du_lieu.empty
        ):

            return du_lieu.copy()

    except Exception:
        pass

    # --------------------------------------------------------
    # FALLBACK:
    # keyword symbols_list / symbol
    # --------------------------------------------------------

    try:

        du_lieu = market.quote(
            symbols_list=danh_sach_ma
        )

        if (
            isinstance(
                du_lieu,
                pd.DataFrame,
            )
            and not du_lieu.empty
        ):

            return du_lieu.copy()

    except Exception:
        pass

    try:

        du_lieu = market.quote(
            symbol=danh_sach_ma
        )

        if (
            isinstance(
                du_lieu,
                pd.DataFrame,
            )
            and not du_lieu.empty
        ):

            return du_lieu.copy()

    except Exception:
        pass

    return pd.DataFrame()


# ============================================================
# CHUẨN HÓA QUOTE
# ============================================================

def _chuan_hoa_quote(
    du_lieu,
):
    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return pd.DataFrame()

    df = du_lieu.copy()

    # --------------------------------------------------------
    # Schema chuẩn Vnstock mới
    # --------------------------------------------------------

    cot_ma = _tim_cot(
        df,
        "symbol",
        "ticker",
        "code",
    )

    if cot_ma is None:
        return pd.DataFrame()

    cot_gia = _tim_cot(
        df,
        "last_price",
        "price",
        "match_price",
        "current_price",
        "close",
    )

    cot_tham_chieu = _tim_cot(
        df,
        "reference_price",
        "ref_price",
        "reference",
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

    cot_thay_doi = _tim_cot(
        df,
        "price_change_1d",
        "change",
        "price_change",
        "change_value",
    )

    cot_phan_tram = _tim_cot(
        df,
        "price_change_percent_1d",
        "change_percent",
        "change_pct",
        "price_change_percent",
    )

    cot_volume = _tim_cot(
        df,
        "volume",
        "match_volume",
        "accumulated_volume",
        "total_volume",
    )

    cot_value = _tim_cot(
        df,
        "total_value",
        "value",
        "trading_value",
        "accumulated_value",
    )

    cot_cao_52w = _tim_cot(
        df,
        "high_52w",
    )

    cot_thap_52w = _tim_cot(
        df,
        "low_52w",
    )

    cot_von_hoa = _tim_cot(
        df,
        "market_cap",
    )

    rows = []

    for _, dong in df.iterrows():

        ma = _chuan_hoa_ma(
            dong[cot_ma]
        )

        if not ma:
            continue

        gia = (
            _so(
                dong[cot_gia]
            )
            if cot_gia
            else np.nan
        )

        # Không nhận 0 / âm làm giá hợp lệ.
        if (
            not np.isfinite(gia)
            or gia <= 0
        ):
            continue

        tham_chieu = (
            _so(
                dong[
                    cot_tham_chieu
                ]
            )
            if cot_tham_chieu
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

        # ----------------------------------------------------
        # Tự tính khi nguồn không trả change.
        # ----------------------------------------------------

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
                gia
                - tham_chieu
            ) / tham_chieu * 100

        rows.append(
            {
                "ma": ma,

                "gia": gia,

                "gia_tham_chieu": (
                    tham_chieu
                ),

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

                "thay_doi": (
                    thay_doi
                ),

                "thay_doi_pct": (
                    phan_tram
                ),

                "khoi_luong": (
                    _so(
                        dong[
                            cot_volume
                        ]
                    )
                    if cot_volume
                    else np.nan
                ),

                "gia_tri_giao_dich": (
                    _so(
                        dong[
                            cot_value
                        ]
                    )
                    if cot_value
                    else np.nan
                ),

                "cao_52_tuan": (
                    _so(
                        dong[
                            cot_cao_52w
                        ]
                    )
                    if cot_cao_52w
                    else np.nan
                ),

                "thap_52_tuan": (
                    _so(
                        dong[
                            cot_thap_52w
                        ]
                    )
                    if cot_thap_52w
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
# BẢNG GIÁ TOÀN THỊ TRƯỜNG
# ============================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False,
)
def lay_bang_gia_toan_thi_truong(
    force_reload: bool = False,
):
    _ = force_reload

    market = _tao_market()

    universe = lay_universe()

    if (
        universe is None
        or universe.empty
    ):

        raise RuntimeError(
            "Universe cổ phiếu đang rỗng."
        )

    danh_sach_ma = (
        universe[
            "ma"
        ]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )

    if not danh_sach_ma:

        raise RuntimeError(
            "Không có mã cổ phiếu để lấy quote."
        )

    # --------------------------------------------------------
    # BƯỚC 1:
    # Thử BULK TOÀN BỘ một lần.
    # --------------------------------------------------------

    quote_goc = _goi_quote(
        market,
        danh_sach_ma,
    )

    # --------------------------------------------------------
    # BƯỚC 2:
    # Nếu backend từ chối request lớn,
    # fallback batch 500 mã.
    # --------------------------------------------------------

    if (
        quote_goc is None
        or quote_goc.empty
    ):

        cac_quote = []

        for bat_dau in range(
            0,
            len(danh_sach_ma),
            FALLBACK_BATCH_SIZE,
        ):

            mot_lo = danh_sach_ma[
                bat_dau:
                bat_dau
                + FALLBACK_BATCH_SIZE
            ]

            du_lieu_lo = _goi_quote(
                market,
                mot_lo,
            )

            if (
                du_lieu_lo is not None
                and not du_lieu_lo.empty
            ):

                cac_quote.append(
                    du_lieu_lo
                )

        if cac_quote:

            quote_goc = pd.concat(
                cac_quote,
                ignore_index=True,
            )

    if (
        quote_goc is None
        or quote_goc.empty
    ):

        raise RuntimeError(
            "Nguồn Market không trả về dữ liệu bảng giá."
        )

    quote = _chuan_hoa_quote(
        quote_goc
    )

    if quote.empty:

        raise RuntimeError(
            "Không có giá hợp lệ sau khi chuẩn hóa quote."
        )

    # --------------------------------------------------------
    # GHÉP THÔNG TIN REFERENCE
    # --------------------------------------------------------

    bang_tham_chieu = universe[
        [
            "ma",
            "ten_doanh_nghiep",
            "san",
            "ma_nganh",
            "ten_nganh",
        ]
    ].copy()

    bang_gia = quote.merge(
        bang_tham_chieu,
        on="ma",
        how="left",
    )

    # --------------------------------------------------------
    # CHUẨN HÓA TEXT
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
    # TRẠNG THÁI
    # --------------------------------------------------------

    thay_doi_pct = pd.to_numeric(
        bang_gia[
            "thay_doi_pct"
        ],
        errors="coerce",
    )

    bang_gia[
        "trang_thai"
    ] = np.select(
        [
            thay_doi_pct > 0.05,
            thay_doi_pct < -0.05,
        ],
        [
            "Tăng",
            "Giảm",
        ],
        default="Đứng giá",
    )

    # --------------------------------------------------------
    # GIÁ TRỊ GIAO DỊCH
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

    # Không tự nhân price * volume nếu nguồn đã có
    # total_value nhưng khác đơn vị.
    # Chỉ giữ giá trị nguồn trả về.

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

    # --------------------------------------------------------
    # TỪ KHÓA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SÀN
    # --------------------------------------------------------

    if (
        san != "Tất cả"
        and "san" in df.columns
    ):

        df = df[
            df["san"]
            .eq(
                str(san).upper()
            )
        ]

    # --------------------------------------------------------
    # NGÀNH
    # --------------------------------------------------------

    if (
        nganh != "Tất cả"
        and "ten_nganh" in df.columns
    ):

        df = df[
            df[
                "ten_nganh"
            ]
            .eq(
                str(nganh)
            )
        ]

    # --------------------------------------------------------
    # TRẠNG THÁI
    # --------------------------------------------------------

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
# THỐNG KÊ TOÀN THỊ TRƯỜNG
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
            "phan_tram_tang": np.nan,
            "phan_tram_giam": np.nan,
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
            tang / tong * 100
            if tong
            else np.nan
        ),
        "phan_tram_giam": (
            giam / tong * 100
            if tong
            else np.nan
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
            thong_ke["tang"]
            - thong_ke["giam"]
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
    so_luong=20,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    df = bang_gia[
        bang_gia[
            "thay_doi_pct"
        ].notna()
    ].copy()

    return (
        df.sort_values(
            "thay_doi_pct",
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
# TOP GIẢM
# ============================================================

def top_giam(
    bang_gia,
    so_luong=20,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    df = bang_gia[
        bang_gia[
            "thay_doi_pct"
        ].notna()
    ].copy()

    return (
        df.sort_values(
            "thay_doi_pct",
            ascending=True,
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
    so_luong=20,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    df = bang_gia[
        bang_gia[
            "khoi_luong"
        ].notna()
        &
        (
            pd.to_numeric(
                bang_gia[
                    "khoi_luong"
                ],
                errors="coerce",
            ) > 0
        )
    ].copy()

    return (
        df.sort_values(
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
    so_luong=20,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    df = bang_gia[
        bang_gia[
            "gia_tri_giao_dich"
        ].notna()
    ].copy()

    return (
        df.sort_values(
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
    ):
        return pd.DataFrame()

    df = bang_gia.copy()

    df[
        "thay_doi_pct"
    ] = pd.to_numeric(
        df[
            "thay_doi_pct"
        ],
        errors="coerce",
    )

    ket_qua = (
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

    return ket_qua


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

    df[
        "thay_doi_pct"
    ] = pd.to_numeric(
        df[
            "thay_doi_pct"
        ],
        errors="coerce",
    )

    ket_qua = (
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

    return ket_qua


# ============================================================
# NGOẠI
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

    if (
        "nuoc_ngoai_mua"
        not in bang_gia.columns
        or
        "nuoc_ngoai_ban"
        not in bang_gia.columns
    ):
        return {
            "co_du_lieu": False,
            "mua": np.nan,
            "ban": np.nan,
            "rong": np.nan,
        }

    mua = pd.to_numeric(
        bang_gia[
            "nuoc_ngoai_mua"
        ],
        errors="coerce",
    )

    ban = pd.to_numeric(
        bang_gia[
            "nuoc_ngoai_ban"
        ],
        errors="coerce",
    )

    if (
        mua.notna().sum()
        == 0
        and ban.notna().sum()
        == 0
    ):
        return {
            "co_du_lieu": False,
            "mua": np.nan,
            "ban": np.nan,
            "rong": np.nan,
        }

    tong_mua = mua.sum(
        min_count=1
    )

    tong_ban = ban.sum(
        min_count=1
    )

    if (
        np.isfinite(
            _so(
                tong_mua
            )
        )
        and np.isfinite(
            _so(
                tong_ban
            )
        )
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
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return {
            "so_ma": 0,
            "cap_nhat": None,
            "nguon": "Vnstock Market",
        }

    return {
        "so_ma": int(
            bang_gia[
                "ma"
            ].nunique()
        ),
        "cap_nhat": pd.Timestamp.now(),
        "nguon": (
            "Vnstock Market · bảng giá toàn thị trường"
        ),
    }


# ============================================================
# API CHÍNH
# ============================================================

def lay_market_overview(
    force_reload: bool = False,
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
            lay_universe(
                force_reload=force_reload
            )
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
            )
        ),
    }
