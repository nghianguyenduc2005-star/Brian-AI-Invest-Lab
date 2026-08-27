from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CẤU HÌNH
# ============================================================

CACHE_SECONDS = 60
QUOTE_BATCH_SIZE = 100


# ============================================================
# TIỆN ÍCH CHUNG
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


def _chuan_hoa_ma(
    gia_tri,
):
    if gia_tri is None:
        return ""

    ma = str(
        gia_tri
    ).strip().upper()

    if ma.endswith(".VN"):
        ma = ma[:-3]

    return ma.replace(
        " ",
        "",
    )


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

    except Exception:
        return None


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

    except Exception:
        return None


# ============================================================
# UNIVERSE TOÀN THỊ TRƯỜNG
# ============================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False,
)
def lay_universe(
    force_reload: bool = False,
):
    _ = force_reload

    reference = _tao_reference()

    if reference is None:
        raise RuntimeError(
            "Không khởi tạo được Vnstock Reference."
        )

    du_lieu = None

    # --------------------------------------------------------
    # Ưu tiên toàn bộ danh sách cổ phiếu
    # --------------------------------------------------------

    try:
        du_lieu = (
            reference
            .equity
            .list()
        )
    except Exception:
        du_lieu = None

    if (
        du_lieu is None
        or not isinstance(
            du_lieu,
            pd.DataFrame,
        )
        or du_lieu.empty
    ):

        # ----------------------------------------------------
        # Fallback theo sàn
        # ----------------------------------------------------

        try:
            du_lieu_san = (
                reference
                .equity
                .list_by_exchange()
            )

            if isinstance(
                du_lieu_san,
                dict,
            ):

                cac_bang = []

                for ten_san, bang in du_lieu_san.items():

                    if not isinstance(
                        bang,
                        pd.DataFrame,
                    ):
                        continue

                    bang = bang.copy()

                    cot_ma = _tim_cot(
                        bang,
                        "symbol",
                        "ticker",
                        "code",
                    )

                    if cot_ma is None:
                        continue

                    bang["_ma"] = (
                        bang[cot_ma]
                        .map(
                            _chuan_hoa_ma
                        )
                    )

                    bang["_san"] = (
                        str(
                            ten_san
                        ).upper()
                    )

                    cac_bang.append(
                        bang[
                            [
                                "_ma",
                                "_san",
                            ]
                        ]
                    )

                if cac_bang:

                    du_lieu = pd.concat(
                        cac_bang,
                        ignore_index=True,
                    )

            elif isinstance(
                du_lieu_san,
                pd.DataFrame,
            ):

                du_lieu = (
                    du_lieu_san.copy()
                )

        except Exception:
            du_lieu = None

    if (
        du_lieu is None
        or du_lieu.empty
    ):
        raise RuntimeError(
            "Không lấy được danh sách cổ phiếu toàn thị trường."
        )

    du_lieu = du_lieu.copy()

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
            "Không tìm thấy cột mã cổ phiếu."
        )

    cot_ten = _tim_cot(
        du_lieu,
        "organ_name",
        "company_name",
        "name",
        "company",
    )

    cot_san = _tim_cot(
        du_lieu,
        "exchange",
        "exchange_name",
        "floor",
        "_san",
    )

    cot_nganh = _tim_cot(
        du_lieu,
        "icb_name",
        "industry_name",
        "industry",
    )

    cot_ma_nganh = _tim_cot(
        du_lieu,
        "icb_code",
        "industry_code",
    )

    ket_qua = []

    for _, dong in du_lieu.iterrows():

        ma = _chuan_hoa_ma(
            dong[cot_ma]
        )

        if not ma:
            continue

        ket_qua.append(
            {
                "ma": ma,

                "ten_doanh_nghiep": (
                    str(
                        dong[cot_ten]
                    ).strip()
                    if (
                        cot_ten
                        and pd.notna(
                            dong[cot_ten]
                        )
                    )
                    else ""
                ),

                "san": (
                    str(
                        dong[cot_san]
                    ).upper().strip()
                    if (
                        cot_san
                        and pd.notna(
                            dong[cot_san]
                        )
                    )
                    else ""
                ),

                "ten_nganh": (
                    str(
                        dong[cot_nganh]
                    ).strip()
                    if (
                        cot_nganh
                        and pd.notna(
                            dong[cot_nganh]
                        )
                    )
                    else ""
                ),

                "ma_nganh": (
                    str(
                        dong[cot_ma_nganh]
                    ).strip()
                    if (
                        cot_ma_nganh
                        and pd.notna(
                            dong[cot_ma_nganh]
                        )
                    )
                    else ""
                ),
            }
        )

    universe = pd.DataFrame(
        ket_qua
    )

    if universe.empty:
        raise RuntimeError(
            "Universe cổ phiếu sau chuẩn hóa đang rỗng."
        )

    universe = (
        universe
        .drop_duplicates(
            subset=["ma"]
        )
        .reset_index(
            drop=True
        )
    )

    def _san(
        value,
    ):
        value = str(
            value or ""
        ).upper().strip()

        if value in {
            "HSX",
            "HOSE",
            "HO CHI MINH",
            "HOCHIMINH",
        }:
            return "HOSE"

        if value in {
            "HNX",
            "HA NOI",
            "HANOI",
        }:
            return "HNX"

        if value in {
            "UPCOM",
        }:
            return "UPCOM"

        return value

    universe[
        "san"
    ] = universe[
        "san"
    ].map(
        _san
    )

    # --------------------------------------------------------
    # Bổ sung ngành nếu nguồn list() chưa có
    # --------------------------------------------------------

    try:

        da_co_nganh = (
            universe[
                "ten_nganh"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .any()
        )

        if not da_co_nganh:

            bang_nganh = (
                reference
                .equity
                .list_by_industry(
                    lang="vi"
                )
            )

            if (
                isinstance(
                    bang_nganh,
                    pd.DataFrame,
                )
                and not bang_nganh.empty
            ):

                cot_nganh_ma = _tim_cot(
                    bang_nganh,
                    "symbol",
                    "ticker",
                    "code",
                )

                cot_nganh_ten = _tim_cot(
                    bang_nganh,
                    "icb_name",
                    "industry_name",
                    "industry",
                )

                if cot_nganh_ma is not None:

                    bang_nganh = (
                        bang_nganh.copy()
                    )

                    bang_nganh[
                        "ma"
                    ] = (
                        bang_nganh[
                            cot_nganh_ma
                        ]
                        .map(
                            _chuan_hoa_ma
                        )
                    )

                    cot_ghep = [
                        "ma"
                    ]

                    if cot_nganh_ten is not None:

                        bang_nganh[
                            "ten_nganh"
                        ] = (
                            bang_nganh[
                                cot_nganh_ten
                            ]
                            .fillna("")
                            .astype(str)
                        )

                        cot_ghep.append(
                            "ten_nganh"
                        )

                    bang_nganh = (
                        bang_nganh[
                            cot_ghep
                        ]
                        .drop_duplicates(
                            "ma"
                        )
                    )

                    universe = universe.merge(
                        bang_nganh,
                        on="ma",
                        how="left",
                        suffixes=(
                            "",
                            "_moi",
                        ),
                    )

                    if (
                        "ten_nganh_moi"
                        in universe.columns
                    ):

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
                                    "ten_nganh_moi"
                                ]
                            )
                            .fillna("")
                        )

                        universe = (
                            universe.drop(
                                columns=[
                                    "ten_nganh_moi"
                                ]
                            )
                        )

    except Exception:
        pass

    return universe


# ============================================================
# CHUYỂN ĐỔI QUOTE
# ============================================================

def _chuan_hoa_quote(
    du_lieu,
):
    if (
        du_lieu is None
        or not isinstance(
            du_lieu,
            pd.DataFrame,
        )
        or du_lieu.empty
    ):
        return pd.DataFrame()

    df = du_lieu.copy()

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
        "match_price",
        "price",
        "last_price",
        "current_price",
        "close",
        "close_price",
    )

    cot_tham_chieu = _tim_cot(
        df,
        "ref_price",
        "reference_price",
        "reference",
        "gia_tham_chieu",
    )

    cot_mo = _tim_cot(
        df,
        "open",
        "open_price",
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
        "change",
        "price_change",
        "change_value",
    )

    cot_phan_tram = _tim_cot(
        df,
        "change_percent",
        "price_change_percent",
        "change_pct",
        "price_change_percent_1d",
    )

    cot_khoi_luong = _tim_cot(
        df,
        "match_volume",
        "volume",
        "total_volume",
    )

    cot_gia_tri = _tim_cot(
        df,
        "total_value",
        "value",
        "trading_value",
        "value_traded",
    )

    cot_mua_ngoai = _tim_cot(
        df,
        "foreign_buy_volume",
        "foreign_buy",
    )

    cot_ban_ngoai = _tim_cot(
        df,
        "foreign_sell_volume",
        "foreign_sell",
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

        if not np.isfinite(
            gia
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
                / tham_chieu
                - 1
            ) * 100

        if (
            np.isfinite(
                phan_tram
            )
            and abs(
                phan_tram
            ) < 1
        ):
            phan_tram *= 100

        rows.append(
            {
                "ma": ma,
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
                "nuoc_ngoai_mua": (
                    _so(
                        dong[
                            cot_mua_ngoai
                        ]
                    )
                    if cot_mua_ngoai
                    else np.nan
                ),
                "nuoc_ngoai_ban": (
                    _so(
                        dong[
                            cot_ban_ngoai
                        ]
                    )
                    if cot_ban_ngoai
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
    )


# ============================================================
# LẤY QUOTE HÀNG LOẠT
# ============================================================

def _lay_quote_mot_lo(
    market,
    danh_sach_ma,
):
    if not danh_sach_ma:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Cách 1: Market.quote(symbols)
    # --------------------------------------------------------

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
            return du_lieu

    except Exception:
        pass

    # --------------------------------------------------------
    # Cách 2: Market.quote(symbols)
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
            return du_lieu

    except Exception:
        pass

    return pd.DataFrame()


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
    """
    Lấy bảng giá toàn thị trường.

    Không nhập tay mã.

    Universe:
        Reference.equity.list()

    Quote:
        Market.quote(...)

    Cache:
        60 giây
    """

    _ = force_reload

    market = _tao_market()

    if market is None:
        raise RuntimeError(
            "Không khởi tạo được Vnstock Market."
        )

    universe = lay_universe()

    if (
        universe is None
        or universe.empty
    ):
        raise RuntimeError(
            "Không có universe cổ phiếu."
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
            "Không có mã cổ phiếu hợp lệ."
        )

    cac_quote = []

    # --------------------------------------------------------
    # Lấy theo lô để giảm kích thước request
    # --------------------------------------------------------

    for bat_dau in range(
        0,
        len(danh_sach_ma),
        QUOTE_BATCH_SIZE,
    ):

        mot_lo = danh_sach_ma[
            bat_dau:
            bat_dau
            + QUOTE_BATCH_SIZE
        ]

        du_lieu_lo = _lay_quote_mot_lo(
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

    # --------------------------------------------------------
    # Nếu bulk không hoạt động, không giả số.
    # --------------------------------------------------------

    if not cac_quote:

        raise RuntimeError(
            "Nguồn Market chưa trả về dữ liệu quote "
            "toàn thị trường. Không dùng dữ liệu giả."
        )

    quote_goc = pd.concat(
        cac_quote,
        ignore_index=True,
    )

    quote = _chuan_hoa_quote(
        quote_goc
    )

    if quote.empty:

        raise RuntimeError(
            "Không chuẩn hóa được dữ liệu bảng giá."
        )

    # --------------------------------------------------------
    # Ghép universe
    # --------------------------------------------------------

    bang_ghep = universe[
        [
            "ma",
            "ten_doanh_nghiep",
            "san",
            "ma_nganh",
            "ten_nganh",
        ]
    ].copy()

    bang_gia = quote.merge(
        bang_ghep,
        on="ma",
        how="left",
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
    # Giá trị giao dịch dự phòng
    # --------------------------------------------------------

    thieu_gia_tri = (
        bang_gia[
            "gia_tri_giao_dich"
        ].isna()
        & bang_gia[
            "gia"
        ].notna()
        & bang_gia[
            "khoi_luong"
        ].notna()
    )

    bang_gia.loc[
        thieu_gia_tri,
        "gia_tri_giao_dich",
    ] = (
        bang_gia.loc[
            thieu_gia_tri,
            "gia",
        ]
        * bang_gia.loc[
            thieu_gia_tri,
            "khoi_luong",
        ]
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

    tu_khoa = str(
        tu_khoa or ""
    ).strip()

    if tu_khoa:

        mask_ma = (
            df["ma"]
            .fillna("")
            .astype(str)
            .str.contains(
                tu_khoa.upper(),
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
        huong != "Tất cả"
        and "trang_thai" in df.columns
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
# TOP CỔ PHIẾU
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

    return (
        bang_gia[
            bang_gia[
                "thay_doi_pct"
            ].notna()
        ]
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
    bang_gia,
    so_luong=20,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    return (
        bang_gia[
            bang_gia[
                "thay_doi_pct"
            ].notna()
        ]
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
    bang_gia,
    so_luong=20,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    return (
        bang_gia[
            bang_gia[
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
        .copy()
    )


def top_gia_tri_giao_dich(
    bang_gia,
    so_luong=20,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    return (
        bang_gia[
            bang_gia[
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
        .copy()
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
# THỐNG KÊ THEO NGÀNH
# ============================================================

def thong_ke_theo_nganh(
    bang_gia,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    df = bang_gia.copy()

    if "ten_nganh" not in df.columns:
        return pd.DataFrame()

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
            _so(tong_mua)
        )
        and np.isfinite(
            _so(tong_ban)
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
            "Vnstock Market · bảng giá thị trường"
        ),
    }


# ============================================================
# HÀM TỔNG
# ============================================================

def lay_market_overview(
    force_reload: bool = False,
):
    """
    API chính cho pages/market.py.
    """

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

        "universe": lay_universe(
            force_reload=force_reload
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
