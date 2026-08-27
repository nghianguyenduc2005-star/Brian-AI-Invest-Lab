from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    from vnstock import Market

    NGUON_VNSTOCK_CO_SAN = True
except Exception:
    Market = None
    NGUON_VNSTOCK_CO_SAN = False


# ============================================================
# CẤU HÌNH
# ============================================================

THOI_GIAN_LUU_CO_PHIEU = 300
THOI_GIAN_LUU_VNINDEX = 300
THOI_GIAN_LUU_GIA_MOI = 30


# ============================================================
# CHUẨN HÓA MÃ
# ============================================================

def normalize_symbol(symbol):
    if symbol is None:
        return "HPG"

    ma = str(symbol).strip().upper()

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


def display_symbol(symbol):
    if symbol is None:
        return ""

    ma = str(symbol).strip().upper()

    if ma.endswith(".VN"):
        ma = ma[:-3]

    return ma


# ============================================================
# KHỞI TẠO NGUỒN DỮ LIỆU
# ============================================================

@st.cache_resource(
    show_spinner=False
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
        )


# ============================================================
# SỐ NGÀY THEO KHUNG THỜI GIAN
# ============================================================

def _so_ngay_theo_ky(
    ky_hieu
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

    # --------------------------------------------------------
    # MultiIndex
    # --------------------------------------------------------

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
                    str(cot[-1])
                )
            else:
                cot_moi.append(
                    str(cot)
                )

        du_lieu.columns = cot_moi

    # --------------------------------------------------------
    # Chuẩn hóa tên cột
    # --------------------------------------------------------

    anh_xa = {}

    for cot in du_lieu.columns:

        ten = str(
            cot
        ).strip().lower()

        if ten in {
            "time",
            "date",
            "datetime",
            "timestamp",
            "tradingdate",
            "trading_date",
        }:
            anh_xa[cot] = "Time"

        elif ten in {
            "open",
            "open_price",
            "openprice",
        }:
            anh_xa[cot] = "Open"

        elif ten in {
            "high",
            "high_price",
            "highprice",
        }:
            anh_xa[cot] = "High"

        elif ten in {
            "low",
            "low_price",
            "lowprice",
        }:
            anh_xa[cot] = "Low"

        elif ten in {
            "close",
            "close_price",
            "closeprice",
            "last",
            "lastprice",
            "last_price",
        }:
            anh_xa[cot] = "Close"

        elif ten in {
            "volume",
            "vol",
            "total_volume",
            "matchvolume",
            "match_volume",
        }:
            anh_xa[cot] = "Volume"

        elif ten in {
            "value",
            "trading_value",
            "value_traded",
            "matchvalue",
            "match_value",
        }:
            anh_xa[cot] = "Value"

    du_lieu = du_lieu.rename(
        columns=anh_xa
    )

    # --------------------------------------------------------
    # Thời gian
    # --------------------------------------------------------

    if "Time" in du_lieu.columns:

        du_lieu["Time"] = pd.to_datetime(
            du_lieu["Time"],
            errors="coerce",
        )

        du_lieu = du_lieu.set_index(
            "Time"
        )

    else:

        du_lieu.index = pd.to_datetime(
            du_lieu.index,
            errors="coerce",
        )

    du_lieu = du_lieu[
        ~du_lieu.index.isna()
    ].copy()

    du_lieu = du_lieu.sort_index()

    du_lieu = du_lieu[
        ~du_lieu.index.duplicated(
            keep="last"
        )
    ].copy()

    # --------------------------------------------------------
    # Kiểm tra cột bắt buộc
    # --------------------------------------------------------

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
            + ", ".join(thieu)
        )

    # --------------------------------------------------------
    # Không có Volume thì tạo 0
    # --------------------------------------------------------

    if "Volume" not in du_lieu.columns:
        du_lieu["Volume"] = 0.0

    # --------------------------------------------------------
    # Ép kiểu số
    # --------------------------------------------------------

    for cot in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

        du_lieu[cot] = pd.to_numeric(
            du_lieu[cot],
            errors="coerce",
        )

    # ========================================================
    # QUAN TRỌNG:
    # CHUẨN HÓA ĐƠN VỊ GIÁ CỔ PHIẾU
    #
    # Một số dữ liệu cổ phiếu Việt Nam trả:
    #
    #     MSR = 45
    #
    # trong khi giao diện cần:
    #
    #     45,000 VND
    #
    # Chỉ cổ phiếu mới được xét.
    # VN-INDEX tuyệt đối không nhân 1,000.
    # ========================================================

    if la_co_phieu:

        gia_trung_vi = (
            du_lieu["Close"]
            .dropna()
            .median()
        )

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

                du_lieu[cot] = (
                    du_lieu[cot]
                    * 1000
                )

    # --------------------------------------------------------
    # Loại dữ liệu giá lỗi
    # --------------------------------------------------------

    du_lieu.loc[
        du_lieu["Open"] <= 0,
        "Open",
    ] = np.nan

    du_lieu.loc[
        du_lieu["High"] <= 0,
        "High",
    ] = np.nan

    du_lieu.loc[
        du_lieu["Low"] <= 0,
        "Low",
    ] = np.nan

    du_lieu.loc[
        du_lieu["Close"] <= 0,
        "Close",
    ] = np.nan

    du_lieu.loc[
        du_lieu["Volume"] < 0,
        "Volume",
    ] = np.nan

    # --------------------------------------------------------
    # Loại vô cực
    # --------------------------------------------------------

    du_lieu = du_lieu.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # Chỉ cần OHLC hợp lệ
    # --------------------------------------------------------

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

    thay_doi = chuoi_gia.diff()

    tang = thay_doi.clip(
        lower=0
    )

    giam = -thay_doi.clip(
        upper=0
    )

    trung_binh_tang = tang.ewm(
        alpha=1 / chu_ky,
        adjust=False,
        min_periods=chu_ky,
    ).mean()

    trung_binh_giam = giam.ewm(
        alpha=1 / chu_ky,
        adjust=False,
        min_periods=chu_ky,
    ).mean()

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
            (trung_binh_giam == 0)
            & (trung_binh_tang > 0)
        ),
        100,
    )

    ket_qua = ket_qua.where(
        ~(
            (trung_binh_giam == 0)
            & (trung_binh_tang == 0)
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

    du_lieu = _chuan_hoa_bang_gia(
        du_lieu,
        la_co_phieu=la_co_phieu,
    ).copy()

    gia = du_lieu[
        "Close"
    ]

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    du_lieu["Return"] = (
        gia.pct_change()
    )

    du_lieu["ReturnPct"] = (
        du_lieu["Return"]
        * 100
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    du_lieu["RSI"] = rsi(
        gia,
        14,
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

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
            ).mean()
        )

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

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
            gia.rolling(
                so_phien
            ).mean()
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    du_lieu["MACD"] = (
        du_lieu["EMA12"]
        - du_lieu["EMA26"]
    )

    du_lieu["MACD_Signal"] = (
        du_lieu["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    du_lieu["MACD_Hist"] = (
        du_lieu["MACD"]
        - du_lieu["MACD_Signal"]
    )

    # --------------------------------------------------------
    # Bollinger
    # --------------------------------------------------------

    do_lech = (
        gia
        .rolling(20)
        .std()
    )

    du_lieu["Bollinger_Mid"] = (
        du_lieu["SMA20"]
    )

    du_lieu["Bollinger_Upper"] = (
        du_lieu["SMA20"]
        + 2 * do_lech
    )

    du_lieu["Bollinger_Lower"] = (
        du_lieu["SMA20"]
        - 2 * do_lech
    )

    du_lieu["Bollinger_Width"] = (
        (
            du_lieu["Bollinger_Upper"]
            - du_lieu["Bollinger_Lower"]
        )
        / du_lieu["Bollinger_Mid"]
        * 100
    )

    # --------------------------------------------------------
    # Biến động
    # --------------------------------------------------------

    for so_phien in [
        5,
        20,
        60,
    ]:

        du_lieu[
            f"Volatility{so_phien}"
        ] = (
            du_lieu["Return"]
            .rolling(so_phien)
            .std()
            * np.sqrt(252)
            * 100
        )

    du_lieu["Volatility_20D"] = (
        du_lieu["Volatility20"]
        / 100
    )

    # --------------------------------------------------------
    # Khối lượng
    # --------------------------------------------------------

    for so_phien in [
        5,
        20,
        50,
    ]:

        du_lieu[
            f"Volume_SMA{so_phien}"
        ] = (
            du_lieu["Volume"]
            .rolling(
                so_phien
            )
            .mean()
        )

    du_lieu["Volume_Change"] = (
        du_lieu["Volume"]
        .pct_change()
    )

    du_lieu["Relative_Volume"] = (
        du_lieu["Volume"]
        / du_lieu["Volume_SMA20"]
    )

    # --------------------------------------------------------
    # Biên độ
    # --------------------------------------------------------

    du_lieu["Range"] = (
        du_lieu["High"]
        - du_lieu["Low"]
    )

    du_lieu["Range_Percent"] = (
        du_lieu["Range"]
        / du_lieu["Close"]
        * 100
    )

    # --------------------------------------------------------
    # ATR14
    # --------------------------------------------------------

    bien_1 = (
        du_lieu["High"]
        - du_lieu["Low"]
    )

    bien_2 = (
        du_lieu["High"]
        - du_lieu["Close"].shift(1)
    ).abs()

    bien_3 = (
        du_lieu["Low"]
        - du_lieu["Close"].shift(1)
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

    du_lieu["ATR14"] = (
        bien_do_that
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # Động lượng
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Đỉnh / đáy
    # --------------------------------------------------------

    for so_phien in [
        20,
        50,
        252,
    ]:

        du_lieu[
            f"High{so_phien}"
        ] = (
            du_lieu["High"]
            .rolling(
                so_phien
            )
            .max()
        )

        du_lieu[
            f"Low{so_phien}"
        ] = (
            du_lieu["Low"]
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
# LẤY DỮ LIỆU CỔ PHIẾU
# ============================================================

def _lay_co_phieu_vnstock(
    ma,
    ky_hieu="1y",
):

    nguon = _tao_nguon_thi_truong()

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
            days=int(so_ngay)
        )
    )

    du_lieu = (
        nguon
        .equity(
            ma
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
            f"Không có dữ liệu OHLCV cho {ma}."
        )

    return _chuan_hoa_bang_gia(
        du_lieu,
        la_co_phieu=True,
    )


# ============================================================
# LẤY DỮ LIỆU VN-INDEX
# ============================================================

def _lay_vnindex_vnstock():

    nguon = _tao_nguon_thi_truong()

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

    du_lieu_goc = _lay_co_phieu_vnstock(
        ma,
        period,
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

    # Cho chart biết đây là cổ phiếu
    du_lieu.attrs[
        "la_co_phieu"
    ] = True

    return du_lieu


# ============================================================
# HÀM CHÍNH — VN-INDEX
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

    du_lieu = _lay_vnindex_vnstock()

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
        du_lieu.iloc[-1]
    )

    gia = float(
        dong_cuoi[
            "Close"
        ]
    )

    if len(du_lieu) >= 2:

        gia_truoc = float(
            du_lieu[
                "Close"
            ].iloc[-2]
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
        "thoi_gian": du_lieu.index[-1],
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
        du_lieu.iloc[-1]
    )

    if len(du_lieu) >= 2:
        dong_truoc = (
            du_lieu.iloc[-2]
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

    def _so(
        ten_cot
    ):

        try:

            gia_tri = float(
                dong_cuoi[
                    ten_cot
                ]
            )

            if pd.notna(
                gia_tri
            ):
                return gia_tri

        except Exception:
            pass

        return np.nan

    return {
        "price": gia,
        "change_1d": thay_doi,
        "return_1d": _so(
            "ReturnPct"
        ),
        "rsi": _so(
            "RSI"
        ),
        "macd": _so(
            "MACD"
        ),
        "sma20": _so(
            "SMA20"
        ),
        "sma50": _so(
            "SMA50"
        ),
        "volatility20": _so(
            "Volatility20"
        ),
        "volume": _so(
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
# SENTIMENT TIN TỨC
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
# OLS
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

    sach = du_lieu[
        cac_cot
        + ["Return"]
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if len(sach) < 50:
        return None

    try:

        X = sm.add_constant(
            sach[cac_cot],
            has_constant="add",
        )

        y = sach[
            "Return"
        ]

        return sm.OLS(
            y,
            X,
        ).fit()

    except Exception:
        return None


# ============================================================
# RANDOM FOREST
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

    sach = du_lieu[
        cac_cot
        + ["Return"]
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    sach["Target"] = (
        sach["Return"]
        .shift(-1)
    )

    sach = sach.dropna()

    if len(sach) < 80:
        return None

    try:

        X = sach[
            cac_cot
        ].astype(float)

        y = sach[
            "Target"
        ].astype(float)

        vi_tri = int(
            len(sach)
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
            X.iloc[:vi_tri],
            y.iloc[:vi_tri],
        )

        du_bao = float(
            mo_hinh.predict(
                X.iloc[[-1]]
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
# BUILD QUANT
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

    sach = du_lieu.copy()

    sach["Target"] = (
        sach["Return"]
        .shift(-1)
    )

    sach = sach.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    sach = sach.dropna(
        subset=(
            cac_cot
            + ["Target"]
        )
    )

    if len(sach) < 60:
        return None

    X = sach[
        cac_cot
    ].astype(float)

    y = sach[
        "Target"
    ].astype(float)

    vi_tri = int(
        len(sach)
        * 0.8
    )

    if vi_tri < 30:
        return None

    if vi_tri >= len(sach):
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
                X.iloc[[-1]]
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
