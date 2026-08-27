from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import feedparser
except Exception:
    feedparser = None


# ============================================================
# CẤU HÌNH
# ============================================================

THOI_GIAN_LUU_DU_LIEU = 300
THOI_GIAN_LUU_TIN = 600


# ============================================================
# CHUẨN HÓA MÃ
# ============================================================

def normalize_symbol(ma):
    if ma is None:
        return "HPG"

    ma = str(ma).strip().upper()
    ma = re.sub(r"\s+", "", ma)

    if not ma:
        return "HPG"

    return ma


def display_symbol(ma):
    if ma is None:
        return ""

    ma = str(ma).strip().upper()

    if ma.endswith(".VN"):
        ma = ma[:-3]

    return ma


def _tao_danh_sach_mã(ma):
    ma = normalize_symbol(ma)

    danh_sach = []

    if ma.endswith(".VN"):
        ma_goc = ma[:-3]

        danh_sach.append(ma)

        if ma_goc:
            danh_sach.append(ma_goc)

    else:
        if re.fullmatch(r"[A-Z0-9]{2,5}", ma):
            danh_sach.append(f"{ma}.VN")

        danh_sach.append(ma)

    ket_qua = []

    for gia_tri in danh_sach:
        if gia_tri not in ket_qua:
            ket_qua.append(gia_tri)

    return ket_qua


# ============================================================
# CHUẨN HÓA PERIOD
# ============================================================

def _chuan_hoa_khoang_thoi_gian(khoang):
    if khoang is None:
        return "1y"

    khoang = str(khoang).strip().lower()

    cac_khoang_hop_le = {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }

    if khoang not in cac_khoang_hop_le:
        return "1y"

    return khoang


# ============================================================
# RSI
# ============================================================

def rsi(
    chuoi_gia,
    chu_ky=14,
):
    chu_ky = int(chu_ky)

    thay_doi = chuoi_gia.diff()

    tang = thay_doi.clip(lower=0)
    giam = -thay_doi.clip(upper=0)

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
        / trung_binh_giam.replace(0, np.nan)
    )

    ket_qua = 100 - (
        100 / (1 + ti_so)
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
# CHUẨN HÓA DATAFRAME
# ============================================================

def _chuan_hoa_bang_gia(
    du_lieu,
):
    if du_lieu is None:
        raise ValueError(
            "Nguồn dữ liệu không trả về dữ liệu."
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
            "Nguồn dữ liệu trả về bảng rỗng."
        )

    du_lieu = du_lieu.copy()

    # --------------------------------------------------------
    # MultiIndex
    # --------------------------------------------------------

    if isinstance(
        du_lieu.columns,
        pd.MultiIndex,
    ):

        muc_can_dung = None

        for so_muc in range(
            du_lieu.columns.nlevels
        ):

            tap_cac_ten = {
                str(x).strip().lower()
                for x in du_lieu.columns
                .get_level_values(so_muc)
            }

            if "close" in tap_cac_ten:
                muc_can_dung = so_muc
                break

        if muc_can_dung is not None:

            du_lieu.columns = (
                du_lieu.columns
                .get_level_values(
                    muc_can_dung
                )
            )

        else:

            du_lieu.columns = [
                (
                    str(cot[-1])
                    if isinstance(
                        cot,
                        tuple,
                    )
                    else str(cot)
                )
                for cot in du_lieu.columns
            ]

    # --------------------------------------------------------
    # Chuẩn hóa tên cột
    # --------------------------------------------------------

    anh_xa = {}

    for cot in du_lieu.columns:

        ten = str(cot).strip().lower()

        if ten == "open":
            anh_xa[cot] = "Open"

        elif ten == "high":
            anh_xa[cot] = "High"

        elif ten == "low":
            anh_xa[cot] = "Low"

        elif ten == "close":
            anh_xa[cot] = "Close"

        elif ten in [
            "adj close",
            "adj_close",
            "adjusted close",
        ]:
            anh_xa[cot] = "Adj Close"

        elif ten == "volume":
            anh_xa[cot] = "Volume"

    du_lieu = du_lieu.rename(
        columns=anh_xa
    )

    # --------------------------------------------------------
    # Kiểm tra cột
    # --------------------------------------------------------

    cac_cot_bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
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
    # Ép kiểu
    # --------------------------------------------------------

    for cot in cac_cot_bat_buoc:

        du_lieu[cot] = pd.to_numeric(
            du_lieu[cot],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Thời gian
    # --------------------------------------------------------

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

    du_lieu = du_lieu.dropna(
        subset=["Close"]
    )

    if du_lieu.empty:
        raise ValueError(
            "Không còn dữ liệu giá hợp lệ."
        )

    return du_lieu


# ============================================================
# THÊM CHỈ BÁO
# ============================================================

def add_indicators(
    du_lieu,
):
    du_lieu = _chuan_hoa_bang_gia(
        du_lieu
    )

    du_lieu = du_lieu.copy()

    gia_dong = du_lieu["Close"]

    # --------------------------------------------------------
    # LỢI SUẤT
    # --------------------------------------------------------

    du_lieu["Return"] = (
        gia_dong.pct_change()
    )

    du_lieu["ReturnPct"] = (
        du_lieu["Return"] * 100
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    du_lieu["RSI"] = rsi(
        gia_dong,
        14,
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    du_lieu["EMA9"] = (
        gia_dong
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA12"] = (
        gia_dong
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA20"] = (
        gia_dong
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA26"] = (
        gia_dong
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA50"] = (
        gia_dong
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    du_lieu["SMA5"] = (
        gia_dong
        .rolling(5)
        .mean()
    )

    du_lieu["SMA10"] = (
        gia_dong
        .rolling(10)
        .mean()
    )

    du_lieu["SMA20"] = (
        gia_dong
        .rolling(20)
        .mean()
    )

    du_lieu["SMA50"] = (
        gia_dong
        .rolling(50)
        .mean()
    )

    du_lieu["SMA100"] = (
        gia_dong
        .rolling(100)
        .mean()
    )

    du_lieu["SMA200"] = (
        gia_dong
        .rolling(200)
        .mean()
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
    # BOLLINGER
    # --------------------------------------------------------

    do_lech = (
        gia_dong
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
    # BIẾN ĐỘNG
    # --------------------------------------------------------

    du_lieu["Volatility5"] = (
        du_lieu["Return"]
        .rolling(5)
        .std()
        * np.sqrt(252)
        * 100
    )

    du_lieu["Volatility20"] = (
        du_lieu["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    du_lieu["Volatility60"] = (
        du_lieu["Return"]
        .rolling(60)
        .std()
        * np.sqrt(252)
        * 100
    )

    du_lieu["Volatility_20D"] = (
        du_lieu["Volatility20"]
        / 100
    )

    # --------------------------------------------------------
    # KHỐI LƯỢNG
    # --------------------------------------------------------

    du_lieu["Volume_SMA5"] = (
        du_lieu["Volume"]
        .rolling(5)
        .mean()
    )

    du_lieu["Volume_SMA20"] = (
        du_lieu["Volume"]
        .rolling(20)
        .mean()
    )

    du_lieu["Volume_SMA50"] = (
        du_lieu["Volume"]
        .rolling(50)
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
    # BIÊN ĐỘ
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
    # ATR
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

    bien_that = pd.concat(
        [
            bien_1,
            bien_2,
            bien_3,
        ],
        axis=1,
    ).max(axis=1)

    du_lieu["ATR14"] = (
        bien_that
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # ĐỘNG LƯỢNG
    # --------------------------------------------------------

    du_lieu["Momentum5"] = (
        gia_dong
        / gia_dong.shift(5)
        - 1
    )

    du_lieu["Momentum10"] = (
        gia_dong
        / gia_dong.shift(10)
        - 1
    )

    du_lieu["Momentum20"] = (
        gia_dong
        / gia_dong.shift(20)
        - 1
    )

    # --------------------------------------------------------
    # ĐỈNH / ĐÁY
    # --------------------------------------------------------

    du_lieu["High20"] = (
        du_lieu["High"]
        .rolling(20)
        .max()
    )

    du_lieu["Low20"] = (
        du_lieu["Low"]
        .rolling(20)
        .min()
    )

    du_lieu["High50"] = (
        du_lieu["High"]
        .rolling(50)
        .max()
    )

    du_lieu["Low50"] = (
        du_lieu["Low"]
        .rolling(50)
        .min()
    )

    du_lieu["High252"] = (
        du_lieu["High"]
        .rolling(252)
        .max()
    )

    du_lieu["Low252"] = (
        du_lieu["Low"]
        .rolling(252)
        .min()
    )

    # --------------------------------------------------------
    # KHOẢNG CÁCH ĐỈNH ĐÁY
    # --------------------------------------------------------

    du_lieu["Distance_From_High20"] = (
        (
            gia_dong
            / du_lieu["High20"]
            - 1
        ) * 100
    )

    du_lieu["Distance_From_Low20"] = (
        (
            gia_dong
            / du_lieu["Low20"]
            - 1
        ) * 100
    )

    du_lieu["Distance_From_High252"] = (
        (
            gia_dong
            / du_lieu["High252"]
            - 1
        ) * 100
    )

    du_lieu["Distance_From_Low252"] = (
        (
            gia_dong
            / du_lieu["Low252"]
            - 1
        ) * 100
    )

    # --------------------------------------------------------
    # TÍN HIỆU XU HƯỚNG
    # --------------------------------------------------------

    du_lieu["Above_SMA20"] = (
        gia_dong
        > du_lieu["SMA20"]
    )

    du_lieu["Above_SMA50"] = (
        gia_dong
        > du_lieu["SMA50"]
    )

    du_lieu["SMA20_Above_SMA50"] = (
        du_lieu["SMA20"]
        > du_lieu["SMA50"]
    )

    du_lieu["SMA50_Above_SMA200"] = (
        du_lieu["SMA50"]
        > du_lieu["SMA200"]
    )

    # --------------------------------------------------------
    # LÀM SẠCH
    # --------------------------------------------------------

    du_lieu = du_lieu.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return du_lieu


# ============================================================
# TẢI DỮ LIỆU YAHOO
# ============================================================

def _tai_yahoo(
    mã,
    khoang_thoi_gian="1y",
):

    khoang_thoi_gian = (
        _chuan_hoa_khoang_thoi_gian(
            khoang_thoi_gian
        )
    )

    loi_cuoi = None

    # --------------------------------------------------------
    # Cách 1
    # --------------------------------------------------------

    try:

        doi_tuong = yf.Ticker(
            mã
        )

        du_lieu = doi_tuong.history(
            period=khoang_thoi_gian,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

        if (
            isinstance(
                du_lieu,
                pd.DataFrame,
            )
            and not du_lieu.empty
        ):

            return _chuan_hoa_bang_gia(
                du_lieu
            )

    except Exception as loi:
        loi_cuoi = loi

    # --------------------------------------------------------
    # Cách 2
    # --------------------------------------------------------

    try:

        du_lieu = yf.download(
            mã,
            period=khoang_thoi_gian,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if (
            isinstance(
                du_lieu,
                pd.DataFrame,
            )
            and not du_lieu.empty
        ):

            return _chuan_hoa_bang_gia(
                du_lieu
            )

    except Exception as loi:
        loi_cuoi = loi

    if loi_cuoi:

        raise ValueError(
            f"Lỗi nguồn dữ liệu {mã}: "
            f"{loi_cuoi}"
        )

    raise ValueError(
        f"Không tìm thấy dữ liệu {mã}."
    )


# ============================================================
# DỮ LIỆU CỔ PHIẾU
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_DU_LIEU,
    show_spinner=False,
)
def load_market_data(
    symbol,
    period="1y",
):

    mã = normalize_symbol(
        symbol
    )

    danh_sach_mã = _tao_danh_sach_mã(
        mã
    )

    loi = []

    for ticker in danh_sach_mã:

        try:

            du_lieu_tho = _tai_yahoo(
                ticker,
                period,
            )

            du_lieu = add_indicators(
                du_lieu_tho
            )

            if du_lieu.empty:
                continue

            du_lieu.attrs["symbol"] = (
                ticker
            )

            du_lieu.attrs["display_symbol"] = (
                display_symbol(ticker)
            )

            du_lieu.attrs["source"] = (
                "Yahoo Finance"
            )

            return du_lieu

        except Exception as exc:

            loi.append(
                f"{ticker}: {exc}"
            )

    chi_tiet = " | ".join(loi)

    raise ValueError(
        f"Không lấy được dữ liệu "
        f"{display_symbol(mã)}. "
        f"{chi_tiet}"
    )


# ============================================================
# VN-INDEX
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_DU_LIEU,
    show_spinner=False,
)
def load_vnindex_data():

    # Yahoo sử dụng mã này cho VN-INDEX
    danh_sach_chi_so = [
        "^VNINDEX",
        "VNINDEX.VN",
        "VNINDEX",
    ]

    loi = []

    for mã in danh_sach_chi_so:

        try:

            du_lieu = _tai_yahoo(
                mã,
                "1y",
            )

            if du_lieu.empty:
                continue

            du_lieu = add_indicators(
                du_lieu
            )

            if du_lieu.empty:
                continue

            du_lieu.attrs["symbol"] = (
                "VN-INDEX"
            )

            du_lieu.attrs["source"] = (
                "Yahoo Finance"
            )

            return du_lieu

        except Exception as exc:

            loi.append(
                f"{mã}: {exc}"
            )

    chi_tiet = " | ".join(loi)

    raise ValueError(
        "Không lấy được dữ liệu VN-INDEX. "
        + chi_tiet
    )


# ============================================================
# GIÁ HIỆN TẠI
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_latest_price(
    symbol,
):

    mã = normalize_symbol(
        symbol
    )

    for ticker in _tao_danh_sach_mã(
        mã
    ):

        try:

            du_lieu = _tai_yahoo(
                ticker,
                "5d",
            )

            if du_lieu.empty:
                continue

            gia_moi = float(
                du_lieu["Close"]
                .dropna()
                .iloc[-1]
            )

            if len(du_lieu) >= 2:

                gia_truoc = float(
                    du_lieu["Close"]
                    .dropna()
                    .iloc[-2]
                )

            else:
                gia_truoc = gia_moi

            if gia_truoc != 0:

                thay_doi = (
                    gia_moi
                    / gia_truoc
                    - 1
                ) * 100

            else:
                thay_doi = 0.0

            khoi_luong = 0

            if "Volume" in du_lieu.columns:

                try:
                    khoi_luong = int(
                        du_lieu[
                            "Volume"
                        ]
                        .iloc[-1]
                    )
                except Exception:
                    khoi_luong = 0

            return {
                "ma": display_symbol(
                    ticker
                ),
                "gia": gia_moi,
                "thay_doi": thay_doi,
                "khoi_luong": khoi_luong,
                "thoi_gian": du_lieu.index[-1],
            }

        except Exception:
            continue

    return {}


# ============================================================
# THÔNG TIN DOANH NGHIỆP
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def load_company_info(
    symbol,
):

    mã = normalize_symbol(
        symbol
    )

    for ticker in _tao_danh_sach_mã(
        mã
    ):

        try:

            doi_tuong = yf.Ticker(
                ticker
            )

            try:

                thong_tin_nhanh = dict(
                    doi_tuong.fast_info
                )

                if thong_tin_nhanh:
                    return thong_tin_nhanh

            except Exception:
                pass

            try:

                thong_tin = doi_tuong.info

                if isinstance(
                    thong_tin,
                    dict,
                ):
                    return thong_tin

            except Exception:
                pass

        except Exception:
            continue

    return {}


# ============================================================
# SNAPSHOT
# ============================================================

def market_snapshot(
    du_lieu,
):

    if du_lieu is None or du_lieu.empty:
        return {}

    dong_cuoi = du_lieu.iloc[-1]

    if len(du_lieu) >= 2:
        dong_truoc = du_lieu.iloc[-2]
    else:
        dong_truoc = dong_cuoi

    try:
        gia = float(
            dong_cuoi["Close"]
        )
    except Exception:
        gia = np.nan

    try:
        gia_truoc = float(
            dong_truoc["Close"]
        )

        thay_doi = (
            gia / gia_truoc - 1
        ) * 100

    except Exception:
        thay_doi = np.nan

    def lay_so(
        ten_cot,
    ):

        try:

            gia_tri = float(
                dong_cuoi[ten_cot]
            )

            if pd.isna(
                gia_tri
            ):
                return np.nan

            return gia_tri

        except Exception:
            return np.nan

    return {
        "price": gia,
        "change_1d": thay_doi,
        "return_1d": (
            lay_so("Return") * 100
        ),
        "rsi": lay_so("RSI"),
        "macd": lay_so("MACD"),
        "sma20": lay_so("SMA20"),
        "sma50": lay_so("SMA50"),
        "volatility20": lay_so(
            "Volatility20"
        ),
        "volume": lay_so("Volume"),
    }


# ============================================================
# DANH SÁCH MÃ TỰ ĐỘNG
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def load_symbol_list():

    ma = set()

    try:

        # Yahoo Finance không có endpoint chính thức
        # trả toàn bộ mã Việt Nam.
        #
        # Vì vậy không bịa danh sách mã.
        #
        # Danh sách thực tế sẽ được chấp nhận tự động
        # khi người dùng nhập mã.

        return []

    except Exception:
        return []


def is_valid_symbol(
    symbol,
):

    mã = normalize_symbol(
        symbol
    )

    if not mã:
        return False

    # Không khóa người dùng vào danh sách viết tay.
    return True


# ============================================================
# TIN TỨC
# ============================================================

def _lam_sach_html(
    chuoi,
):

    if chuoi is None:
        return ""

    chuoi = str(
        chuoi
    )

    chuoi = re.sub(
        r"<[^>]+>",
        " ",
        chuoi,
    )

    chuoi = re.sub(
        r"\s+",
        " ",
        chuoi,
    )

    return chuoi.strip()


def _chuyen_thoi_gian(
    gia_tri,
):

    if not gia_tri:
        return None

    try:

        if hasattr(
            gia_tri,
            "tm_year",
        ):

            return datetime(
                gia_tri.tm_year,
                gia_tri.tm_mon,
                gia_tri.tm_mday,
                gia_tri.tm_hour,
                gia_tri.tm_min,
                gia_tri.tm_sec,
                tzinfo=timezone.utc,
            )

    except Exception:
        pass

    try:

        ket_qua = (
            parsedate_to_datetime(
                str(gia_tri)
            )
        )

        if ket_qua.tzinfo is None:

            ket_qua = ket_qua.replace(
                tzinfo=timezone.utc
            )

        return ket_qua

    except Exception:
        pass

    try:

        ket_qua = pd.to_datetime(
            gia_tri,
            utc=True,
            errors="coerce",
        )

        if pd.isna(
            ket_qua
        ):
            return None

        return ket_qua.to_pydatetime()

    except Exception:
        return None


@st.cache_data(
    ttl=THOI_GIAN_LUU_TIN,
    show_spinner=False,
)
def fetch_news(
    symbol,
    limit=8,
):

    try:
        limit = int(
            float(limit)
        )
    except Exception:
        limit = 8

    limit = max(
        1,
        min(
            limit,
            50,
        ),
    )

    ma = display_symbol(
        symbol
    )

    truy_van = quote(
        f'"{ma}" cổ phiếu OR "{ma}" chứng khoán'
    )

    dia_chi = (
        "https://news.google.com/rss/search?"
        f"q={truy_van}"
        "&hl=vi"
        "&gl=VN"
        "&ceid=VN:vi"
    )

    ket_qua = []

    if feedparser is None:
        return ket_qua

    try:

        bang_tin = feedparser.parse(
            dia_chi
        )

        for muc in getattr(
            bang_tin,
            "entries",
            [],
        ):

            tieu_de = _lam_sach_html(
                muc.get(
                    "title",
                    "",
                )
            )

            lien_ket = str(
                muc.get(
                    "link",
                    "",
                )
            ).strip()

            thoi_gian_goc = (
                muc.get(
                    "published",
                    "",
                )
                or muc.get(
                    "updated",
                    "",
                )
            )

            thoi_gian = _chuyen_thoi_gian(
                thoi_gian_goc
            )

            if thoi_gian:

                hien_thi_thoi_gian = (
                    thoi_gian.astimezone()
                    .strftime(
                        "%d/%m/%Y %H:%M"
                    )
                )

            else:

                hien_thi_thoi_gian = str(
                    thoi_gian_goc
                )

            nguon = "Google News"

            try:

                nguon = (
                    muc.get(
                        "source",
                        {}
                    ).get(
                        "title",
                        "Google News",
                    )
                    or "Google News"
                )

            except Exception:
                nguon = "Google News"

            tom_tat = _lam_sach_html(
                muc.get(
                    "summary",
                    "",
                )
            )

            if not tieu_de:
                continue

            ket_qua.append(
                {
                    "title": tieu_de,
                    "link": lien_ket,
                    "published": hien_thi_thoi_gian,
                    "summary": tom_tat,
                    "source": nguon,
                }
            )

            if len(
                ket_qua
            ) >= limit:
                break

    except Exception:
        return []

    return ket_qua


# ============================================================
# TƯƠNG THÍCH CODE CŨ
# ============================================================

def get_news(
    ticker,
    days=7,
):

    tin = fetch_news(
        ticker,
        20,
    )

    if not tin:
        return []

    try:
        days = int(
            float(days)
        )
    except Exception:
        days = 7

    days = max(
        1,
        min(
            days,
            365,
        ),
    )

    moc_thoi_gian = (
        datetime.now(
            timezone.utc
        )
        - pd.Timedelta(
            days=days
        )
    )

    ket_qua = []

    for muc in tin:

        thoi_gian = _chuyen_thoi_gian(
            muc.get(
                "published",
                "",
            )
        )

        if (
            thoi_gian is None
            or thoi_gian >= moc_thoi_gian
        ):
            ket_qua.append(
                muc
            )

    return ket_qua[:20]


# ============================================================
# TƯƠNG THÍCH TÊN HÀM CŨ
# ============================================================

def market_data(
    symbol,
    period="1y",
):

    return load_market_data(
        symbol,
        period,
    )


def load_full_stock_data(
    symbol,
):

    du_lieu = load_market_data(
        symbol,
        "1y",
    )

    gia_moi = load_latest_price(
        symbol
    )

    thong_tin = load_company_info(
        symbol
    )

    return {
        "symbol": normalize_symbol(
            symbol
        ),
        "data": du_lieu,
        "latest": gia_moi,
        "company": thong_tin,
    }
