from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from vnstock import Market

    THI_TRUONG = Market()

except Exception:
    THI_TRUONG = None


# ============================================================
# CẤU HÌNH
# ============================================================

THOI_GIAN_LUU_CO_PHIEU = 300
THOI_GIAN_LUU_VNINDEX = 60


# ============================================================
# CHUẨN HÓA MÃ
# ============================================================

def normalize_symbol(symbol):

    if symbol is None:
        return "HPG"

    symbol = str(symbol).strip().upper()
    symbol = re.sub(r"\s+", "", symbol)

    if not symbol:
        return "HPG"

    return symbol


def display_symbol(symbol):

    if symbol is None:
        return ""

    symbol = str(symbol).strip().upper()

    if symbol.endswith(".VN"):
        symbol = symbol[:-3]

    return symbol


# ============================================================
# RSI
# ============================================================

def rsi(series, period=14):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    thay_doi = series.diff()

    tang = thay_doi.clip(
        lower=0
    )

    giam = -thay_doi.clip(
        upper=0
    )

    trung_binh_tang = tang.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    trung_binh_giam = giam.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    ti_so = (
        trung_binh_tang
        / trung_binh_giam.replace(
            0,
            np.nan,
        )
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
# CHUẨN HÓA BẢNG OHLCV
# ============================================================

def _chuan_hoa_bang_gia(du_lieu):

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

        for muc in range(
            du_lieu.columns.nlevels
        ):

            cac_ten = {
                str(x).strip().lower()
                for x in du_lieu.columns
                .get_level_values(muc)
            }

            if "close" in cac_ten:

                du_lieu.columns = (
                    du_lieu.columns
                    .get_level_values(muc)
                )

                break

    # --------------------------------------------------------
    # Tên cột
    # --------------------------------------------------------

    anh_xa = {}

    for cot in du_lieu.columns:

        ten = str(cot).strip().lower()

        if ten in [
            "time",
            "date",
            "datetime",
            "timestamp",
        ]:
            anh_xa[cot] = "Time"

        elif ten == "open":
            anh_xa[cot] = "Open"

        elif ten == "high":
            anh_xa[cot] = "High"

        elif ten == "low":
            anh_xa[cot] = "Low"

        elif ten == "close":
            anh_xa[cot] = "Close"

        elif ten == "volume":
            anh_xa[cot] = "Volume"

        elif ten in [
            "value",
            "value_traded",
            "trading_value",
        ]:
            anh_xa[cot] = "Value"

    du_lieu = du_lieu.rename(
        columns=anh_xa
    )

    # --------------------------------------------------------
    # Time
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

    # --------------------------------------------------------
    # OHLCV
    # --------------------------------------------------------

    cac_cot = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    thieu = [
        cot
        for cot in cac_cot
        if cot not in du_lieu.columns
    ]

    if thieu:
        raise ValueError(
            "Thiếu cột: "
            + ", ".join(thieu)
        )

    for cot in cac_cot:

        du_lieu[cot] = pd.to_numeric(
            du_lieu[cot],
            errors="coerce",
        )

    du_lieu = du_lieu.dropna(
        subset=["Close"]
    )

    if du_lieu.empty:
        raise ValueError(
            "Không có giá đóng cửa hợp lệ."
        )

    return du_lieu


# ============================================================
# CHỈ BÁO
# ============================================================

def add_indicators(du_lieu):

    du_lieu = _chuan_hoa_bang_gia(
        du_lieu
    )

    du_lieu = du_lieu.copy()

    gia = du_lieu["Close"]

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    du_lieu["Return"] = (
        gia.pct_change()
    )

    du_lieu["ReturnPct"] = (
        du_lieu["Return"] * 100
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

    du_lieu["EMA9"] = (
        gia.ewm(
            span=9,
            adjust=False,
        ).mean()
    )

    du_lieu["EMA12"] = (
        gia.ewm(
            span=12,
            adjust=False,
        ).mean()
    )

    du_lieu["EMA20"] = (
        gia.ewm(
            span=20,
            adjust=False,
        ).mean()
    )

    du_lieu["EMA26"] = (
        gia.ewm(
            span=26,
            adjust=False,
        ).mean()
    )

    du_lieu["EMA50"] = (
        gia.ewm(
            span=50,
            adjust=False,
        ).mean()
    )

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    du_lieu["SMA5"] = (
        gia.rolling(5).mean()
    )

    du_lieu["SMA10"] = (
        gia.rolling(10).mean()
    )

    du_lieu["SMA20"] = (
        gia.rolling(20).mean()
    )

    du_lieu["SMA50"] = (
        gia.rolling(50).mean()
    )

    du_lieu["SMA100"] = (
        gia.rolling(100).mean()
    )

    du_lieu["SMA200"] = (
        gia.rolling(200).mean()
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
    # Volume
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

    bien_do_that = pd.concat(
        [
            bien_1,
            bien_2,
            bien_3,
        ],
        axis=1,
    ).max(axis=1)

    du_lieu["ATR14"] = (
        bien_do_that
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # Động lượng
    # --------------------------------------------------------

    du_lieu["Momentum5"] = (
        gia
        / gia.shift(5)
        - 1
    )

    du_lieu["Momentum10"] = (
        gia
        / gia.shift(10)
        - 1
    )

    du_lieu["Momentum20"] = (
        gia
        / gia.shift(20)
        - 1
    )

    # --------------------------------------------------------
    # Đỉnh / đáy
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
    # Khoảng cách đỉnh đáy
    # --------------------------------------------------------

    du_lieu["Distance_From_High20"] = (
        (
            gia
            / du_lieu["High20"]
            - 1
        ) * 100
    )

    du_lieu["Distance_From_Low20"] = (
        (
            gia
            / du_lieu["Low20"]
            - 1
        ) * 100
    )

    du_lieu["Distance_From_High252"] = (
        (
            gia
            / du_lieu["High252"]
            - 1
        ) * 100
    )

    du_lieu["Distance_From_Low252"] = (
        (
            gia
            / du_lieu["Low252"]
            - 1
        ) * 100
    )

    du_lieu = du_lieu.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return du_lieu


# ============================================================
# DỮ LIỆU CỔ PHIẾU - VNSTOCK
# ============================================================

def _lay_co_phieu_vnstock(
    symbol,
    period="1y",
):

    if THI_TRUONG is None:
        raise RuntimeError(
            "Không khởi tạo được Vnstock."
        )

    ma = normalize_symbol(
        symbol
    )

    if ma.endswith(".VN"):
        ma = ma[:-3]

    # Vnstock v4 hỗ trợ start/end cho OHLCV.
    # Tính ngày bằng số nguyên, không bao giờ
    # truyền chuỗi vào timedelta.

    bang_ngay = {
        "1d": 1,
        "5d": 5,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
    }

    period = str(
        period
    ).lower().strip()

    so_ngay = bang_ngay.get(
        period,
        365,
    )

    ngay_ket_thuc = datetime.now()

    ngay_bat_dau = (
        ngay_ket_thuc
        - timedelta(
            days=int(so_ngay)
        )
    )

    du_lieu = (
        THI_TRUONG
        .equity
        .ohlcv(
            symbol=ma,
            start=ngay_bat_dau.strftime(
                "%Y-%m-%d"
            ),
            end=(ngay_ket_thuc + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            interval="1D",
        )
    )

    return _chuan_hoa_bang_gia(
        du_lieu
    )


# ============================================================
# DỮ LIỆU CỔ PHIẾU - YAHOO DỰ PHÒNG
# ============================================================

def _lay_co_phieu_yahoo(
    symbol,
    period="1y",
):

    symbol = normalize_symbol(
        symbol
    )

    danh_sach = []

    if symbol.endswith(".VN"):

        danh_sach.append(
            symbol
        )

        goc = symbol[:-3]

        if goc:
            danh_sach.append(
                goc
            )

    else:

        if re.fullmatch(
            r"[A-Z0-9]{2,5}",
            symbol,
        ):

            danh_sach.append(
                f"{symbol}.VN"
            )

        danh_sach.append(
            symbol
        )

    danh_sach = list(
        dict.fromkeys(
            danh_sach
        )
    )

    loi = []

    for ticker in danh_sach:

        try:

            du_lieu = yf.Ticker(
                ticker
            ).history(
                period=str(period),
                interval="1d",
                auto_adjust=False,
                actions=False,
            )

            if (
                du_lieu is not None
                and not du_lieu.empty
            ):

                return _chuan_hoa_bang_gia(
                    du_lieu
                )

        except Exception as exc:

            loi.append(
                f"{ticker}: {exc}"
            )

    raise ValueError(
        "Không lấy được dữ liệu "
        f"{display_symbol(symbol)}. "
        + " | ".join(loi)
    )


# ============================================================
# HÀM CHÍNH CỔ PHIẾU
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_CO_PHIEU,
    show_spinner=False,
)
def load_market_data(
    symbol,
    period="1y",
):

    symbol = normalize_symbol(
        symbol
    )

    # Ưu tiên Vnstock với mã Việt Nam
    if (
        THI_TRUONG is not None
        and re.fullmatch(
            r"[A-Z0-9]{2,5}(\.VN)?",
            symbol,
        )
    ):

        try:

            du_lieu = _lay_co_phieu_vnstock(
                symbol,
                period,
            )

            du_lieu = add_indicators(
                du_lieu
            )

            du_lieu.attrs["symbol"] = (
                display_symbol(symbol)
            )

            du_lieu.attrs["source"] = (
                "Vnstock"
            )

            return du_lieu

        except Exception:
            pass

    # Fallback Yahoo
    du_lieu = _lay_co_phieu_yahoo(
        symbol,
        period,
    )

    du_lieu = add_indicators(
        du_lieu
    )

    du_lieu.attrs["symbol"] = (
        display_symbol(symbol)
    )

    du_lieu.attrs["source"] = (
        "Yahoo Finance"
    )

    return du_lieu


# ============================================================
# VN-INDEX
# ============================================================
#
# TUYỆT ĐỐI KHÔNG DÙNG YAHOO CHO VN-INDEX.
#
# Nguồn:
#     Vnstock Market -> Index -> VNINDEX
#
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_VNINDEX,
    show_spinner=False,
)
def load_vnindex_data():

    if THI_TRUONG is None:

        raise RuntimeError(
            "Không khởi tạo được Vnstock Market."
        )

    ngay_ket_thuc = datetime.now()

    # Lấy dư 1 năm để đủ dữ liệu
    # cho SMA200 và các chỉ báo.
    ngay_bat_dau = (
        ngay_ket_thuc
        - timedelta(
            days=450
        )
    )

    try:

        du_lieu = (
            THI_TRUONG
            .index(
                "VNINDEX"
            )
            .ohlcv(
                start=ngay_bat_dau.strftime(
                    "%Y-%m-%d"
                ),
                end=(ngay_ket_thuc + timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                interval="1D",
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Vnstock không lấy được VN-INDEX: "
            f"{exc}"
        )

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        raise RuntimeError(
            "Vnstock trả về VN-INDEX rỗng."
        )

    du_lieu = _chuan_hoa_bang_gia(
        du_lieu
    )

    du_lieu = add_indicators(
        du_lieu
    )

    du_lieu.attrs["symbol"] = (
        "VNINDEX"
    )

    du_lieu.attrs["source"] = (
        "Vnstock KBS"
    )

    return du_lieu


# ============================================================
# GIÁ HIỆN TẠI
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_latest_price(symbol):

    try:

        du_lieu = load_market_data(
            symbol,
            "5d",
        )

        if du_lieu.empty:
            return {}

        gia_moi = float(
            du_lieu["Close"].iloc[-1]
        )

        if len(du_lieu) >= 2:

            gia_truoc = float(
                du_lieu["Close"].iloc[-2]
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

            khoi_luong = float(
                du_lieu[
                    "Volume"
                ].iloc[-1]
            )

        return {
            "ma": display_symbol(
                symbol
            ),
            "gia": gia_moi,
            "thay_doi": thay_doi,
            "khoi_luong": khoi_luong,
            "thoi_gian": du_lieu.index[-1],
        }

    except Exception:
        return {}


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

    cuoi = du_lieu.iloc[-1]

    if len(du_lieu) >= 2:
        truoc = du_lieu.iloc[-2]
    else:
        truoc = cuoi

    gia = float(
        cuoi["Close"]
    )

    gia_truoc = float(
        truoc["Close"]
    )

    if gia_truoc != 0:

        thay_doi = (
            gia
            / gia_truoc
            - 1
        ) * 100

    else:

        thay_doi = 0.0

    return {
        "price": gia,
        "change_1d": thay_doi,
        "return_1d": float(
            cuoi["Return"]
        ) * 100,
        "rsi": float(
            cuoi["RSI"]
        )
        if pd.notna(cuoi["RSI"])
        else np.nan,
        "macd": float(
            cuoi["MACD"]
        )
        if pd.notna(cuoi["MACD"])
        else np.nan,
        "sma20": float(
            cuoi["SMA20"]
        )
        if pd.notna(cuoi["SMA20"])
        else np.nan,
        "sma50": float(
            cuoi["SMA50"]
        )
        if pd.notna(cuoi["SMA50"])
        else np.nan,
        "volatility20": float(
            cuoi["Volatility20"]
        )
        if pd.notna(cuoi["Volatility20"])
        else np.nan,
        "volume": float(
            cuoi["Volume"]
        )
        if pd.notna(cuoi["Volume"])
        else np.nan,
    }
