from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# Vnstock 4.x
# ============================================================

try:
    from vnstock import Market

    THI_TRUONG_VNSTOCK = Market()

except Exception:
    THI_TRUONG_VNSTOCK = None


# ============================================================
# TIỆN ÍCH
# ============================================================

def normalize_symbol(symbol: str) -> str:

    if symbol is None:
        return "HPG"

    symbol = str(symbol).strip().upper()

    symbol = re.sub(
        r"\s+",
        "",
        symbol,
    )

    if not symbol:
        return "HPG"

    return symbol


def display_symbol(symbol: str) -> str:

    if symbol is None:
        return ""

    symbol = str(symbol).strip().upper()

    if symbol.endswith(".VN"):
        symbol = symbol[:-3]

    return symbol


def _so(value, mac_dinh=np.nan):

    try:
        value = float(value)

        if pd.isna(value):
            return mac_dinh

        return value

    except Exception:
        return mac_dinh


def _tach_ma_viet_nam(symbol):

    symbol = normalize_symbol(symbol)

    if symbol.endswith(".VN"):
        return symbol[:-3]

    return symbol


# ============================================================
# RSI
# ============================================================

def rsi(
    series: pd.Series,
    period: int = 14,
):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    result = 100 - (
        100 / (1 + rs)
    )

    result = result.where(
        ~(
            (avg_loss == 0)
            & (avg_gain > 0)
        ),
        100,
    )

    result = result.where(
        ~(
            (avg_loss == 0)
            & (avg_gain == 0)
        ),
        50,
    )

    return result


# ============================================================
# CHUẨN HÓA OHLCV
# ============================================================

def _chuan_hoa_ohlcv(
    du_lieu,
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

        for cap in range(
            du_lieu.columns.nlevels
        ):

            ten_cot = {
                str(x).strip().lower()
                for x in du_lieu.columns
                .get_level_values(cap)
            }

            if "close" in ten_cot:

                du_lieu.columns = (
                    du_lieu.columns
                    .get_level_values(cap)
                )

                break

    # --------------------------------------------------------
    # Chuẩn hóa tên cột
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

        elif ten in [
            "open",
            "open_price",
        ]:
            anh_xa[cot] = "Open"

        elif ten in [
            "high",
            "high_price",
        ]:
            anh_xa[cot] = "High"

        elif ten in [
            "low",
            "low_price",
        ]:
            anh_xa[cot] = "Low"

        elif ten in [
            "close",
            "close_price",
        ]:
            anh_xa[cot] = "Close"

        elif ten in [
            "volume",
            "total_volume",
        ]:
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
    # Một số nguồn không trả Time mà dùng index
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
    # Kiểm tra OHLCV
    # --------------------------------------------------------

    bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    thieu = [
        cot
        for cot in bat_buoc
        if cot not in du_lieu.columns
    ]

    if thieu:
        raise ValueError(
            "Thiếu cột: "
            + ", ".join(thieu)
        )

    # --------------------------------------------------------
    # Ép kiểu số
    # --------------------------------------------------------

    for cot in bat_buoc:

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

    # --------------------------------------------------------
    # Đơn vị giá
    #
    # Một số nguồn trả 22.05 thay vì 22,050.
    # Cổ phiếu VN thường hiển thị theo đồng.
    # --------------------------------------------------------

    trung_binh_gia = _so(
        du_lieu["Close"].median(),
        np.nan,
    )

    if (
        not np.isnan(trung_binh_gia)
        and trung_binh_gia < 1000
        and trung_binh_gia > 0
    ):

        for cot in [
            "Open",
            "High",
            "Low",
            "Close",
        ]:

            du_lieu[cot] = (
                du_lieu[cot] * 1000
            )

    return du_lieu


# ============================================================
# CHỈ BÁO
# ============================================================

def add_indicators(
    du_lieu,
):

    du_lieu = _chuan_hoa_ohlcv(
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
        gia
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA12"] = (
        gia
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA20"] = (
        gia
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA26"] = (
        gia
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    du_lieu["EMA50"] = (
        gia
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
    # BOLLINGER
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

    # --------------------------------------------------------
    # BIẾN ĐỘNG
    # --------------------------------------------------------

    du_lieu["Volatility20"] = (
        du_lieu["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    du_lieu["Volatility5"] = (
        du_lieu["Return"]
        .rolling(5)
        .std()
        * np.sqrt(252)
        * 100
    )

    # --------------------------------------------------------
    # KHỐI LƯỢNG
    # --------------------------------------------------------

    du_lieu["Volume_SMA20"] = (
        du_lieu["Volume"]
        .rolling(20)
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
    # RANGE
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

    phan_1 = (
        du_lieu["High"]
        - du_lieu["Low"]
    )

    phan_2 = (
        du_lieu["High"]
        - du_lieu["Close"].shift(1)
    ).abs()

    phan_3 = (
        du_lieu["Low"]
        - du_lieu["Close"].shift(1)
    ).abs()

    bien_do_that = pd.concat(
        [
            phan_1,
            phan_2,
            phan_3,
        ],
        axis=1,
    ).max(axis=1)

    du_lieu["ATR14"] = (
        bien_do_that
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # ĐỘNG LƯỢNG
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
    # TỶ LỆ VỊ TRÍ
    # --------------------------------------------------------

    bien_20 = (
        du_lieu["High20"]
        - du_lieu["Low20"]
    ).replace(
        0,
        np.nan,
    )

    du_lieu["Position20"] = (
        (
            gia
            - du_lieu["Low20"]
        )
        / bien_20
        * 100
    )

    # --------------------------------------------------------
    # TÍN HIỆU XU HƯỚNG
    # --------------------------------------------------------

    du_lieu["Above_SMA20"] = (
        gia
        > du_lieu["SMA20"]
    )

    du_lieu["Above_SMA50"] = (
        gia
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

    du_lieu = du_lieu.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return du_lieu


# ============================================================
# LẤY DỮ LIỆU CỔ PHIẾU BẰNG VNSTOCK
# ============================================================

def _lay_co_phieu_vnstock(
    symbol,
    period="1y",
):

    if THI_TRUONG_VNSTOCK is None:
        raise RuntimeError(
            "Không khởi tạo được Vnstock Market."
        )

    ma = _tach_ma_viet_nam(
        symbol
    )

    # Vnstock v4 Unified UI:
    # market.equity.ohlcv(...)
    #
    # Dùng length để tránh tự xử lý timedelta.

    anh_xa_period = {
        "1d": "1D",
        "5d": "5D",
        "1mo": "1M",
        "3mo": "3M",
        "6mo": "6M",
        "1y": "1Y",
        "2y": "2Y",
        "5y": "5Y",
        "10y": "10Y",
        "max": "MAX",
    }

    period = str(
        period
    ).strip().lower()

    length = anh_xa_period.get(
        period,
        "1Y",
    )

    du_lieu = (
        THI_TRUONG_VNSTOCK
        .equity
        .ohlcv(
            symbol=ma,
            length=length,
            interval="1D",
        )
    )

    return _chuan_hoa_ohlcv(
        du_lieu
    )


# ============================================================
# FALLBACK YAHOO
# ============================================================

def _lay_co_phieu_yahoo(
    symbol,
    period="1y",
):

    danh_sach = []

    symbol = normalize_symbol(
        symbol
    )

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
                isinstance(
                    du_lieu,
                    pd.DataFrame,
                )
                and not du_lieu.empty
            ):

                return _chuan_hoa_ohlcv(
                    du_lieu
                )

        except Exception as exc:

            loi.append(
                f"{ticker}: {exc}"
            )

        try:

            du_lieu = yf.download(
                ticker,
                period=str(period),
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

                return _chuan_hoa_ohlcv(
                    du_lieu
                )

        except Exception as exc:

            loi.append(
                f"{ticker}: {exc}"
            )

    raise ValueError(
        "Không lấy được dữ liệu cổ phiếu. "
        + " | ".join(loi)
    )


# ============================================================
# HÀM CHÍNH LẤY DỮ LIỆU CỔ PHIẾU
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_market_data(
    symbol,
    period="1y",
):

    symbol = normalize_symbol(
        symbol
    )

    period = str(
        period
    ).strip().lower()

    # --------------------------------------------------------
    # Ưu tiên Vnstock cho chứng khoán Việt Nam
    # --------------------------------------------------------

    if (
        re.fullmatch(
            r"[A-Z0-9]{2,5}(\.VN)?",
            symbol,
        )
        and THI_TRUONG_VNSTOCK is not None
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
                _tach_ma_viet_nam(
                    symbol
                )
            )

            du_lieu.attrs["source"] = (
                "Vnstock"
            )

            return du_lieu

        except Exception:
            pass

    # --------------------------------------------------------
    # Fallback Yahoo
    # --------------------------------------------------------

    du_lieu = _lay_co_phieu_yahoo(
        symbol,
        period,
    )

    du_lieu = add_indicators(
        du_lieu
    )

    du_lieu.attrs["symbol"] = symbol
    du_lieu.attrs["source"] = (
        "Yahoo Finance"
    )

    return du_lieu


# ============================================================
# VN-INDEX
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_vnindex_data():

    if THI_TRUONG_VNSTOCK is None:

        raise RuntimeError(
            "Không khởi tạo được Vnstock Market."
        )

    loi = []

    # ========================================================
    # Vnstock v4
    # ========================================================

    try:

        du_lieu = (
            THI_TRUONG_VNSTOCK
            .index(
                "VNINDEX"
            )
            .ohlcv(
                length="1Y",
                interval="1D",
            )
        )

        if (
            du_lieu is not None
            and not du_lieu.empty
        ):

            du_lieu = _chuan_hoa_ohlcv(
                du_lieu
            )

            du_lieu = add_indicators(
                du_lieu
            )

            du_lieu.attrs["symbol"] = (
                "VNINDEX"
            )

            du_lieu.attrs["source"] = (
                "Vnstock"
            )

            return du_lieu

    except Exception as exc:

        loi.append(
            f"Vnstock index: {exc}"
        )

    # ========================================================
    # Fallback: API Quote cũ / index quote
    # ========================================================

    try:

        from vnstock.api.quote import (
            Quote,
        )

        # Không dùng cho VNINDEX nếu nguồn hiện tại
        # không xác nhận hỗ trợ; chỉ thử nếu khả dụng.

        quote = Quote(
            symbol="VNINDEX",
            source="VCI",
        )

        du_lieu = quote.history(
            length="1Y",
            interval="1D",
        )

        if (
            du_lieu is not None
            and not du_lieu.empty
        ):

            du_lieu = _chuan_hoa_ohlcv(
                du_lieu
            )

            du_lieu = add_indicators(
                du_lieu
            )

            du_lieu.attrs["symbol"] = (
                "VNINDEX"
            )

            du_lieu.attrs["source"] = (
                "Vnstock Quote"
            )

            return du_lieu

    except Exception as exc:

        loi.append(
            f"Vnstock Quote: {exc}"
        )

    raise ValueError(
        "Không lấy được VN-INDEX. "
        + " | ".join(loi)
    )


# ============================================================
# GIÁ MỚI NHẤT
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_latest_price(
    symbol,
):

    try:

        du_lieu = load_market_data(
            symbol,
            "5d",
        )

        if du_lieu.empty:
            return {}

        gia = _so(
            du_lieu["Close"].iloc[-1],
            np.nan,
        )

        if len(du_lieu) >= 2:

            gia_truoc = _so(
                du_lieu["Close"].iloc[-2],
                np.nan,
            )

        else:
            gia_truoc = gia

        if (
            not np.isnan(gia)
            and not np.isnan(gia_truoc)
            and gia_truoc != 0
        ):

            thay_doi = (
                gia
                / gia_truoc
                - 1
            ) * 100

        else:

            thay_doi = np.nan

        khoi_luong = _so(
            du_lieu["Volume"].iloc[-1],
            0,
        )

        return {
            "ma": display_symbol(
                symbol
            ),
            "gia": gia,
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

    if du_lieu is None or du_lieu.empty:
        return {}

    cuoi = du_lieu.iloc[-1]

    if len(du_lieu) >= 2:
        truoc = du_lieu.iloc[-2]
    else:
        truoc = cuoi

    gia = _so(
        cuoi.get(
            "Close"
        ),
        np.nan,
    )

    gia_truoc = _so(
        truoc.get(
            "Close"
        ),
        np.nan,
    )

    if (
        not np.isnan(gia)
        and not np.isnan(gia_truoc)
        and gia_truoc != 0
    ):

        thay_doi = (
            gia
            / gia_truoc
            - 1
        ) * 100

    else:

        thay_doi = np.nan

    return {
        "price": gia,
        "change_1d": thay_doi,
        "return_1d": (
            _so(
                cuoi.get(
                    "Return"
                ),
                np.nan,
            )
            * 100
        ),
        "rsi": _so(
            cuoi.get(
                "RSI"
            ),
            np.nan,
        ),
        "macd": _so(
            cuoi.get(
                "MACD"
            ),
            np.nan,
        ),
        "sma20": _so(
            cuoi.get(
                "SMA20"
            ),
            np.nan,
        ),
        "sma50": _so(
            cuoi.get(
                "SMA50"
            ),
            np.nan,
        ),
        "volatility20": _so(
            cuoi.get(
                "Volatility20"
            ),
            np.nan,
        ),
        "volume": _so(
            cuoi.get(
                "Volume"
            ),
            np.nan,
        ),
    }
