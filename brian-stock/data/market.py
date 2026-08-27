import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# CHUYỂN ĐỔI MÃ CỔ PHIẾU
# ============================================================

def normalize_symbol(symbol: str) -> str:
    if symbol is None:
        return "HPG.VN"

    ma = str(symbol).strip().upper().replace(" ", "")

    if not ma:
        return "HPG.VN"

    if ma.endswith(".VN"):
        return ma

    # Mã cổ phiếu Việt Nam
    if re.fullmatch(r"[A-Z0-9]{2,5}", ma):
        return ma + ".VN"

    return ma


def display_symbol(symbol: str) -> str:
    if not symbol:
        return ""

    return str(symbol).upper().replace(".VN", "")


# ============================================================
# CHUYỂN KHOẢNG THỜI GIAN THÀNH SỐ NGÀY
# ============================================================

def period_to_days(period) -> int:
    """
    Chuyển các giá trị:
    1mo, 3mo, 6mo, 1y, 2y, 5y
    thành số ngày thực tế để dùng cho timedelta.
    """

    if isinstance(period, (int, float)):
        return max(1, int(period))

    period = str(period).strip().lower()

    bang_ngay = {
        "1d": 1,
        "5d": 5,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 365,
        "2y": 730,
        "3y": 1095,
        "5y": 1825,
        "10y": 3650,
    }

    if period in bang_ngay:
        return bang_ngay[period]

    ket_qua = re.fullmatch(r"(\d+)(d|mo|y)", period)

    if ket_qua:
        so_luong = int(ket_qua.group(1))
        don_vi = ket_qua.group(2)

        if don_vi == "d":
            return so_luong

        if don_vi == "mo":
            return so_luong * 31

        if don_vi == "y":
            return so_luong * 365

    return 365


# ============================================================
# LẤY DỮ LIỆU THỊ TRƯỜNG
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(symbol: str, period="1y") -> pd.DataFrame:

    ma = normalize_symbol(symbol)

    # Tuyệt đối không đưa chuỗi "1y", "3mo"... vào timedelta(days=...)
    so_ngay = period_to_days(period)

    # Thêm vùng đệm để đảm bảo đủ dữ liệu tính chỉ báo
    so_ngay_lay_du_lieu = max(so_ngay + 120, 180)

    ngay_ket_thuc = datetime.now(timezone.utc).date()
    ngay_bat_dau = ngay_ket_thuc - timedelta(days=so_ngay_lay_du_lieu)

    df = None
    loi_cuoi = None

    # ========================================================
    # CÁCH 1: LẤY THEO KHOẢNG NGÀY
    # ========================================================

    try:
        df = yf.download(
            ma,
            start=ngay_bat_dau.strftime("%Y-%m-%d"),
            end=(ngay_ket_thuc + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        loi_cuoi = exc
        df = None

    # ========================================================
    # CÁCH 2: DỰ PHÒNG BẰNG PERIOD CỦA YAHOO
    # ========================================================

    if df is None or df.empty:

        try:
            df = yf.download(
                ma,
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            loi_cuoi = exc
            df = None

    if df is None or df.empty:
        if loi_cuoi:
            raise ValueError(
                f"Không lấy được dữ liệu {display_symbol(ma)}: {loi_cuoi}"
            )

        raise ValueError(
            f"Không tìm thấy dữ liệu cho {display_symbol(ma)}."
        )

    # ========================================================
    # XỬ LÝ CỘT NHIỀU TẦNG
    # ========================================================

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().title() for c in df.columns]

    # ========================================================
    # KIỂM TRA CỘT BẮT BUỘC
    # ========================================================

    cot_bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for cot in cot_bat_buoc:
        if cot not in df.columns:
            raise ValueError(
                f"Dữ liệu {display_symbol(ma)} thiếu cột {cot}."
            )

    # ========================================================
    # CHUẨN HÓA NGÀY
    # ========================================================

    df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].copy()

    # ========================================================
    # ÉP KIỂU SỐ
    # ========================================================

    for cot in cot_bat_buoc:
        df[cot] = pd.to_numeric(
            df[cot],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    ).copy()

    # ========================================================
    # CHỈ GIỮ ĐÚNG KHOẢNG NGƯỜI DÙNG CHỌN
    # ========================================================

    if len(df) > 0:

        ngay_cuoi = df.index.max()

        ngay_gioi_han = ngay_cuoi - pd.Timedelta(
            days=so_ngay
        )

        df = df[
            df.index >= ngay_gioi_han
        ].copy()

    # ========================================================
    # LỢI NHUẬN
    # ========================================================

    df["Return"] = df["Close"].pct_change()

    # ========================================================
    # RSI 14
    # ========================================================

    thay_doi = df["Close"].diff()

    tang = thay_doi.clip(lower=0)
    giam = -thay_doi.clip(upper=0)

    trung_binh_tang = tang.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    trung_binh_giam = giam.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    ti_so = (
        trung_binh_tang /
        trung_binh_giam.replace(0, np.nan)
    )

    df["RSI"] = 100 - (
        100 / (1 + ti_so)
    )

    # ========================================================
    # MACD
    # ========================================================

    trung_binh_mu_12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    trung_binh_mu_26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = (
        trung_binh_mu_12 -
        trung_binh_mu_26
    )

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Hist"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    # ========================================================
    # ĐƯỜNG TRUNG BÌNH
    # ========================================================

    df["SMA20"] = df["Close"].rolling(
        window=20
    ).mean()

    df["SMA50"] = df["Close"].rolling(
        window=50
    ).mean()

    df["SMA200"] = df["Close"].rolling(
        window=200
    ).mean()

    # ========================================================
    # BIẾN ĐỘNG
    # ========================================================

    df["Volatility20"] = (
        df["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    # ========================================================
    # KHỐI LƯỢNG
    # ========================================================

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    df["Volume_SMA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # ========================================================
    # KHOẢNG GIÁ
    # ========================================================

    df["Range"] = (
        df["High"] -
        df["Low"]
    )

    df["Range_Percent"] = (
        df["Range"] /
        df["Close"]
        .replace(0, np.nan)
        * 100
    )

    # ========================================================
    # KHỐI LƯỢNG TƯƠNG ĐỐI
    # ========================================================

    df["Relative_Volume"] = (
        df["Volume"] /
        df["Volume_SMA20"]
        .replace(0, np.nan)
    )

    # ========================================================
    # GIÁ CAO NHẤT / THẤP NHẤT
    # ========================================================

    df["High20"] = (
        df["High"]
        .rolling(20)
        .max()
    )

    df["Low20"] = (
        df["Low"]
        .rolling(20)
        .min()
    )

    df["High50"] = (
        df["High"]
        .rolling(50)
        .max()
    )

    df["Low50"] = (
        df["Low"]
        .rolling(50)
        .min()
    )

    # ========================================================
    # TỶ LỆ VỊ TRÍ GIÁ TRONG BIÊN 20 PHIÊN
    # ========================================================

    bien_20 = (
        df["High20"] -
        df["Low20"]
    ).replace(0, np.nan)

    df["Vi_Tri_Gia_20"] = (
        (df["Close"] - df["Low20"]) /
        bien_20
        * 100
    )

    # ========================================================
    # LOẠI GIÁ TRỊ VÔ HẠN
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Không drop toàn bộ dữ liệu vì SMA200 có thể
    # chưa đủ dữ liệu ở đầu chuỗi.
    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Return",
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "SMA20",
            "SMA50",
            "Volatility20",
            "Volume_Change",
        ]
    ).copy()

    if df.empty:
        raise ValueError(
            f"Dữ liệu {display_symbol(ma)} không đủ để tính các chỉ báo."
        )

    return df.sort_index()


# ============================================================
# THÔNG TIN CÔNG TY
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_company_info(symbol: str) -> dict:

    ma = normalize_symbol(symbol)

    try:
        thong_tin = yf.Ticker(ma).fast_info

        if thong_tin is None:
            return {}

        return dict(thong_tin)

    except Exception:
        return {}


# ============================================================
# GIÁ MỚI NHẤT
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def load_latest_price(symbol: str) -> dict:

    ma = normalize_symbol(symbol)

    try:
        df = yf.download(
            ma,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return {}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [
            str(c).strip().title()
            for c in df.columns
        ]

        df = df.dropna(
            subset=["Close"]
        )

        if df.empty:
            return {}

        gia_moi_nhat = float(
            df["Close"].iloc[-1]
        )

        if len(df) >= 2:
            gia_truoc = float(
                df["Close"].iloc[-2]
            )
        else:
            gia_truoc = gia_moi_nhat

        thay_doi = (
            gia_moi_nhat / gia_truoc - 1
        ) * 100 if gia_truoc else 0.0

        khoi_luong = 0

        if "Volume" in df.columns:
            try:
                khoi_luong = int(
                    float(df["Volume"].iloc[-1])
                )
            except Exception:
                khoi_luong = 0

        return {
            "ma": display_symbol(ma),
            "gia": gia_moi_nhat,
            "thay_doi": thay_doi,
            "khoi_luong": khoi_luong,
            "thoi_gian": df.index[-1],
        }

    except Exception:
        return {}
