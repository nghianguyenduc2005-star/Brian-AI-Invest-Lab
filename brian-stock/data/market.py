# ============================================================
# BRIAN STOCK - DATA MARKET
# FILE GỐC DUY NHẤT CHO DỮ LIỆU THỊ TRƯỜNG
# ============================================================

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# KẾT NỐI VNSTOCK
# ============================================================

try:
    from vnstock.api.quote import Quote
except Exception:
    try:
        from vnstock import Quote
    except Exception:
        Quote = None


try:
    from vnstock.api.listing import Listing
except Exception:
    try:
        from vnstock import Listing
    except Exception:
        Listing = None


try:
    from vnstock.api.trading import Trading
except Exception:
    try:
        from vnstock import Trading
    except Exception:
        Trading = None


try:
    from vnstock.api.company import Company
except Exception:
    try:
        from vnstock import Company
    except Exception:
        Company = None


try:
    from vnstock.api.financial import Finance
except Exception:
    try:
        from vnstock import Finance
    except Exception:
        Finance = None


# ============================================================
# CẤU HÌNH
# ============================================================

NGUON_UU_TIEN = ["KBS", "VCI"]

KHOANG_THOI_GIAN = {
    "3 tháng": 90,
    "6 tháng": 180,
    "1 năm": 365,
    "2 năm": 730,
    "3 năm": 1095,
    "5 năm": 1825,
    "10 năm": 3650,
}


# ============================================================
# CHUẨN HÓA MÃ
# ============================================================

def chuan_hoa_ma(ma):
    if ma is None:
        return ""

    ma = str(ma).strip().upper()
    ma = ma.replace(" ", "")
    ma = ma.replace(".VN", "")

    return ma


def hien_thi_ma(ma):
    return chuan_hoa_ma(ma)


# ============================================================
# KIỂM TRA MÃ
# KHÔNG CẦN DANH SÁCH MÃ CỨNG
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def lay_danh_sach_ma():
    """
    Tự động lấy toàn bộ mã cổ phiếu đang có trong nguồn dữ liệu.

    Không viết tay:
    HPG, FPT, VCB...
    """

    if Listing is None:
        raise ImportError(
            "Không thể nạp mô-đun Listing của Vnstock."
        )

    loi_cuoi = None

    for nguon in NGUON_UU_TIEN:

        try:
            bo_danh_sach = Listing(
                source=nguon,
                show_log=False,
            )

            du_lieu = bo_danh_sach.all_symbols()

            if du_lieu is None:
                continue

            du_lieu = pd.DataFrame(du_lieu)

            if du_lieu.empty:
                continue

            cot_ma = None

            for ten_cot in du_lieu.columns:
                if str(ten_cot).lower() == "symbol":
                    cot_ma = ten_cot
                    break

            if cot_ma is None:
                continue

            du_lieu[cot_ma] = (
                du_lieu[cot_ma]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            du_lieu = du_lieu[
                du_lieu[cot_ma].str.fullmatch(
                    r"[A-Z0-9]{2,10}",
                    na=False,
                )
            ].copy()

            du_lieu = du_lieu.drop_duplicates(
                subset=[cot_ma]
            )

            du_lieu = du_lieu.sort_values(
                by=cot_ma
            ).reset_index(drop=True)

            return du_lieu

        except Exception as loi:
            loi_cuoi = loi
            continue

    raise RuntimeError(
        f"Không lấy được danh sách mã từ Vnstock: {loi_cuoi}"
    )


# ============================================================
# CHỈ LẤY MÃ
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def lay_toan_bo_ma():
    du_lieu = lay_danh_sach_ma()

    cot_ma = next(
        (
            cot
            for cot in du_lieu.columns
            if str(cot).lower() == "symbol"
        ),
        None,
    )

    if cot_ma is None:
        return []

    return (
        du_lieu[cot_ma]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# TÌM MÃ
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def tim_ma(thu_tu_tim_kiem):
    thu_tu_tim_kiem = str(
        thu_tu_tim_kiem or ""
    ).strip().upper()

    if not thu_tu_tim_kiem:
        return pd.DataFrame()

    du_lieu = lay_danh_sach_ma()

    ket_qua = du_lieu.copy()

    cot_ma = next(
        (
            cot
            for cot in ket_qua.columns
            if str(cot).lower() == "symbol"
        ),
        None,
    )

    if cot_ma is None:
        return pd.DataFrame()

    dieu_kien = (
        ket_qua[cot_ma]
        .astype(str)
        .str.upper()
        .str.contains(
            thu_tu_tim_kiem,
            regex=False,
            na=False,
        )
    )

    return ket_qua[dieu_kien].reset_index(
        drop=True
    )


# ============================================================
# KIỂM TRA MÃ CÓ THẬT HAY KHÔNG
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def ma_co_ton_tai(ma):
    ma = chuan_hoa_ma(ma)

    if not ma:
        return False

    try:
        danh_sach = set(lay_toan_bo_ma())
        return ma in danh_sach
    except Exception:
        return False


# ============================================================
# TÍNH RSI
# ============================================================

def tinh_rsi(gia_dong_cua, chu_ky=14):

    thay_doi = gia_dong_cua.diff()

    tang = thay_doi.clip(lower=0)

    giam = -thay_doi.clip(upper=0)

    trung_binh_tang = tang.ewm(
        alpha=1 / chu_ky,
        adjust=False,
    ).mean()

    trung_binh_giam = giam.ewm(
        alpha=1 / chu_ky,
        adjust=False,
    ).mean()

    ti_so = (
        trung_binh_tang
        / trung_binh_giam.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + ti_so)
    )


# ============================================================
# THÊM TOÀN BỘ CHỈ BÁO KỸ THUẬT CƠ BẢN
# ============================================================

def them_chi_bao(du_lieu):

    du_lieu = du_lieu.copy()

    # --------------------------------------------------------
    # GIÁ
    # --------------------------------------------------------

    if "Close" not in du_lieu.columns:
        raise ValueError(
            "Dữ liệu không có cột giá đóng cửa."
        )

    du_lieu = du_lieu.dropna(
        subset=["Close"]
    ).copy()

    gia = pd.to_numeric(
        du_lieu["Close"],
        errors="coerce",
    )

    cao_nhat = pd.to_numeric(
        du_lieu["High"],
        errors="coerce",
    )

    thap_nhat = pd.to_numeric(
        du_lieu["Low"],
        errors="coerce",
    )

    khoi_luong = pd.to_numeric(
        du_lieu["Volume"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # LỢI SUẤT
    # --------------------------------------------------------

    du_lieu["Return"] = gia.pct_change()

    du_lieu["Return_1D"] = gia.pct_change(1)

    du_lieu["Return_3D"] = gia.pct_change(3)

    du_lieu["Return_5D"] = gia.pct_change(5)

    du_lieu["Return_10D"] = gia.pct_change(10)

    du_lieu["Return_20D"] = gia.pct_change(20)

    du_lieu["Return_60D"] = gia.pct_change(60)

    du_lieu["Return_120D"] = gia.pct_change(120)

    du_lieu["Return_252D"] = gia.pct_change(252)

    # --------------------------------------------------------
    # TRUNG BÌNH ĐỘNG
    # --------------------------------------------------------

    du_lieu["SMA5"] = gia.rolling(5).mean()

    du_lieu["SMA10"] = gia.rolling(10).mean()

    du_lieu["SMA20"] = gia.rolling(20).mean()

    du_lieu["SMA30"] = gia.rolling(30).mean()

    du_lieu["SMA50"] = gia.rolling(50).mean()

    du_lieu["SMA100"] = gia.rolling(100).mean()

    du_lieu["SMA150"] = gia.rolling(150).mean()

    du_lieu["SMA200"] = gia.rolling(200).mean()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    du_lieu["EMA5"] = gia.ewm(
        span=5,
        adjust=False,
    ).mean()

    du_lieu["EMA10"] = gia.ewm(
        span=10,
        adjust=False,
    ).mean()

    du_lieu["EMA12"] = gia.ewm(
        span=12,
        adjust=False,
    ).mean()

    du_lieu["EMA20"] = gia.ewm(
        span=20,
        adjust=False,
    ).mean()

    du_lieu["EMA26"] = gia.ewm(
        span=26,
        adjust=False,
    ).mean()

    du_lieu["EMA50"] = gia.ewm(
        span=50,
        adjust=False,
    ).mean()

    du_lieu["EMA100"] = gia.ewm(
        span=100,
        adjust=False,
    ).mean()

    du_lieu["EMA200"] = gia.ewm(
        span=200,
        adjust=False,
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    du_lieu["RSI"] = tinh_rsi(
        gia,
        14,
    )

    du_lieu["RSI_7"] = tinh_rsi(
        gia,
        7,
    )

    du_lieu["RSI_21"] = tinh_rsi(
        gia,
        21,
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
    # BIẾN ĐỘNG
    # --------------------------------------------------------

    du_lieu["Volatility5"] = (
        du_lieu["Return"]
        .rolling(5)
        .std()
        * np.sqrt(252)
        * 100
    )

    du_lieu["Volatility10"] = (
        du_lieu["Return"]
        .rolling(10)
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

    # --------------------------------------------------------
    # DẢI BOLLINGER
    # --------------------------------------------------------

    trung_binh_bollinger = (
        gia.rolling(20).mean()
    )

    do_lech_bollinger = (
        gia.rolling(20).std()
    )

    du_lieu["Bollinger_Giua"] = (
        trung_binh_bollinger
    )

    du_lieu["Bollinger_Tren"] = (
        trung_binh_bollinger
        + 2 * do_lech_bollinger
    )

    du_lieu["Bollinger_Duoi"] = (
        trung_binh_bollinger
        - 2 * do_lech_bollinger
    )

    du_lieu["Bollinger_Rong"] = (
        (
            du_lieu["Bollinger_Tren"]
            - du_lieu["Bollinger_Duoi"]
        )
        / trung_binh_bollinger
    )

    du_lieu["Bollinger_Vi_Tri"] = (
        (
            gia
            - du_lieu["Bollinger_Duoi"]
        )
        / (
            du_lieu["Bollinger_Tren"]
            - du_lieu["Bollinger_Duoi"]
        )
    )

    # --------------------------------------------------------
    # TRUE RANGE
    # --------------------------------------------------------

    bien_do_1 = (
        cao_nhat - thap_nhat
    )

    bien_do_2 = (
        cao_nhat
        - gia.shift(1)
    ).abs()

    bien_do_3 = (
        thap_nhat
        - gia.shift(1)
    ).abs()

    true_range = pd.concat(
        [
            bien_do_1,
            bien_do_2,
            bien_do_3,
        ],
        axis=1,
    ).max(axis=1)

    du_lieu["True_Range"] = true_range

    du_lieu["ATR14"] = (
        true_range
        .rolling(14)
        .mean()
    )

    du_lieu["ATR20"] = (
        true_range
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # KHỐI LƯỢNG
    # --------------------------------------------------------

    du_lieu["Volume_SMA5"] = (
        khoi_luong
        .rolling(5)
        .mean()
    )

    du_lieu["Volume_SMA10"] = (
        khoi_luong
        .rolling(10)
        .mean()
    )

    du_lieu["Volume_SMA20"] = (
        khoi_luong
        .rolling(20)
        .mean()
    )

    du_lieu["Volume_SMA50"] = (
        khoi_luong
        .rolling(50)
        .mean()
    )

    du_lieu["Volume_Change"] = (
        khoi_luong.pct_change()
    )

    du_lieu["Volume_Ratio"] = (
        khoi_luong
        / du_lieu["Volume_SMA20"]
    )

    # --------------------------------------------------------
    # DÒNG TIỀN ĐƠN GIẢN
    # --------------------------------------------------------

    du_lieu["Money_Flow"] = (
        gia * khoi_luong
    )

    du_lieu["Money_Flow_SMA20"] = (
        du_lieu["Money_Flow"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # VỊ TRÍ GIÁ TRONG BIÊN ĐỘ
    # --------------------------------------------------------

    du_lieu["Cao_Nhat_20"] = (
        cao_nhat
        .rolling(20)
        .max()
    )

    du_lieu["Thap_Nhat_20"] = (
        thap_nhat
        .rolling(20)
        .min()
    )

    du_lieu["Cao_Nhat_50"] = (
        cao_nhat
        .rolling(50)
        .max()
    )

    du_lieu["Thap_Nhat_50"] = (
        thap_nhat
        .rolling(50)
        .min()
    )

    du_lieu["Cao_Nhat_200"] = (
        cao_nhat
        .rolling(200)
        .max()
    )

    du_lieu["Thap_Nhat_200"] = (
        thap_nhat
        .rolling(200)
        .min()
    )

    du_lieu["Vi_Tri_20"] = (
        (
            gia
            - du_lieu["Thap_Nhat_20"]
        )
        / (
            du_lieu["Cao_Nhat_20"]
            - du_lieu["Thap_Nhat_20"]
        )
    )

    du_lieu["Vi_Tri_52Tuan"] = (
        (
            gia
            - du_lieu["Thap_Nhat_200"]
        )
        / (
            du_lieu["Cao_Nhat_200"]
            - du_lieu["Thap_Nhat_200"]
        )
    )

    # --------------------------------------------------------
    # ĐỘNG LƯỢNG
    # --------------------------------------------------------

    du_lieu["Khoang_Cach_SMA20"] = (
        gia / du_lieu["SMA20"] - 1
    ) * 100

    du_lieu["Khoang_Cach_SMA50"] = (
        gia / du_lieu["SMA50"] - 1
    ) * 100

    du_lieu["Khoang_Cach_SMA200"] = (
        gia / du_lieu["SMA200"] - 1
    ) * 100

    du_lieu["Khoang_Cach_EMA20"] = (
        gia / du_lieu["EMA20"] - 1
    ) * 100

    du_lieu["Khoang_Cach_EMA50"] = (
        gia / du_lieu["EMA50"] - 1
    ) * 100

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    dinh_tich_luy = gia.cummax()

    du_lieu["Dinh_Tich_Luy"] = (
        dinh_tich_luy
    )

    du_lieu["Drawdown"] = (
        gia / dinh_tich_luy - 1
    ) * 100

    # --------------------------------------------------------
    # SỨC MẠNH NẾN
    # --------------------------------------------------------

    du_lieu["Bien_Do_Nen"] = (
        cao_nhat - thap_nhat
    )

    du_lieu["Than_Nen"] = (
        gia - du_lieu["Open"]
    )

    du_lieu["Ty_Le_Than_Nen"] = (
        (
            gia
            - du_lieu["Open"]
        ).abs()
        / (
            cao_nhat - thap_nhat
        ).replace(0, np.nan)
    )

    # --------------------------------------------------------
    # SẮP XẾP THỜI GIAN
    # --------------------------------------------------------

    du_lieu = du_lieu.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return du_lieu


# ============================================================
# LẤY DỮ LIỆU LỊCH SỬ
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def tai_lich_su(
    ma,
    so_ngay=365,
    khung="1D",
):

    ma = chuan_hoa_ma(ma)

    if not ma:
        raise ValueError(
            "Chưa nhập mã cổ phiếu."
        )

    if Quote is None:
        raise ImportError(
            "Không thể nạp Quote của Vnstock."
        )

    ngay_ket_thuc = datetime.now().date()

    ngay_bat_dau = (
        ngay_ket_thuc
        - timedelta(days=so_ngay)
    )

    loi_cuoi = None

    for nguon in NGUON_UU_TIEN:

        try:

            bo_gia = Quote(
                symbol=ma,
                source=nguon,
                show_log=False,
            )

            du_lieu = bo_gia.history(
                start=str(ngay_bat_dau),
                end=str(
                    ngay_ket_thuc
                    + timedelta(days=1)
                ),
                interval=khung,
            )

            if du_lieu is None:
                continue

            du_lieu = pd.DataFrame(
                du_lieu
            )

            if du_lieu.empty:
                continue

            # Chuẩn hóa tên cột
            anh_xa = {}

            for cot in du_lieu.columns:

                ten = str(cot).strip().lower()

                if ten in ["time", "date", "datetime"]:
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

            du_lieu = du_lieu.rename(
                columns=anh_xa
            )

            cot_bat_buoc = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            if not all(
                cot in du_lieu.columns
                for cot in cot_bat_buoc
            ):
                continue

            if "Time" in du_lieu.columns:

                du_lieu["Time"] = pd.to_datetime(
                    du_lieu["Time"],
                    errors="coerce",
                )

                du_lieu = du_lieu.set_index(
                    "Time"
                )

            elif not isinstance(
                du_lieu.index,
                pd.DatetimeIndex,
            ):

                du_lieu.index = pd.to_datetime(
                    du_lieu.index,
                    errors="coerce",
                )

            for cot in cot_bat_buoc:

                du_lieu[cot] = pd.to_numeric(
                    du_lieu[cot],
                    errors="coerce",
                )

            du_lieu = du_lieu.dropna(
                subset=["Close"]
            )

            du_lieu = du_lieu[
                ~du_lieu.index.isna()
            ]

            du_lieu = du_lieu.sort_index()

            du_lieu = them_chi_bao(
                du_lieu
            )

            if du_lieu.empty:
                continue

            du_lieu.attrs["ma"] = ma
            du_lieu.attrs["nguon"] = nguon

            return du_lieu

        except Exception as loi:

            loi_cuoi = loi
            continue

    raise RuntimeError(
        f"Không lấy được dữ liệu {ma}. "
        f"Nguồn cuối cùng báo lỗi: {loi_cuoi}"
    )


# ============================================================
# DỮ LIỆU HIỆN TẠI
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def lay_gia_hien_tai(ma):

    ma = chuan_hoa_ma(ma)

    if not ma:
        raise ValueError(
            "Mã cổ phiếu không hợp lệ."
        )

    if Trading is not None:

        for nguon in NGUON_UU_TIEN:

            try:

                bang_gia = Trading(
                    source=nguon,
                    show_log=False,
                )

                du_lieu = bang_gia.price_board(
                    [ma]
                )

                if du_lieu is not None:

                    du_lieu = pd.DataFrame(
                        du_lieu
                    )

                    if not du_lieu.empty:

                        du_lieu.attrs[
                            "nguon"
                        ] = nguon

                        return du_lieu

            except Exception:
                pass

    # --------------------------------------------------------
    # DỰ PHÒNG BẰNG DỮ LIỆU LỊCH SỬ GẦN NHẤT
    # --------------------------------------------------------

    du_lieu = tai_lich_su(
        ma,
        so_ngay=10,
        khung="1D",
    )

    dong_cuoi = du_lieu.iloc[-1]

    return pd.DataFrame(
        [
            {
                "symbol": ma,
                "price": dong_cuoi["Close"],
                "volume": dong_cuoi["Volume"],
                "change": (
                    dong_cuoi["Return"]
                    * 100
                    if pd.notna(
                        dong_cuoi["Return"]
                    )
                    else np.nan
                ),
            }
        ]
    )


# ============================================================
# TỔNG HỢP TOÀN BỘ CHỈ SỐ PHÂN TÍCH
# ============================================================

def tong_hop_chi_so(du_lieu):

    if du_lieu is None or du_lieu.empty:
        return {}

    dong_cuoi = du_lieu.iloc[-1]

    def lay(ten):
        gia_tri = dong_cuoi.get(
            ten,
            np.nan,
        )

        try:
            return float(gia_tri)
        except Exception:
            return np.nan

    return {

        "gia": lay("Close"),

        "gia_mo": lay("Open"),

        "gia_cao": lay("High"),

        "gia_thap": lay("Low"),

        "khoi_luong": lay("Volume"),

        "thay_doi_1_ngay":
            lay("Return_1D") * 100,

        "thay_doi_5_ngay":
            lay("Return_5D") * 100,

        "thay_doi_20_ngay":
            lay("Return_20D") * 100,

        "thay_doi_60_ngay":
            lay("Return_60D") * 100,

        "thay_doi_1_nam":
            lay("Return_252D") * 100,

        "rsi":
            lay("RSI"),

        "rsi_7":
            lay("RSI_7"),

        "rsi_21":
            lay("RSI_21"),

        "macd":
            lay("MACD"),

        "macd_tin_hieu":
            lay("MACD_Signal"),

        "macd_lich_su":
            lay("MACD_Hist"),

        "sma5":
            lay("SMA5"),

        "sma10":
            lay("SMA10"),

        "sma20":
            lay("SMA20"),

        "sma50":
            lay("SMA50"),

        "sma100":
            lay("SMA100"),

        "sma200":
            lay("SMA200"),

        "ema20":
            lay("EMA20"),

        "ema50":
            lay("EMA50"),

        "ema200":
            lay("EMA200"),

        "bien_dong_5":
            lay("Volatility5"),

        "bien_dong_20":
            lay("Volatility20"),

        "bien_dong_60":
            lay("Volatility60"),

        "atr14":
            lay("ATR14"),

        "atr20":
            lay("ATR20"),

        "bollinger_tren":
            lay("Bollinger_Tren"),

        "bollinger_giua":
            lay("Bollinger_Giua"),

        "bollinger_duoi":
            lay("Bollinger_Duoi"),

        "bollinger_rong":
            lay("Bollinger_Rong"),

        "vi_tri_bollinger":
            lay("Bollinger_Vi_Tri"),

        "ty_le_khoi_luong":
            lay("Volume_Ratio"),

        "dong_tien":
            lay("Money_Flow"),

        "drawdown":
            lay("Drawdown"),

        "vi_tri_20":
            lay("Vi_Tri_20"),

        "vi_tri_52_tuan":
            lay("Vi_Tri_52Tuan"),

        "khoang_cach_sma20":
            lay("Khoang_Cach_SMA20"),

        "khoang_cach_sma50":
            lay("Khoang_Cach_SMA50"),

        "khoang_cach_sma200":
            lay("Khoang_Cach_SMA200"),

        "khoang_cach_ema20":
            lay("Khoang_Cach_EMA20"),

        "khoang_cach_ema50":
            lay("Khoang_Cach_EMA50"),

    }


# ============================================================
# THÔNG TIN DOANH NGHIỆP
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def lay_thong_tin_doanh_nghiep(ma):

    ma = chuan_hoa_ma(ma)

    if Company is None:
        return pd.DataFrame()

    for nguon in NGUON_UU_TIEN:

        try:

            cong_ty = Company(
                symbol=ma,
                source=nguon,
                show_log=False,
            )

            if hasattr(cong_ty, "overview"):
                du_lieu = cong_ty.overview()
            elif hasattr(cong_ty, "info"):
                du_lieu = cong_ty.info()
            else:
                continue

            if du_lieu is None:
                continue

            du_lieu = pd.DataFrame(
                du_lieu
            )

            if not du_lieu.empty:
                return du_lieu

        except Exception:
            continue

    return pd.DataFrame()


# ============================================================
# BÁO CÁO TÀI CHÍNH
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def lay_bao_cao_tai_chinh(
    ma,
    loai_bao_cao="ratio",
    ky="year",
):

    ma = chuan_hoa_ma(ma)

    if Finance is None:
        return pd.DataFrame()

    for nguon in NGUON_UU_TIEN:

        try:

            tai_chinh = Finance(
                symbol=ma,
                source=nguon,
                show_log=False,
            )

            if loai_bao_cao == "ratio":

                du_lieu = tai_chinh.ratio(
                    period=ky,
                    lang="vi",
                )

            elif loai_bao_cao == "balance_sheet":

                du_lieu = tai_chinh.balance_sheet(
                    period=ky,
                    lang="vi",
                )

            elif loai_bao_cao == "income_statement":

                du_lieu = tai_chinh.income_statement(
                    period=ky,
                    lang="vi",
                )

            elif loai_bao_cao == "cash_flow":

                du_lieu = tai_chinh.cash_flow(
                    period=ky,
                )

            else:
                return pd.DataFrame()

            if du_lieu is None:
                continue

            du_lieu = pd.DataFrame(
                du_lieu
            )

            if not du_lieu.empty:
                return du_lieu

        except Exception:
            continue

    return pd.DataFrame()


# ============================================================
# HÀM CŨ - GIỮ TƯƠNG THÍCH VỚI CÁC FILE HIỆN TẠI
# ============================================================

def normalize_symbol(symbol):
    return chuan_hoa_ma(symbol)


def display_symbol(symbol):
    return hien_thi_ma(symbol)


def load_market_data(
    symbol,
    period="1y",
):

    so_ngay = KHOANG_THOI_GIAN.get(
        period,
        365,
    )

    return tai_lich_su(
        symbol,
        so_ngay=so_ngay,
        khung="1D",
    )


def load_company_info(symbol):
    du_lieu = lay_thong_tin_doanh_nghiep(
        symbol
    )

    if du_lieu is None:
        return {}

    if isinstance(
        du_lieu,
        pd.DataFrame,
    ):

        if du_lieu.empty:
            return {}

        return du_lieu.iloc[
            0
        ].to_dict()

    if isinstance(
        du_lieu,
        dict,
    ):
        return du_lieu

    return {}


# ============================================================
# KIỂM TRA NHANH FILE
# ============================================================

def kiem_tra_ket_noi():

    ket_qua = {
        "vnstock": Quote is not None,
        "danh_sach_ma": Listing is not None,
        "bang_gia": Trading is not None,
        "doanh_nghiep": Company is not None,
        "tai_chinh": Finance is not None,
    }

    return ket_qua
