from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# KẾT NỐI DNSE
# ============================================================

try:
    from dnse import DNSEClient
except Exception:
    DNSEClient = None


THOI_GIAN_LUU_CO_PHIEU = 300
THOI_GIAN_LUU_VNINDEX = 60
THOI_GIAN_LUU_GIA_MOI = 15


# ============================================================
# CHUẨN HÓA MÃ
# ============================================================

def normalize_symbol(symbol):
    if symbol is None:
        return "HPG"

    ma = str(symbol).strip().upper()
    ma = re.sub(r"\s+", "", ma)

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
# LẤY KHÓA DNSE
# ============================================================

def _lay_cau_hinh_dnse():
    khoa = None
    bi_mat = None

    try:
        khoa = st.secrets.get(
            "DNSE_API_KEY"
        )

        bi_mat = st.secrets.get(
            "DNSE_API_SECRET"
        )
    except Exception:
        pass

    if not khoa or not bi_mat:

        khoa = st.session_state.get(
            "DNSE_API_KEY"
        )

        bi_mat = st.session_state.get(
            "DNSE_API_SECRET"
        )

    if not khoa or not bi_mat:
        return None, None

    return (
        str(khoa).strip(),
        str(bi_mat).strip(),
    )


# ============================================================
# TẠO KHÁCH DNSE
# ============================================================

def _tao_khach_dnse():

    if DNSEClient is None:
        raise RuntimeError(
            "Chưa cài thư viện DNSE OpenAPI."
        )

    khoa, bi_mat = _lay_cau_hinh_dnse()

    if not khoa or not bi_mat:
        raise RuntimeError(
            "Chưa cấu hình DNSE_API_KEY và DNSE_API_SECRET "
            "trong Streamlit Secrets."
        )

    return DNSEClient(
        api_key=khoa,
        api_secret=bi_mat,
        base_url="https://openapi.dnse.com.vn",
        api_version="2026-05-07",
    )


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
# TÌM CỘT
# ============================================================

def _tim_cot(
    du_lieu,
    cac_ten,
):
    if du_lieu is None:
        return None

    if du_lieu.empty:
        return None

    ban_do = {
        str(cot).strip().lower(): cot
        for cot in du_lieu.columns
    }

    for ten in cac_ten:

        cot = ban_do.get(
            str(ten).strip().lower()
        )

        if cot is not None:
            return cot

    return None


# ============================================================
# CHUẨN HÓA DỮ LIỆU DNSE
# ============================================================

def _chuan_hoa_bang_gia(
    du_lieu,
):

    if du_lieu is None:
        raise ValueError(
            "DNSE không trả về dữ liệu."
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
            "DNSE trả về bảng dữ liệu rỗng."
        )

    du_lieu = du_lieu.copy()

    # --------------------------------------------------------
    # Nếu dữ liệu có dạng MultiIndex
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

        if ten in [
            "time",
            "datetime",
            "timestamp",
            "date",
            "tradingdate",
            "trading_date",
        ]:
            anh_xa[cot] = "Time"

        elif ten in [
            "open",
            "open_price",
            "openprice",
        ]:
            anh_xa[cot] = "Open"

        elif ten in [
            "high",
            "high_price",
            "highprice",
        ]:
            anh_xa[cot] = "High"

        elif ten in [
            "low",
            "low_price",
            "lowprice",
        ]:
            anh_xa[cot] = "Low"

        elif ten in [
            "close",
            "close_price",
            "closeprice",
            "last",
        ]:
            anh_xa[cot] = "Close"

        elif ten in [
            "volume",
            "vol",
            "total_volume",
        ]:
            anh_xa[cot] = "Volume"

        elif ten in [
            "value",
            "trading_value",
            "value_traded",
        ]:
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
            utc=True,
        )

        du_lieu = du_lieu.set_index(
            "Time"
        )

    else:

        du_lieu.index = pd.to_datetime(
            du_lieu.index,
            errors="coerce",
            utc=True,
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
    # Bắt buộc OHLCV
    # --------------------------------------------------------

    cot_bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    thieu = [
        cot
        for cot in cot_bat_buoc
        if cot not in du_lieu.columns
    ]

    if thieu:
        raise ValueError(
            "DNSE thiếu cột: "
            + ", ".join(thieu)
            + ". Cột nhận được: "
            + ", ".join(
                map(
                    str,
                    du_lieu.columns,
                )
            )
        )

    # --------------------------------------------------------
    # Ép kiểu số
    # --------------------------------------------------------

    for cot in cot_bat_buoc:

        du_lieu[cot] = pd.to_numeric(
            du_lieu[cot],
            errors="coerce",
        )

    du_lieu = du_lieu.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # Không cho giá <= 0 đi vào hệ thống
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
            "DNSE không có OHLC hợp lệ."
        )

    return du_lieu


# ============================================================
# CHỈ BÁO
# ============================================================

def add_indicators(
    du_lieu,
):

    du_lieu = _chuan_hoa_bang_gia(
        du_lieu
    )

    du_lieu = du_lieu.copy()

    gia = du_lieu["Close"]

    # --------------------------------------------------------
    # Lợi suất
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

    du_lieu["EMA100"] = (
        gia.ewm(
            span=100,
            adjust=False,
        ).mean()
    )

    du_lieu["EMA200"] = (
        gia.ewm(
            span=200,
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
    # Khối lượng
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
    # Khoảng cách với đỉnh / đáy
    # --------------------------------------------------------

    du_lieu["Distance_From_High20"] = (
        (
            gia
            / du_lieu["High20"]
            - 1
        )
        * 100
    )

    du_lieu["Distance_From_Low20"] = (
        (
            gia
            / du_lieu["Low20"]
            - 1
        )
        * 100
    )

    du_lieu["Distance_From_High252"] = (
        (
            gia
            / du_lieu["High252"]
            - 1
        )
        * 100
    )

    du_lieu["Distance_From_Low252"] = (
        (
            gia
            / du_lieu["Low252"]
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
# KHOẢNG THỜI GIAN
# ============================================================

def _so_ngay_theo_ky(
    ky_hieu,
):

    ky_hieu = str(
        ky_hieu or "1y"
    ).strip().lower()

    bang = {
        "1d": 3,
        "5d": 10,
        "1mo": 45,
        "3mo": 120,
        "6mo": 240,
        "1y": 450,
        "2y": 850,
        "3y": 1200,
        "5y": 1900,
        "10y": 3800,
        "max": 3800,
    }

    return bang.get(
        ky_hieu,
        450,
    )


# ============================================================
# CHUYỂN THỜI GIAN SANG UNIX
# ============================================================

def _unix(
    thoi_gian,
):
    if thoi_gian.tzinfo is None:
        thoi_gian = thoi_gian.replace(
            tzinfo=timezone.utc
        )

    return int(
        thoi_gian.timestamp()
    )


# ============================================================
# GIẢI MÃ BODY DNSE
# ============================================================

def _tach_du_lieu_dnse(
    thanh_cong,
    noi_dung,
):

    if int(thanh_cong) >= 400:
        raise RuntimeError(
            f"DNSE HTTP {thanh_cong}: "
            f"{noi_dung}"
        )

    if noi_dung is None:
        return None

    if isinstance(
        noi_dung,
        pd.DataFrame,
    ):
        return noi_dung

    if isinstance(
        noi_dung,
        list,
    ):
        return noi_dung

    if isinstance(
        noi_dung,
        dict,
    ):

        # Các dạng thường gặp
        cac_khoa = [
            "data",
            "content",
            "items",
            "results",
            "candles",
            "ohlc",
        ]

        for khoa in cac_khoa:

            if khoa not in noi_dung:
                continue

            gia_tri = noi_dung[
                khoa
            ]

            if isinstance(
                gia_tri,
                dict,
            ):

                ket_qua = (
                    _tach_du_lieu_dnse(
                        200,
                        gia_tri,
                    )
                )

                if ket_qua is not None:
                    return ket_qua

            elif isinstance(
                gia_tri,
                list,
            ):

                return gia_tri

        return noi_dung

    return noi_dung


# ============================================================
# LẤY OHLC TỪ DNSE
# ============================================================

def _lay_ohlc_dnse(
    ma,
    ky_hieu="1y",
):

    khach = _tao_khach_dnse()

    so_ngay = _so_ngay_theo_ky(
        ky_hieu
    )

    ngay_cuoi = datetime.now(
        timezone.utc
    )

    ngay_dau = (
        ngay_cuoi
        - timedelta(
            days=int(so_ngay)
        )
    )

    thanh_cong, noi_dung = (
        khach.get_ohlc(
            bar_type="STOCK",
            query={
                "symbol": ma,
                "resolution": "1",
                "from": _unix(
                    ngay_dau
                ),
                "to": _unix(
                    ngay_cuoi
                ),
            },
            dry_run=False,
        )
    )

    than = _tach_du_lieu_dnse(
        thanh_cong,
        noi_dung,
    )

    if than is None:
        raise RuntimeError(
            f"DNSE không trả dữ liệu OHLC cho {ma}."
        )

    if isinstance(
        than,
        dict,
    ):
        # Thử lấy phần mảng dữ liệu bên trong
        for khoa in [
            "data",
            "content",
            "items",
            "results",
            "candles",
            "ohlc",
        ]:

            if isinstance(
                than.get(khoa),
                list,
            ):

                than = than[
                    khoa
                ]

                break

    du_lieu = pd.DataFrame(
        than
    )

    if du_lieu.empty:

        raise RuntimeError(
            f"DNSE trả OHLC rỗng cho {ma}."
        )

    return _chuan_hoa_bang_gia(
        du_lieu
    )


# ============================================================
# GIÁ KHỚP GẦN NHẤT DNSE
# ============================================================

def _lay_gia_moi_nhat_dnse(
    ma,
):

    khach = _tao_khach_dnse()

    # DNSE yêu cầu board theo dữ liệu thị trường.
    # Thử G1 trước vì ví dụ SDK chính thức sử dụng G1.
    cac_bang = [
        "G1",
        "G3",
        "G4",
        "G7",
    ]

    loi = []

    for bang in cac_bang:

        try:

            thanh_cong, noi_dung = (
                khach.get_latest_trade(
                    symbol=ma,
                    board_id=bang,
                    dry_run=False,
                )
            )

            if int(
                thanh_cong
            ) >= 400:
                loi.append(
                    f"{bang}: HTTP {thanh_cong}"
                )
                continue

            than = _tach_du_lieu_dnse(
                thanh_cong,
                noi_dung,
            )

            if isinstance(
                than,
                dict,
            ):

                return than

            if isinstance(
                than,
                list,
            ) and than:

                return than[-1]

        except Exception as exc:

            loi.append(
                f"{bang}: {exc}"
            )

    raise RuntimeError(
        "Không lấy được giao dịch gần nhất từ DNSE. "
        + " | ".join(loi)
    )


# ============================================================
# CHUẨN HÓA GIÁ GẦN NHẤT
# ============================================================

def _lay_gia_tu_giao_dich(
    du_lieu,
):

    if du_lieu is None:
        return None

    if isinstance(
        du_lieu,
        list,
    ):

        if not du_lieu:
            return None

        du_lieu = du_lieu[-1]

    if not isinstance(
        du_lieu,
        dict,
    ):
        return None

    ban_do = {
        str(k).strip().lower(): v
        for k, v in du_lieu.items()
    }

    cac_ten_gia = [
        "last",
        "lastprice",
        "last_price",
        "price",
        "matchprice",
        "match_price",
        "tradeprice",
        "trade_price",
        "close",
    ]

    for ten in cac_ten_gia:

        if ten not in ban_do:
            continue

        try:

            gia = float(
                ban_do[ten]
            )

            if (
                np.isfinite(gia)
                and gia > 0
            ):
                return gia

        except Exception:
            continue

    return None


# ============================================================
# DỮ LIỆU CỔ PHIẾU
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

    # Chỉ dùng DNSE cho cổ phiếu Việt Nam.
    du_lieu_goc = _lay_ohlc_dnse(
        ma,
        period,
    )

    du_lieu = add_indicators(
        du_lieu_goc
    )

    if du_lieu.empty:
        raise ValueError(
            f"DNSE không có dữ liệu hợp lệ cho {ma}."
        )

    du_lieu.attrs["symbol"] = ma
    du_lieu.attrs["display_symbol"] = (
        display_symbol(ma)
    )
    du_lieu.attrs["source"] = "DNSE"

    return du_lieu


# ============================================================
# VN-INDEX
# ============================================================
#
# DNSE /price/ohlc hỗ trợ cả chỉ số thị trường.
# Thử các tên thường dùng theo thứ tự.
# Không gọi Yahoo.
# ============================================================

@st.cache_data(
    ttl=THOI_GIAN_LUU_VNINDEX,
    show_spinner=False,
)
def load_vnindex_data():

    khach = _tao_khach_dnse()

    ngay_cuoi = datetime.now(
        timezone.utc
    )

    ngay_dau = (
        ngay_cuoi
        - timedelta(
            days=450
        )
    )

    cac_ma_chi_so = [
        "VNINDEX",
        "VN-INDEX",
        "^VNINDEX",
    ]

    cac_loi = []

    for ma_chi_so in cac_ma_chi_so:

        try:

            thanh_cong, noi_dung = (
                khach.get_ohlc(
                    bar_type="INDEX",
                    query={
                        "symbol": ma_chi_so,
                        "resolution": "1",
                        "from": _unix(
                            ngay_dau
                        ),
                        "to": _unix(
                            ngay_cuoi
                        ),
                    },
                    dry_run=False,
                )
            )

            than = _tach_du_lieu_dnse(
                thanh_cong,
                noi_dung,
            )

            if isinstance(
                than,
                dict,
            ):

                for khoa in [
                    "data",
                    "content",
                    "items",
                    "results",
                    "candles",
                    "ohlc",
                ]:

                    if isinstance(
                        than.get(khoa),
                        list,
                    ):

                        than = than[
                            khoa
                        ]

                        break

            if (
                isinstance(
                    than,
                    list,
                )
                and than
            ):

                du_lieu = pd.DataFrame(
                    than
                )

            else:

                du_lieu = pd.DataFrame(
                    than
                )

            if du_lieu.empty:
                continue

            du_lieu = _chuan_hoa_bang_gia(
                du_lieu
            )

            # Index đôi khi không có volume.
            if "Volume" not in du_lieu.columns:

                du_lieu["Volume"] = 0.0

            du_lieu = add_indicators(
                du_lieu
            )

            if du_lieu.empty:
                continue

            du_lieu.attrs["symbol"] = (
                "VNINDEX"
            )

            du_lieu.attrs["source"] = (
                "DNSE"
            )

            return du_lieu

        except Exception as exc:

            cac_loi.append(
                f"{ma_chi_so}: {exc}"
            )

    raise RuntimeError(
        "Không lấy được VN-INDEX từ DNSE. "
        + " | ".join(cac_loi)
    )


# ============================================================
# GIÁ HIỆN TẠI
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

    try:

        giao_dich = (
            _lay_gia_moi_nhat_dnse(
                ma
            )
        )

        gia_moi = (
            _lay_gia_tu_giao_dich(
                giao_dich
            )
        )

        if gia_moi is not None:

            try:

                du_lieu = load_market_data(
                    ma,
                    "5d",
                )

                if (
                    du_lieu is not None
                    and not du_lieu.empty
                ):

                    dong_cuoi = (
                        du_lieu.iloc[-1]
                    )

                    gia_cu = float(
                        dong_cuoi[
                            "Close"
                        ]
                    )

                    thay_doi = (
                        gia_moi
                        / gia_cu
                        - 1
                    ) * 100

                    return {
                        "ma": display_symbol(
                            ma
                        ),
                        "gia": gia_moi,
                        "thay_doi": thay_doi,
                        "khoi_luong": float(
                            dong_cuoi[
                                "Volume"
                            ]
                        ),
                    }

            except Exception:
                pass

            return {
                "ma": display_symbol(
                    ma
                ),
                "gia": gia_moi,
                "thay_doi": np.nan,
                "khoi_luong": np.nan,
            }

        raise RuntimeError(
            "DNSE không trả giá giao dịch."
        )

    except Exception as exc:

        raise RuntimeError(
            f"Không lấy được giá mới nhất {ma}: {exc}"
        )


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

    dong_cuoi = du_lieu.iloc[-1]

    if len(du_lieu) >= 2:

        dong_truoc = (
            du_lieu.iloc[-2]
        )

    else:

        dong_truoc = (
            dong_cuoi
        )

    gia = float(
        dong_cuoi["Close"]
    )

    gia_truoc = float(
        dong_truoc["Close"]
    )

    if gia_truoc != 0:

        thay_doi = (
            gia
            / gia_truoc
            - 1
        ) * 100

    else:

        thay_doi = np.nan

    def _gia_tri(
        ten_cot,
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
        "return_1d": _gia_tri(
            "ReturnPct"
        ),
        "rsi": _gia_tri(
            "RSI"
        ),
        "macd": _gia_tri(
            "MACD"
        ),
        "sma20": _gia_tri(
            "SMA20"
        ),
        "sma50": _gia_tri(
            "SMA50"
        ),
        "volatility20": _gia_tri(
            "Volatility20"
        ),
        "volume": _gia_tri(
            "Volume"
        ),
    }


# ============================================================
# GIỮ TÊN HÀM CŨ CHO CÁC FILE KHÁC
# ============================================================

def market_data(
    symbol,
    period="1y",
):
    return load_market_data(
        symbol,
        period,
    )


def _safe_int(
    gia_tri,
    mac_dinh,
):
    try:
        return int(
            float(gia_tri)
        )
    except Exception:
        return int(mac_dinh)


def get_news(
    ticker,
    days=7,
):
    return []


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
        "vi phạm",
    ]

    diem_tich_cuc = sum(
        tu
        in van_ban
        for tu
        in tu_tich_cuc
    )

    diem_tieu_cuc = sum(
        tu
        in van_ban
        for tu
        in tu_tieu_cuc
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
# MÔ HÌNH ĐỊNH LƯỢNG
# ============================================================

def build_quant(
    du_lieu,
):

    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return None

    cot_dac_trung = [
        "RSI",
        "MACD",
        "MACD_Hist",
        "Volatility20",
        "Volume_Change",
        "Return",
    ]

    if any(
        cot not in du_lieu.columns
        for cot in cot_dac_trung
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

    lam_viec = du_lieu.copy()

    lam_viec["Target"] = (
        lam_viec["Return"]
        .shift(-1)
    )

    lam_viec = lam_viec.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    lam_viec = lam_viec.dropna(
        subset=(
            cot_dac_trung
            + ["Target"]
        )
    )

    if len(lam_viec) < 60:
        return None

    X = lam_viec[
        cot_dac_trung
    ].astype(float)

    y = lam_viec[
        "Target"
    ].astype(float)

    vi_tri = int(
        len(lam_viec)
        * 0.8
    )

    if vi_tri < 30:
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

        mo_hinh_ols = sm.OLS(
            y_train,
            sm.add_constant(
                X_train,
                has_constant="add",
            ),
        ).fit(
            cov_type="HC3"
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

            r2 = float("nan")

        du_bao_tiep = float(
            mo_hinh_rung.predict(
                X.iloc[[-1]]
            )[0]
        )

        tam_quan_trong = pd.Series(
            mo_hinh_rung.feature_importances_,
            index=cot_dac_trung,
        ).sort_values(
            ascending=False
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


# ============================================================
# OLS TƯƠNG THÍCH
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

        y = sach["Return"]

        return sm.OLS(
            y,
            X,
        ).fit()

    except Exception:
        return None


# ============================================================
# RANDOM FOREST TƯƠNG THÍCH
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
