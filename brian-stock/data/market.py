from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CẤU HÌNH DNSE
# ============================================================

ĐỊA_CHỈ_DNSE = "https://api.dnse.com.vn/market-data/v1"

THỜI_GIAN_CHỜ = 20

ĐỘ_PHÂN_GIẢI_NGÀY = "1D"

SỐ_NGÀY_MẶC_ĐỊNH = {
    "1 tháng": 31,
    "3 tháng": 93,
    "6 tháng": 186,
    "1 năm": 366,
    "2 năm": 731,
    "3 năm": 1096,
    "5 năm": 1826,
}


# ============================================================
# CHUẨN HÓA MÃ
# ============================================================

def chuẩn_hóa_mã(mã: str) -> str:
    """
    Chuẩn hóa mã cổ phiếu theo định dạng DNSE.

    Ví dụ:
        HPG      -> HPG
        hpg      -> HPG
        HPG.VN   -> HPG
        " HPG "  -> HPG
    """

    if mã is None:
        return ""

    mã_sạch = str(mã).strip().upper()

    mã_sạch = mã_sạch.replace(" ", "")
    mã_sạch = mã_sạch.replace(".VN", "")

    return mã_sạch


def hiển_thị_mã(mã: str) -> str:
    """
    Trả về mã sạch để hiển thị trên giao diện.
    """

    return chuẩn_hóa_mã(mã)


# ============================================================
# CHUYỂN THỜI GIAN
# ============================================================

def chuyển_sang_dấu_thời_gian(ngày_giờ: datetime) -> int:
    """
    Chuyển thời gian sang Unix timestamp tính bằng giây.
    """

    if ngày_giờ.tzinfo is None:
        ngày_giờ = ngày_giờ.replace(tzinfo=timezone.utc)

    return int(ngày_giờ.timestamp())


def xác_định_khoảng_thời_gian(khoảng_dữ_liệu: str) -> tuple[int, int]:
    """
    Xác định thời gian bắt đầu và kết thúc.
    """

    số_ngày = SỐ_NGÀY_MẶC_ĐỊNH.get(
        khoảng_dữ_liệu,
        366,
    )

    thời_gian_kết_thúc = datetime.now(timezone.utc)
    thời_gian_bắt_đầu = (
        thời_gian_kết_thúc
        - timedelta(days=số_ngày)
    )

    return (
        chuyển_sang_dấu_thời_gian(thời_gian_bắt_đầu),
        chuyển_sang_dấu_thời_gian(thời_gian_kết_thúc),
    )


# ============================================================
# GỌI DNSE
# ============================================================

def gọi_dnse(
    mã: str,
    độ_phân_giải: str = "1D",
    dấu_thời_gian_bắt_đầu: int | None = None,
    dấu_thời_gian_kết_thúc: int | None = None,
) -> Any:
    """
    Gọi API OHLC của DNSE.

    Không sử dụng vnstock.
    Không sử dụng Yahoo Finance.
    Không tạo dữ liệu giả.
    """

    mã_sạch = chuẩn_hóa_mã(mã)

    if not mã_sạch:
        raise ValueError("Mã cổ phiếu đang trống.")

    if dấu_thời_gian_bắt_đầu is None:
        dấu_thời_gian_bắt_đầu = chuyển_sang_dấu_thời_gian(
            datetime.now(timezone.utc) - timedelta(days=366)
        )

    if dấu_thời_gian_kết_thúc is None:
        dấu_thời_gian_kết_thúc = chuyển_sang_dấu_thời_gian(
            datetime.now(timezone.utc)
        )

    địa_chỉ = f"{ĐỊA_CHỈ_DNSE}/price/ohlc"

    tham_số = {
        "symbol": mã_sạch,
        "resolution": độ_phân_giải,
        "from": dấu_thời_gian_bắt_đầu,
        "to": dấu_thời_gian_kết_thúc,
    }

    tiêu_đề = {
        "Accept": "application/json",
        "User-Agent": "Brian-Stock-Investment-Research/1.0",
    }

    lỗi_cuối = None

    for lần_thử in range(3):

        try:

            phản_hồi = requests.get(
                địa_chỉ,
                params=tham_số,
                headers=tiêu_đề,
                timeout=THỜI_GIAN_CHỜ,
            )

            if phản_hồi.status_code == 200:
                return phản_hồi.json()

            if phản_hồi.status_code in (429, 500, 502, 503, 504):
                lỗi_cuối = (
                    f"DNSE trả HTTP {phản_hồi.status_code}: "
                    f"{phản_hồi.text[:500]}"
                )

                time.sleep(1.5 * (lần_thử + 1))
                continue

            raise ValueError(
                f"DNSE HTTP {phản_hồi.status_code}: "
                f"{phản_hồi.text[:1000]}"
            )

        except requests.RequestException as lỗi:

            lỗi_cuối = str(lỗi)

            time.sleep(1.5 * (lần_thử + 1))

    raise ConnectionError(
        f"Không thể kết nối DNSE sau 3 lần thử. "
        f"Chi tiết: {lỗi_cuối}"
    )


# ============================================================
# TÌM DANH SÁCH NẾN TRONG PHẢN HỒI
# ============================================================

def tìm_danh_sách_nến(dữ_liệu: Any) -> list:
    """
    DNSE có thể trả dữ liệu qua nhiều lớp cấu trúc.
    Hàm này tìm danh sách nến một cách an toàn.
    """

    if isinstance(dữ_liệu, list):
        return dữ_liệu

    if isinstance(dữ_liệu, dict):

        khóa_ưu_tiên = [
            "data",
            "items",
            "content",
            "result",
            "results",
            "candles",
            "ohlc",
        ]

        for khóa in khóa_ưu_tiên:

            giá_trị = dữ_liệu.get(khóa)

            if isinstance(giá_trị, list):
                return giá_trị

            if isinstance(giá_trị, dict):
                danh_sách = tìm_danh_sách_nến(giá_trị)

                if danh_sách:
                    return danh_sách

    return []


# ============================================================
# CHUẨN HÓA MỘT DÒNG DỮ LIỆU
# ============================================================

def lấy_giá_trị(dòng: Any, *tên: str) -> Any:

    if isinstance(dòng, dict):

        for khóa in tên:

            if khóa in dòng:
                return dòng[khóa]

        return None

    if isinstance(dòng, (list, tuple)):

        return None

    return None


def chuẩn_hóa_lịch_sử(dữ_liệu: Any) -> pd.DataFrame:
    """
    Chuyển dữ liệu DNSE về bảng chuẩn.
    """

    danh_sách_nến = tìm_danh_sách_nến(dữ_liệu)

    if not danh_sách_nến:
        return pd.DataFrame()

    các_dòng = []

    for dòng in danh_sách_nến:

        if isinstance(dòng, dict):

            thời_gian = lấy_giá_trị(
                dòng,
                "t",
                "time",
                "timestamp",
                "datetime",
                "date",
            )

            giá_mở = lấy_giá_trị(
                dòng,
                "o",
                "open",
                "Open",
            )

            giá_cao = lấy_giá_trị(
                dòng,
                "h",
                "high",
                "High",
            )

            giá_thấp = lấy_giá_trị(
                dòng,
                "l",
                "low",
                "Low",
            )

            giá_đóng = lấy_giá_trị(
                dòng,
                "c",
                "close",
                "Close",
            )

            khối_lượng = lấy_giá_trị(
                dòng,
                "v",
                "volume",
                "Volume",
            )

            các_dòng.append(
                {
                    "Thời gian": thời_gian,
                    "Mở cửa": giá_mở,
                    "Cao nhất": giá_cao,
                    "Thấp nhất": giá_thấp,
                    "Đóng cửa": giá_đóng,
                    "Khối lượng": khối_lượng,
                }
            )

    bảng = pd.DataFrame(các_dòng)

    if bảng.empty:
        return bảng

    bảng["Thời gian"] = pd.to_numeric(
        bảng["Thời gian"],
        errors="coerce",
    )

    # DNSE trả thời gian Unix.
    bảng["Ngày"] = pd.to_datetime(
        bảng["Thời gian"],
        unit="s",
        errors="coerce",
        utc=True,
    ).dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)

    for cột in [
        "Mở cửa",
        "Cao nhất",
        "Thấp nhất",
        "Đóng cửa",
        "Khối lượng",
    ]:
        bảng[cột] = pd.to_numeric(
            bảng[cột],
            errors="coerce",
        )

    bảng = bảng.dropna(
        subset=[
            "Ngày",
            "Mở cửa",
            "Cao nhất",
            "Thấp nhất",
            "Đóng cửa",
        ]
    )

    bảng = bảng.sort_values("Ngày")
    bảng = bảng.drop_duplicates(
        subset=["Ngày"],
        keep="last",
    )

    bảng = bảng.set_index("Ngày")

    return bảng[
        [
            "Mở cửa",
            "Cao nhất",
            "Thấp nhất",
            "Đóng cửa",
            "Khối lượng",
        ]
    ].copy()


# ============================================================
# CHUẨN HÓA ĐƠN VỊ GIÁ
# ============================================================

def chuẩn_hóa_đơn_vị_giá(bảng: pd.DataFrame) -> pd.DataFrame:

    bảng = bảng.copy()

    for cột in [
        "Mở cửa",
        "Cao nhất",
        "Thấp nhất",
        "Đóng cửa",
    ]:

        if cột not in bảng.columns:
            continue

        trung_vị = bảng[cột].median()

        # Một số nguồn dữ liệu có thể trả giá theo nghìn đồng.
        # Nếu trung vị < 1000 thì chuyển về đồng.
        if pd.notna(trung_vị) and 0 < trung_vị < 1000:
            bảng[cột] = bảng[cột] * 1000

    return bảng


# ============================================================
# TÍNH CHỈ BÁO
# ============================================================

def thêm_chỉ_báo(bảng: pd.DataFrame) -> pd.DataFrame:

    bảng = bảng.copy()

    đóng = bảng["Đóng cửa"]
    cao = bảng["Cao nhất"]
    thấp = bảng["Thấp nhất"]
    khối_lượng = bảng["Khối lượng"]

    # --------------------------------------------------------
    # THAY ĐỔI GIÁ
    # --------------------------------------------------------

    bảng["Thay đổi"] = đóng.diff()

    bảng["Tỷ suất 1 ngày"] = đóng.pct_change()

    bảng["Tỷ suất 5 ngày"] = đóng.pct_change(5)

    bảng["Tỷ suất 10 ngày"] = đóng.pct_change(10)

    bảng["Tỷ suất 20 ngày"] = đóng.pct_change(20)

    bảng["Tỷ suất 60 ngày"] = đóng.pct_change(60)

    bảng["Tỷ suất 120 ngày"] = đóng.pct_change(120)

    bảng["Tỷ suất 252 ngày"] = đóng.pct_change(252)

    # --------------------------------------------------------
    # LỢI NHUẬN TÍCH LŨY
    # --------------------------------------------------------

    bảng["Lợi nhuận tích lũy"] = (
        (1 + bảng["Tỷ suất 1 ngày"].fillna(0)).cumprod() - 1
    )

    # --------------------------------------------------------
    # TRUNG BÌNH ĐỘNG ĐƠN
    # --------------------------------------------------------

    for số_ngày in [
        5,
        10,
        20,
        50,
        100,
        150,
        200,
    ]:

        bảng[f"Trung bình động {số_ngày}"] = (
            đóng.rolling(số_ngày).mean()
        )

    # --------------------------------------------------------
    # TRUNG BÌNH ĐỘNG LŨY THỪA
    # --------------------------------------------------------

    for số_ngày in [
        5,
        10,
        12,
        20,
        26,
        50,
        100,
        200,
    ]:

        bảng[f"Trung bình lũy thừa {số_ngày}"] = (
            đóng.ewm(
                span=số_ngày,
                adjust=False,
            ).mean()
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    chênh_lệch = đóng.diff()

    tăng = chênh_lệch.clip(lower=0)

    giảm = -chênh_lệch.clip(upper=0)

    trung_bình_tăng = tăng.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    trung_bình_giảm = giảm.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    tỷ_số_sức_mạnh = (
        trung_bình_tăng
        / trung_bình_giảm.replace(0, np.nan)
    )

    bảng["RSI 14"] = (
        100 - 100 / (1 + tỷ_số_sức_mạnh)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    trung_bình_12 = đóng.ewm(
        span=12,
        adjust=False,
    ).mean()

    trung_bình_26 = đóng.ewm(
        span=26,
        adjust=False,
    ).mean()

    bảng["MACD"] = trung_bình_12 - trung_bình_26

    bảng["Đường tín hiệu MACD"] = (
        bảng["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    bảng["Biểu đồ MACD"] = (
        bảng["MACD"]
        - bảng["Đường tín hiệu MACD"]
    )

    # --------------------------------------------------------
    # DẢI BOLLINGER
    # --------------------------------------------------------

    trung_bình_20 = đóng.rolling(20).mean()

    độ_lệch_20 = đóng.rolling(20).std()

    bảng["Dải giữa Bollinger"] = trung_bình_20

    bảng["Dải trên Bollinger"] = (
        trung_bình_20 + 2 * độ_lệch_20
    )

    bảng["Dải dưới Bollinger"] = (
        trung_bình_20 - 2 * độ_lệch_20
    )

    bảng["Độ rộng Bollinger"] = (
        (
            bảng["Dải trên Bollinger"]
            - bảng["Dải dưới Bollinger"]
        )
        / trung_bình_20
    )

    bảng["Vị trí trong Bollinger"] = (
        (
            đóng
            - bảng["Dải dưới Bollinger"]
        )
        / (
            bảng["Dải trên Bollinger"]
            - bảng["Dải dưới Bollinger"]
        )
    )

    # --------------------------------------------------------
    # TRUE RANGE / ATR
    # --------------------------------------------------------

    biên_độ_1 = cao - thấp

    biên_độ_2 = (
        cao - đóng.shift(1)
    ).abs()

    biên_độ_3 = (
        thấp - đóng.shift(1)
    ).abs()

    biên_độ_thật = pd.concat(
        [
            biên_độ_1,
            biên_độ_2,
            biên_độ_3,
        ],
        axis=1,
    ).max(axis=1)

    bảng["Biên độ thật"] = biên_độ_thật

    bảng["ATR 14"] = (
        biên_độ_thật
        .rolling(14)
        .mean()
    )

    bảng["ATR 20"] = (
        biên_độ_thật
        .rolling(20)
        .mean()
    )

    bảng["ATR phần trăm"] = (
        bảng["ATR 14"] / đóng * 100
    )

    # --------------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------------

    thấp_14 = thấp.rolling(14).min()

    cao_14 = cao.rolling(14).max()

    bảng["Stochastic K"] = (
        100
        * (đóng - thấp_14)
        / (cao_14 - thấp_14)
    )

    bảng["Stochastic D"] = (
        bảng["Stochastic K"]
        .rolling(3)
        .mean()
    )

    # --------------------------------------------------------
    # WILLIAMS %R
    # --------------------------------------------------------

    bảng["Williams R"] = (
        -100
        * (cao_14 - đóng)
        / (cao_14 - thấp_14)
    )

    # --------------------------------------------------------
    # CCI
    # --------------------------------------------------------

    giá_điển_hình = (
        cao + thấp + đóng
    ) / 3

    trung_bình_cci = (
        giá_điển_hình
        .rolling(20)
        .mean()
    )

    độ_lệch_cci = (
        giá_điển_hình
        .rolling(20)
        .apply(
            lambda x: np.mean(
                np.abs(x - np.mean(x))
            ),
            raw=True,
        )
    )

    bảng["CCI 20"] = (
        (
            giá_điển_hình
            - trung_bình_cci
        )
        / (
            0.015
            * độ_lệch_cci
        )
    )

    # --------------------------------------------------------
    # ROC / ĐỘNG LƯỢNG
    # --------------------------------------------------------

    bảng["Động lượng 10"] = (
        đóng - đóng.shift(10)
    )

    bảng["Động lượng 20"] = (
        đóng - đóng.shift(20)
    )

    bảng["ROC 10"] = (
        đóng.pct_change(10) * 100
    )

    bảng["ROC 20"] = (
        đóng.pct_change(20) * 100
    )

    # --------------------------------------------------------
    # BIẾN ĐỘNG
    # --------------------------------------------------------

    lợi_nhuận_ngày = bảng["Tỷ suất 1 ngày"]

    for số_ngày in [
        5,
        10,
        20,
        30,
        60,
        120,
        252,
    ]:

        bảng[f"Biến động {số_ngày} ngày"] = (
            lợi_nhuận_ngày
            .rolling(số_ngày)
            .std()
            * np.sqrt(252)
            * 100
        )

    # --------------------------------------------------------
    # KHỐI LƯỢNG
    # --------------------------------------------------------

    for số_ngày in [
        5,
        10,
        20,
        50,
    ]:

        bảng[f"Khối lượng trung bình {số_ngày}"] = (
            khối_lượng
            .rolling(số_ngày)
            .mean()
        )

    bảng["Thay đổi khối lượng"] = (
        khối_lượng.pct_change()
    )

    bảng["Tỷ lệ khối lượng"] = (
        khối_lượng
        / bảng["Khối lượng trung bình 20"]
    )

    # --------------------------------------------------------
    # OBV
    # --------------------------------------------------------

    hướng_giá = np.sign(
        đóng.diff().fillna(0)
    )

    bảng["OBV"] = (
        hướng_giá * khối_lượng
    ).cumsum()

    # --------------------------------------------------------
    # MỨC CAO / THẤP
    # --------------------------------------------------------

    for số_ngày in [
        20,
        50,
        100,
        200,
        252,
    ]:

        bảng[f"Cao nhất {số_ngày} ngày"] = (
            cao.rolling(số_ngày).max()
        )

        bảng[f"Thấp nhất {số_ngày} ngày"] = (
            thấp.rolling(số_ngày).min()
        )

    # --------------------------------------------------------
    # VỊ TRÍ SO VỚI ĐỈNH / ĐÁY
    # --------------------------------------------------------

    đỉnh_252 = bảng["Cao nhất 252 ngày"]

    đáy_252 = bảng["Thấp nhất 252 ngày"]

    bảng["Khoảng cách đỉnh 252 ngày"] = (
        đóng / đỉnh_252 - 1
    ) * 100

    bảng["Khoảng cách đáy 252 ngày"] = (
        đóng / đáy_252 - 1
    ) * 100

    # --------------------------------------------------------
    # ADX / DI
    # --------------------------------------------------------

    tăng_cực_đại = cao.diff()

    giảm_cực_đại = -thấp.diff()

    hướng_tăng = np.where(
        (tăng_cực_đại > giảm_cực_đại)
        & (tăng_cực_đại > 0),
        tăng_cực_đại,
        0,
    )

    hướng_giảm = np.where(
        (giảm_cực_đại > tăng_cực_đại)
        & (giảm_cực_đại > 0),
        giảm_cực_đại,
        0,
    )

    hướng_tăng = pd.Series(
        hướng_tăng,
        index=bảng.index,
    )

    hướng_giảm = pd.Series(
        hướng_giảm,
        index=bảng.index,
    )

    trung_bình_biên_độ = (
        biên_độ_thật
        .rolling(14)
        .mean()
    )

    bảng["DI dương"] = (
        100
        * hướng_tăng.rolling(14).mean()
        / trung_bình_biên_độ
    )

    bảng["DI âm"] = (
        100
        * hướng_giảm.rolling(14).mean()
        / trung_bình_biên_độ
    )

    chỉ_số_hướng = (
        (
            bảng["DI dương"]
            - bảng["DI âm"]
        ).abs()
        / (
            bảng["DI dương"]
            + bảng["DI âm"]
        )
    ) * 100

    bảng["ADX 14"] = (
        chỉ_số_hướng
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # TÍN HIỆU CƠ BẢN
    # --------------------------------------------------------

    bảng["Giá trên MA20"] = (
        đóng > bảng["Trung bình động 20"]
    )

    bảng["Giá trên MA50"] = (
        đóng > bảng["Trung bình động 50"]
    )

    bảng["Giá trên MA200"] = (
        đóng > bảng["Trung bình động 200"]
    )

    bảng["MA20 trên MA50"] = (
        bảng["Trung bình động 20"]
        > bảng["Trung bình động 50"]
    )

    bảng["MA50 trên MA200"] = (
        bảng["Trung bình động 50"]
        > bảng["Trung bình động 200"]
    )

    bảng["MACD dương"] = (
        bảng["MACD"]
        > bảng["Đường tín hiệu MACD"]
    )

    bảng["RSI quá mua"] = (
        bảng["RSI 14"] >= 70
    )

    bảng["RSI quá bán"] = (
        bảng["RSI 14"] <= 30
    )

    # --------------------------------------------------------
    # ĐIỂM XU HƯỚNG
    # --------------------------------------------------------

    điểm_xu_hướng = pd.Series(
        0.0,
        index=bảng.index,
    )

    điểm_xu_hướng += np.where(
        đóng > bảng["Trung bình động 20"],
        1,
        -1,
    )

    điểm_xu_hướng += np.where(
        đóng > bảng["Trung bình động 50"],
        1,
        -1,
    )

    điểm_xu_hướng += np.where(
        đóng > bảng["Trung bình động 200"],
        1,
        -1,
    )

    điểm_xu_hướng += np.where(
        bảng["MACD"] > bảng["Đường tín hiệu MACD"],
        1,
        -1,
    )

    điểm_xu_hướng += np.where(
        bảng["RSI 14"] < 30,
        1,
        np.where(
            bảng["RSI 14"] > 70,
            -1,
            0,
        ),
    )

    điểm_xu_hướng += np.where(
        bảng["ADX 14"] > 25,
        np.sign(
            bảng["DI dương"]
            - bảng["DI âm"]
        ),
        0,
    )

    bảng["Điểm xu hướng"] = điểm_xu_hướng

    bảng["Trạng thái kỹ thuật"] = np.select(
        [
            bảng["Điểm xu hướng"] >= 4,
            bảng["Điểm xu hướng"] >= 2,
            bảng["Điểm xu hướng"] <= -4,
            bảng["Điểm xu hướng"] <= -2,
        ],
        [
            "TÍCH CỰC",
            "HƠI TÍCH CỰC",
            "THẬN TRỌNG",
            "HƠI THẬN TRỌNG",
        ],
        default="TRUNG TÍNH",
    )

    return bảng


# ============================================================
# KIỂM TRA DỮ LIỆU
# ============================================================

def kiểm_tra_dữ_liệu(bảng: pd.DataFrame) -> pd.DataFrame:

    bảng = bảng.copy()

    if bảng.empty:
        return bảng

    điều_kiện_hợp_lệ = (
        (bảng["Mở cửa"] > 0)
        & (bảng["Cao nhất"] > 0)
        & (bảng["Thấp nhất"] > 0)
        & (bảng["Đóng cửa"] > 0)
        & (bảng["Cao nhất"] >= bảng["Thấp nhất"])
        & (bảng["Cao nhất"] >= bảng["Mở cửa"])
        & (bảng["Cao nhất"] >= bảng["Đóng cửa"])
        & (bảng["Thấp nhất"] <= bảng["Mở cửa"])
        & (bảng["Thấp nhất"] <= bảng["Đóng cửa"])
    )

    bảng = bảng.loc[điều_kiện_hợp_lệ].copy()

    return bảng


# ============================================================
# HÀM TẢI DỮ LIỆU CHÍNH
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def tải_dữ_liệu_thị_trường(
    mã: str,
    khoảng_dữ_liệu: str = "1 năm",
) -> pd.DataFrame:

    mã_sạch = chuẩn_hóa_mã(mã)

    if not mã_sạch:
        raise ValueError(
            "Bạn chưa nhập mã cổ phiếu."
        )

    dấu_bắt_đầu, dấu_kết_thúc = (
        xác_định_khoảng_thời_gian(
            khoảng_dữ_liệu
        )
    )

    dữ_liệu_thô = gọi_dnse(
        mã=mã_sạch,
        độ_phân_giải=ĐỘ_PHÂN_GIẢI_NGÀY,
        dấu_thời_gian_bắt_đầu=dấu_bắt_đầu,
        dấu_thời_gian_kết_thúc=dấu_kết_thúc,
    )

    bảng = chuẩn_hóa_lịch_sử(
        dữ_liệu_thô
    )

    if bảng.empty:
        raise ValueError(
            f"DNSE không trả dữ liệu OHLC cho mã {mã_sạch}."
        )

    bảng = chuẩn_hóa_đơn_vị_giá(bảng)

    bảng = kiểm_tra_dữ_liệu(bảng)

    if bảng.empty:
        raise ValueError(
            f"Dữ liệu {mã_sạch} sau khi kiểm tra không còn dòng hợp lệ."
        )

    bảng = thêm_chỉ_báo(bảng)

    bảng = bảng.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return bảng


# ============================================================
# TƯƠNG THÍCH VỚI CÁC FILE CŨ
# ============================================================

# Các tên này giúp project cũ không chết ngay khi chuyển sang
# bộ hàm mới.

def normalize_symbol(mã: str) -> str:
    return chuẩn_hóa_mã(mã)


def display_symbol(mã: str) -> str:
    return hiển_thị_mã(mã)


def load_market_data(
    mã: str,
    khoảng_dữ_liệu: str = "1y",
) -> pd.DataFrame:

    ánh_xạ_khoảng = {
        "1mo": "1 tháng",
        "3mo": "3 tháng",
        "6mo": "6 tháng",
        "1y": "1 năm",
        "2y": "2 năm",
        "3y": "3 năm",
        "5y": "5 năm",
    }

    khoảng = ánh_xạ_khoảng.get(
        khoảng_dữ_liệu,
        khoảng_dữ_liệu,
    )

    return tải_dữ_liệu_thị_trường(
        mã,
        khoảng,
    )
