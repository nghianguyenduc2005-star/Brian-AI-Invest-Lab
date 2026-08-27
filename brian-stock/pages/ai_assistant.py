from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from data.market import (
    normalize_symbol,
    display_symbol,
    load_market_data,
    market_snapshot,
    build_quant,
    run_ols,
    run_random_forest,
)
from components.charts import price_volume_chart


# ============================================================
# HÀM PHỤ
# ============================================================

def _so(
    gia_tri,
    mac_dinh=None,
):
    try:
        gia_tri = float(gia_tri)

        if pd.isna(gia_tri):
            return mac_dinh

        return gia_tri

    except Exception:
        return mac_dinh


def _phan_tram(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return f"{gia_tri:+.2f}%"


def _dinh_dang_so(
    gia_tri,
    so_chu_so=2,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return f"{gia_tri:,.{so_chu_so}f}"


def _dinh_dang_khoi_luong(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    if gia_tri >= 1_000_000_000:
        return f"{gia_tri / 1_000_000_000:.2f} tỷ"

    if gia_tri >= 1_000_000:
        return f"{gia_tri / 1_000_000:.2f} triệu"

    if gia_tri >= 1_000:
        return f"{gia_tri / 1_000:.2f} nghìn"

    return f"{gia_tri:,.0f}"


def _mau_danh_gia(
    gia_tri,
):
    if gia_tri == "TÍCH CỰC":
        return "#00d4a8"

    if gia_tri == "TIÊU CỰC":
        return "#ff4d5a"

    if gia_tri == "TRUNG TÍNH":
        return "#f59e0b"

    return "#94a3b8"


# ============================================================
# PHÂN TÍCH KỸ THUẬT
# ============================================================

def _phan_tich_ky_thuat(
    du_lieu,
):
    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return {}

    dong_cuoi = du_lieu.iloc[-1]

    gia = _so(
        dong_cuoi.get("Close"),
        None,
    )

    rsi = _so(
        dong_cuoi.get("RSI"),
        None,
    )

    macd = _so(
        dong_cuoi.get("MACD"),
        None,
    )

    macd_tin_hieu = _so(
        dong_cuoi.get("MACD_Signal"),
        None,
    )

    macd_hist = _so(
        dong_cuoi.get("MACD_Hist"),
        None,
    )

    sma20 = _so(
        dong_cuoi.get("SMA20"),
        None,
    )

    sma50 = _so(
        dong_cuoi.get("SMA50"),
        None,
    )

    sma200 = _so(
        dong_cuoi.get("SMA200"),
        None,
    )

    ema20 = _so(
        dong_cuoi.get("EMA20"),
        None,
    )

    ema50 = _so(
        dong_cuoi.get("EMA50"),
        None,
    )

    khau_luong = _so(
        dong_cuoi.get("Volume"),
        None,
    )

    khoi_luong_tb20 = _so(
        dong_cuoi.get("Volume_SMA20"),
        None,
    )

    bien_dong = _so(
        dong_cuoi.get("Volatility20"),
        None,
    )

    atr14 = _so(
        dong_cuoi.get("ATR14"),
        None,
    )

    momentum5 = _so(
        dong_cuoi.get("Momentum5"),
        None,
    )

    momentum20 = _so(
        dong_cuoi.get("Momentum20"),
        None,
    )

    bollinger_tren = _so(
        dong_cuoi.get("Bollinger_Upper"),
        None,
    )

    bollinger_duoi = _so(
        dong_cuoi.get("Bollinger_Lower"),
        None,
    )

    diem_tich_cuc = 0
    diem_tieu_cuc = 0

    tin_hieu = []

    # --------------------------------------------------------
    # Xu hướng SMA
    # --------------------------------------------------------

    if (
        gia is not None
        and sma20 is not None
    ):

        if gia > sma20:
            diem_tich_cuc += 1
            tin_hieu.append(
                "Giá nằm trên trung bình 20 phiên."
            )
        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "Giá nằm dưới trung bình 20 phiên."
            )

    if (
        gia is not None
        and sma50 is not None
    ):

        if gia > sma50:
            diem_tich_cuc += 1
            tin_hieu.append(
                "Giá nằm trên trung bình 50 phiên."
            )
        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "Giá nằm dưới trung bình 50 phiên."
            )

    if (
        gia is not None
        and sma200 is not None
    ):

        if gia > sma200:
            diem_tich_cuc += 1
            tin_hieu.append(
                "Giá nằm trên trung bình 200 phiên."
            )
        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "Giá nằm dưới trung bình 200 phiên."
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi is not None:

        if rsi >= 70:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "RSI ở vùng quá mua."
            )

        elif rsi <= 30:
            diem_tich_cuc += 1
            tin_hieu.append(
                "RSI ở vùng quá bán, có khả năng xuất hiện nhịp hồi."
            )

        elif rsi >= 50:
            diem_tich_cuc += 1
            tin_hieu.append(
                "RSI duy trì trên vùng cân bằng."
            )

        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "RSI nằm dưới vùng cân bằng."
            )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        macd is not None
        and macd_tin_hieu is not None
    ):

        if macd > macd_tin_hieu:
            diem_tich_cuc += 1
            tin_hieu.append(
                "MACD nằm trên đường tín hiệu."
            )

        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "MACD nằm dưới đường tín hiệu."
            )

    if macd_hist is not None:

        if macd_hist > 0:
            diem_tich_cuc += 1
            tin_hieu.append(
                "Động lượng MACD đang dương."
            )

        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "Động lượng MACD đang âm."
            )

    # --------------------------------------------------------
    # Thanh khoản
    # --------------------------------------------------------

    if (
        khau_luong is not None
        and khoi_luong_tb20 is not None
        and khoi_luong_tb20 > 0
    ):

        ty_le_khoi_luong = (
            khau_luong
            / khoi_luong_tb20
        )

        if ty_le_khoi_luong >= 1.5:
            diem_tich_cuc += 1
            tin_hieu.append(
                "Khối lượng cao hơn đáng kể so với trung bình 20 phiên."
            )

        elif ty_le_khoi_luong < 0.7:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "Khối lượng thấp hơn đáng kể so với trung bình 20 phiên."
            )

        else:
            tin_hieu.append(
                "Khối lượng ở mức tương đối bình thường."
            )

    # --------------------------------------------------------
    # Động lượng
    # --------------------------------------------------------

    if momentum5 is not None:

        if momentum5 > 0:
            diem_tich_cuc += 1
            tin_hieu.append(
                "Động lượng 5 phiên đang tăng."
            )

        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "Động lượng 5 phiên đang giảm."
            )

    if momentum20 is not None:

        if momentum20 > 0:
            diem_tich_cuc += 1
            tin_hieu.append(
                "Động lượng 20 phiên đang tăng."
            )

        else:
            diem_tieu_cuc += 1
            tin_hieu.append(
                "Động lượng 20 phiên đang giảm."
            )

    # --------------------------------------------------------
    # Bollinger
    # --------------------------------------------------------

    if (
        gia is not None
        and bollinger_tren is not None
        and bollinger_duoi is not None
    ):

        if gia >= bollinger_tren:
            tin_hieu.append(
                "Giá đang sát hoặc vượt dải Bollinger trên."
            )
            diem_tieu_cuc += 1

        elif gia <= bollinger_duoi:
            tin_hieu.append(
                "Giá đang sát hoặc dưới dải Bollinger dưới."
            )
            diem_tich_cuc += 1

        else:
            tin_hieu.append(
                "Giá đang nằm trong dải Bollinger."
            )

    # --------------------------------------------------------
    # Xu hướng tổng hợp
    # --------------------------------------------------------

    if (
        diem_tich_cuc
        >= diem_tieu_cuc + 3
    ):
        danh_gia = "TÍCH CỰC"

    elif (
        diem_tieu_cuc
        >= diem_tich_cuc + 3
    ):
        danh_gia = "TIÊU CỰC"

    else:
        danh_gia = "TRUNG TÍNH"

    return {
        "danh_gia": danh_gia,
        "diem_tich_cuc": diem_tich_cuc,
        "diem_tieu_cuc": diem_tieu_cuc,
        "tin_hieu": tin_hieu,
        "gia": gia,
        "rsi": rsi,
        "macd": macd,
        "macd_tin_hieu": macd_tin_hieu,
        "macd_hist": macd_hist,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "ema20": ema20,
        "ema50": ema50,
        "volume": khau_luong,
        "volume_tb20": khoi_luong_tb20,
        "volatility20": bien_dong,
        "atr14": atr14,
        "momentum5": momentum5,
        "momentum20": momentum20,
        "bollinger_tren": bollinger_tren,
        "bollinger_duoi": bollinger_duoi,
    }


# ============================================================
# PHÂN TÍCH ĐỊNH LƯỢNG
# ============================================================

def _phan_tich_dinh_luong(
    du_lieu,
):
    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return None

    ket_qua = build_quant(
        du_lieu
    )

    if ket_qua is None:
        return None

    try:

        mo_hinh_ols = ket_qua[0]
        mo_hinh_rung = ket_qua[1]
        chi_so = ket_qua[2]
        du_bao = float(
            ket_qua[3]
        )
        tam_quan_trong = ket_qua[4]

    except Exception:
        return None

    return {
        "ols": mo_hinh_ols,
        "rung": mo_hinh_rung,
        "chi_so": chi_so,
        "du_bao": du_bao,
        "tam_quan_trong": tam_quan_trong,
    }


# ============================================================
# HIỂN THỊ ĐỊNH LƯỢNG
# ============================================================

def _hien_thi_dinh_luong(
    ket_qua,
):

    if ket_qua is None:

        st.info(
            "Chưa đủ dữ liệu hợp lệ để xây dựng mô hình định lượng."
        )

        return

    chi_so = ket_qua[
        "chi_so"
    ]

    du_bao = (
        ket_qua[
            "du_bao"
        ]
        * 100
    )

    sai_so = _so(
        chi_so.get("MAE"),
        None,
    )

    r2 = _so(
        chi_so.get("R2"),
        None,
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Dự báo lợi suất kế tiếp",
            f"{du_bao:+.2f}%",
        )

    with b:

        if sai_so is None:

            st.metric(
                "Sai số trung bình",
                "—",
            )

        else:

            st.metric(
                "Sai số trung bình",
                f"{sai_so * 100:.3f}%",
            )

    with c:

        if r2 is None or np.isnan(r2):

            st.metric(
                "R² kiểm tra",
                "—",
            )

        else:

            st.metric(
                "R² kiểm tra",
                f"{r2:.3f}",
            )

    with d:

        if du_bao > 0:

            st.metric(
                "Hướng mô hình",
                "TĂNG",
            )

        elif du_bao < 0:

            st.metric(
                "Hướng mô hình",
                "GIẢM",
            )

        else:

            st.metric(
                "Hướng mô hình",
                "ĐI NGANG",
            )

    # --------------------------------------------------------
    # Độ quan trọng biến
    # --------------------------------------------------------

    tam_quan_trong = ket_qua[
        "tam_quan_trong"
    ]

    if isinstance(
        tam_quan_trong,
        pd.Series,
    ):

        bang_tam_quan_trong = (
            tam_quan_trong
            .rename(
                "Mức độ quan trọng"
            )
            .to_frame()
        )

        bang_tam_quan_trong[
            "Mức độ quan trọng"
        ] = (
            bang_tam_quan_trong[
                "Mức độ quan trọng"
            ]
            * 100
        )

        anh_xa = {
            "RSI": "Sức mạnh tương đối",
            "MACD": "MACD",
            "MACD_Hist": "Động lượng MACD",
            "Volatility20": "Biến động 20 phiên",
            "Volume_Change": "Thay đổi khối lượng",
            "Return": "Lợi suất",
        }

        bang_tam_quan_trong.index = [
            anh_xa.get(
                str(x),
                str(x),
            )
            for x
            in bang_tam_quan_trong.index
        ]

        st.markdown(
            "#### Yếu tố ảnh hưởng mô hình"
        )

        st.bar_chart(
            bang_tam_quan_trong
        )


# ============================================================
# HIỂN THỊ OLS
# ============================================================

def _hien_thi_ols(
    du_lieu,
):

    mo_hinh = run_ols(
        du_lieu
    )

    if mo_hinh is None:

        st.info(
            "Chưa đủ dữ liệu để chạy mô hình hồi quy."
        )

        return

    ket_qua = []

    anh_xa = {
        "const": "Hằng số",
        "Volume_Change": "Thay đổi khối lượng",
        "RSI": "Sức mạnh tương đối",
        "MACD": "MACD",
        "Volatility20": "Biến động 20 phiên",
    }

    try:

        for ten in mo_hinh.params.index:

            he_so = float(
                mo_hinh.params[
                    ten
                ]
            )

            pvalue = float(
                mo_hinh.pvalues[
                    ten
                ]
            )

            ket_qua.append(
                {
                    "Yếu tố": anh_xa.get(
                        str(ten),
                        str(ten),
                    ),
                    "Hệ số": he_so,
                    "Giá trị p": pvalue,
                }
            )

        bang = pd.DataFrame(
            ket_qua
        )

        st.dataframe(
            bang,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            f"R² mô hình: {mo_hinh.rsquared:.3f}"
        )

    except Exception as loi:

        st.warning(
            f"Không thể hiển thị kết quả hồi quy: {loi}"
        )


# ============================================================
# HIỂN THỊ RỪNG NGẪU NHIÊN
# ============================================================

def _hien_thi_rung(
    du_lieu,
):

    ket_qua = run_random_forest(
        du_lieu
    )

    if ket_qua is None:

        st.info(
            "Chưa đủ dữ liệu để chạy mô hình cây quyết định."
        )

        return

    du_bao = (
        float(
            ket_qua[
                "prediction"
            ]
        )
        * 100
    )

    st.metric(
        "Dự báo lợi suất kế tiếp",
        f"{du_bao:+.2f}%",
    )

    tam_quan_trong = ket_qua[
        "importance"
    ]

    if tam_quan_trong:

        anh_xa = {
            "Volume_Change": "Thay đổi khối lượng",
            "RSI": "Sức mạnh tương đối",
            "MACD": "MACD",
            "Volatility20": "Biến động 20 phiên",
        }

        bang = pd.DataFrame(
            {
                "Yếu tố": [
                    anh_xa.get(
                        str(k),
                        str(k),
                    )
                    for k
                    in tam_quan_trong
                ],
                "Mức độ quan trọng": [
                    float(v) * 100
                    for v
                    in tam_quan_trong.values()
                ],
            }
        )

        st.dataframe(
            bang.sort_values(
                "Mức độ quan trọng",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# RENDER
# ============================================================

def render_stock_analysis():

    # ========================================================
    # TIÊU ĐỀ
    # ========================================================

    st.markdown(
        '<div class="section-title">🔬 Phân tích cổ phiếu</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # NHẬP MÃ
    # ========================================================

    ma_mac_dinh = (
        st.session_state.get(
            "stock_analysis_symbol",
            "HPG",
        )
    )

    ma_nhap = st.text_input(
        "Mã cổ phiếu",
        value=ma_mac_dinh,
        placeholder="Nhập mã, ví dụ HPG hoặc MSR",
        key="stock_analysis_input",
    )

    cot_1, cot_2 = st.columns(
        [3, 1]
    )

    with cot_1:

        khung_thoi_gian = st.selectbox(
            "Khoảng thời gian",
            [
                "3mo",
                "6mo",
                "1y",
                "2y",
                "5y",
                "10y",
            ],
            index=2,
            key="stock_analysis_period",
        )

    with cot_2:

        tai_lai = st.button(
            "Tải dữ liệu",
            type="primary",
            key="stock_analysis_load",
        )

    if tai_lai:

        ma_sach = normalize_symbol(
            ma_nhap
        )

        st.session_state[
            "stock_analysis_symbol"
        ] = ma_sach

        st.rerun()

    ma_dang_xem = normalize_symbol(
        st.session_state.get(
            "stock_analysis_symbol",
            ma_nhap,
        )
    )

    # ========================================================
    # LẤY DỮ LIỆU
    # ========================================================

    try:

        with st.spinner(
            f"Đang tải dữ liệu {display_symbol(ma_dang_xem)}..."
        ):

            du_lieu = load_market_data(
                ma_dang_xem,
                khung_thoi_gian,
            )

    except Exception as loi:

        st.error(
            f"Không tải được dữ liệu {display_symbol(ma_dang_xem)}: {loi}"
        )

        return

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        st.error(
            "Nguồn dữ liệu trả về rỗng."
        )

        return

    # ========================================================
    # SNAPSHOT
    # ========================================================

    anh_chup = market_snapshot(
        du_lieu
    )

    phan_tich = _phan_tich_ky_thuat(
        du_lieu
    )

    # ========================================================
    # TÊN MÃ
    # ========================================================

    st.markdown(
        f'<div class="hero">'
        f'<div class="eyebrow">PHÂN TÍCH CHUYÊN SÂU</div>'
        f'<h1>{html.escape(display_symbol(ma_dang_xem))}</h1>'
        f'<p>Dữ liệu thị trường thật · Phân tích kỹ thuật · Định lượng</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # TỔNG QUAN
    # ========================================================

    st.markdown(
        '<div class="section-title">📌 Tổng quan</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:

        gia = _so(
            anh_chup.get(
                "price"
            ),
            None,
        )

        st.metric(
            "Giá",
            (
                f"{gia:,.0f}"
                if gia is not None
                else "—"
            ),
        )

    with b:

        thay_doi = _so(
            anh_chup.get(
                "change_1d"
            ),
            None,
        )

        st.metric(
            "1D",
            (
                f"{thay_doi:+.2f}%"
                if thay_doi is not None
                else "—"
            ),
        )

    with c:

        rsi_hien_tai = _so(
            anh_chup.get(
                "rsi"
            ),
            None,
        )

        st.metric(
            "RSI",
            (
                f"{rsi_hien_tai:.1f}"
                if rsi_hien_tai is not None
                else "—"
            ),
        )

    with d:

        khoi_luong = _so(
            anh_chup.get(
                "volume"
            ),
            None,
        )

        st.metric(
            "Khối lượng",
            _dinh_dang_khoi_luong(
                khoi_luong
            ),
        )

    # ========================================================
    # ĐÁNH GIÁ TỔNG HỢP
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Đánh giá kỹ thuật</div>',
        unsafe_allow_html=True,
    )

    danh_gia = phan_tich.get(
        "danh_gia",
        "TRUNG TÍNH",
    )

    st.markdown(
        f"""
        <div style="
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #243447;
            background: #0b1722;
            margin-bottom: 15px;
        ">
            <div style="
                font-size: 12px;
                opacity: 0.7;
                margin-bottom: 5px;
            ">
                TÍN HIỆU TỔNG HỢP
            </div>

            <div style="
                font-size: 28px;
                font-weight: 700;
                color: {_mau_danh_gia(danh_gia)};
            ">
                {danh_gia}
            </div>

            <div style="
                margin-top: 8px;
                font-size: 13px;
            ">
                Tích cực: {phan_tich.get("diem_tich_cuc", 0)}
                ·
                Tiêu cực: {phan_tich.get("diem_tieu_cuc", 0)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # CHI TIẾT CHỈ BÁO
    # ========================================================

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "SMA20",
            _dinh_dang_so(
                phan_tich.get("sma20"),
                0,
            ),
        )

    with b:

        st.metric(
            "SMA50",
            _dinh_dang_so(
                phan_tich.get("sma50"),
                0,
            ),
        )

    with c:

        st.metric(
            "EMA20",
            _dinh_dang_so(
                phan_tich.get("ema20"),
                0,
            ),
        )

    with d:

        st.metric(
            "EMA50",
            _dinh_dang_so(
                phan_tich.get("ema50"),
                0,
            ),
        )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "MACD",
            _dinh_dang_so(
                phan_tich.get("macd"),
                3,
            ),
        )

    with b:

        st.metric(
            "Tín hiệu MACD",
            _dinh_dang_so(
                phan_tich.get(
                    "macd_tin_hieu"
                ),
                3,
            ),
        )

    with c:

        st.metric(
            "ATR14",
            _dinh_dang_so(
                phan_tich.get(
                    "atr14"
                ),
                0,
            ),
        )

    with d:

        st.metric(
            "Biến động 20 phiên",
            _phan_tram(
                phan_tich.get(
                    "volatility20"
                )
            ),
        )

    # ========================================================
    # ĐỘNG LƯỢNG
    # ========================================================

    st.markdown(
        '<div class="section-title">⚡ Động lượng</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "5 phiên",
            _phan_tram(
                (
                    phan_tich.get(
                        "momentum5"
                    )
                    * 100
                    if phan_tich.get(
                        "momentum5"
                    )
                    is not None
                    else None
                )
            ),
        )

    with b:

        st.metric(
            "20 phiên",
            _phan_tram(
                (
                    phan_tich.get(
                        "momentum20"
                    )
                    * 100
                    if phan_tich.get(
                        "momentum20"
                    )
                    is not None
                    else None
                )
            ),
        )

    with c:

        st.metric(
            "Khối lượng / TB20",
            (
                (
                    f"{phan_tich['volume'] / phan_tich['volume_tb20']:.2f}x"
                )
                if (
                    phan_tich.get(
                        "volume"
                    )
                    is not None
                    and phan_tich.get(
                        "volume_tb20"
                    )
                    not in [
                        None,
                        0,
                    ]
                )
                else "—"
            ),
        )

    with d:

        st.metric(
            "RSI",
            (
                f"{phan_tich['rsi']:.1f}"
                if phan_tich.get(
                    "rsi"
                )
                is not None
                else "—"
            ),
        )

    # ========================================================
    # CÁC NHẬN ĐỊNH
    # ========================================================

    st.markdown(
        "#### Các tín hiệu đang hoạt động"
    )

    for tin_hieu in phan_tich.get(
        "tin_hieu",
        [],
    ):

        st.write(
            f"• {tin_hieu}"
        )

    # ========================================================
    # BIỂU ĐỒ
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Biểu đồ phân tích</div>',
        unsafe_allow_html=True,
    )

    try:

        bieu_do = price_volume_chart(
            du_lieu
        )

        if bieu_do is not None:

            st.plotly_chart(
                bieu_do,
                width="stretch",
                config={
                    "displaylogo": False,
                },
            )

    except Exception as loi:

        st.warning(
            f"Không thể hiển thị biểu đồ: {loi}"
        )

    # ========================================================
    # ĐỊNH LƯỢNG
    # ========================================================

    st.markdown(
        '<div class="section-title">🧮 Định lượng</div>',
        unsafe_allow_html=True,
    )

    ket_qua_dinh_luong = (
        _phan_tich_dinh_luong(
            du_lieu
        )
    )

    _hien_thi_dinh_luong(
        ket_qua_dinh_luong
    )

    # ========================================================
    # HỒI QUY
    # ========================================================

    with st.expander(
        "📐 Hồi quy",
        expanded=False,
    ):

        _hien_thi_ols(
            du_lieu
        )

    # ========================================================
    # MÔ HÌNH CÂY
    # ========================================================

    with st.expander(
        "🌲 Mô hình cây quyết định",
        expanded=False,
    ):

        _hien_thi_rung(
            du_lieu
        )

    # ========================================================
    # DỮ LIỆU GỐC
    # ========================================================

    with st.expander(
        "📋 Dữ liệu phiên gần nhất",
        expanded=False,
    ):

        so_dong = st.slider(
            "Số phiên",
            min_value=5,
            max_value=min(
                100,
                len(du_lieu),
            ),
            value=min(
                20,
                len(du_lieu),
            ),
            key="so_phien_hien_thi",
        )

        cac_cot_hien_thi = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "ReturnPct",
            "RSI",
            "MACD",
            "SMA20",
            "SMA50",
            "EMA20",
            "EMA50",
            "Volatility20",
        ]

        cac_cot_thuc_te = [
            cot
            for cot
            in cac_cot_hien_thi
            if cot in du_lieu.columns
        ]

        bang = du_lieu[
            cac_cot_thuc_te
        ].tail(
            so_dong
        ).copy()

        anh_xa_cot = {
            "Open": "Mở cửa",
            "High": "Cao nhất",
            "Low": "Thấp nhất",
            "Close": "Đóng cửa",
            "Volume": "Khối lượng",
            "ReturnPct": "Thay đổi %",
            "RSI": "RSI",
            "MACD": "MACD",
            "SMA20": "TB20",
            "SMA50": "TB50",
            "EMA20": "TLT20",
            "EMA50": "TLT50",
            "Volatility20": "Biến động 20 phiên",
        }

        bang = bang.rename(
            columns=anh_xa_cot
        )

        st.dataframe(
            bang,
            width="stretch",
        )
