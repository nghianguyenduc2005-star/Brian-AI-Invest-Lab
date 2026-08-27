from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from components.cards import metric_card
from components.charts import price_volume_chart
from data.market import (
    normalize_symbol,
    display_symbol,
    load_market_data,
    load_vnindex_data,
    market_snapshot,
)
from data.news import fetch_market_news


# ============================================================
# TIỆN ÍCH
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


def _tim_cot(
    du_lieu,
    danh_sach_ten,
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

    for ten in danh_sach_ten:

        cot = anh_xa.get(
            str(ten).strip().lower()
        )

        if cot is not None:
            return cot

    return None


# ============================================================
# ĐỊNH DẠNG ĐƠN VỊ
# ============================================================

def _khoi_luong_co_phieu(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    if gia_tri >= 1_000_000_000:
        return (
            f"{gia_tri / 1_000_000_000:.2f} tỷ cổ phiếu"
        )

    if gia_tri >= 1_000_000:
        return (
            f"{gia_tri / 1_000_000:.2f} triệu cổ phiếu"
        )

    if gia_tri >= 1_000:
        return (
            f"{gia_tri / 1_000:.2f} nghìn cổ phiếu"
        )

    return (
        f"{gia_tri:,.0f} cổ phiếu"
    )


def _gia_tri_giao_dich(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    if gia_tri >= 1_000_000_000_000:
        return (
            f"{gia_tri / 1_000_000_000_000:.2f} nghìn tỷ đồng"
        )

    if gia_tri >= 1_000_000_000:
        return (
            f"{gia_tri / 1_000_000_000:.2f} tỷ đồng"
        )

    if gia_tri >= 1_000_000:
        return (
            f"{gia_tri / 1_000_000:.2f} triệu đồng"
        )

    if gia_tri >= 1_000:
        return (
            f"{gia_tri / 1_000:.2f} nghìn đồng"
        )

    return (
        f"{gia_tri:,.0f} đồng"
    )


def _diem_vn_index(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:,.2f} điểm"
    )


def _thay_doi_diem(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:+,.2f} điểm"
    )


def _thay_doi_phan_tram(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:+.2f}%"
    )


def _gia_co_phieu(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:,.0f} đồng/cổ phiếu"
    )


def _bien_dong(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:.2f}%"
    )


def _rsi(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:.1f} điểm"
    )


def _macd(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:.3f}"
    )


def _atr(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:,.0f} đồng/cổ phiếu"
    )


# ============================================================
# LẤY VN-INDEX
# ============================================================

def _lay_vn_index():

    try:

        du_lieu = load_vnindex_data()

    except Exception as loi:

        return {
            "loi": str(
                loi
            )
        }

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        return {
            "loi": "Nguồn VN-INDEX trả về dữ liệu rỗng."
        }

    du_lieu = du_lieu.copy()

    # --------------------------------------------------------
    # Tìm cột đóng cửa
    # --------------------------------------------------------

    cot_dong_cua = _tim_cot(
        du_lieu,
        [
            "Close",
            "close",
            "last",
            "price",
            "index",
            "Đóng cửa",
            "đóng cửa",
        ],
    )

    if cot_dong_cua is None:

        return {
            "loi": "Không tìm thấy cột điểm VN-INDEX."
        }

    du_lieu[
        cot_dong_cua
    ] = pd.to_numeric(
        du_lieu[
            cot_dong_cua
        ],
        errors="coerce",
    )

    du_lieu = du_lieu.dropna(
        subset=[
            cot_dong_cua
        ]
    )

    if du_lieu.empty:

        return {
            "loi": "VN-INDEX không có điểm hợp lệ."
        }

    diem = float(
        du_lieu[
            cot_dong_cua
        ].iloc[-1]
    )

    # --------------------------------------------------------
    # Thay đổi
    # --------------------------------------------------------

    if len(du_lieu) >= 2:

        diem_truoc = float(
            du_lieu[
                cot_dong_cua
            ].iloc[-2]
        )

        thay_doi = (
            diem
            - diem_truoc
        )

        if diem_truoc != 0:

            phan_tram = (
                thay_doi
                / diem_truoc
                * 100
            )

        else:

            phan_tram = 0.0

    else:

        diem_truoc = None
        thay_doi = None
        phan_tram = None

    # --------------------------------------------------------
    # Khối lượng
    # --------------------------------------------------------

    cot_khoi_luong = _tim_cot(
        du_lieu,
        [
            "Volume",
            "volume",
            "vol",
            "Khối lượng",
            "khối lượng",
        ],
    )

    khoi_luong = None

    if cot_khoi_luong is not None:

        du_lieu[
            cot_khoi_luong
        ] = pd.to_numeric(
            du_lieu[
                cot_khoi_luong
            ],
            errors="coerce",
        )

        khoi_luong = _so(
            du_lieu[
                cot_khoi_luong
            ].iloc[-1],
            None,
        )

    # --------------------------------------------------------
    # Giá trị giao dịch
    # --------------------------------------------------------

    cot_gia_tri = _tim_cot(
        du_lieu,
        [
            "Value",
            "value",
            "value_traded",
            "trading_value",
            "Giá trị",
            "giá trị",
        ],
    )

    gia_tri = None

    if cot_gia_tri is not None:

        du_lieu[
            cot_gia_tri
        ] = pd.to_numeric(
            du_lieu[
                cot_gia_tri
            ],
            errors="coerce",
        )

        gia_tri = _so(
            du_lieu[
                cot_gia_tri
            ].iloc[-1],
            None,
        )

    return {
        "diem": diem,
        "diem_truoc": diem_truoc,
        "thay_doi": thay_doi,
        "phan_tram": phan_tram,
        "khoi_luong": khoi_luong,
        "gia_tri": gia_tri,
        "du_lieu": du_lieu,
        "loi": None,
    }


# ============================================================
# HERO
# ============================================================

def render_dashboard():

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                BRIAN STOCK · INVESTMENT INTELLIGENCE
            </div>

            <h1>
                Góc nhìn dữ liệu cho nhà đầu tư
            </h1>

            <p>
                Dữ liệu thị trường thực, chỉ báo kỹ thuật,
                thanh khoản và tin tức được tải trực tiếp
                từ nguồn dữ liệu.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # VN-INDEX
    # ========================================================

    vn_index = _lay_vn_index()

    if vn_index.get("loi"):

        vn_diem = "—"
        vn_thay_doi = "—"
        vn_phan_tram = "—"
        vn_khoi_luong = "—"
        vn_gia_tri = "—"

    else:

        vn_diem = _diem_vn_index(
            vn_index.get(
                "diem"
            )
        )

        vn_thay_doi = _thay_doi_diem(
            vn_index.get(
                "thay_doi"
            )
        )

        vn_phan_tram = _thay_doi_phan_tram(
            vn_index.get(
                "phan_tram"
            )
        )

        vn_khoi_luong = _khoi_luong_co_phieu(
            vn_index.get(
                "khoi_luong"
            )
        )

        vn_gia_tri = _gia_tri_giao_dich(
            vn_index.get(
                "gia_tri"
            )
        )

    # ========================================================
    # THEO DÕI NHANH
    # ========================================================

    st.markdown(
        '<div class="section-title">📌 Theo dõi nhanh</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "VN-INDEX",
            vn_diem,
            "Điểm chỉ số thị trường",
        )

    with c2:

        metric_card(
            "Khối lượng",
            vn_khoi_luong,
            "Khối lượng giao dịch toàn thị trường",
        )

    with c3:

        metric_card(
            "Giá trị giao dịch",
            vn_gia_tri,
            "Tổng giá trị giao dịch",
        )

    with c4:

        metric_card(
            "Tin tức",
            "LIVE",
            "Nguồn tin thị trường mới",
        )

    # ========================================================
    # VN-INDEX CHI TIẾT
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 VN-INDEX</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Điểm",
            vn_diem,
        )

    with b:

        st.metric(
            "Thay đổi",
            vn_thay_doi,
        )

    with c:

        st.metric(
            "Thay đổi 1D",
            vn_phan_tram,
        )

    with d:

        st.metric(
            "Khối lượng",
            vn_khoi_luong,
        )

    if vn_index.get("loi"):

        st.warning(
            "VN-INDEX chưa tải được: "
            + str(
                vn_index[
                    "loi"
                ]
            )
        )

    # ========================================================
    # THEO DÕI CỔ PHIẾU
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Theo dõi cổ phiếu</div>',
        unsafe_allow_html=True,
    )

    ma_mac_dinh = st.session_state.get(
        "dashboard_symbol",
        "HPG",
    )

    ma_nhap = st.text_input(
        "Mã cổ phiếu",
        value=ma_mac_dinh,
        label_visibility="collapsed",
        placeholder=(
            "Nhập mã cổ phiếu, ví dụ HPG, MSR, VNM"
        ),
        key="dashboard_stock_input",
    )

    if st.button(
        "Tải dữ liệu",
        type="primary",
        key="dashboard_load_button",
    ):

        ma_sach = normalize_symbol(
            ma_nhap
        )

        if not ma_sach:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

        else:

            st.session_state[
                "dashboard_symbol"
            ] = ma_sach

            st.rerun()

    ma_dang_xem = normalize_symbol(
        st.session_state.get(
            "dashboard_symbol",
            ma_nhap,
        )
    )

    # ========================================================
    # DỮ LIỆU CỔ PHIẾU
    # ========================================================

    try:

        du_lieu = load_market_data(
            ma_dang_xem,
            "1y",
        )

    except Exception as loi:

        st.error(
            f"Không lấy được dữ liệu "
            f"{display_symbol(ma_dang_xem)}: "
            f"{loi}"
        )

        du_lieu = None

    if (
        du_lieu is not None
        and not du_lieu.empty
    ):

        anh_chup = market_snapshot(
            du_lieu
        )

        gia = _so(
            anh_chup.get(
                "price"
            ),
            None,
        )

        thay_doi_1d = _so(
            anh_chup.get(
                "change_1d"
            ),
            None,
        )

        rsi = _so(
            anh_chup.get(
                "rsi"
            ),
            None,
        )

        khoi_luong = _so(
            anh_chup.get(
                "volume"
            ),
            None,
        )

        sma20 = _so(
            anh_chup.get(
                "sma20"
            ),
            None,
        )

        sma50 = _so(
            anh_chup.get(
                "sma50"
            ),
            None,
        )

        macd = _so(
            anh_chup.get(
                "macd"
            ),
            None,
        )

        bien_dong = _so(
            anh_chup.get(
                "volatility20"
            ),
            None,
        )

        dong_cuoi = (
            du_lieu.iloc[-1]
        )

        atr14 = _so(
            dong_cuoi.get(
                "ATR14"
            ),
            None,
        )

        st.markdown(
            f"""
            <div class="section-title">
                📈 {html.escape(
                    display_symbol(
                        ma_dang_xem
                    )
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================
        # HÀNG 1
        # ====================================================

        a, b, c, d = st.columns(4)

        with a:

            st.metric(
                "Giá",
                _gia_co_phieu(
                    gia
                ),
            )

        with b:

            st.metric(
                "Thay đổi 1D",
                _thay_doi_phan_tram(
                    thay_doi_1d
                ),
            )

        with c:

            st.metric(
                "RSI",
                _rsi(
                    rsi
                ),
            )

        with d:

            st.metric(
                "Khối lượng",
                _khoi_luong_co_phieu(
                    khoi_luong
                ),
            )

        # ====================================================
        # HÀNG 2
        # ====================================================

        e, f, g, h = st.columns(4)

        with e:

            st.metric(
                "Trung bình 20 phiên",
                _gia_co_phieu(
                    sma20
                ),
            )

        with f:

            st.metric(
                "Trung bình 50 phiên",
                _gia_co_phieu(
                    sma50
                ),
            )

        with g:

            st.metric(
                "MACD",
                _macd(
                    macd
                ),
            )

        with h:

            st.metric(
                "Biến động 20 phiên",
                _bien_dong(
                    bien_dong
                ),
            )

        # ====================================================
        # HÀNG 3
        # ====================================================

        i, j, k, l = st.columns(4)

        with i:

            st.metric(
                "ATR 14 phiên",
                _atr(
                    atr14
                ),
            )

        with j:

            gia_tri_tb20 = _so(
                dong_cuoi.get(
                    "Volume_SMA20"
                ),
                None,
            )

            st.metric(
                "Khối lượng TB20",
                _khoi_luong_co_phieu(
                    gia_tri_tb20
                ),
            )

        with k:

            ty_le_khoi_luong = None

            if (
                khoi_luong is not None
                and gia_tri_tb20 is not None
                and gia_tri_tb20 != 0
            ):

                ty_le_khoi_luong = (
                    khoi_luong
                    / gia_tri_tb20
                )

            st.metric(
                "Khối lượng / TB20",
                (
                    f"{ty_le_khoi_luong:.2f} lần"
                    if ty_le_khoi_luong is not None
                    else "—"
                ),
            )

        with l:

            dong_mo = _so(
                dong_cuoi.get(
                    "Open"
                ),
                None,
            )

            st.metric(
                "Giá mở cửa",
                _gia_co_phieu(
                    dong_mo
                ),
            )

        # ====================================================
        # BIỂU ĐỒ
        # ====================================================

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

        except Exception as loi_bieu_do:

            st.warning(
                "Không thể hiển thị biểu đồ: "
                f"{loi_bieu_do}"
            )

    elif du_lieu is None:

        pass

    else:

        st.warning(
            f"Không tìm thấy dữ liệu thật cho "
            f"{display_symbol(ma_dang_xem)}."
        )

    # ========================================================
    # TIN MỚI
    # ========================================================

    st.markdown(
        '<div class="section-title">📰 Tin mới</div>',
        unsafe_allow_html=True,
    )

    try:

        tin_tuc = fetch_market_news(
            6
        )

    except Exception as loi:

        st.info(
            f"Chưa thể tải tin tức: {loi}"
        )

        tin_tuc = []

    if not tin_tuc:

        st.info(
            "Hiện chưa lấy được tin tức mới."
        )

    else:

        for tin in tin_tuc:

            tieu_de = html.escape(
                str(
                    tin.get(
                        "title",
                        "Không có tiêu đề",
                    )
                )
            )

            nguon = html.escape(
                str(
                    tin.get(
                        "source",
                        "Nguồn không xác định",
                    )
                )
            )

            thoi_gian = html.escape(
                str(
                    tin.get(
                        "published",
                        "",
                    )
                )
            )

            lien_ket = str(
                tin.get(
                    "link",
                    "",
                )
            ).strip()

            st.markdown(
                f"""
                <div class="news-card">
                    <div class="news-title">
                        {tieu_de}
                    </div>

                    <div class="news-meta">
                        {nguon} · {thoi_gian}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if lien_ket:

                st.markdown(
                    f"[Đọc bài ↗]({lien_ket})"
                )
