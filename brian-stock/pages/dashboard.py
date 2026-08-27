import html

import pandas as pd
import streamlit as st

from components.cards import metric_card
from components.charts import price_volume_chart
from data.market import (
    normalize_symbol,
    load_market_data,
    load_vnindex_data,
)
from data.news import fetch_market_news


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def _so(gia_tri, mac_dinh=None):
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

    for cot in du_lieu.columns:

        ten_cot = str(
            cot
        ).strip().lower()

        for ten_tim in danh_sach_ten:

            if ten_cot == str(
                ten_tim
            ).strip().lower():

                return cot

    return None


def _dinh_dang_khoi_luong(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        0,
    )

    if gia_tri >= 1_000_000_000:
        return (
            f"{gia_tri / 1_000_000_000:.2f} tỷ"
        )

    if gia_tri >= 1_000_000:
        return (
            f"{gia_tri / 1_000_000:.2f} triệu"
        )

    if gia_tri >= 1_000:
        return (
            f"{gia_tri / 1_000:.2f} nghìn"
        )

    return f"{gia_tri:,.0f}"


# ============================================================
# LẤY VN-INDEX
# ============================================================

def _lay_vn_index():

    try:

        du_lieu = load_vnindex_data()

        if (
            du_lieu is None
            or du_lieu.empty
        ):
            return None, "Nguồn VN-INDEX trả về dữ liệu rỗng."

        du_lieu = du_lieu.copy()

        # ----------------------------------------------------
        # Tìm cột đóng cửa
        # ----------------------------------------------------

        cot_dong_cua = _tim_cot(
            du_lieu,
            [
                "Close",
                "close",
                "Đóng cửa",
                "đóng cửa",
                "last",
                "price",
            ],
        )

        if cot_dong_cua is None:
            return (
                None,
                "Không tìm thấy cột giá VN-INDEX."
            )

        du_lieu[cot_dong_cua] = pd.to_numeric(
            du_lieu[cot_dong_cua],
            errors="coerce",
        )

        du_lieu = du_lieu.dropna(
            subset=[
                cot_dong_cua
            ]
        )

        if du_lieu.empty:
            return (
                None,
                "Không còn giá VN-INDEX hợp lệ."
            )

        # ----------------------------------------------------
        # Điểm hiện tại
        # ----------------------------------------------------

        diem_hien_tai = float(
            du_lieu[
                cot_dong_cua
            ].iloc[-1]
        )

        # ----------------------------------------------------
        # Thay đổi
        # ----------------------------------------------------

        thay_doi_diem = 0.0
        thay_doi_phan_tram = 0.0

        if len(
            du_lieu
        ) >= 2:

            diem_truoc = float(
                du_lieu[
                    cot_dong_cua
                ].iloc[-2]
            )

            thay_doi_diem = (
                diem_hien_tai
                - diem_truoc
            )

            if diem_truoc != 0:

                thay_doi_phan_tram = (
                    thay_doi_diem
                    / diem_truoc
                    * 100
                )

        # ----------------------------------------------------
        # Nếu nguồn đã có Change
        # ----------------------------------------------------

        cot_thay_doi = _tim_cot(
            du_lieu,
            [
                "Change",
                "change",
                "Thay đổi",
            ],
        )

        if cot_thay_doi is not None:

            gia_tri = _so(
                du_lieu[
                    cot_thay_doi
                ].iloc[-1],
                None,
            )

            if gia_tri is not None:
                thay_doi_diem = gia_tri

        # ----------------------------------------------------
        # Nếu nguồn đã có ChangePercent
        # ----------------------------------------------------

        cot_phan_tram = _tim_cot(
            du_lieu,
            [
                "ChangePercent",
                "change_percent",
                "Percent",
                "percent",
                "Phần trăm thay đổi",
            ],
        )

        if cot_phan_tram is not None:

            gia_tri = _so(
                du_lieu[
                    cot_phan_tram
                ].iloc[-1],
                None,
            )

            if gia_tri is not None:

                # Có nguồn trả 0.65 thay vì 0.65%
                if abs(gia_tri) < 1:
                    thay_doi_phan_tram = (
                        gia_tri
                        * 100
                    )
                else:
                    thay_doi_phan_tram = gia_tri

        # ----------------------------------------------------
        # Khối lượng
        # ----------------------------------------------------

        cot_khoi_luong = _tim_cot(
            du_lieu,
            [
                "Volume",
                "volume",
                "Khối lượng",
            ],
        )

        khoi_luong = None

        if cot_khoi_luong is not None:

            khoi_luong = _so(
                du_lieu[
                    cot_khoi_luong
                ].iloc[-1],
                None,
            )

        return (
            {
                "diem": diem_hien_tai,
                "thay_doi": thay_doi_diem,
                "phan_tram": thay_doi_phan_tram,
                "khoi_luong": khoi_luong,
                "du_lieu": du_lieu,
            },
            None,
        )

    except Exception as loi:

        return (
            None,
            str(loi),
        )


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard():

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        '<div class="hero">'
        '<div class="eyebrow">BRIAN STOCK · INVESTMENT INTELLIGENCE</div>'
        '<h1>Góc nhìn dữ liệu cho nhà đầu tư</h1>'
        '<p>Dashboard nghiên cứu thị trường, cổ phiếu, tin tức và AI. '
        'Dữ liệu được tải trực tiếp khi cần, không dùng dữ liệu ngẫu nhiên.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # VN-INDEX
    # ========================================================

    vn_index, loi_vn_index = _lay_vn_index()

    if vn_index is None:

        vn_diem = "—"
        vn_thay_doi = "—"
        vn_1d = "—"
        vn_khoi_luong = "—"

    else:

        vn_diem = (
            f"{vn_index['diem']:,.2f}"
        )

        vn_thay_doi = (
            f"{vn_index['thay_doi']:+,.2f}"
        )

        vn_1d = (
            f"{vn_index['phan_tram']:+.2f}%"
        )

        if vn_index[
            "khoi_luong"
        ] is None:

            vn_khoi_luong = "—"

        else:

            vn_khoi_luong = (
                _dinh_dang_khoi_luong(
                    vn_index[
                        "khoi_luong"
                    ]
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
            "Chỉ số thị trường",
        )

    with c2:
        metric_card(
            "Thanh khoản",
            vn_khoi_luong,
            "Khối lượng giao dịch",
        )

    with c3:
        metric_card(
            "Tin tức",
            "LIVE",
            "Google News + nguồn Việt Nam",
        )

    with c4:
        metric_card(
            "AI",
            "READY",
            "Chỉ gọi khi yêu cầu",
        )

    # ========================================================
    # THÔNG BÁO LỖI VN-INDEX
    # ========================================================

    if (
        vn_index is None
        and loi_vn_index
    ):

        with st.expander(
            "Chi tiết lỗi VN-INDEX",
            expanded=False,
        ):
            st.write(
                loi_vn_index
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
            "1D",
            vn_1d,
        )

    with d:
        st.metric(
            "Thanh khoản",
            vn_khoi_luong,
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
        placeholder="Nhập mã cổ phiếu, ví dụ HPG",
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

            st.session_state.dashboard_symbol = (
                ma_sach
            )

            st.rerun()

    ma_dang_xem = st.session_state.get(
        "dashboard_symbol",
        "HPG",
    )

    # ========================================================
    # DỮ LIỆU CỔ PHIẾU
    # ========================================================

    try:

        du_lieu = load_market_data(
            ma_dang_xem,
            "1y",
        )

        if (
            du_lieu is None
            or du_lieu.empty
        ):

            st.warning(
                f"Không tìm thấy dữ liệu thật "
                f"cho {ma_dang_xem}."
            )

        else:

            dong_cuoi = du_lieu.iloc[-1]

            gia = _so(
                dong_cuoi.get("Close"),
                None,
            )

            # ------------------------------------------------
            # 1D
            # ------------------------------------------------

            thay_doi_1d = None

            if len(
                du_lieu
            ) >= 2:

                gia_truoc = _so(
                    du_lieu[
                        "Close"
                    ].iloc[-2],
                    None,
                )

                if (
                    gia is not None
                    and gia_truoc is not None
                    and gia_truoc != 0
                ):

                    thay_doi_1d = (
                        gia
                        / gia_truoc
                        - 1
                    ) * 100

            # ------------------------------------------------
            # RSI
            # ------------------------------------------------

            rsi_value = _so(
                dong_cuoi.get(
                    "RSI"
                ),
                None,
            )

            # ------------------------------------------------
            # Volume
            # ------------------------------------------------

            khoi_luong = _so(
                dong_cuoi.get(
                    "Volume"
                ),
                0,
            )

            # ------------------------------------------------
            # Tiêu đề
            # ------------------------------------------------

            st.markdown(
                f'<div class="section-title">📈 {html.escape(str(ma_dang_xem))}</div>',
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # BẢNG CHÍNH
            # ------------------------------------------------

            a, b, c, d = st.columns(4)

            with a:

                st.metric(
                    "Giá",
                    (
                        f"{gia:,.0f}"
                        if gia is not None
                        else "—"
                    ),
                )

            with b:

                st.metric(
                    "1D",
                    (
                        f"{thay_doi_1d:+.2f}%"
                        if thay_doi_1d is not None
                        else "—"
                    ),
                )

            with c:

                st.metric(
                    "RSI",
                    (
                        f"{rsi_value:.1f}"
                        if rsi_value is not None
                        else "—"
                    ),
                )

            with d:

                st.metric(
                    "Volume",
                    f"{khoi_luong:,.0f}",
                )

            # ------------------------------------------------
            # CHỈ BÁO
            # ------------------------------------------------

            e, f, g, h = st.columns(4)

            sma20 = _so(
                dong_cuoi.get(
                    "SMA20"
                ),
                None,
            )

            sma50 = _so(
                dong_cuoi.get(
                    "SMA50"
                ),
                None,
            )

            macd = _so(
                dong_cuoi.get(
                    "MACD"
                ),
                None,
            )

            bien_dong = _so(
                dong_cuoi.get(
                    "Volatility20"
                ),
                None,
            )

            with e:

                st.metric(
                    "SMA20",
                    (
                        f"{sma20:,.0f}"
                        if sma20 is not None
                        else "—"
                    ),
                )

            with f:

                st.metric(
                    "SMA50",
                    (
                        f"{sma50:,.0f}"
                        if sma50 is not None
                        else "—"
                    ),
                )

            with g:

                st.metric(
                    "MACD",
                    (
                        f"{macd:,.3f}"
                        if macd is not None
                        else "—"
                    ),
                )

            with h:

                st.metric(
                    "Biến động 20D",
                    (
                        f"{bien_dong:.2f}%"
                        if bien_dong is not None
                        else "—"
                    ),
                )

            # ------------------------------------------------
            # BIỂU ĐỒ
            # ------------------------------------------------

            try:

                bieu_do = price_volume_chart(
                    du_lieu
                )

                if bieu_do is not None:

                    st.plotly_chart(
                        bieu_do,
                        width="stretch",
                        config={
                            "displaylogo": False
                        },
                    )

            except Exception as loi:

                st.warning(
                    "Không thể hiển thị biểu đồ: "
                    f"{loi}"
                )

    except Exception as loi:

        st.warning(
            f"Không tải được dữ liệu "
            f"{ma_dang_xem}. "
            f"Chi tiết: {loi}"
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

                # QUAN TRỌNG:
                # HTML phải viết sát trái, không thụt đầu dòng.
                st.markdown(
                    f'<div class="news-card">'
                    f'<div class="news-title">{tieu_de}</div>'
                    f'<div class="news-meta">{nguon} · {thoi_gian}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if lien_ket:

                    st.markdown(
                        f"[Đọc bài ↗]({lien_ket})"
                    )

    except Exception as loi:

        st.info(
            f"Chưa thể tải tin tức: {loi}"
        )
