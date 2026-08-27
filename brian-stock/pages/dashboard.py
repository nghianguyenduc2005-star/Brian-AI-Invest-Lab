from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import metric_card
from components.charts import price_volume_chart

from data.market import (
    display_symbol,
    load_latest_price,
    load_market_data,
    load_vnindex_data,
    load_vnindex_market_summary,
    market_snapshot,
    normalize_symbol,
)

from data.news import fetch_market_news


# ============================================================
# TIỆN ÍCH
# ============================================================

def _so(
    value,
    mac_dinh=None,
):
    try:
        value = float(value)

        if pd.isna(value):
            return mac_dinh

        return value

    except Exception:
        return mac_dinh


def _tim_cot(
    df,
    cac_ten,
):
    if (
        df is None
        or df.empty
    ):
        return None

    ban_do = {
        str(cot).strip().lower(): cot
        for cot in df.columns
    }

    for ten in cac_ten:

        cot = ban_do.get(
            str(ten).strip().lower()
        )

        if cot is not None:
            return cot

    return None


# ============================================================
# ĐỊNH DẠNG KHỐI LƯỢNG
# ============================================================

def _dinh_dang_khoi_luong(
    value,
):
    value = _so(
        value,
        None,
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu cổ phiếu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn cổ phiếu"
        )

    return (
        f"{value:,.0f} cổ phiếu"
    )


# ============================================================
# ĐỊNH DẠNG GIÁ TRỊ GIAO DỊCH
# ============================================================

def _dinh_dang_gia_tri(
    value,
):
    value = _so(
        value,
        None,
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:

        return (
            f"{value / 1_000_000_000_000:.2f} nghìn tỷ đồng"
        )

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ đồng"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu đồng"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn đồng"
        )

    return (
        f"{value:,.0f} đồng"
    )


# ============================================================
# ĐỊNH DẠNG GIÁ
# ============================================================

def _dinh_dang_gia(
    value,
):
    value = _so(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:,.0f} đồng/cổ phiếu"
    )


# ============================================================
# ĐỊNH DẠNG RSI
# ============================================================

def _dinh_dang_rsi(
    value,
):
    value = _so(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:.1f} điểm"
    )


# ============================================================
# ĐỊNH DẠNG MACD
# ============================================================

def _dinh_dang_macd(
    value,
):
    value = _so(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:.3f}"
    )


# ============================================================
# ĐỊNH DẠNG %
# ============================================================

def _dinh_dang_phan_tram(
    value,
):
    value = _so(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:+.2f}%"
    )


# ============================================================
# VN-INDEX
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
            "loi": (
                "Nguồn VN-INDEX trả về dữ liệu rỗng."
            )
        }

    du_lieu = du_lieu.copy()

    # ========================================================
    # ĐIỂM
    # ========================================================

    cot_diem = _tim_cot(
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

    if cot_diem is None:

        return {
            "loi": (
                "Không tìm thấy cột điểm VN-INDEX."
            )
        }

    du_lieu[cot_diem] = pd.to_numeric(
        du_lieu[cot_diem],
        errors="coerce",
    )

    du_lieu = du_lieu.dropna(
        subset=[
            cot_diem
        ]
    )

    if du_lieu.empty:

        return {
            "loi": (
                "VN-INDEX không có điểm hợp lệ."
            )
        }

    diem = float(
        du_lieu[
            cot_diem
        ].iloc[-1]
    )

    # ========================================================
    # THAY ĐỔI
    # ========================================================

    if len(
        du_lieu
    ) >= 2:

        diem_truoc = float(
            du_lieu[
                cot_diem
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

        thay_doi = None
        phan_tram = None

    # ========================================================
    # KHỐI LƯỢNG
    # ========================================================

    cot_khoi_luong = _tim_cot(
        du_lieu,
        [
            "Volume",
            "volume",
            "Vol",
            "total_volume",
            "match_volume",
            "matchvolume",
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

    # ========================================================
    # GIÁ TRỊ GIAO DỊCH
    #
    # Ưu tiên:
    #   1. cột Value trong OHLCV nếu có
    #   2. index.summary()
    #   3. index.trade_history()
    #
    # Không dùng điểm VN-INDEX × khối lượng.
    # ========================================================

    cot_gia_tri = _tim_cot(
        du_lieu,
        [
            "Value",
            "value",
            "ValueTraded",
            "value_traded",
            "trading_value",
            "traded_value",
            "match_value",
            "matchvalue",
            "turnover",
            "Turnover",
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

    # --------------------------------------------------------
    # Nếu OHLCV không có Value thì gọi summary riêng.
    # --------------------------------------------------------

    if gia_tri is None:

        try:

            summary = (
                load_vnindex_market_summary()
            )

        except Exception:

            summary = {}

        gia_tri_summary = summary.get(
            "gia_tri"
        )

        if gia_tri_summary is not None:

            gia_tri = _so(
                gia_tri_summary,
                None,
            )

        khoi_luong_summary = summary.get(
            "khoi_luong"
        )

        if (
            khoi_luong_summary is not None
            and khoi_luong is None
        ):

            khoi_luong = _so(
                khoi_luong_summary,
                None,
            )

    return {
        "diem": diem,
        "thay_doi": thay_doi,
        "phan_tram": phan_tram,
        "khoi_luong": khoi_luong,
        "gia_tri": gia_tri,
        "du_lieu": du_lieu,
        "loi": None,
    }


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard():

    # ========================================================
    # HERO
    # ========================================================

    st.caption(
        "BRIAN STOCK · INVESTMENT INTELLIGENCE"
    )

    st.title(
        "Góc nhìn dữ liệu cho nhà đầu tư"
    )

    st.write(
        "Dashboard nghiên cứu thị trường, "
        "cổ phiếu, tin tức và AI. "
        "Dữ liệu được tải trực tiếp khi cần, "
        "không dùng dữ liệu ngẫu nhiên."
    )

    # ========================================================
    # VN-INDEX
    # ========================================================

    vn_index = _lay_vn_index()

    if vn_index.get(
        "loi"
    ):

        vn_diem = "—"
        vn_thay_doi = "—"
        vn_1d = "—"
        vn_khoi_luong = "—"
        vn_gia_tri = "—"

    else:

        diem = _so(
            vn_index.get(
                "diem"
            )
        )

        thay_doi = _so(
            vn_index.get(
                "thay_doi"
            )
        )

        phan_tram = _so(
            vn_index.get(
                "phan_tram"
            )
        )

        vn_diem = (
            f"{diem:,.2f} điểm"
            if diem is not None
            else "—"
        )

        vn_thay_doi = (
            f"{thay_doi:+,.2f} điểm"
            if thay_doi is not None
            else "—"
        )

        vn_1d = (
            f"{phan_tram:+.2f}%"
            if phan_tram is not None
            else "—"
        )

        vn_khoi_luong = (
            _dinh_dang_khoi_luong(
                vn_index.get(
                    "khoi_luong"
                )
            )
        )

        vn_gia_tri = (
            _dinh_dang_gia_tri(
                vn_index.get(
                    "gia_tri"
                )
            )
        )

    # ========================================================
    # THEO DÕI NHANH
    # ========================================================

    st.subheader(
        "📌 Theo dõi nhanh"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

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

    st.subheader(
        "📊 VN-INDEX"
    )

    a, b, c, d = st.columns(
        4
    )

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
            vn_1d,
        )

    with d:

        st.metric(
            "Khối lượng",
            vn_khoi_luong,
        )

    if vn_index.get(
        "loi"
    ):

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

    st.subheader(
        "📈 Theo dõi cổ phiếu"
    )

    ma_mac_dinh = (
        st.session_state.get(
            "dashboard_symbol",
            "HPG",
        )
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
        width="content",
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

        volume_tb20 = _so(
            dong_cuoi.get(
                "Volume_SMA20"
            ),
            None,
        )

        # ====================================================
        # TIÊU ĐỀ MÃ
        # ====================================================

        st.subheader(
            f"📈 {display_symbol(ma_dang_xem)}"
        )

        # ====================================================
        # THÔNG TIN
        # ====================================================

        a, b, c, d = st.columns(
            4
        )

        with a:

            st.metric(
                "Giá",
                _dinh_dang_gia(
                    gia
                ),
            )

        with b:

            st.metric(
                "Thay đổi 1D",
                _dinh_dang_phan_tram(
                    thay_doi_1d
                ),
            )

        with c:

            st.metric(
                "RSI",
                _dinh_dang_rsi(
                    rsi
                ),
            )

        with d:

            st.metric(
                "Khối lượng",
                _dinh_dang_khoi_luong(
                    khoi_luong
                ),
            )

        # ====================================================
        # CHỈ BÁO
        # ====================================================

        e, f, g, h = st.columns(
            4
        )

        with e:

            st.metric(
                "Trung bình 20 phiên",
                _dinh_dang_gia(
                    sma20
                ),
            )

        with f:

            st.metric(
                "Trung bình 50 phiên",
                _dinh_dang_gia(
                    sma50
                ),
            )

        with g:

            st.metric(
                "MACD",
                _dinh_dang_macd(
                    macd
                ),
            )

        with h:

            st.metric(
                "Biến động 20 phiên",
                (
                    f"{bien_dong:.2f}%"
                    if bien_dong is not None
                    else "—"
                ),
            )

        # ====================================================
        # CHỈ BÁO BỔ SUNG
        # ====================================================

        i, j, k, l = st.columns(
            4
        )

        with i:

            st.metric(
                "ATR 14 phiên",
                _dinh_dang_gia(
                    atr14
                ),
            )

        with j:

            st.metric(
                "Khối lượng TB20",
                _dinh_dang_khoi_luong(
                    volume_tb20
                ),
            )

        with k:

            ty_le_volume = None

            if (
                khoi_luong is not None
                and volume_tb20 is not None
                and volume_tb20 != 0
            ):

                ty_le_volume = (
                    khoi_luong
                    / volume_tb20
                )

            st.metric(
                "Khối lượng / TB20",
                (
                    f"{ty_le_volume:.2f} lần"
                    if ty_le_volume is not None
                    else "—"
                ),
            )

        with l:

            gia_mo = _so(
                dong_cuoi.get(
                    "Open"
                ),
                None,
            )

            st.metric(
                "Giá mở cửa",
                _dinh_dang_gia(
                    gia_mo
                ),
            )

        # ====================================================
        # BIỂU ĐỒ
        # ====================================================

        st.subheader(
            "📊 Biểu đồ kỹ thuật"
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

        except Exception as loi_bieu_do:

            st.warning(
                "Không thể hiển thị biểu đồ: "
                f"{loi_bieu_do}"
            )

    elif du_lieu is not None:

        st.warning(
            f"Không tìm thấy dữ liệu thật cho "
            f"{display_symbol(ma_dang_xem)}."
        )

    # ========================================================
    # TIN MỚI
    # ========================================================

    st.subheader(
        "📰 Tin mới"
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

            tieu_de = str(
                tin.get(
                    "title",
                    "Không có tiêu đề",
                )
            ).strip()

            nguon = str(
                tin.get(
                    "source",
                    "Nguồn không xác định",
                )
            ).strip()

            thoi_gian = str(
                tin.get(
                    "published",
                    "",
                )
            ).strip()

            lien_ket = str(
                tin.get(
                    "link",
                    "",
                )
            ).strip()

            with st.container(
                border=True,
            ):

                st.markdown(
                    f"**{tieu_de}**"
                )

                if (
                    nguon
                    or thoi_gian
                ):

                    if (
                        nguon
                        and thoi_gian
                    ):

                        st.caption(
                            f"{nguon} · {thoi_gian}"
                        )

                    elif nguon:

                        st.caption(
                            nguon
                        )

                    else:

                        st.caption(
                            thoi_gian
                        )

                if lien_ket:

                    st.markdown(
                        f"[Đọc bài ↗]({lien_ket})"
                    )
