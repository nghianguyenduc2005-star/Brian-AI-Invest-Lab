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


def _lay_gia_tri(dong, cac_ten):
    for ten in cac_ten:
        if ten in dong.index:
            try:
                gia_tri = float(dong[ten])
                if pd.notna(gia_tri):
                    return gia_tri
            except Exception:
                pass

    return None


def _lay_vn_index():
    try:
        du_lieu = load_vnindex_data()

        if du_lieu is None or du_lieu.empty:
            return None

        du_lieu = du_lieu.copy()

        # Chuẩn hóa tên cột phổ biến
        anh_xa = {}

        for cot in du_lieu.columns:
            ten = str(cot).strip().lower()

            if ten in ["close", "last", "price", "index"]:
                anh_xa[cot] = "Close"

            elif ten in ["open"]:
                anh_xa[cot] = "Open"

            elif ten in ["high"]:
                anh_xa[cot] = "High"

            elif ten in ["low"]:
                anh_xa[cot] = "Low"

            elif ten in ["volume"]:
                anh_xa[cot] = "Volume"

            elif ten in ["change", "price_change"]:
                anh_xa[cot] = "Change"

            elif ten in ["change_percent", "percent_change"]:
                anh_xa[cot] = "ChangePercent"

        du_lieu = du_lieu.rename(
            columns=anh_xa
        )

        if "Close" not in du_lieu.columns:
            return None

        du_lieu["Close"] = pd.to_numeric(
            du_lieu["Close"],
            errors="coerce"
        )

        du_lieu = du_lieu.dropna(
            subset=["Close"]
        )

        if du_lieu.empty:
            return None

        dong_cuoi = du_lieu.iloc[-1]

        diem = float(
            dong_cuoi["Close"]
        )

        # Tự tính thay đổi từ hai phiên gần nhất
        thay_doi_diem = 0.0
        thay_doi_phan_tram = 0.0

        if len(du_lieu) >= 2:

            diem_truoc = float(
                du_lieu["Close"].iloc[-2]
            )

            thay_doi_diem = (
                diem - diem_truoc
            )

            if diem_truoc != 0:
                thay_doi_phan_tram = (
                    thay_doi_diem
                    / diem_truoc
                    * 100
                )

        if "Change" in du_lieu.columns:
            try:
                gia_tri = float(
                    dong_cuoi["Change"]
                )

                if pd.notna(gia_tri):
                    thay_doi_diem = gia_tri
            except Exception:
                pass

        if "ChangePercent" in du_lieu.columns:
            try:
                gia_tri = float(
                    dong_cuoi["ChangePercent"]
                )

                if pd.notna(gia_tri):
                    thay_doi_phan_tram = gia_tri
            except Exception:
                pass

        khoi_luong = None

        if "Volume" in du_lieu.columns:

            try:
                khoi_luong = float(
                    dong_cuoi["Volume"]
                )

            except Exception:
                khoi_luong = None

        return {
            "diem": diem,
            "thay_doi": thay_doi_diem,
            "thay_doi_phan_tram": thay_doi_phan_tram,
            "khoi_luong": khoi_luong,
            "du_lieu": du_lieu,
        }

    except Exception:
        return None


def render_dashboard():

    # ========================================================
    # HERO
    # ========================================================

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
            Dashboard nghiên cứu thị trường,
            cổ phiếu, tin tức và AI.
            Dữ liệu được tải khi cần,
            không dùng dữ liệu random.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # VN-INDEX + THANH KHOẢN
    # ========================================================

    thong_tin_vn_index = _lay_vn_index()

    if thong_tin_vn_index is not None:

        diem_vn_index = (
            thong_tin_vn_index["diem"]
        )

        thay_doi_vn_index = (
            thong_tin_vn_index["thay_doi"]
        )

        thay_doi_vn_index_phan_tram = (
            thong_tin_vn_index[
                "thay_doi_phan_tram"
            ]
        )

        khoi_luong_vn_index = (
            thong_tin_vn_index[
                "khoi_luong"
            ]
        )

        gia_tri_vn_index = (
            f"{diem_vn_index:,.2f}"
        )

        gia_tri_thay_doi = (
            f"{thay_doi_vn_index:+,.2f}"
        )

        gia_tri_phan_tram = (
            f"{thay_doi_vn_index_phan_tram:+.2f}%"
        )

        if khoi_luong_vn_index is not None:

            if khoi_luong_vn_index >= 1_000_000_000:
                thanh_khoan_text = (
                    f"{khoi_luong_vn_index / 1_000_000_000:.2f} tỷ"
                )

            elif khoi_luong_vn_index >= 1_000_000:
                thanh_khoan_text = (
                    f"{khoi_luong_vn_index / 1_000_000:.2f} triệu"
                )

            elif khoi_luong_vn_index >= 1_000:
                thanh_khoan_text = (
                    f"{khoi_luong_vn_index / 1_000:.2f} nghìn"
                )

            else:
                thanh_khoan_text = (
                    f"{khoi_luong_vn_index:,.0f}"
                )

        else:
            thanh_khoan_text = "—"

    else:

        gia_tri_vn_index = "—"
        gia_tri_thay_doi = "—"
        gia_tri_phan_tram = "—"
        thanh_khoan_text = "—"

    # ========================================================
    # QUICK OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">📌 Theo dõi nhanh</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "VN-INDEX",
            gia_tri_vn_index,
            "Chỉ số thị trường",
        )

    with c2:
        metric_card(
            "Thanh khoản",
            thanh_khoan_text,
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
            gia_tri_vn_index,
        )

    with b:
        st.metric(
            "Thay đổi",
            gia_tri_thay_doi,
        )

    with c:
        st.metric(
            "1D",
            gia_tri_phan_tram,
        )

    with d:
        st.metric(
            "Thanh khoản",
            thanh_khoan_text,
        )

    # ========================================================
    # THEO DÕI CỔ PHIẾU
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Theo dõi cổ phiếu</div>',
        unsafe_allow_html=True,
    )

    gia_tri_mac_dinh = st.session_state.get(
        "dashboard_input",
        "HPG",
    )

    ma_nhap = st.text_input(
        "Mã cổ phiếu",
        value=gia_tri_mac_dinh,
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

        if ma_sach:

            st.session_state.dashboard_symbol = (
                ma_sach
            )

            st.session_state.dashboard_input = (
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

        if du_lieu is None or du_lieu.empty:

            st.warning(
                f"Không lấy được dữ liệu "
                f"{ma_dang_xem}."
            )

        else:

            dong_cuoi = du_lieu.iloc[-1]

            gia = float(
                dong_cuoi["Close"]
            )

            if len(du_lieu) >= 2:

                gia_truoc = float(
                    du_lieu["Close"].iloc[-2]
                )

                thay_doi_1d = (
                    gia / gia_truoc - 1
                ) * 100

            else:
                thay_doi_1d = 0.0

            rsi_value = (
                float(dong_cuoi["RSI"])
                if pd.notna(
                    dong_cuoi["RSI"]
                )
                else None
            )

            khoi_luong = (
                float(dong_cuoi["Volume"])
                if pd.notna(
                    dong_cuoi["Volume"]
                )
                else 0
            )

            st.markdown(
                f"""
                <div class="section-title">
                  📈 {html.escape(ma_dang_xem)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            a, b, c, d = st.columns(4)

            with a:
                st.metric(
                    "Giá",
                    f"{gia:,.0f}",
                )

            with b:
                st.metric(
                    "1D",
                    f"{thay_doi_1d:+.2f}%",
                )

            with c:
                if rsi_value is not None:
                    st.metric(
                        "RSI",
                        f"{rsi_value:.1f}",
                    )
                else:
                    st.metric(
                        "RSI",
                        "—",
                    )

            with d:
                st.metric(
                    "Volume",
                    f"{khoi_luong:,.0f}",
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

            except Exception as loi_bieu_do:

                st.warning(
                    "Không thể hiển thị biểu đồ: "
                    f"{loi_bieu_do}"
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
                )

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

    except Exception as loi:

        st.info(
            f"Chưa thể tải tin tức: {loi}"
        )
