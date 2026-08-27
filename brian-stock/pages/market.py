from __future__ import annotations

import pandas as pd
import streamlit as st

from data.market_overview import (
    lay_market_overview,
    loc_bang_gia,
)


# ============================================================
# ĐỊNH DẠNG
# ============================================================

def _gia(value):
    try:
        value = float(value)

        if pd.isna(value):
            return "—"

        return f"{value:,.2f}"

    except Exception:
        return "—"


def _phan_tram(value):
    try:
        value = float(value)

        if pd.isna(value):
            return "—"

        return f"{value:+.2f}%"

    except Exception:
        return "—"


def _so_luong(value):
    try:
        value = float(value)

        if pd.isna(value):
            return "—"

        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"

        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} triệu cổ phiếu"

        if value >= 1_000:
            return f"{value / 1_000:.2f} nghìn cổ phiếu"

        return f"{value:,.0f} cổ phiếu"

    except Exception:
        return "—"


def _gia_tri(value):
    try:
        value = float(value)

        if pd.isna(value):
            return "—"

        if value >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f} nghìn tỷ đồng"

        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f} tỷ đồng"

        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} triệu đồng"

        if value >= 1_000:
            return f"{value / 1_000:.2f} nghìn đồng"

        return f"{value:,.0f} đồng"

    except Exception:
        return "—"


def _mau_trang_thai(value):
    try:
        value = float(value)

        if pd.isna(value):
            return "🟡"

        if value > 0.05:
            return "🟢"

        if value < -0.05:
            return "🔴"

        return "🟡"

    except Exception:
        return "🟡"


# ============================================================
# BẢNG DỮ LIỆU HIỂN THỊ
# ============================================================

def _tao_bang_hien_thi(df):

    if df is None or df.empty:
        return pd.DataFrame()

    bang = df.copy()

    bang["Trạng thái"] = (
        bang["thay_doi_pct"]
        .apply(_mau_trang_thai)
    )

    bang["Mã"] = bang["ma"]

    bang["Sàn"] = (
        bang["san"]
        .replace("", "—")
    )

    bang["Ngành"] = (
        bang["ten_nganh"]
        .replace("", "—")
    )

    bang["Giá"] = (
        bang["gia"]
        .apply(_gia)
    )

    bang["Thay đổi"] = (
        bang["thay_doi_pct"]
        .apply(_phan_tram)
    )

    bang["Khối lượng"] = (
        bang["khoi_luong"]
        .apply(_so_luong)
    )

    bang["Giá trị giao dịch"] = (
        bang["gia_tri_giao_dich"]
        .apply(_gia_tri)
    )

    bang["Tham chiếu"] = (
        bang["gia_tham_chieu"]
        .apply(_gia)
    )

    return bang[
        [
            "Trạng thái",
            "Mã",
            "Sàn",
            "Ngành",
            "Giá",
            "Thay đổi",
            "Khối lượng",
            "Giá trị giao dịch",
            "Tham chiếu",
        ]
    ].copy()


# ============================================================
# TRANG THỊ TRƯỜNG
# ============================================================

def render_market():

    # ========================================================
    # TIÊU ĐỀ
    # ========================================================

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                BRIAN STOCK · MARKET OVERVIEW
            </div>

            <h1>
                Thị trường
            </h1>

            <p>
                Bảng giá toàn thị trường được cập nhật
                tự động. Dữ liệu được làm mới theo chu kỳ
                60 giây và không sử dụng dữ liệu giả.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # TẢI DỮ LIỆU
    # ========================================================

    try:

        du_lieu_tong = (
            lay_market_overview()
        )

    except Exception as loi:

        st.error(
            f"Không thể tải bảng giá thị trường: {loi}"
        )

        return

    bang_gia = (
        du_lieu_tong.get(
            "bang_gia",
            pd.DataFrame(),
        )
    )

    if (
        bang_gia is None
        or bang_gia.empty
    ):

        st.warning(
            "Nguồn dữ liệu chưa trả về bảng giá."
        )

        return

    # ========================================================
    # THÔNG TIN TỔNG QUAN
    # ========================================================

    thong_ke = (
        du_lieu_tong.get(
            "thong_ke",
            {},
        )
    )

    tam_ly = du_lieu_tong.get(
        "tam_ly"
    )

    nhan_tam_ly = du_lieu_tong.get(
        "nhan_tam_ly",
        "Trung tính",
    )

    nguon = (
        du_lieu_tong.get(
            "nguon",
            {},
        )
    )

    # ========================================================
    # HEADER BẢNG GIÁ
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Bảng giá toàn thị trường'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # THỐNG KÊ NHANH
    # ========================================================

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Số mã có dữ liệu",
            f"{thong_ke.get('co_du_lieu_gia', 0):,} mã",
        )

    with b:

        st.metric(
            "Tăng",
            f"{thong_ke.get('tang', 0):,} mã",
        )

    with c:

        st.metric(
            "Đứng giá",
            f"{thong_ke.get('dung_gia', 0):,} mã",
        )

    with d:

        st.metric(
            "Giảm",
            f"{thong_ke.get('giam', 0):,} mã",
        )

    # ========================================================
    # BỘ LỌC
    # ========================================================

    st.markdown(
        "#### 🔎 Bộ lọc"
    )

    col_1, col_2, col_3, col_4 = st.columns(
        [2.2, 1, 1.5, 1]
    )

    with col_1:

        tu_khoa = st.text_input(
            "Tìm mã hoặc tên doanh nghiệp",
            placeholder=(
                "Ví dụ: HPG, MSR, Vinamilk..."
            ),
            key="thi_truong_tu_khoa",
        )

    with col_2:

        cac_san = [
            "Tất cả"
        ]

        if "san" in bang_gia.columns:

            cac_san += [
                x
                for x
                in sorted(
                    bang_gia["san"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                if x
            ]

        san = st.selectbox(
            "Sàn",
            cac_san,
            key="thi_truong_san",
        )

    with col_3:

        cac_nganh = [
            "Tất cả"
        ]

        if "ten_nganh" in bang_gia.columns:

            cac_nganh += [
                x
                for x
                in sorted(
                    bang_gia["ten_nganh"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                if x
            ]

        nganh = st.selectbox(
            "Ngành",
            cac_nganh,
            key="thi_truong_nganh",
        )

    with col_4:

        huong = st.selectbox(
            "Trạng thái",
            [
                "Tất cả",
                "Tăng",
                "Đứng giá",
                "Giảm",
            ],
            key="thi_truong_huong",
        )

    # ========================================================
    # LỌC
    # ========================================================

    bang_loc = loc_bang_gia(
        bang_gia,
        san=san,
        tu_khoa=tu_khoa,
        nganh=nganh,
        huong=huong,
    )

    # ========================================================
    # SẮP XẾP
    # ========================================================

    cot_1, cot_2 = st.columns(
        [2, 1]
    )

    with cot_1:

        kieu_sap_xep = st.selectbox(
            "Sắp xếp theo",
            [
                "Mã cổ phiếu",
                "Tăng/giảm (%)",
                "Khối lượng",
                "Giá trị giao dịch",
                "Giá",
            ],
            key="thi_truong_sap_xep",
        )

    with cot_2:

        thu_tu = st.selectbox(
            "Thứ tự",
            [
                "Giảm dần",
                "Tăng dần",
            ],
            key="thi_truong_thu_tu",
        )

    anh_xa_sap_xep = {
        "Mã cổ phiếu": "ma",
        "Tăng/giảm (%)": "thay_doi_pct",
        "Khối lượng": "khoi_luong",
        "Giá trị giao dịch": "gia_tri_giao_dich",
        "Giá": "gia",
    }

    cot_sap_xep = anh_xa_sap_xep[
        kieu_sap_xep
    ]

    if cot_sap_xep in bang_loc.columns:

        bang_loc = bang_loc.sort_values(
            cot_sap_xep,
            ascending=(
                thu_tu == "Tăng dần"
            ),
            na_position="last",
        )

    # ========================================================
    # PHÂN TRANG NHẸ
    # ========================================================

    tong_so_dong = len(
        bang_loc
    )

    st.caption(
        f"Đang hiển thị {tong_so_dong:,} mã"
    )

    # ========================================================
    # BẢNG
    # ========================================================

    bang_hien_thi = _tao_bang_hien_thi(
        bang_loc
    )

    if bang_hien_thi.empty:

        st.info(
            "Không có mã nào phù hợp với bộ lọc."
        )

    else:

        st.dataframe(
            bang_hien_thi,
            width="stretch",
            height=650,
            hide_index=True,
            column_config={
                "Trạng thái": st.column_config.TextColumn(
                    "TT",
                    width="small",
                ),
                "Mã": st.column_config.TextColumn(
                    "Mã",
                    width="small",
                ),
                "Sàn": st.column_config.TextColumn(
                    "Sàn",
                    width="small",
                ),
                "Ngành": st.column_config.TextColumn(
                    "Ngành",
                    width="medium",
                ),
                "Giá": st.column_config.TextColumn(
                    "Giá",
                    width="small",
                ),
                "Thay đổi": st.column_config.TextColumn(
                    "Thay đổi",
                    width="small",
                ),
                "Khối lượng": st.column_config.TextColumn(
                    "Khối lượng",
                    width="medium",
                ),
                "Giá trị giao dịch": st.column_config.TextColumn(
                    "Giá trị giao dịch",
                    width="medium",
                ),
                "Tham chiếu": st.column_config.TextColumn(
                    "Tham chiếu",
                    width="small",
                ),
            },
        )

    # ========================================================
    # TÂM LÝ
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🧭 Tâm lý thị trường'
        '</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Điểm tâm lý",
            (
                f"{tam_ly:.0f}/100"
                if tam_ly is not None
                else "—"
            ),
        )

    with b:

        st.metric(
            "Trạng thái",
            nhan_tam_ly,
        )

    with c:

        st.metric(
            "Tỷ lệ mã tăng",
            (
                f"{thong_ke.get('phan_tram_tang', 0):.1f}%"
                if thong_ke.get(
                    "phan_tram_tang"
                ) is not None
                else "—"
            ),
        )

    # ========================================================
    # TOP TĂNG / GIẢM / KHỐI LƯỢNG
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🏆 Cổ phiếu nổi bật'
        '</div>',
        unsafe_allow_html=True,
    )

    top_tang = du_lieu_tong.get(
        "top_tang",
        pd.DataFrame(),
    )

    top_giam = du_lieu_tong.get(
        "top_giam",
        pd.DataFrame(),
    )

    top_khoi_luong = du_lieu_tong.get(
        "top_khoi_luong",
        pd.DataFrame(),
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:

        st.markdown(
            "#### 🟢 Tăng mạnh"
        )

        st.dataframe(
            _tao_bang_hien_thi(
                top_tang
            ),
            width="stretch",
            hide_index=True,
            height=420,
        )

    with col_b:

        st.markdown(
            "#### 🔴 Giảm mạnh"
        )

        st.dataframe(
            _tao_bang_hien_thi(
                top_giam
            ),
            width="stretch",
            hide_index=True,
            height=420,
        )

    with col_c:

        st.markdown(
            "#### 💰 Khối lượng lớn"
        )

        st.dataframe(
            _tao_bang_hien_thi(
                top_khoi_luong
            ),
            width="stretch",
            hide_index=True,
            height=420,
        )

    # ========================================================
    # NGÀNH
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📚 Diễn biến nhóm ngành'
        '</div>',
        unsafe_allow_html=True,
    )

    theo_nganh = du_lieu_tong.get(
        "theo_nganh",
        pd.DataFrame(),
    )

    if (
        theo_nganh is not None
        and not theo_nganh.empty
    ):

        bang_nganh = theo_nganh.copy()

        bang_nganh[
            "Biến động bình quân"
        ] = (
            bang_nganh[
                "bien_dong_binh_quan"
            ]
            .apply(_phan_tram)
        )

        bang_nganh[
            "Khối lượng"
        ] = (
            bang_nganh[
                "tong_khoi_luong"
            ]
            .apply(_so_luong)
        )

        bang_nganh[
            "Giá trị giao dịch"
        ] = (
            bang_nganh[
                "tong_gia_tri"
            ]
            .apply(_gia_tri)
        )

        bang_nganh[
            "Ngành"
        ] = bang_nganh[
            "ten_nganh"
        ]

        bang_nganh[
            "Số mã"
        ] = bang_nganh[
            "so_ma"
        ]

        bang_nganh[
            "Tăng"
        ] = bang_nganh[
            "tang"
        ]

        bang_nganh[
            "Đứng giá"
        ] = bang_nganh[
            "dung_gia"
        ]

        bang_nganh[
            "Giảm"
        ] = bang_nganh[
            "giam"
        ]

        st.dataframe(
            bang_nganh[
                [
                    "Ngành",
                    "Số mã",
                    "Tăng",
                    "Đứng giá",
                    "Giảm",
                    "Biến động bình quân",
                    "Khối lượng",
                    "Giá trị giao dịch",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "Chưa có đủ dữ liệu ngành."
        )

    # ========================================================
    # DÒNG TIỀN NƯỚC NGOÀI
    # ========================================================

    nuoc_ngoai = du_lieu_tong.get(
        "nuoc_ngoai",
        {},
    )

    if nuoc_ngoai.get(
        "co_du_lieu",
        False,
    ):

        st.markdown(
            '<div class="section-title">'
            '🌏 Giao dịch nước ngoài'
            '</div>',
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Nước ngoài mua",
                _so_luong(
                    nuoc_ngoai.get(
                        "mua"
                    )
                ),
            )

        with b:

            st.metric(
                "Nước ngoài bán",
                _so_luong(
                    nuoc_ngoai.get(
                        "ban"
                    )
                ),
            )

        with c:

            st.metric(
                "Mua ròng",
                _so_luong(
                    nuoc_ngoai.get(
                        "rong"
                    )
                ),
            )

    # ========================================================
    # NGUỒN
    # ========================================================

    so_ma = nguon.get(
        "so_ma",
        len(bang_gia),
    )

    cap_nhat = nguon.get(
        "cap_nhat"
    )

    if cap_nhat is not None:

        try:

            cap_nhat_text = pd.Timestamp(
                cap_nhat
            ).strftime(
                "%d/%m/%Y %H:%M:%S"
            )

        except Exception:

            cap_nhat_text = str(
                cap_nhat
            )

    else:

        cap_nhat_text = "—"

    st.caption(
        f"Nguồn: {nguon.get('nguon', 'Dữ liệu thị trường')}"
        f" · {so_ma:,} mã"
        f" · Cập nhật: {cap_nhat_text}"
        f" · Làm mới dữ liệu mỗi 60 giây"
    )
