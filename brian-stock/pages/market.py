from __future__ import annotations

import pandas as pd
import streamlit as st

from data.market_overview import (
    lay_market_overview,
    loc_bang_gia,
)


# ============================================================
# CẤU HÌNH
# ============================================================

SO_DONG_MAC_DINH = 100


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


def _format_gia(
    value,
):
    value = _so(
        value
    )

    if value is None:
        return "—"

    return f"{value:,.0f}"


def _format_phan_tram(
    value,
):
    value = _so(
        value
    )

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def _format_khoi_luong(
    value,
):
    value = _so(
        value
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000:
        return (
            f"{value / 1_000_000_000:.2f} "
            "tỷ cổ phiếu"
        )

    if value >= 1_000_000:
        return (
            f"{value / 1_000_000:.2f} "
            "triệu cổ phiếu"
        )

    if value >= 1_000:
        return (
            f"{value / 1_000:.2f} "
            "nghìn cổ phiếu"
        )

    return f"{value:,.0f} cổ phiếu"


def _format_gia_tri(
    value,
):
    value = _so(
        value
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:
        return (
            f"{value / 1_000_000_000_000:.2f} "
            "nghìn tỷ đồng"
        )

    if value >= 1_000_000_000:
        return (
            f"{value / 1_000_000_000:.2f} "
            "tỷ đồng"
        )

    if value >= 1_000_000:
        return (
            f"{value / 1_000_000:.2f} "
            "triệu đồng"
        )

    if value >= 1_000:
        return (
            f"{value / 1_000:.2f} "
            "nghìn đồng"
        )

    return f"{value:,.0f} đồng"


def _ky_hieu(
    value,
):
    value = _so(
        value
    )

    if value is None:
        return "⚪"

    if value > 0.05:
        return "🟢"

    if value < -0.05:
        return "🔴"

    return "🟡"


# ============================================================
# CHUYỂN BẢNG DỮ LIỆU THÀNH BẢNG HIỂN THỊ
# ============================================================

def _tao_bang_hien_thi(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    df = df.copy()

    # --------------------------------------------------------
    # Bảo đảm cột tồn tại
    # --------------------------------------------------------

    cac_cot = {
        "ma": "",
        "ten_doanh_nghiep": "",
        "san": "",
        "ten_nganh": "",
        "gia": None,
        "gia_tham_chieu": None,
        "gia_mo_cua": None,
        "gia_cao_nhat": None,
        "gia_thap_nhat": None,
        "thay_doi_pct": None,
        "khoi_luong": None,
        "gia_tri_giao_dich": None,
    }

    for ten_cot, gia_tri in cac_cot.items():

        if ten_cot not in df.columns:
            df[ten_cot] = gia_tri

    bang = pd.DataFrame(
        index=df.index
    )

    bang["TT"] = (
        df["thay_doi_pct"]
        .apply(_ky_hieu)
    )

    bang["Mã"] = (
        df["ma"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    bang["Doanh nghiệp"] = (
        df["ten_doanh_nghiep"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(
            "",
            "—",
        )
    )

    bang["Sàn"] = (
        df["san"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(
            "",
            "—",
        )
    )

    bang["Ngành"] = (
        df["ten_nganh"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(
            "",
            "—",
        )
    )

    bang["Giá"] = (
        df["gia"]
        .apply(_format_gia)
    )

    bang["Tham chiếu"] = (
        df["gia_tham_chieu"]
        .apply(_format_gia)
    )

    bang["Mở cửa"] = (
        df["gia_mo_cua"]
        .apply(_format_gia)
    )

    bang["Cao nhất"] = (
        df["gia_cao_nhat"]
        .apply(_format_gia)
    )

    bang["Thấp nhất"] = (
        df["gia_thap_nhat"]
        .apply(_format_gia)
    )

    bang["Thay đổi"] = (
        df["thay_doi_pct"]
        .apply(_format_phan_tram)
    )

    bang["Khối lượng"] = (
        df["khoi_luong"]
        .apply(_format_khoi_luong)
    )

    bang["Giá trị giao dịch"] = (
        df["gia_tri_giao_dich"]
        .apply(_format_gia_tri)
    )

    return bang.reset_index(
        drop=True
    )


# ============================================================
# CHUYỂN BẢNG TOP
# ============================================================

def _tao_bang_top(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    df = df.copy()

    cac_cot = {
        "ma": "",
        "gia": None,
        "thay_doi_pct": None,
        "khoi_luong": None,
        "gia_tri_giao_dich": None,
    }

    for ten_cot, gia_tri in cac_cot.items():

        if ten_cot not in df.columns:
            df[ten_cot] = gia_tri

    bang = pd.DataFrame(
        index=df.index
    )

    bang["Mã"] = (
        df["ma"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    bang["Giá"] = (
        df["gia"]
        .apply(_format_gia)
    )

    bang["Thay đổi"] = (
        df["thay_doi_pct"]
        .apply(_format_phan_tram)
    )

    bang["Khối lượng"] = (
        df["khoi_luong"]
        .apply(_format_khoi_luong)
    )

    bang["Giá trị giao dịch"] = (
        df["gia_tri_giao_dich"]
        .apply(_format_gia_tri)
    )

    return bang.reset_index(
        drop=True
    )


# ============================================================
# LOAD SESSION
# ============================================================

def _co_du_lieu_session():
    return (
        "market_overview"
        in st.session_state
        and isinstance(
            st.session_state[
                "market_overview"
            ],
            dict,
        )
    )


def _tai_du_lieu(
    force_reload=False,
):
    """
    Quan trọng:

    - Nếu đã có dữ liệu trong session:
      KHÔNG gọi API lại.
    - Chỉ gọi API khi:
      + mở page lần đầu
      + bấm cập nhật dữ liệu
    """

    if (
        not force_reload
        and _co_du_lieu_session()
    ):
        return st.session_state[
            "market_overview"
        ]

    with st.spinner(
        "Đang tải dữ liệu thị trường..."
    ):

        du_lieu = (
            lay_market_overview(
                force_reload=force_reload
            )
        )

    st.session_state[
        "market_overview"
    ] = du_lieu

    return du_lieu


# ============================================================
# DANH SÁCH SÀN
# ============================================================

def _lay_danh_sach_san(
    df,
):
    ket_qua = [
        "Tất cả"
    ]

    if (
        df is None
        or df.empty
        or "san" not in df.columns
    ):
        return ket_qua

    gia_tri = (
        df["san"]
        .fillna("")
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    gia_tri = [
        x
        for x in gia_tri
        if x
    ]

    ket_qua.extend(
        sorted(
            gia_tri
        )
    )

    return ket_qua


# ============================================================
# DANH SÁCH NGÀNH
# ============================================================

def _lay_danh_sach_nganh(
    df,
):
    ket_qua = [
        "Tất cả"
    ]

    if (
        df is None
        or df.empty
        or "ten_nganh"
        not in df.columns
    ):
        return ket_qua

    gia_tri = (
        df["ten_nganh"]
        .fillna("")
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    gia_tri = [
        x
        for x in gia_tri
        if x
    ]

    ket_qua.extend(
        sorted(
            gia_tri
        )
    )

    return ket_qua


# ============================================================
# SẮP XẾP LOCAL
# ============================================================

def _sap_xep_bang(
    df,
    kieu_sap_xep,
    thu_tu,
):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    df = df.copy()

    anh_xa = {
        "Tăng/giảm (%)": "thay_doi_pct",
        "Khối lượng": "khoi_luong",
        "Giá trị giao dịch": "gia_tri_giao_dich",
        "Giá": "gia",
        "Mã cổ phiếu": "ma",
    }

    cot = anh_xa.get(
        kieu_sap_xep
    )

    if cot is None:
        return df

    if cot not in df.columns:
        return df

    ascending = (
        thu_tu == "Tăng dần"
    )

    # --------------------------------------------------------
    # Mã cổ phiếu
    # --------------------------------------------------------

    if cot == "ma":

        return (
            df
            .sort_values(
                by=cot,
                ascending=ascending,
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Số
    # --------------------------------------------------------

    df[cot] = pd.to_numeric(
        df[cot],
        errors="coerce",
    )

    return (
        df
        .sort_values(
            by=cot,
            ascending=ascending,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CALLBACK CẬP NHẬT
# ============================================================

def _cap_nhat_du_lieu():
    st.session_state[
        "market_force_reload"
    ] = True


# ============================================================
# RENDER MARKET
# ============================================================

def render_market():

    # ========================================================
    # TIÊU ĐỀ
    # ========================================================

    st.title(
        "📊 Thị trường"
    )

    st.caption(
        "Bảng giá thị trường · "
        "dữ liệu cache 60 giây"
    )

    # ========================================================
    # NÚT CẬP NHẬT
    # ========================================================

    cot_nut, cot_thong_tin = (
        st.columns(
            [
                1.2,
                4,
            ]
        )
    )

    with cot_nut:

        st.button(
            "🔄 Cập nhật dữ liệu",
            type="primary",
            width="stretch",
            key="market_update_button",
            on_click=_cap_nhat_du_lieu,
        )

    with cot_thong_tin:

        if _co_du_lieu_session():

            st.caption(
                "Dữ liệu đã tải · "
                "đổi bộ lọc không gọi API lại."
            )

    force_reload = bool(
        st.session_state.pop(
            "market_force_reload",
            False,
        )
    )

    # ========================================================
    # LOAD
    # ========================================================

    try:

        du_lieu_tong = _tai_du_lieu(
            force_reload=force_reload
        )

    except Exception as loi:

        st.error(
            "Không thể tải dữ liệu thị trường."
        )

        st.code(
            str(loi)
        )

        return

    # ========================================================
    # BẢNG GIÁ
    # ========================================================

    bang_gia = du_lieu_tong.get(
        "bang_gia",
        pd.DataFrame(),
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
    # THỐNG KÊ
    # ========================================================

    thong_ke = du_lieu_tong.get(
        "thong_ke",
        {},
    )

    so_ma = int(
        thong_ke.get(
            "co_du_lieu_gia",
            len(bang_gia),
        )
        or 0
    )

    tang = int(
        thong_ke.get(
            "tang",
            0,
        )
        or 0
    )

    dung_gia = int(
        thong_ke.get(
            "dung_gia",
            0,
        )
        or 0
    )

    giam = int(
        thong_ke.get(
            "giam",
            0,
        )
        or 0
    )

    # ========================================================
    # TỔNG QUAN
    # ========================================================

    st.subheader(
        "📌 Tổng quan thị trường"
    )

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Có dữ liệu",
            f"{so_ma:,} mã",
        )

    with b:
        st.metric(
            "Tăng",
            f"{tang:,} mã",
        )

    with c:
        st.metric(
            "Đứng giá",
            f"{dung_gia:,} mã",
        )

    with d:
        st.metric(
            "Giảm",
            f"{giam:,} mã",
        )

    # ========================================================
    # BỘ LỌC
    # ========================================================

    st.subheader(
        "🔎 Bộ lọc bảng giá"
    )

    f1, f2, f3 = st.columns(
        [
            2.5,
            1.0,
            2.0,
        ]
    )

    with f1:

        tu_khoa = st.text_input(
            "Tìm mã / doanh nghiệp",
            placeholder=(
                "Ví dụ: HPG, FPT, VHM..."
            ),
            key="market_filter_keyword",
        )

    with f2:

        danh_sach_san = (
            _lay_danh_sach_san(
                bang_gia
            )
        )

        san = st.selectbox(
            "Sàn",
            danh_sach_san,
            key="market_filter_exchange",
        )

    with f3:

        danh_sach_nganh = (
            _lay_danh_sach_nganh(
                bang_gia
            )
        )

        nganh = st.selectbox(
            "Ngành",
            danh_sach_nganh,
            key="market_filter_industry",
        )

    f4, f5, f6 = st.columns(
        [
            1.2,
            2.0,
            1.2,
        ]
    )

    with f4:

        trang_thai = st.selectbox(
            "Trạng thái",
            [
                "Tất cả",
                "Tăng",
                "Đứng giá",
                "Giảm",
            ],
            key="market_filter_status",
        )

    with f5:

        kieu_sap_xep = st.selectbox(
            "Sắp xếp theo",
            [
                "Tăng/giảm (%)",
                "Khối lượng",
                "Giá trị giao dịch",
                "Giá",
                "Mã cổ phiếu",
            ],
            key="market_filter_sort",
        )

    with f6:

        thu_tu = st.selectbox(
            "Thứ tự",
            [
                "Giảm dần",
                "Tăng dần",
            ],
            key="market_filter_order",
        )

    # ========================================================
    # SỐ DÒNG
    # ========================================================

    so_dong = st.select_slider(
        "Số dòng hiển thị",
        options=[
            50,
            100,
            200,
            500,
        ],
        value=SO_DONG_MAC_DINH,
        key="market_rows",
    )

    # ========================================================
    # LỌC LOCAL
    # ========================================================

    try:

        bang_loc = loc_bang_gia(
            bang_gia,
            san=san,
            tu_khoa=tu_khoa,
            nganh=nganh,
            huong=trang_thai,
        )

    except Exception as loi:

        st.error(
            "Không thể áp dụng bộ lọc."
        )

        st.code(
            str(loi)
        )

        bang_loc = pd.DataFrame()

    # ========================================================
    # SẮP XẾP LOCAL
    # ========================================================

    bang_loc = _sap_xep_bang(
        bang_loc,
        kieu_sap_xep,
        thu_tu,
    )

    # ========================================================
    # KẾT QUẢ
    # ========================================================

    st.caption(
        "Tổng "
        f"{len(bang_gia):,}"
        " mã · lọc còn "
        f"{len(bang_loc):,}"
        " mã"
    )

    if bang_loc.empty:

        st.info(
            "Không có cổ phiếu phù hợp với bộ lọc."
        )

    else:

        bang_hien_thi = (
            _tao_bang_hien_thi(
                bang_loc.head(
                    so_dong
                )
            )
        )

        st.dataframe(
            bang_hien_thi,
            width="stretch",
            height=680,
            hide_index=True,
            column_config={
                "TT": st.column_config.TextColumn(
                    "TT",
                    width="small",
                ),
                "Mã": st.column_config.TextColumn(
                    "Mã",
                    width="small",
                ),
                "Doanh nghiệp": st.column_config.TextColumn(
                    "Doanh nghiệp",
                    width="large",
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
                "Tham chiếu": st.column_config.TextColumn(
                    "TC",
                    width="small",
                ),
                "Mở cửa": st.column_config.TextColumn(
                    "Mở",
                    width="small",
                ),
                "Cao nhất": st.column_config.TextColumn(
                    "Cao",
                    width="small",
                ),
                "Thấp nhất": st.column_config.TextColumn(
                    "Thấp",
                    width="small",
                ),
                "Thay đổi": st.column_config.TextColumn(
                    "+/-",
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
            },
        )

    # ========================================================
    # THANH KHOẢN
    # ========================================================

    st.subheader(
        "💰 Thanh khoản"
    )

    a, b = st.columns(2)

    with a:

        st.metric(
            "Tổng khối lượng",
            _format_khoi_luong(
                thong_ke.get(
                    "tong_khoi_luong"
                )
            ),
        )

    with b:

        st.metric(
            "Tổng giá trị giao dịch",
            _format_gia_tri(
                thong_ke.get(
                    "tong_gia_tri"
                )
            ),
        )

    # ========================================================
    # TÂM LÝ
    # ========================================================

    st.subheader(
        "🧭 Tâm lý thị trường"
    )

    diem_tam_ly = _so(
        du_lieu_tong.get(
            "tam_ly",
            50,
        ),
        50,
    )

    nhan_tam_ly = str(
        du_lieu_tong.get(
            "nhan_tam_ly",
            "Trung tính",
        )
    )

    phan_tram_tang = _so(
        thong_ke.get(
            "phan_tram_tang",
            0,
        ),
        0,
    )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Điểm tâm lý",
            f"{diem_tam_ly:.0f}/100",
        )

    with b:

        st.metric(
            "Trạng thái",
            nhan_tam_ly,
        )

    with c:

        st.metric(
            "Tỷ lệ mã tăng",
            f"{phan_tram_tang:.1f}%",
        )

    # ========================================================
    # TOP
    # ========================================================

    st.subheader(
        "🏆 Cổ phiếu nổi bật"
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

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            "#### 🟢 Tăng mạnh"
        )

        bang_top_tang = (
            _tao_bang_top(
                top_tang
            )
        )

        st.dataframe(
            bang_top_tang,
            width="stretch",
            height=360,
            hide_index=True,
        )

    with b:

        st.markdown(
            "#### 🔴 Giảm mạnh"
        )

        bang_top_giam = (
            _tao_bang_top(
                top_giam
            )
        )

        st.dataframe(
            bang_top_giam,
            width="stretch",
            height=360,
            hide_index=True,
        )

    with c:

        st.markdown(
            "#### 💧 Khối lượng lớn"
        )

        bang_top_khoi_luong = (
            _tao_bang_top(
                top_khoi_luong
            )
        )

        st.dataframe(
            bang_top_khoi_luong,
            width="stretch",
            height=360,
            hide_index=True,
        )

    # ========================================================
    # NGÀNH
    # ========================================================

    st.subheader(
        "📚 Diễn biến nhóm ngành"
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
            "Ngành"
        ] = (
            bang_nganh[
                "ten_nganh"
            ]
            .fillna("")
            .astype(str)
        )

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

        bang_nganh[
            "Biến động"
        ] = (
            bang_nganh[
                "bien_dong_binh_quan"
            ]
            .apply(
                _format_phan_tram
            )
        )

        bang_nganh[
            "Khối lượng"
        ] = (
            bang_nganh[
                "tong_khoi_luong"
            ]
            .apply(
                _format_khoi_luong
            )
        )

        bang_nganh[
            "Giá trị giao dịch"
        ] = (
            bang_nganh[
                "tong_gia_tri"
            ]
            .apply(
                _format_gia_tri
            )
        )

        st.dataframe(
            bang_nganh[
                [
                    "Ngành",
                    "Số mã",
                    "Tăng",
                    "Đứng giá",
                    "Giảm",
                    "Biến động",
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
    # GIAO DỊCH NƯỚC NGOÀI
    # ========================================================

    nuoc_ngoai = du_lieu_tong.get(
        "nuoc_ngoai",
        {},
    )

    if nuoc_ngoai.get(
        "co_du_lieu",
        False,
    ):

        st.subheader(
            "🌏 Giao dịch nước ngoài"
        )

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Nước ngoài mua",
                _format_khoi_luong(
                    nuoc_ngoai.get(
                        "mua"
                    )
                ),
            )

        with b:

            st.metric(
                "Nước ngoài bán",
                _format_khoi_luong(
                    nuoc_ngoai.get(
                        "ban"
                    )
                ),
            )

        with c:

            st.metric(
                "Mua ròng",
                _format_khoi_luong(
                    nuoc_ngoai.get(
                        "rong"
                    )
                ),
            )

    # ========================================================
    # NGUỒN
    # ========================================================

    nguon = du_lieu_tong.get(
        "nguon",
        {},
    )

    ten_nguon = str(
        nguon.get(
            "nguon",
            "Vnstock",
        )
    )

    so_ma_nguon = int(
        nguon.get(
            "so_ma",
            len(bang_gia),
        )
        or 0
    )

    st.caption(
        "Nguồn: "
        + ten_nguon
        + " · "
        + f"{so_ma_nguon:,}"
        + " mã · cập nhật theo cache"
    )
