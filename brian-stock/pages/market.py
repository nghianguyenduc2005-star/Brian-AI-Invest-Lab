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
# SESSION STATE
# ============================================================

def _khoi_tao_session():

    if "market_loaded" not in st.session_state:
        st.session_state["market_loaded"] = False

    if "market_overview" not in st.session_state:
        st.session_state["market_overview"] = None


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
    value = _so(value)

    if value is None:
        return "—"

    return f"{value:,.0f}"


def _format_phan_tram(
    value,
):
    value = _so(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def _format_khoi_luong(
    value,
):
    value = _so(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn"
        )

    return f"{value:,.0f}"


def _format_gia_tri(
    value,
):
    value = _so(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:

        return (
            f"{value / 1_000_000_000_000:.2f} nghìn tỷ"
        )

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn"
        )

    return f"{value:,.0f}"


def _ky_hieu(
    value,
):
    value = _so(value)

    if value is None:
        return "⚪"

    if value > 0.05:
        return "🟢"

    if value < -0.05:
        return "🔴"

    return "🟡"


# ============================================================
# BẢNG HIỂN THỊ CHÍNH
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
        .replace("", "—")
    )

    bang["Sàn"] = (
        df["san"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "—")
    )

    bang["Ngành"] = (
        df["ten_nganh"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "—")
    )

    bang["Giá"] = (
        df["gia"]
        .apply(_format_gia)
    )

    bang["TC"] = (
        df["gia_tham_chieu"]
        .apply(_format_gia)
    )

    bang["Mở"] = (
        df["gia_mo_cua"]
        .apply(_format_gia)
    )

    bang["Cao"] = (
        df["gia_cao_nhat"]
        .apply(_format_gia)
    )

    bang["Thấp"] = (
        df["gia_thap_nhat"]
        .apply(_format_gia)
    )

    bang["+/-"] = (
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
# BẢNG TOP
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

    for ten_cot, gia_tri in {
        "ma": "",
        "gia": None,
        "thay_doi_pct": None,
        "khoi_luong": None,
        "gia_tri_giao_dich": None,
    }.items():

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

    bang["Giá trị"] = (
        df["gia_tri_giao_dich"]
        .apply(_format_gia_tri)
    )

    return bang.reset_index(
        drop=True
    )


# ============================================================
# LẤY DATAFRAME
# ============================================================

def _lay_dataframe(
    overview,
):
    if not isinstance(
        overview,
        dict,
    ):
        return pd.DataFrame()

    df = overview.get(
        "bang_gia",
        pd.DataFrame(),
    )

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
    ):
        return pd.DataFrame()

    return df


# ============================================================
# OPTIONS
# ============================================================

def _lay_danh_sach_san(
    df,
):

    result = ["Tất cả"]

    if (
        df is None
        or df.empty
        or "san" not in df.columns
    ):
        return result

    values = (
        df["san"]
        .fillna("")
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    values = sorted(
        x for x in values if x
    )

    result.extend(values)

    return result


def _lay_danh_sach_nganh(
    df,
):

    result = ["Tất cả"]

    if (
        df is None
        or df.empty
        or "ten_nganh" not in df.columns
    ):
        return result

    values = (
        df["ten_nganh"]
        .fillna("")
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    values = sorted(
        x for x in values if x
    )

    result.extend(values)

    return result


# ============================================================
# SẮP XẾP
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

    mapping = {
        "Tăng/giảm (%)": "thay_doi_pct",
        "Khối lượng": "khoi_luong",
        "Giá trị giao dịch": "gia_tri_giao_dich",
        "Giá": "gia",
        "Mã cổ phiếu": "ma",
    }

    column = mapping.get(
        kieu_sap_xep
    )

    if (
        column is None
        or column not in df.columns
    ):
        return df

    ascending = (
        thu_tu == "Tăng dần"
    )

    if column != "ma":

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return (
        df
        .sort_values(
            by=column,
            ascending=ascending,
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# LOAD
# ============================================================

def _load_market_data(
    force_reload=False,
):

    if (
        not force_reload
        and st.session_state.get(
            "market_loaded",
            False,
        )
    ):

        return st.session_state.get(
            "market_overview"
        )

    with st.spinner(
        "Đang tải dữ liệu thị trường..."
    ):

        overview = lay_market_overview(
            force_reload=force_reload
        )

    st.session_state[
        "market_overview"
    ] = overview

    st.session_state[
        "market_loaded"
    ] = True

    return overview


# ============================================================
# RENDER MARKET
# ============================================================

def render_market():

    _khoi_tao_session()

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "📊 Thị trường"
    )

    st.caption(
        "Bảng giá toàn thị trường · "
        "Dữ liệu chỉ tải khi bấm cập nhật"
    )

    # ========================================================
    # CHƯA LOAD
    # ========================================================

    if not st.session_state.get(
        "market_loaded",
        False,
    ):

        st.info(
            "Bấm «Cập nhật dữ liệu» để tải bảng giá thị trường."
        )

        if st.button(
            "🔄 Cập nhật dữ liệu",
            type="primary",
            key="market_first_load",
            use_container_width=True,
        ):

            try:

                _load_market_data(
                    force_reload=True
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Không thể tải dữ liệu thị trường."
                )

                st.code(
                    str(error)
                )

                return

        return

    # ========================================================
    # UPDATE
    # ========================================================

    c1, c2 = st.columns(
        [
            1.2,
            4,
        ]
    )

    with c1:

        refresh = st.button(
            "🔄 Cập nhật dữ liệu",
            type="primary",
            key="market_refresh",
            use_container_width=True,
        )

    with c2:

        st.caption(
            "Đổi bộ lọc không gọi API. "
            "Chỉ nút cập nhật mới tải lại dữ liệu."
        )

    if refresh:

        try:

            _load_market_data(
                force_reload=True
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Không thể cập nhật dữ liệu."
            )

            st.code(
                str(error)
            )

            return

    # ========================================================
    # GET DATA
    # ========================================================

    overview = st.session_state.get(
        "market_overview"
    )

    bang_gia = _lay_dataframe(
        overview
    )

    if bang_gia.empty:

        st.warning(
            "Chưa có dữ liệu bảng giá."
        )

        return

    # ========================================================
    # SUMMARY
    # ========================================================

    stats = (
        overview.get(
            "thong_ke",
            {},
        )
        if isinstance(
            overview,
            dict,
        )
        else {}
    )

    total = int(
        stats.get(
            "co_du_lieu_gia",
            len(bang_gia),
        )
        or 0
    )

    tang = int(
        stats.get(
            "tang",
            0,
        )
        or 0
    )

    dung = int(
        stats.get(
            "dung_gia",
            0,
        )
        or 0
    )

    giam = int(
        stats.get(
            "giam",
            0,
        )
        or 0
    )

    st.subheader(
        "📌 Tổng quan thị trường"
    )

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Có dữ liệu",
            f"{total:,} mã",
        )

    with b:
        st.metric(
            "Tăng",
            f"{tang:,} mã",
        )

    with c:
        st.metric(
            "Đứng giá",
            f"{dung:,} mã",
        )

    with d:
        st.metric(
            "Giảm",
            f"{giam:,} mã",
        )

    # ========================================================
    # FILTER
    # ========================================================

    st.subheader(
        "🔎 Bộ lọc bảng giá"
    )

    f1, f2, f3 = st.columns(
        [
            2.4,
            1,
            2,
        ]
    )

    with f1:

        keyword = st.text_input(
            "Tìm mã / doanh nghiệp",
            placeholder="Ví dụ HPG, FPT, VHM...",
            key="market_keyword",
        )

    with f2:

        exchange = st.selectbox(
            "Sàn",
            _lay_danh_sach_san(
                bang_gia
            ),
            key="market_exchange",
        )

    with f3:

        industry = st.selectbox(
            "Ngành",
            _lay_danh_sach_nganh(
                bang_gia
            ),
            key="market_industry",
        )

    f4, f5, f6 = st.columns(
        [
            1.2,
            2,
            1.2,
        ]
    )

    with f4:

        status = st.selectbox(
            "Trạng thái",
            [
                "Tất cả",
                "Tăng",
                "Đứng giá",
                "Giảm",
            ],
            key="market_status",
        )

    with f5:

        sort_by = st.selectbox(
            "Sắp xếp theo",
            [
                "Tăng/giảm (%)",
                "Khối lượng",
                "Giá trị giao dịch",
                "Giá",
                "Mã cổ phiếu",
            ],
            key="market_sort",
        )

    with f6:

        order = st.selectbox(
            "Thứ tự",
            [
                "Giảm dần",
                "Tăng dần",
            ],
            key="market_order",
        )

    rows = st.select_slider(
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
    # FILTER LOCAL
    # ========================================================

    try:

        filtered = loc_bang_gia(
            bang_gia,
            san=exchange,
            tu_khoa=keyword,
            nganh=industry,
            huong=status,
        )

    except Exception as error:

        st.error(
            "Lỗi khi lọc dữ liệu."
        )

        st.code(
            str(error)
        )

        filtered = pd.DataFrame()

    # ========================================================
    # SORT LOCAL
    # ========================================================

    filtered = _sap_xep_bang(
        filtered,
        sort_by,
        order,
    )

    # ========================================================
    # TABLE
    # ========================================================

    st.caption(
        f"Tổng {len(bang_gia):,} mã · "
        f"đang hiển thị {len(filtered):,} mã"
    )

    if filtered.empty:

        st.info(
            "Không có cổ phiếu phù hợp với bộ lọc."
        )

    else:

        display_df = _tao_bang_hien_thi(
            filtered.head(rows)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
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
                "TC": st.column_config.TextColumn(
                    "TC",
                    width="small",
                ),
                "Mở": st.column_config.TextColumn(
                    "Mở",
                    width="small",
                ),
                "Cao": st.column_config.TextColumn(
                    "Cao",
                    width="small",
                ),
                "Thấp": st.column_config.TextColumn(
                    "Thấp",
                    width="small",
                ),
                "+/-": st.column_config.TextColumn(
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

    l1, l2 = st.columns(2)

    with l1:

        st.metric(
            "Tổng khối lượng",
            _format_khoi_luong(
                stats.get(
                    "tong_khoi_luong"
                )
            ),
        )

    with l2:

        st.metric(
            "Tổng giá trị giao dịch",
            _format_gia_tri(
                stats.get(
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

    score = _so(
        overview.get(
            "tam_ly",
            50,
        ),
        50,
    )

    label = str(
        overview.get(
            "nhan_tam_ly",
            "Trung tính",
        )
    )

    percent_up = _so(
        stats.get(
            "phan_tram_tang",
            0,
        ),
        0,
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Điểm tâm lý",
            f"{score:.0f}/100",
        )

    with p2:

        st.metric(
            "Trạng thái",
            label,
        )

    with p3:

        st.metric(
            "Tỷ lệ mã tăng",
            f"{percent_up:.1f}%",
        )

    # ========================================================
    # TOP
    # ========================================================

    st.subheader(
        "🏆 Cổ phiếu nổi bật"
    )

    top_up = overview.get(
        "top_tang",
        pd.DataFrame(),
    )

    top_down = overview.get(
        "top_giam",
        pd.DataFrame(),
    )

    top_volume = overview.get(
        "top_khoi_luong",
        pd.DataFrame(),
    )

    t1, t2, t3 = st.columns(3)

    with t1:

        st.markdown(
            "#### 🟢 Tăng mạnh"
        )

        st.dataframe(
            _tao_bang_top(
                top_up
            ),
            use_container_width=True,
            height=360,
            hide_index=True,
        )

    with t2:

        st.markdown(
            "#### 🔴 Giảm mạnh"
        )

        st.dataframe(
            _tao_bang_top(
                top_down
            ),
            use_container_width=True,
            height=360,
            hide_index=True,
        )

    with t3:

        st.markdown(
            "#### 💧 Khối lượng lớn"
        )

        st.dataframe(
            _tao_bang_top(
                top_volume
            ),
            use_container_width=True,
            height=360,
            hide_index=True,
        )

    # ========================================================
    # NGÀNH
    # ========================================================

    st.subheader(
        "📚 Diễn biến nhóm ngành"
    )

    by_industry = overview.get(
        "theo_nganh",
        pd.DataFrame(),
    )

    if (
        by_industry is not None
        and not by_industry.empty
    ):

        industry_df = by_industry.copy()

        industry_df[
            "Ngành"
        ] = (
            industry_df[
                "ten_nganh"
            ]
            .fillna("")
            .astype(str)
        )

        industry_df[
            "Số mã"
        ] = industry_df[
            "so_ma"
        ]

        industry_df[
            "Tăng"
        ] = industry_df[
            "tang"
        ]

        industry_df[
            "Đứng giá"
        ] = industry_df[
            "dung_gia"
        ]

        industry_df[
            "Giảm"
        ] = industry_df[
            "giam"
        ]

        industry_df[
            "Biến động"
        ] = (
            industry_df[
                "bien_dong_binh_quan"
            ]
            .apply(
                _format_phan_tram
            )
        )

        industry_df[
            "Khối lượng"
        ] = (
            industry_df[
                "tong_khoi_luong"
            ]
            .apply(
                _format_khoi_luong
            )
        )

        industry_df[
            "Giá trị giao dịch"
        ] = (
            industry_df[
                "tong_gia_tri"
            ]
            .apply(
                _format_gia_tri
            )
        )

        st.dataframe(
            industry_df[
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
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Chưa có đủ dữ liệu ngành."
        )

    # ========================================================
    # NƯỚC NGOÀI
    # ========================================================

    foreign = overview.get(
        "nuoc_ngoai",
        {},
    )

    if foreign.get(
        "co_du_lieu",
        False,
    ):

        st.subheader(
            "🌏 Giao dịch nước ngoài"
        )

        x1, x2, x3 = st.columns(3)

        with x1:

            st.metric(
                "Nước ngoài mua",
                _format_khoi_luong(
                    foreign.get(
                        "mua"
                    )
                ),
            )

        with x2:

            st.metric(
                "Nước ngoài bán",
                _format_khoi_luong(
                    foreign.get(
                        "ban"
                    )
                ),
            )

        with x3:

            st.metric(
                "Mua ròng",
                _format_khoi_luong(
                    foreign.get(
                        "rong"
                    )
                ),
            )

    # ========================================================
    # SOURCE
    # ========================================================

    source = overview.get(
        "nguon",
        {},
    )

    source_name = str(
        source.get(
            "nguon",
            "Vnstock",
        )
    )

    source_count = int(
        source.get(
            "so_ma",
            len(bang_gia),
        )
        or 0
    )

    st.caption(
        f"Nguồn: {source_name} · "
        f"{source_count:,} mã"
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_market_overview():
    render_market()
