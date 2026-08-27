from __future__ import annotations

import pandas as pd
import streamlit as st

from data.market_overview import (
    lay_market_overview,
    loc_bang_gia,
)


# ============================================================
# TIỆN ÍCH
# ============================================================

def _so(value, mac_dinh=None):
    try:
        value = float(value)

        if pd.isna(value):
            return mac_dinh

        return value

    except Exception:
        return mac_dinh


def _cot(df, ten, mac_dinh=None):
    if ten in df.columns:
        return df[ten]

    return pd.Series(
        mac_dinh,
        index=df.index,
    )


def _gia(value):
    value = _so(value)

    if value is None:
        return "—"

    return f"{value:,.2f}"


def _phan_tram(value):
    value = _so(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def _khoi_luong(value):
    value = _so(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu cổ phiếu"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn cổ phiếu"

    return f"{value:,.0f} cổ phiếu"


def _gia_tri(value):
    value = _so(value)

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


def _ky_hieu(value):
    value = _so(value)

    if value is None:
        return "⚪"

    if value > 0.05:
        return "🟢"

    if value < -0.05:
        return "🔴"

    return "🟡"


def _metric(cot, nhan, value):
    with cot:
        st.metric(
            nhan,
            value,
        )


# ============================================================
# BẢNG GIÁ CHÍNH
# ============================================================

def _tao_bang_gia(df):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    df = df.copy()

    bang = pd.DataFrame(
        index=df.index
    )

    bang["TT"] = _cot(
        df,
        "thay_doi_pct",
    ).apply(
        _ky_hieu
    )

    bang["Mã"] = (
        _cot(
            df,
            "ma",
            "",
        )
        .fillna("")
        .astype(str)
        .str.upper()
    )

    bang["Doanh nghiệp"] = (
        _cot(
            df,
            "ten_doanh_nghiep",
            "",
        )
        .fillna("")
        .astype(str)
        .replace(
            "",
            "—",
        )
    )

    bang["Sàn"] = (
        _cot(
            df,
            "san",
            "",
        )
        .fillna("")
        .astype(str)
        .replace(
            "",
            "—",
        )
    )

    bang["Ngành"] = (
        _cot(
            df,
            "ten_nganh",
            "",
        )
        .fillna("")
        .astype(str)
        .replace(
            "",
            "—",
        )
    )

    bang["Giá"] = _cot(
        df,
        "gia",
    ).apply(
        _gia
    )

    bang["Tham chiếu"] = _cot(
        df,
        "gia_tham_chieu",
    ).apply(
        _gia
    )

    bang["Mở cửa"] = _cot(
        df,
        "gia_mo_cua",
    ).apply(
        _gia
    )

    bang["Cao nhất"] = _cot(
        df,
        "gia_cao_nhat",
    ).apply(
        _gia
    )

    bang["Thấp nhất"] = _cot(
        df,
        "gia_thap_nhat",
    ).apply(
        _gia
    )

    bang["Thay đổi"] = _cot(
        df,
        "thay_doi_pct",
    ).apply(
        _phan_tram
    )

    bang["Khối lượng"] = _cot(
        df,
        "khoi_luong",
    ).apply(
        _khoi_luong
    )

    bang["Giá trị giao dịch"] = _cot(
        df,
        "gia_tri_giao_dich",
    ).apply(
        _gia_tri
    )

    return bang.reset_index(
        drop=True
    )


# ============================================================
# BẢNG TOP
# ============================================================

def _tao_bang_top(df):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    df = df.copy()

    bang = pd.DataFrame(
        index=df.index
    )

    bang["Mã"] = (
        _cot(
            df,
            "ma",
            "",
        )
        .fillna("")
        .astype(str)
        .str.upper()
    )

    bang["Giá"] = _cot(
        df,
        "gia",
    ).apply(
        _gia
    )

    bang["Thay đổi"] = _cot(
        df,
        "thay_doi_pct",
    ).apply(
        _phan_tram
    )

    bang["Khối lượng"] = _cot(
        df,
        "khoi_luong",
    ).apply(
        _khoi_luong
    )

    bang["Giá trị giao dịch"] = _cot(
        df,
        "gia_tri_giao_dich",
    ).apply(
        _gia_tri
    )

    return bang.reset_index(
        drop=True
    )


# ============================================================
# RESET BỘ LỌC
# ============================================================

def _dat_lai_bo_loc():
    st.session_state[
        "thi_truong_tu_khoa"
    ] = ""

    st.session_state[
        "thi_truong_san"
    ] = "Tất cả"

    st.session_state[
        "thi_truong_nganh"
    ] = "Tất cả"

    st.session_state[
        "thi_truong_trang_thai"
    ] = "Tất cả"

    st.session_state[
        "thi_truong_sap_xep"
    ] = "Tăng/giảm (%)"

    st.session_state[
        "thi_truong_thu_tu"
    ] = "Giảm dần"

    st.session_state[
        "thi_truong_so_dong"
    ] = 100


# ============================================================
# RENDER MARKET
# ============================================================

def render_market():

    st.title(
        "📊 Thị trường"
    )

    st.caption(
        "Bảng giá toàn thị trường · "
        "dữ liệu cache 60 giây"
    )

    # ========================================================
    # LẤY DỮ LIỆU
    # ========================================================

    try:

        with st.spinner(
            "Đang tải dữ liệu thị trường..."
        ):

            du_lieu = (
                lay_market_overview()
            )

    except Exception as loi:

        st.error(
            "Không thể tải dữ liệu thị trường."
        )

        st.code(
            str(loi)
        )

        return

    bang_gia = (
        du_lieu.get(
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
    # THỐNG KÊ
    # ========================================================

    thong_ke = (
        du_lieu.get(
            "thong_ke",
            {},
        )
    )

    tong_ma = int(
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
        "📌 Tổng quan"
    )

    a, b, c, d = st.columns(4)

    _metric(
        a,
        "Có dữ liệu",
        f"{tong_ma:,} mã",
    )

    _metric(
        b,
        "Tăng",
        f"{tang:,} mã",
    )

    _metric(
        c,
        "Đứng giá",
        f"{dung_gia:,} mã",
    )

    _metric(
        d,
        "Giảm",
        f"{giam:,} mã",
    )

    # ========================================================
    # THANH KHOẢN
    # ========================================================

    a, b = st.columns(2)

    _metric(
        a,
        "Tổng khối lượng",
        _khoi_luong(
            thong_ke.get(
                "tong_khoi_luong"
            )
        ),
    )

    _metric(
        b,
        "Tổng giá trị giao dịch",
        _gia_tri(
            thong_ke.get(
                "tong_gia_tri"
            )
        ),
    )

    # ========================================================
    # BẢNG GIÁ
    # ========================================================

    st.subheader(
        "💹 Bảng giá toàn thị trường"
    )

    # ========================================================
    # DANH SÁCH SÀN
    # ========================================================

    danh_sach_san = [
        "Tất cả"
    ]

    if "san" in bang_gia.columns:

        gia_tri_san = (
            bang_gia[
                "san"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        danh_sach_san.extend(
            sorted(
                [
                    x
                    for x
                    in gia_tri_san.unique()
                    if x
                ]
            )
        )

    # ========================================================
    # DANH SÁCH NGÀNH
    # ========================================================

    danh_sach_nganh = [
        "Tất cả"
    ]

    if "ten_nganh" in bang_gia.columns:

        gia_tri_nganh = (
            bang_gia[
                "ten_nganh"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        danh_sach_nganh.extend(
            sorted(
                [
                    x
                    for x
                    in gia_tri_nganh.unique()
                    if x
                ]
            )
        )

    # ========================================================
    # BỘ LỌC HÀNG 1
    # ========================================================

    f1, f2, f3 = st.columns(
        [
            2.4,
            1.0,
            1.8,
        ]
    )

    with f1:

        tu_khoa = st.text_input(
            "Tìm mã / doanh nghiệp",
            placeholder=(
                "Ví dụ: HPG, VHM, FPT..."
            ),
            key="thi_truong_tu_khoa",
        )

    with f2:

        san = st.selectbox(
            "Sàn",
            danh_sach_san,
            key="thi_truong_san",
        )

    with f3:

        nganh = st.selectbox(
            "Ngành",
            danh_sach_nganh,
            key="thi_truong_nganh",
        )

    # ========================================================
    # BỘ LỌC HÀNG 2
    # ========================================================

    f4, f5, f6, f7 = st.columns(
        [
            1.1,
            1.8,
            1.1,
            1.0,
        ]
    )

    with f4:

        huong = st.selectbox(
            "Trạng thái",
            [
                "Tất cả",
                "Tăng",
                "Đứng giá",
                "Giảm",
            ],
            key="thi_truong_trang_thai",
        )

    with f5:

        sap_xep = st.selectbox(
            "Sắp xếp theo",
            [
                "Tăng/giảm (%)",
                "Giá trị giao dịch",
                "Khối lượng",
                "Giá",
                "Mã cổ phiếu",
            ],
            key="thi_truong_sap_xep",
        )

    with f6:

        thu_tu = st.selectbox(
            "Thứ tự",
            [
                "Giảm dần",
                "Tăng dần",
            ],
            key="thi_truong_thu_tu",
        )

    with f7:

        st.write("")

        st.button(
            "↺ Đặt lại",
            on_click=_dat_lai_bo_loc,
            width="stretch",
            key="thi_truong_dat_lai",
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
            1000,
        ],
        value=100,
        key="thi_truong_so_dong",
    )

    # ========================================================
    # LỌC
    # ========================================================

    try:

        bang_loc = loc_bang_gia(
            bang_gia,
            san=san,
            tu_khoa=tu_khoa,
            nganh=nganh,
            huong=huong,
        )

    except Exception as loi:

        st.error(
            "Không thể lọc bảng giá."
        )

        st.code(
            str(loi)
        )

        bang_loc = pd.DataFrame()

    # ========================================================
    # SẮP XẾP
    # ========================================================

    anh_xa_sap_xep = {
        "Tăng/giảm (%)": (
            "thay_doi_pct"
        ),
        "Giá trị giao dịch": (
            "gia_tri_giao_dich"
        ),
        "Khối lượng": (
            "khoi_luong"
        ),
        "Giá": (
            "gia"
        ),
        "Mã cổ phiếu": (
            "ma"
        ),
    }

    cot_sap_xep = (
        anh_xa_sap_xep[
            sap_xep
        ]
    )

    if (
        not bang_loc.empty
        and cot_sap_xep
        in bang_loc.columns
    ):

        if cot_sap_xep == "ma":

            bang_loc = (
                bang_loc
                .sort_values(
                    by="ma",
                    ascending=(
                        thu_tu
                        == "Tăng dần"
                    ),
                    na_position="last",
                )
            )

        else:

            bang_loc[
                cot_sap_xep
            ] = pd.to_numeric(
                bang_loc[
                    cot_sap_xep
                ],
                errors="coerce",
            )

            bang_loc = (
                bang_loc
                .sort_values(
                    by=cot_sap_xep,
                    ascending=(
                        thu_tu
                        == "Tăng dần"
                    ),
                    na_position="last",
                )
            )

    # ========================================================
    # HIỂN THỊ KẾT QUẢ
    # ========================================================

    so_dong_thuc_te = min(
        len(bang_loc),
        so_dong,
    )

    st.caption(
        "Lọc còn "
        f"{len(bang_loc):,}"
        " mã · hiển thị "
        f"{so_dong_thuc_te:,}"
        " mã"
    )

    if bang_loc.empty:

        st.info(
            "Không có cổ phiếu phù hợp với bộ lọc."
        )

    else:

        bang_hien_thi = _tao_bang_gia(
            bang_loc.head(
                so_dong
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
    # TÂM LÝ
    # ========================================================

    st.subheader(
        "🧭 Tâm lý thị trường"
    )

    diem_tam_ly = _so(
        du_lieu.get(
            "tam_ly",
            50,
        ),
        50,
    )

    nhan_tam_ly = str(
        du_lieu.get(
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

    _metric(
        a,
        "Điểm tâm lý",
        f"{diem_tam_ly:.0f}/100",
    )

    _metric(
        b,
        "Trạng thái",
        nhan_tam_ly,
    )

    _metric(
        c,
        "Tỷ lệ mã tăng",
        f"{phan_tram_tang:.1f}%",
    )

    # ========================================================
    # CỔ PHIẾU NỔI BẬT
    # ========================================================

    st.subheader(
        "🏆 Cổ phiếu nổi bật"
    )

    top_tang = du_lieu.get(
        "top_tang",
        pd.DataFrame(),
    )

    top_giam = du_lieu.get(
        "top_giam",
        pd.DataFrame(),
    )

    top_khoi_luong = du_lieu.get(
        "top_khoi_luong",
        pd.DataFrame(),
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            "#### 🟢 Tăng mạnh"
        )

        st.dataframe(
            _tao_bang_top(
                top_tang
            ),
            width="stretch",
            height=400,
            hide_index=True,
        )

    with b:

        st.markdown(
            "#### 🔴 Giảm mạnh"
        )

        st.dataframe(
            _tao_bang_top(
                top_giam
            ),
            width="stretch",
            height=400,
            hide_index=True,
        )

    with c:

        st.markdown(
            "#### 💧 Khối lượng lớn"
        )

        st.dataframe(
            _tao_bang_top(
                top_khoi_luong
            ),
            width="stretch",
            height=400,
            hide_index=True,
        )

    # ========================================================
    # NGÀNH
    # ========================================================

    st.subheader(
        "📚 Diễn biến nhóm ngành"
    )

    theo_nganh = du_lieu.get(
        "theo_nganh",
        pd.DataFrame(),
    )

    if (
        theo_nganh is not None
        and not theo_nganh.empty
    ):

        bang_nganh = (
            theo_nganh.copy()
        )

        bang_nganh["Ngành"] = _cot(
            bang_nganh,
            "ten_nganh",
            "",
        ).fillna("").astype(str)

        bang_nganh["Số mã"] = _cot(
            bang_nganh,
            "so_ma",
            0,
        )

        bang_nganh["Tăng"] = _cot(
            bang_nganh,
            "tang",
            0,
        )

        bang_nganh["Đứng giá"] = _cot(
            bang_nganh,
            "dung_gia",
            0,
        )

        bang_nganh["Giảm"] = _cot(
            bang_nganh,
            "giam",
            0,
        )

        bang_nganh[
            "Biến động bình quân"
        ] = _cot(
            bang_nganh,
            "bien_dong_binh_quan",
        ).apply(
            _phan_tram
        )

        bang_nganh[
            "Khối lượng"
        ] = _cot(
            bang_nganh,
            "tong_khoi_luong",
        ).apply(
            _khoi_luong
        )

        bang_nganh[
            "Giá trị giao dịch"
        ] = _cot(
            bang_nganh,
            "tong_gia_tri",
        ).apply(
            _gia_tri
        )

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
            "Nguồn dữ liệu hiện chưa có thông tin ngành."
        )

    # ========================================================
    # GIAO DỊCH NƯỚC NGOÀI
    # ========================================================

    nuoc_ngoai = du_lieu.get(
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

        _metric(
            a,
            "Nước ngoài mua",
            _khoi_luong(
                nuoc_ngoai.get(
                    "mua"
                )
            ),
        )

        _metric(
            b,
            "Nước ngoài bán",
            _khoi_luong(
                nuoc_ngoai.get(
                    "ban"
                )
            ),
        )

        _metric(
            c,
            "Mua ròng",
            _khoi_luong(
                nuoc_ngoai.get(
                    "rong"
                )
            ),
        )

    # ========================================================
    # NGUỒN
    # ========================================================

    nguon = du_lieu.get(
        "nguon",
        {},
    )

    ten_nguon = str(
        nguon.get(
            "nguon",
            "Vnstock Market",
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
        + " mã · cập nhật tự động"
    )
