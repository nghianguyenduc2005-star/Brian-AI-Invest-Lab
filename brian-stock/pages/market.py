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


def _dinh_dang_gia(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return f"{gia_tri:,.2f}"


def _dinh_dang_phan_tram(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    return f"{gia_tri:+.2f}%"


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
        return (
            f"{gia_tri / 1_000_000_000:.2f} "
            "tỷ cổ phiếu"
        )

    if gia_tri >= 1_000_000:
        return (
            f"{gia_tri / 1_000_000:.2f} "
            "triệu cổ phiếu"
        )

    if gia_tri >= 1_000:
        return (
            f"{gia_tri / 1_000:.2f} "
            "nghìn cổ phiếu"
        )

    return f"{gia_tri:,.0f} cổ phiếu"


def _dinh_dang_gia_tri(
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
            f"{gia_tri / 1_000_000_000_000:.2f} "
            "nghìn tỷ đồng"
        )

    if gia_tri >= 1_000_000_000:
        return (
            f"{gia_tri / 1_000_000_000:.2f} "
            "tỷ đồng"
        )

    if gia_tri >= 1_000_000:
        return (
            f"{gia_tri / 1_000_000:.2f} "
            "triệu đồng"
        )

    if gia_tri >= 1_000:
        return (
            f"{gia_tri / 1_000:.2f} "
            "nghìn đồng"
        )

    return f"{gia_tri:,.0f} đồng"


def _ky_hieu_trang_thai(
    gia_tri,
):
    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "⚪"

    if gia_tri > 0.05:
        return "🟢"

    if gia_tri < -0.05:
        return "🔴"

    return "🟡"


# ============================================================
# BẢNG HIỂN THỊ CHÍNH
# ============================================================

def _tao_bang_hien_thi(
    bang_gia,
):
    if (
        bang_gia is None
        or not isinstance(
            bang_gia,
            pd.DataFrame,
        )
        or bang_gia.empty
    ):
        return pd.DataFrame()

    bang = bang_gia.copy()

    # --------------------------------------------------------
    # Bảo đảm các cột cần thiết tồn tại
    # --------------------------------------------------------

    cac_cot_mac_dinh = {
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

    for ten_cot, gia_tri_mac_dinh in (
        cac_cot_mac_dinh.items()
    ):
        if ten_cot not in bang.columns:
            bang[ten_cot] = gia_tri_mac_dinh

    # --------------------------------------------------------
    # Cột hiển thị
    # --------------------------------------------------------

    bang["TT"] = (
        bang["thay_doi_pct"]
        .apply(
            _ky_hieu_trang_thai
        )
    )

    bang["Mã"] = (
        bang["ma"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    bang["Doanh nghiệp"] = (
        bang["ten_doanh_nghiep"]
        .fillna("")
        .astype(str)
        .replace(
            "",
            "—",
        )
    )

    bang["Sàn"] = (
        bang["san"]
        .fillna("")
        .astype(str)
        .replace(
            "",
            "—",
        )
    )

    bang["Ngành"] = (
        bang["ten_nganh"]
        .fillna("")
        .astype(str)
        .replace(
            "",
            "—",
        )
    )

    bang["Giá"] = (
        bang["gia"]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Tham chiếu"] = (
        bang["gia_tham_chieu"]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Mở cửa"] = (
        bang["gia_mo_cua"]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Cao nhất"] = (
        bang["gia_cao_nhat"]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Thấp nhất"] = (
        bang["gia_thap_nhat"]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Thay đổi"] = (
        bang["thay_doi_pct"]
        .apply(
            _dinh_dang_phan_tram
        )
    )

    bang["Khối lượng"] = (
        bang["khoi_luong"]
        .apply(
            _dinh_dang_khoi_luong
        )
    )

    bang["Giá trị giao dịch"] = (
        bang["gia_tri_giao_dich"]
        .apply(
            _dinh_dang_gia_tri
        )
    )

    return bang[
        [
            "TT",
            "Mã",
            "Doanh nghiệp",
            "Sàn",
            "Ngành",
            "Giá",
            "Tham chiếu",
            "Mở cửa",
            "Cao nhất",
            "Thấp nhất",
            "Thay đổi",
            "Khối lượng",
            "Giá trị giao dịch",
        ]
    ].reset_index(
        drop=True
    )


# ============================================================
# BẢNG TOP
# ============================================================

def _tao_bang_top(
    bang_gia,
):
    if (
        bang_gia is None
        or not isinstance(
            bang_gia,
            pd.DataFrame,
        )
        or bang_gia.empty
    ):
        return pd.DataFrame()

    bang = bang_gia.copy()

    for cot in [
        "ma",
        "gia",
        "thay_doi_pct",
        "khoi_luong",
        "gia_tri_giao_dich",
    ]:
        if cot not in bang.columns:
            bang[cot] = None

    bang["Mã"] = (
        bang["ma"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    bang["Giá"] = (
        bang["gia"]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Thay đổi"] = (
        bang["thay_doi_pct"]
        .apply(
            _dinh_dang_phan_tram
        )
    )

    bang["Khối lượng"] = (
        bang["khoi_luong"]
        .apply(
            _dinh_dang_khoi_luong
        )
    )

    bang["Giá trị giao dịch"] = (
        bang["gia_tri_giao_dich"]
        .apply(
            _dinh_dang_gia_tri
        )
    )

    return bang[
        [
            "Mã",
            "Giá",
            "Thay đổi",
            "Khối lượng",
            "Giá trị giao dịch",
        ]
    ].reset_index(
        drop=True
    )


# ============================================================
# THẺ METRIC
# ============================================================

def _metric(
    cot,
    tieu_de,
    gia_tri,
):
    with cot:
        st.metric(
            tieu_de,
            gia_tri,
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

    # ========================================================
    # TIÊU ĐỀ
    # ========================================================

    st.title(
        "📊 Thị trường"
    )

    st.caption(
        "Bảng giá toàn thị trường · "
        "dữ liệu làm mới theo chu kỳ 60 giây"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        with st.spinner(
            "Đang tải dữ liệu thị trường..."
        ):

            du_lieu_tong = (
                lay_market_overview()
            )

    except Exception as loi:

        st.error(
            "Không thể tải dữ liệu thị trường."
        )

        st.code(
            str(
                loi
            )
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
    # THỐNG KÊ
    # ========================================================

    thong_ke = (
        du_lieu_tong.get(
            "thong_ke",
            {},
        )
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

    tong_khoi_luong = (
        thong_ke.get(
            "tong_khoi_luong"
        )
    )

    tong_gia_tri = (
        thong_ke.get(
            "tong_gia_tri"
        )
    )

    # ========================================================
    # BẢNG GIÁ
    # ========================================================

    st.subheader(
        "💹 Bảng giá toàn thị trường"
    )

    # ========================================================
    # CÁC GIÁ TRỊ LỌC
    # ========================================================

    danh_sach_san = [
        "Tất cả"
    ]

    if "san" in bang_gia.columns:

        cac_san = (
            bang_gia["san"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        cac_san = [
            x
            for x in cac_san.unique().tolist()
            if x
        ]

        danh_sach_san.extend(
            sorted(
                cac_san
            )
        )

    danh_sach_nganh = [
        "Tất cả"
    ]

    if "ten_nganh" in bang_gia.columns:

        cac_nganh = (
            bang_gia["ten_nganh"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        cac_nganh = [
            x
            for x in cac_nganh.unique().tolist()
            if x
        ]

        danh_sach_nganh.extend(
            sorted(
                cac_nganh
            )
        )

    # ========================================================
    # BỘ LỌC HÀNG 1
    # ========================================================

    cot_tim, cot_san, cot_nganh = (
        st.columns(
            [
                2.4,
                1.0,
                1.8,
            ]
        )
    )

    with cot_tim:

        tu_khoa = st.text_input(
            "Tìm mã / doanh nghiệp",
            placeholder=(
                "Ví dụ: HPG, FPT, VHM..."
            ),
            key="thi_truong_tu_khoa",
        )

    with cot_san:

        san = st.selectbox(
            "Sàn",
            danh_sach_san,
            key="thi_truong_san",
        )

    with cot_nganh:

        nganh = st.selectbox(
            "Ngành",
            danh_sach_nganh,
            key="thi_truong_nganh",
        )

    # ========================================================
    # BỘ LỌC HÀNG 2
    # ========================================================

    cot_trang_thai, cot_sap_xep, cot_thu_tu, cot_reset = (
        st.columns(
            [
                1.1,
                1.8,
                1.1,
                1.0,
            ]
        )
    )

    with cot_trang_thai:

        trang_thai = st.selectbox(
            "Trạng thái",
            [
                "Tất cả",
                "Tăng",
                "Đứng giá",
                "Giảm",
            ],
            key="thi_truong_trang_thai",
        )

    with cot_sap_xep:

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

    with cot_thu_tu:

        thu_tu = st.selectbox(
            "Thứ tự",
            [
                "Giảm dần",
                "Tăng dần",
            ],
            key="thi_truong_thu_tu",
        )

    with cot_reset:

        st.write("")

        st.button(
            "↺ Đặt lại",
            on_click=_dat_lai_bo_loc,
            width="stretch",
            key="thi_truong_reset",
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
            huong=trang_thai,
        )

    except Exception as loi:

        st.error(
            "Lỗi khi lọc bảng giá."
        )

        st.code(
            str(
                loi
            )
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
    # THÔNG TIN KẾT QUẢ
    # ========================================================

    st.caption(
        f"Đang có {len(bang_gia):,} mã · "
        f"lọc còn {len(bang_loc):,} mã · "
        f"hiển thị tối đa {so_dong:,} dòng"
    )

    # ========================================================
    # HIỂN THỊ
    # ========================================================

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
    # ĐỘ RỘNG THỊ TRƯỜNG
    # ========================================================

    st.subheader(
        "📌 Độ rộng thị trường"
    )

    a, b, c, d = st.columns(4)

    _metric(
        a,
        "Tăng",
        f"{tang:,} mã",
    )

    _metric(
        b,
        "Đứng giá",
        f"{dung_gia:,} mã",
    )

    _metric(
        c,
        "Giảm",
        f"{giam:,} mã",
    )

    _metric(
        d,
        "Có dữ liệu",
        f"{so_ma:,} mã",
    )

    # ========================================================
    # THANH KHOẢN
    # ========================================================

    st.subheader(
        "💰 Thanh khoản toàn thị trường"
    )

    a, b = st.columns(2)

    _metric(
        a,
        "Tổng khối lượng",
        _dinh_dang_khoi_luong(
            tong_khoi_luong
        ),
    )

    _metric(
        b,
        "Tổng giá trị giao dịch",
        _dinh_dang_gia_tri(
            tong_gia_tri
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
            "tam_ly"
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
            "phan_tram_tang"
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

    theo_nganh = du_lieu_tong.get(
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
            "Biến động bình quân"
        ] = (
            bang_nganh[
                "bien_dong_binh_quan"
            ]
            .apply(
                _dinh_dang_phan_tram
            )
        )

        bang_nganh[
            "Khối lượng"
        ] = (
            bang_nganh[
                "tong_khoi_luong"
            ]
            .apply(
                _dinh_dang_khoi_luong
            )
        )

        bang_nganh[
            "Giá trị giao dịch"
        ] = (
            bang_nganh[
                "tong_gia_tri"
            ]
            .apply(
                _dinh_dang_gia_tri
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

    nuoc_ngoai = (
        du_lieu_tong.get(
            "nuoc_ngoai",
            {},
        )
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
            _dinh_dang_khoi_luong(
                nuoc_ngoai.get(
                    "mua"
                )
            ),
        )

        _metric(
            b,
            "Nước ngoài bán",
            _dinh_dang_khoi_luong(
                nuoc_ngoai.get(
                    "ban"
                )
            ),
        )

        _metric(
            c,
            "Mua ròng",
            _dinh_dang_khoi_luong(
                nuoc_ngoai.get(
                    "rong"
                )
            ),
        )

    # ========================================================
    # NGUỒN
    # ========================================================

    thong_tin = (
        du_lieu_tong.get(
            "nguon",
            {},
        )
    )

    st.caption(
        f"Nguồn: "
        f"{thong_tin.get(
            'nguon',
            'Vnstock Market',
        )}"
        f" · "
        f"{int(
            thong_tin.get(
                'so_ma',
                len(bang_gia),
            )
            or 0
        ):,} mã"
    )
