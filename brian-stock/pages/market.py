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

def _so(
    gia_tri,
    mac_dinh=None,
):
    try:
        gia_tri = float(
            gia_tri
        )

        if pd.isna(
            gia_tri
        ):
            return mac_dinh

        return gia_tri

    except Exception:
        return mac_dinh


def _dinh_dang_gia(
    gia_tri,
):
    gia_tri = _so(
        gia_tri
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:,.2f}"
    )


def _dinh_dang_phan_tram(
    gia_tri,
):
    gia_tri = _so(
        gia_tri
    )

    if gia_tri is None:
        return "—"

    return (
        f"{gia_tri:+.2f}%"
    )


def _dinh_dang_khoi_luong(
    gia_tri,
):
    gia_tri = _so(
        gia_tri
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


def _dinh_dang_gia_tri(
    gia_tri,
):
    gia_tri = _so(
        gia_tri
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


def _trang_thai_ky_hieu(
    gia_tri,
):
    gia_tri = _so(
        gia_tri
    )

    if gia_tri is None:
        return "⚪"

    if gia_tri > 0.05:
        return "🟢"

    if gia_tri < -0.05:
        return "🔴"

    return "🟡"


# ============================================================
# TẠO BẢNG HIỂN THỊ
# ============================================================

def _tao_bang_hien_thi(
    bang_gia,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    bang = bang_gia.copy()

    # --------------------------------------------------------
    # Cột trạng thái
    # --------------------------------------------------------

    bang["TT"] = (
        bang[
            "thay_doi_pct"
        ]
        .apply(
            _trang_thai_ky_hieu
        )
    )

    # --------------------------------------------------------
    # Cột hiển thị
    # --------------------------------------------------------

    bang["Mã"] = (
        bang["ma"]
        .astype(str)
        .str.upper()
    )

    bang["Tên doanh nghiệp"] = (
        bang[
            "ten_doanh_nghiep"
        ]
        .fillna("")
        .astype(str)
    )

    bang["Sàn"] = (
        bang[
            "san"
        ]
        .fillna("")
        .astype(str)
        .replace(
            "",
            "—",
        )
    )

    bang["Ngành"] = (
        bang[
            "ten_nganh"
        ]
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
        bang[
            "gia_tham_chieu"
        ]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Mở cửa"] = (
        bang[
            "gia_mo_cua"
        ]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Cao nhất"] = (
        bang[
            "gia_cao_nhat"
        ]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Thấp nhất"] = (
        bang[
            "gia_thap_nhat"
        ]
        .apply(
            _dinh_dang_gia
        )
    )

    bang["Thay đổi"] = (
        bang[
            "thay_doi_pct"
        ]
        .apply(
            _dinh_dang_phan_tram
        )
    )

    bang["Khối lượng"] = (
        bang[
            "khoi_luong"
        ]
        .apply(
            _dinh_dang_khoi_luong
        )
    )

    bang["Giá trị giao dịch"] = (
        bang[
            "gia_tri_giao_dich"
        ]
        .apply(
            _dinh_dang_gia_tri
        )
    )

    return bang[
        [
            "TT",
            "Mã",
            "Tên doanh nghiệp",
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
    ].copy()


# ============================================================
# TẠO BẢNG NHỎ
# ============================================================

def _tao_bang_top(
    bang_gia,
):
    if (
        bang_gia is None
        or bang_gia.empty
    ):
        return pd.DataFrame()

    bang = bang_gia.copy()

    bang["Mã"] = (
        bang["ma"]
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
        bang[
            "thay_doi_pct"
        ]
        .apply(
            _dinh_dang_phan_tram
        )
    )

    bang["Khối lượng"] = (
        bang[
            "khoi_luong"
        ]
        .apply(
            _dinh_dang_khoi_luong
        )
    )

    bang["Giá trị giao dịch"] = (
        bang[
            "gia_tri_giao_dich"
        ]
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
    ].copy()


# ============================================================
# THỐNG KÊ THẺ
# ============================================================

def _hien_thi_metric(
    cot,
    tieu_de,
    gia_tri,
    ghi_chu=None,
):
    with cot:

        if ghi_chu:

            st.metric(
                tieu_de,
                gia_tri,
                help=ghi_chu,
            )

        else:

            st.metric(
                tieu_de,
                gia_tri,
            )


# ============================================================
# TRANG THỊ TRƯỜNG
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
        "dữ liệu cập nhật theo chu kỳ 60 giây"
    )

    # ========================================================
    # TẢI DATA ENGINE
    # ========================================================

    try:

        with st.spinner(
            "Đang tải dữ liệu toàn thị trường..."
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

    tong_ma = int(
        thong_ke.get(
            "co_du_lieu_gia",
            len(bang_gia),
        )
    )

    tang = int(
        thong_ke.get(
            "tang",
            0,
        )
    )

    dung_gia = int(
        thong_ke.get(
            "dung_gia",
            0,
        )
    )

    giam = int(
        thong_ke.get(
            "giam",
            0,
        )
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
    # BỘ LỌC
    # ========================================================

    co_1, co_2, co_3 = st.columns(
        [
            2.4,
            1,
            1.8,
        ]
    )

    with co_1:

        tu_khoa = st.text_input(
            "Tìm mã / doanh nghiệp",
            placeholder=(
                "Ví dụ: HPG, VHM, FPT..."
            ),
            key="thi_truong_tu_khoa",
        )

    with co_2:

        danh_sach_san = [
            "Tất cả"
        ]

        if "san" in bang_gia.columns:

            san_duy_nhat = (
                bang_gia["san"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace(
                    "",
                    pd.NA,
                )
                .dropna()
                .unique()
                .tolist()
            )

            danh_sach_san.extend(
                sorted(
                    san_duy_nhat
                )
            )

        san = st.selectbox(
            "Sàn",
            danh_sach_san,
            key="thi_truong_san",
        )

    with co_3:

        danh_sach_nganh = [
            "Tất cả"
        ]

        if "ten_nganh" in bang_gia.columns:

            nganh_duy_nhat = (
                bang_gia[
                    "ten_nganh"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace(
                    "",
                    pd.NA,
                )
                .dropna()
                .unique()
                .tolist()
            )

            danh_sach_nganh.extend(
                sorted(
                    nganh_duy_nhat
                )
            )

        nganh = st.selectbox(
            "Ngành",
            danh_sach_nganh,
            key="thi_truong_nganh",
        )

    # ========================================================
    # HÀNG BỘ LỌC THỨ HAI
    # ========================================================

    co_4, co_5, co_6 = st.columns(
        [
            1,
            1.6,
            1.2,
        ]
    )

    with co_4:

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

    with co_5:

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

    with co_6:

        thu_tu = st.selectbox(
            "Thứ tự",
            [
                "Giảm dần",
                "Tăng dần",
            ],
            key="thi_truong_thu_tu",
        )

    # ========================================================
    # SỐ DÒNG HIỂN THỊ
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
    # ÁP DỤNG LỌC
    # ========================================================

    bang_loc = loc_bang_gia(
        bang_gia,
        san=san,
        tu_khoa=tu_khoa,
        nganh=nganh,
        huong=trang_thai,
    )

    # ========================================================
    # ÁP DỤNG SẮP XẾP
    # ========================================================

    anh_xa_sap_xep = {
        "Tăng/giảm (%)": "thay_doi_pct",
        "Giá trị giao dịch": (
            "gia_tri_giao_dich"
        ),
        "Khối lượng": "khoi_luong",
        "Giá": "gia",
        "Mã cổ phiếu": "ma",
    }

    cot_sap_xep = anh_xa_sap_xep[
        sap_xep
    ]

    if (
        not bang_loc.empty
        and cot_sap_xep
        in bang_loc.columns
    ):

        if cot_sap_xep == "ma":

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
        f"lọc còn {len(bang_loc):,} mã"
    )

    # ========================================================
    # HIỂN THỊ BẢNG
    # ========================================================

    if bang_loc.empty:

        st.info(
            "Không có mã nào phù hợp với bộ lọc."
        )

    else:

        bang_hien_thi = _tao_bang_hien_thi(
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
                "Tên doanh nghiệp": st.column_config.TextColumn(
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
    # THANH THỐNG KÊ
    # ========================================================

    st.subheader(
        "📌 Độ rộng thị trường"
    )

    a, b, c, d = st.columns(4)

    _hien_thi_metric(
        a,
        "Tăng",
        f"{tang:,} mã",
    )

    _hien_thi_metric(
        b,
        "Đứng giá",
        f"{dung_gia:,} mã",
    )

    _hien_thi_metric(
        c,
        "Giảm",
        f"{giam:,} mã",
    )

    _hien_thi_metric(
        d,
        "Có dữ liệu",
        f"{tong_ma:,} mã",
    )

    # ========================================================
    # THANH KHOẢN
    # ========================================================

    st.subheader(
        "💰 Thanh khoản toàn thị trường"
    )

    a, b = st.columns(2)

    _hien_thi_metric(
        a,
        "Tổng khối lượng",
        _dinh_dang_khoi_luong(
            tong_khoi_luong
        ),
    )

    _hien_thi_metric(
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

    nhan_tam_ly = (
        du_lieu_tong.get(
            "nhan_tam_ly",
            "Trung tính",
        )
    )

    a, b, c = st.columns(3)

    _hien_thi_metric(
        a,
        "Điểm tâm lý",
        f"{diem_tam_ly:.0f}/100",
    )

    _hien_thi_metric(
        b,
        "Trạng thái",
        nhan_tam_ly,
    )

    _hien_thi_metric(
        c,
        "Tỷ lệ mã tăng",
        (
            f"{_so(
                thong_ke.get(
                    "phan_tram_tang"
                ),
                0,
            ):.1f}%"
        ),
    )

    # ========================================================
    # CỔ PHIẾU NỔI BẬT
    # ========================================================

    st.subheader(
        "🏆 Cổ phiếu nổi bật"
    )

    col_a, col_b, col_c = st.columns(3)

    top_tang = (
        du_lieu_tong.get(
            "top_tang",
            pd.DataFrame(),
        )
    )

    top_giam = (
        du_lieu_tong.get(
            "top_giam",
            pd.DataFrame(),
        )
    )

    top_khoi_luong = (
        du_lieu_tong.get(
            "top_khoi_luong",
            pd.DataFrame(),
        )
    )

    with col_a:

        st.markdown(
            "#### 🟢 Tăng mạnh"
        )

        st.dataframe(
            _tao_bang_top(
                top_tang
            ),
            width="stretch",
            height=420,
            hide_index=True,
        )

    with col_b:

        st.markdown(
            "#### 🔴 Giảm mạnh"
        )

        st.dataframe(
            _tao_bang_top(
                top_giam
            ),
            width="stretch",
            height=420,
            hide_index=True,
        )

    with col_c:

        st.markdown(
            "#### 💧 Khối lượng lớn"
        )

        st.dataframe(
            _tao_bang_top(
                top_khoi_luong
            ),
            width="stretch",
            height=420,
            hide_index=True,
        )

    # ========================================================
    # NHÓM NGÀNH
    # ========================================================

    st.subheader(
        "📚 Diễn biến nhóm ngành"
    )

    theo_nganh = (
        du_lieu_tong.get(
            "theo_nganh",
            pd.DataFrame(),
        )
    )

    if (
        theo_nganh is not None
        and not theo_nganh.empty
    ):

        bang_nganh = (
            theo_nganh.copy()
        )

        bang_nganh[
            "Biến động"
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

        bang_nganh[
            "Ngành"
        ] = (
            bang_nganh[
                "ten_nganh"
            ]
        )

        bang_nganh[
            "Số mã"
        ] = (
            bang_nganh[
                "so_ma"
            ]
        )

        bang_nganh[
            "Tăng"
        ] = (
            bang_nganh[
                "tang"
            ]
        )

        bang_nganh[
            "Đứng giá"
        ] = (
            bang_nganh[
                "dung_gia"
            ]
        )

        bang_nganh[
            "Giảm"
        ] = (
            bang_nganh[
                "giam"
            ]
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
            "Nguồn dữ liệu hiện chưa có thông tin ngành."
        )

    # ========================================================
    # NGOẠI
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

        _hien_thi_metric(
            a,
            "Nước ngoài mua",
            _dinh_dang_khoi_luong(
                nuoc_ngoai.get(
                    "mua"
                )
            ),
        )

        _hien_thi_metric(
            b,
            "Nước ngoài bán",
            _dinh_dang_khoi_luong(
                nuoc_ngoai.get(
                    "ban"
                )
            ),
        )

        _hien_thi_metric(
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

    thong_tin_nguon = (
        du_lieu_tong.get(
            "nguon",
            {},
        )
    )

    st.caption(
        f"Nguồn: "
        f"{thong_tin_nguon.get(
            'nguon',
            'Vnstock Market',
        )}"
        f" · "
        f"{thong_tin_nguon.get(
            'so_ma',
            len(bang_gia),
        ):,} mã"
    )
