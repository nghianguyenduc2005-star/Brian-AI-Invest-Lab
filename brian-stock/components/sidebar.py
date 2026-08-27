from __future__ import annotations

import base64
import os

import streamlit as st


# ============================================================
# ĐƯỜNG DẪN
# ============================================================

THU_MUC_GOC = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

THU_MUC_ASSETS = os.path.join(
    THU_MUC_GOC,
    "assets",
)


# ============================================================
# TÌM LOGO
# ============================================================

def _tim_logo():

    danh_sach = [
        "logo.png",
        "logo.jpg",
        "logo.jpeg",
        "logo.webp",
        "brian-stock.png",
        "brian_stock.png",
        "brian.png",
    ]

    for ten in danh_sach:

        duong_dan = os.path.join(
            THU_MUC_ASSETS,
            ten,
        )

        if os.path.isfile(
            duong_dan
        ):
            return duong_dan

    return None


# ============================================================
# TÌM ẢNH NỀN
# ============================================================

def _tim_anh_nen():

    danh_sach = [
        "background.jpg",
        "background.jpeg",
        "background.png",
        "background.webp",
        "bg.jpg",
        "bg.jpeg",
        "bg.png",
        "bg.webp",
        "hero.jpg",
        "hero.jpeg",
        "hero.png",
        "hero.webp",
        "nen.jpg",
        "nen.jpeg",
        "nen.png",
        "nen.webp",
    ]

    for ten in danh_sach:

        duong_dan = os.path.join(
            THU_MUC_ASSETS,
            ten,
        )

        if os.path.isfile(
            duong_dan
        ):
            return duong_dan

    # Tự tìm ảnh đầu tiên trong assets
    if os.path.isdir(
        THU_MUC_ASSETS
    ):

        for ten in os.listdir(
            THU_MUC_ASSETS
        ):

            if "logo" in ten.lower():
                continue

            duong_dan = os.path.join(
                THU_MUC_ASSETS,
                ten,
            )

            if not os.path.isfile(
                duong_dan
            ):
                continue

            duoi = os.path.splitext(
                ten
            )[1].lower()

            if duoi in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                return duong_dan

    return None


# ============================================================
# CHUYỂN ẢNH SANG BASE64
# ============================================================

def _anh_base64(
    duong_dan,
):

    if not duong_dan:
        return None

    try:

        with open(
            duong_dan,
            "rb",
        ) as tep:

            noi_dung = tep.read()

        du_lieu = base64.b64encode(
            noi_dung
        ).decode(
            "utf-8"
        )

        duoi = os.path.splitext(
            duong_dan
        )[1].lower()

        if duoi == ".png":
            loai = "image/png"

        elif duoi in {
            ".jpg",
            ".jpeg",
        }:
            loai = "image/jpeg"

        elif duoi == ".webp":
            loai = "image/webp"

        else:
            loai = "image/jpeg"

        return (
            f"data:{loai};base64,"
            f"{du_lieu}"
        )

    except Exception:

        return None


# ============================================================
# NẠP CSS
# ============================================================

def _nap_css():

    anh_nen = _tim_anh_nen()

    anh_nen_base64 = _anh_base64(
        anh_nen
    )

    if anh_nen_base64:

        css_nen = f"""
        background-image:
            linear-gradient(
                rgba(3, 9, 16, 0.80),
                rgba(3, 9, 16, 0.88)
            ),
            url("{anh_nen_base64}");

        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        """

    else:

        css_nen = """
        background:
            linear-gradient(
                135deg,
                #050b12 0%,
                #07131d 50%,
                #090d15 100%
            );
        """

    st.markdown(
        f"""
<style>

.stApp {{
    {css_nen}
    min-height: 100vh;
}}

[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

.main {{
    background: transparent !important;
}}

.block-container {{
    background: transparent !important;
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}

[data-testid="stSidebar"] {{
    background:
        rgba(3, 9, 16, 0.96) !important;
    border-right:
        1px solid
        rgba(90, 120, 145, 0.18);
}}

[data-testid="stSidebarContent"] {{
    background: transparent !important;
}}

/* Ẩn navigation tự động của Streamlit */
[data-testid="stSidebarNav"] {{
    display: none !important;
}}

.hero {{
    background:
        rgba(5, 15, 24, 0.60);
    border:
        1px solid
        rgba(100, 135, 160, 0.30);
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 26px;
    backdrop-filter: blur(8px);
}}

.eyebrow {{
    color: #ff4d4d;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}}

.section-title {{
    color: #ffffff;
    font-size: 21px;
    font-weight: 800;
    margin-top: 24px;
    margin-bottom: 14px;
}}

[data-testid="stMetric"] {{
    background:
        rgba(7, 22, 32, 0.82) !important;
    border:
        1px solid
        rgba(75, 110, 135, 0.38);
    border-radius: 14px;
    padding: 14px 16px;
}}

[data-testid="stMetricLabel"] {{
    color: #91a8b9 !important;
}}

[data-testid="stMetricValue"] {{
    color: #ffffff !important;
}}

div[data-baseweb="input"] {{
    background:
        rgba(39, 39, 49, 0.96) !important;
}}

div[data-baseweb="select"] {{
    background:
        rgba(39, 39, 49, 0.96) !important;
}}

input {{
    color: #ffffff !important;
}}

.stButton > button {{
    border-radius: 9px;
    font-weight: 700;
}}

.news-card {{
    background:
        rgba(5, 19, 30, 0.82);
    border:
        1px solid
        rgba(75, 110, 135, 0.34);
    border-radius: 14px;
    padding: 17px;
    margin-bottom: 12px;
}}

.news-title {{
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    line-height: 1.5;
}}

.news-meta {{
    color: #8fa7b8;
    font-size: 12px;
    margin-top: 7px;
}}

</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    _nap_css()

    if "page" not in st.session_state:
        st.session_state[
            "page"
        ] = "Dashboard"

    with st.sidebar:

        # ====================================================
        # LOGO
        # ====================================================

        logo = _tim_logo()

        if logo:

            st.image(
                logo,
                width=100,
            )

        st.markdown(
            "### BRIAN STOCK"
        )

        st.caption(
            "INVESTMENT INTELLIGENCE"
        )

        st.divider()

        # ====================================================
        # MENU
        # ====================================================

        cac_trang = [
            (
                "Dashboard",
                "📊 Dashboard",
            ),
            (
                "Phân tích cổ phiếu",
                "📈 Phân tích cổ phiếu",
            ),
            (
                "Tin tức thị trường",
                "📰 Tin tức thị trường",
            ),
            (
                "Danh mục",
                "💼 Danh mục",
            ),
            (
                "AI Assistant",
                "🤖 AI Assistant",
            ),
        ]

        for ten_trang, ten_hien_thi in cac_trang:

            dang_chon = (
                st.session_state[
                    "page"
                ]
                == ten_trang
            )

            if st.button(
                ten_hien_thi,
                key=(
                    "sidebar_nut_"
                    + ten_trang
                ),
                width="stretch",
                type=(
                    "primary"
                    if dang_chon
                    else "secondary"
                ),
            ):

                st.session_state[
                    "page"
                ] = ten_trang

                st.rerun()

        st.divider()

        # ====================================================
        # THÔNG TIN
        # ====================================================

        st.caption(
            "BRIAN STOCK"
        )

        st.caption(
            "Dữ liệu thị trường thật"
        )

        st.caption(
            "Phân tích kỹ thuật"
        )

        st.caption(
            "Phân tích định lượng"
        )

        st.caption(
            "Trợ lý AI"
        )
