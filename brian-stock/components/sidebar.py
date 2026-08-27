```python
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

        if os.path.isfile(duong_dan):
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

        if os.path.isfile(duong_dan):
            return duong_dan

    # Nếu tên ảnh khác thì tự quét assets
    if os.path.isdir(
        THU_MUC_ASSETS
    ):

        for ten in os.listdir(
            THU_MUC_ASSETS
        ):

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

                # Không dùng logo làm ảnh nền
                if "logo" not in ten.lower():

                    return duong_dan

    return None


# ============================================================
# NHÚNG ẢNH
# ============================================================

def _anh_base64(
    duong_dan,
):

    if not duong_dan:
        return None

    if not os.path.isfile(
        duong_dan
    ):
        return None

    try:

        with open(
            duong_dan,
            "rb",
        ) as tep:

            du_lieu = base64.b64encode(
                tep.read()
            ).decode(
                "utf-8"
            )

        duoi = os.path.splitext(
            duong_dan
        )[1].lower()

        loai = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(
            duoi,
            "image/jpeg",
        )

        return (
            f"data:{loai};base64,"
            f"{du_lieu}"
        )

    except Exception:
        return None


# ============================================================
# CSS
# ============================================================

def _nap_css():

    anh_nen = _tim_anh_nen()

    du_lieu_anh = _anh_base64(
        anh_nen
    )

    if du_lieu_anh:

        nen = f"""
        background-image:
            linear-gradient(
                rgba(3, 9, 16, 0.78),
                rgba(3, 9, 16, 0.84)
            ),
            url("{du_lieu_anh}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        """

    else:

        nen = """
        background:
            linear-gradient(
                135deg,
                #050b12,
                #07131d,
                #090d15
            );
        """

    st.markdown(
        f"""
        <style>

        /* ==================================================
           NỀN TOÀN APP
           ================================================== */

        .stApp {{
            {nen}
            min-height: 100vh;
        }}

        [data-testid="stAppViewContainer"] {{
            background: transparent !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        [data-testid="stToolbar"] {{
            background: transparent !important;
        }}

        .main {{
            background: transparent !important;
        }}

        .block-container {{
            background: transparent !important;
        }}

        /* ==================================================
           SIDEBAR
           ================================================== */

        [data-testid="stSidebar"] {{
            background:
                rgba(4, 10, 17, 0.96) !important;

            border-right:
                1px solid
                rgba(90, 120, 145, 0.18);
        }}

        [data-testid="stSidebarContent"] {{
            background: transparent !important;
        }}

        /* ==================================================
           HERO
           ================================================== */

        .hero {{
            background:
                rgba(5, 15, 24, 0.64);

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
            margin-bottom: 10px;
        }}

        .hero h1 {{
            color: #ffffff;
            margin: 0;
            font-size: 42px;
            font-weight: 800;
            line-height: 1.1;
        }}

        .hero p {{
            color: #a9bdcc;
            font-size: 15px;
            line-height: 1.6;
            margin-top: 14px;
        }}

        /* ==================================================
           SECTION
           ================================================== */

        .section-title {{
            color: #ffffff;
            font-size: 21px;
            font-weight: 800;
            margin-top: 24px;
            margin-bottom: 14px;
        }}

        /* ==================================================
           METRIC
           ================================================== */

        [data-testid="stMetric"] {{
            background:
                rgba(7, 22, 32, 0.80) !important;

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

        /* ==================================================
           INPUT
           ================================================== */

        div[data-baseweb="input"] {{
            background:
                rgba(39, 39, 49, 0.96) !important;

            border-radius: 9px;
        }}

        input {{
            color: #ffffff !important;
        }}

        /* ==================================================
           SELECT
           ================================================== */

        div[data-baseweb="select"] {{
            background:
                rgba(39, 39, 49, 0.96) !important;

            border-radius: 9px;
        }}

        /* ==================================================
           NÚT
           ================================================== */

        .stButton > button {{
            border-radius: 9px;
            font-weight: 700;
        }}

        /* ==================================================
           TIN
           ================================================== */

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
                width=105,
            )

        st.markdown(
            """
            <div style="
                padding: 4px 2px 15px 2px;
            ">

                <div style="
                    color:#ffffff;
                    font-size:17px;
                    font-weight:800;
                ">
                    BRIAN STOCK
                </div>

                <div style="
                    color:#8499aa;
                    font-size:10px;
                    letter-spacing:1px;
                    margin-top:4px;
                ">
                    INVESTMENT INTELLIGENCE
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ====================================================
        # ĐIỀU HƯỚNG
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
                    "nut_sidebar_"
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

        # ====================================================
        # CHÂN SIDEBAR
        # ====================================================

        st.markdown("---")

        st.markdown(
            """
            <div style="
                color:#718798;
                font-size:10px;
                line-height:1.8;
            ">
                <b style="color:#b9c9d4;">
                    BRIAN STOCK
                </b>
                <br>
                Investment Research Platform
                <br><br>
                Dữ liệu thị trường thật
                <br>
                Phân tích kỹ thuật
                <br>
                Phân tích định lượng
                <br>
                Trợ lý AI
            </div>
            """,
            unsafe_allow_html=True,
        )
```
