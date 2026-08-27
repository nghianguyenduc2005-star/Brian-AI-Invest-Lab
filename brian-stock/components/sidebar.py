from __future__ import annotations

import base64
import os

import streamlit as st


# ============================================================
# ĐƯỜNG DẪN GỐC
# ============================================================

THU_MUC_DU_AN = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

THU_MUC_TAI_NGUYEN = os.path.join(
    THU_MUC_DU_AN,
    "assets",
)


# ============================================================
# TÌM ẢNH NỀN
# ============================================================

def _tim_anh_nen():

    ten_uu_tien = [
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

    for ten_tep in ten_uu_tien:

        duong_dan = os.path.join(
            THU_MUC_TAI_NGUYEN,
            ten_tep,
        )

        if os.path.isfile(
            duong_dan
        ):
            return duong_dan

    # --------------------------------------------------------
    # Tự quét thư mục assets nếu tên ảnh khác
    # --------------------------------------------------------

    if os.path.isdir(
        THU_MUC_TAI_NGUYEN
    ):

        for ten_tep in os.listdir(
            THU_MUC_TAI_NGUYEN
        ):

            duong_dan = os.path.join(
                THU_MUC_TAI_NGUYEN,
                ten_tep,
            )

            if not os.path.isfile(
                duong_dan
            ):
                continue

            phan_mo_rong = os.path.splitext(
                ten_tep
            )[1].lower()

            if phan_mo_rong in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:

                return duong_dan

    return None


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

    for ten_tep in danh_sach:

        duong_dan = os.path.join(
            THU_MUC_TAI_NGUYEN,
            ten_tep,
        )

        if os.path.isfile(
            duong_dan
        ):
            return duong_dan

    return None


# ============================================================
# NHÚNG ẢNH THÀNH BASE64
# ============================================================

def _anh_thanh_base64(
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

        phan_mo_rong = os.path.splitext(
            duong_dan
        )[1].lower()

        anh_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(
            phan_mo_rong,
            "image/jpeg",
        )

        return (
            f"data:{anh_type};base64,"
            f"{du_lieu}"
        )

    except Exception:
        return None


# ============================================================
# NẠP GIAO DIỆN
# ============================================================

@st.cache_data(
    show_spinner=False
)
def _lay_css_anh_nen():

    duong_dan_anh = _tim_anh_nen()

    anh_base64 = _anh_thanh_base64(
        duong_dan_anh
    )

    if not anh_base64:

        return """
        background:
            linear-gradient(
                135deg,
                #050b12 0%,
                #07131d 50%,
                #090d15 100%
            );
        """

    return f"""
    background-image:
        linear-gradient(
            rgba(3, 10, 17, 0.80),
            rgba(3, 10, 17, 0.88)
        ),
        url("{anh_base64}");

    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    """


def _nap_giao_dien():

    css_anh_nen = _lay_css_anh_nen()

    st.markdown(
        f"""
        <style>

        /* ==================================================
           TOÀN ỨNG DỤNG
           ================================================== */

        .stApp {{
            {css_anh_nen}
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

        [data-testid="stDecoration"] {{
            background: transparent !important;
        }}

        .main {{
            background: transparent !important;
        }}

        .block-container {{
            background: transparent !important;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }}

        /* ==================================================
           SIDEBAR
           ================================================== */

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    rgba(4, 10, 17, 0.96),
                    rgba(4, 10, 17, 0.94)
                ) !important;

            border-right:
                1px solid
                rgba(90, 120, 145, 0.20);
        }}

        [data-testid="stSidebarContent"] {{
            background: transparent !important;
        }}

        /* ==================================================
           HERO
           ================================================== */

        .hero {{
            background:
                rgba(4, 15, 25, 0.66);

            border:
                1px solid
                rgba(92, 130, 155, 0.34);

            border-radius: 18px;

            padding: 32px;

            margin-bottom: 28px;

            backdrop-filter: blur(7px);
        }}

        .eyebrow {{
            color: #ff4d4d;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }}

        .hero h1 {{
            margin: 0;
            color: #ffffff;
            font-size: 42px;
            font-weight: 800;
            line-height: 1.1;
        }}

        .hero p {{
            color: #a7bbcb;
            font-size: 15px;
            line-height: 1.6;
            margin-top: 15px;
            margin-bottom: 0;
        }}

        /* ==================================================
           SECTION
           ================================================== */

        .section-title {{
            color: #ffffff;
            font-size: 21px;
            font-weight: 800;
            margin-top: 25px;
            margin-bottom: 14px;
        }}

        /* ==================================================
           CARD
           ================================================== */

        [data-testid="stMetric"] {{
            background:
                rgba(6, 20, 31, 0.82) !important;

            border:
                1px solid
                rgba(74, 111, 138, 0.35);

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
                rgba(39, 39, 49, 0.95) !important;

            border-radius: 9px;
        }}

        div[data-baseweb="select"] {{
            background:
                rgba(39, 39, 49, 0.95) !important;

            border-radius: 9px;
        }}

        input {{
            color: #ffffff !important;
        }}

        /* ==================================================
           NÚT
           ================================================== */

        .stButton > button {{
            border-radius: 9px;
            font-weight: 700;
        }}

        /* ==================================================
           TIN TỨC
           ================================================== */

        .news-card {{
            background:
                rgba(5, 19, 30, 0.84);

            border:
                1px solid
                rgba(74, 111, 138, 0.34);

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
            color: #8da5b7;
            font-size: 12px;
            margin-top: 7px;
        }}

        /* ==================================================
           SIDEBAR THƯƠNG HIỆU
           ================================================== */

        .thuong-hieu {{
            padding:
                5px
                2px
                18px
                2px;
        }}

        .thuong-hieu-ten {{
            color: #ffffff;
            font-size: 17px;
            font-weight: 800;
            margin-top: 7px;
        }}

        .thuong-hieu-phu {{
            color: #8499aa;
            font-size: 10px;
            letter-spacing: 0.7px;
            margin-top: 4px;
        }}

        .thuong-hieu-mo-ta {{
            color: #8195a5;
            font-size: 11px;
            line-height: 1.7;
            margin-top: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    _nap_giao_dien()

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
                width=120,
            )

        st.markdown(
            """
            <div class="thuong-hieu">

                <div class="thuong-hieu-ten">
                    BRIAN STOCK
                </div>

                <div class="thuong-hieu-phu">
                    INVESTMENT INTELLIGENCE
                </div>

                <div class="thuong-hieu-mo-ta">
                    Nền tảng nghiên cứu đầu tư
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "---"
        )

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

        st.markdown(
            "---"
        )

        st.markdown(
            """
            <div style="
                padding: 4px;
                color: #708696;
                font-size: 10px;
                line-height: 1.8;
            ">
                <b style="
                    color: #b8c8d4;
                    font-size: 11px;
                ">
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
