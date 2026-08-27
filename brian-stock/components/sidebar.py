import os

import streamlit as st


# ============================================================
# TÌM ẢNH NỀN
# ============================================================

def _tim_anh_nen():
    danh_sach = [
        "assets/background.jpg",
        "assets/background.jpeg",
        "assets/background.png",
        "assets/bg.jpg",
        "assets/bg.jpeg",
        "assets/bg.png",
        "assets/hero.jpg",
        "assets/hero.jpeg",
        "assets/hero.png",
        "assets/logo.png",
    ]

    thu_muc_goc = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    for duong_dan in danh_sach:

        duong_dan_day_du = os.path.join(
            thu_muc_goc,
            duong_dan,
        )

        if os.path.exists(
            duong_dan_day_du
        ):
            return duong_dan_day_du

    return None


# ============================================================
# CSS TOÀN ỨNG DỤNG
# ============================================================

def _nap_giao_dien():

    anh_nen = _tim_anh_nen()

    if anh_nen:
        anh_nen_uri = (
            "file:///"
            + anh_nen.replace(
                "\\",
                "/",
            )
        )

        css_anh_nen = f"""
        background-image:
            linear-gradient(
                rgba(3, 10, 17, 0.88),
                rgba(3, 10, 17, 0.88)
            ),
            url("{anh_nen_uri}");
        """

    else:

        css_anh_nen = """
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

        /* ==============================================
           NỀN TOÀN ỨNG DỤNG
           ============================================== */

        .stApp {{
            {css_anh_nen}
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main {{
            background: transparent !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background: transparent !important;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0) !important;
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    rgba(4,10,17,0.96),
                    rgba(4,10,17,0.96)
                ) !important;
            border-right:
                1px solid
                rgba(255,255,255,0.08);
        }}

        [data-testid="stSidebarContent"] {{
            padding-top: 1rem;
        }}

        /* ==============================================
           KHỐI NỘI DUNG
           ============================================== */

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        /* ==============================================
           TIÊU ĐỀ
           ============================================== */

        .hero {{
            position: relative;
            padding: 32px;
            margin-bottom: 28px;
            border-radius: 18px;
            border:
                1px solid
                rgba(120,160,190,0.28);
            background:
                rgba(5,15,24,0.72);
            backdrop-filter: blur(8px);
        }}

        .eyebrow {{
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 2px;
            color: #ff514f;
            margin-bottom: 12px;
        }}

        .hero h1 {{
            margin: 0;
            color: #ffffff;
            font-size: 42px;
            font-weight: 800;
            line-height: 1.1;
        }}

        .hero p {{
            color: #a8bdd0;
            font-size: 15px;
            line-height: 1.6;
            margin-top: 14px;
            margin-bottom: 0;
        }}

        /* ==============================================
           TIÊU ĐỀ SECTION
           ============================================== */

        .section-title {{
            margin-top: 26px;
            margin-bottom: 14px;
            color: #ffffff;
            font-size: 21px;
            font-weight: 800;
        }}

        /* ==============================================
           CARD
           ============================================== */

        [data-testid="stMetric"] {{
            background:
                rgba(7,22,32,0.82);
            border:
                1px solid
                rgba(65,100,125,0.42);
            border-radius: 15px;
            padding: 14px 16px;
        }}

        [data-testid="stMetricLabel"] {{
            color: #9ab1c2 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #ffffff !important;
        }}

        /* ==============================================
           INPUT
           ============================================== */

        div[data-baseweb="input"] {{
            background: #252630 !important;
        }}

        div[data-baseweb="select"] {{
            background: #252630 !important;
        }}

        input {{
            color: #ffffff !important;
        }}

        /* ==============================================
           NÚT
           ============================================== */

        .stButton > button {{
            border-radius: 10px;
            font-weight: 700;
        }}

        /* ==============================================
           TIN TỨC
           ============================================== */

        .news-card {{
            padding: 18px;
            margin-bottom: 12px;
            border-radius: 14px;
            background:
                rgba(6,20,30,0.82);
            border:
                1px solid
                rgba(65,100,125,0.38);
        }}

        .news-title {{
            color: #ffffff;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.5;
        }}

        .news-meta {{
            margin-top: 7px;
            color: #8fa7b9;
            font-size: 12px;
        }}

        /* ==============================================
           LOGO
           ============================================== */

        .brian-logo {{
            width: 100%;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 2px 18px 2px;
        }}

        .brian-logo-icon {{
            width: 38px;
            height: 38px;
            object-fit: contain;
        }}

        .brian-logo-title {{
            color: #ffffff;
            font-size: 17px;
            font-weight: 800;
            line-height: 1.1;
        }}

        .brian-logo-sub {{
            margin-top: 4px;
            color: #8399aa;
            font-size: 10px;
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

    # ========================================================
    # KHỞI TẠO TRẠNG THÁI
    # ========================================================

    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"

    with st.sidebar:

        # ====================================================
        # LOGO
        # ====================================================

        logo_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            ),
            "assets",
            "logo.png",
        )

        if os.path.exists(
            logo_path
        ):

            st.image(
                logo_path,
                width=125,
            )

        st.markdown(
            """
            <div class="brian-logo-title">
                BRIAN STOCK
            </div>

            <div class="brian-logo-sub">
                INVESTMENT INTELLIGENCE
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
                st.session_state["page"]
                == ten_trang
            )

            if st.button(
                ten_hien_thi,
                key=f"sidebar_{ten_trang}",
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
        # THÔNG TIN
        # ====================================================

        st.markdown(
            "---"
        )

        st.markdown(
            """
            <div style="
                color:#7f95a6;
                font-size:11px;
                line-height:1.8;
                padding:4px;
            ">
                <b style="color:#c6d5df;">
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
