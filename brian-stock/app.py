import streamlit as st

from config.settings import APP_NAME

from pages.dashboard import render_dashboard
from pages.stock_analysis import render_stock_analysis
from pages.market_news import render_market_news
from pages.portfolio import render_portfolio
from pages.ai_assistant import render_ai_assistant


# ============================================================
# CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# TRẠNG THÁI TRANG
# ============================================================

if "trang" not in st.session_state:
    st.session_state["trang"] = "Dashboard"


# ============================================================
# CSS CƠ BẢN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # THƯƠNG HIỆU
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            padding: 5px 4px 18px 4px;
        ">

            <div style="
                color:#ff4d4d;
                font-size:10px;
                font-weight:800;
                letter-spacing:2px;
            ">
                BRIAN STOCK
            </div>

            <div style="
                color:#ffffff;
                font-size:18px;
                font-weight:800;
                margin-top:5px;
            ">
                INVESTMENT INTELLIGENCE
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # NÚT TRANG
    # --------------------------------------------------------

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
            st.session_state["trang"]
            == ten_trang
        )

        if st.button(
            ten_hien_thi,
            key=f"chon_trang_{ten_trang}",
            width="stretch",
            type=(
                "primary"
                if dang_chon
                else "secondary"
            ),
        ):

            st.session_state[
                "trang"
            ] = ten_trang

            st.rerun()

    # --------------------------------------------------------
    # THÔNG TIN
    # --------------------------------------------------------

    st.markdown("---")

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


# ============================================================
# ĐIỀU HƯỚNG
# ============================================================

trang = st.session_state[
    "trang"
]


if trang == "Dashboard":

    render_dashboard()


elif trang == "Phân tích cổ phiếu":

    render_stock_analysis()


elif trang == "Tin tức thị trường":

    render_market_news()


elif trang == "Danh mục":

    render_portfolio()


elif trang == "AI Assistant":

    render_ai_assistant()


else:

    st.session_state[
        "trang"
    ] = "Dashboard"

    render_dashboard()
