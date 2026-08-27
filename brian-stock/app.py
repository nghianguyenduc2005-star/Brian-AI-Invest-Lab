import streamlit as st

from config.settings import APP_NAME
from components.sidebar import render_sidebar

from pages.dashboard import render_dashboard
from pages.stock_analysis import render_stock_analysis
from pages.market_news import render_market_news
from pages.portfolio import render_portfolio
from pages.ai_assistant import render_ai_assistant


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


render_sidebar()


if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"


trang = st.session_state["page"]


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
    st.session_state["page"] = "Dashboard"
    render_dashboard()
