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
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

page = st.session_state.get("page", "Dashboard")
if page == "Dashboard":
    render_dashboard()
elif page == "Phân tích cổ phiếu":
    render_stock_analysis()
elif page == "Tin tức thị trường":
    render_market_news()
elif page == "Danh mục":
    render_portfolio()
else:
    render_ai_assistant()
