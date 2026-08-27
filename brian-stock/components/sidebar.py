import streamlit as st
from components.theme import inject_theme
from config.settings import LOGO_PATH, DEFAULT_MODELS

def render_sidebar():
    inject_theme()
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=150)
    st.sidebar.markdown("### BRIAN STOCK")
    st.sidebar.caption("Investment Research Platform")
    st.sidebar.markdown("---")

    pages = ["Dashboard", "Phân tích cổ phiếu", "Tin tức thị trường", "Danh mục", "AI Assistant"]
    current = st.session_state.get("page", "Dashboard")
    for p in pages:
        if st.sidebar.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Model")
    st.session_state.ai_model = st.sidebar.selectbox(
        "Model",
        DEFAULT_MODELS,
        index=0,
        help="Chỉ gọi model khi mày bấm gửi/phân tích.",
    )
    st.sidebar.caption("AI không tự chạy khi mở dashboard.")
