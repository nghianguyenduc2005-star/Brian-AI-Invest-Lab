import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Brian AI Invest Lab", page_icon="⚡", layout="wide")

# Custom CSS cho UI hiện đại
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2a2e39; }
    .metric-card { background: #1e222d; border-radius: 8px; padding: 16px; border: 1px solid #2a2e39; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Brian AI Invest Lab — Copilot")
st.caption("Trợ lý Phân tích Chứng khoán Đa Biến Realtime & Nghiên cứu Định lượng Chuyên sâu")

tab1, tab2 = st.tabs(["📈 Phân Tích Chứng Khoán Live", "🔬 Nghiên Cứu Định Lượng (SPSS/Stata)"])

with tab1:
    c1, c2 = st.columns([3, 1])
    with c1:
        symbol = st.text_input("Mã cổ phiếu:", value="HPG").upper()
    with c2:
        st.write("")
        st.write("")
        btn = st.button("🚀 Phân Tích Đa Biến", use_container_width=True)

    st.markdown("---")
    
    # Dashboard chỉ số
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Giá Hiện Tại", value="28,400 VND", delta="+2.3%")
    m2.metric(label="Khối Lượng GD", value="15.2M", delta="Cao hơn trung bình")
    m3.metric(label="RSI (14)", value="58.4", delta="Trung tính")
    m4.metric(label="Dòng Tiền Cá Lớn", value="Tích Cực", delta="Mua ròng")

    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 Biểu đồ biến động giá")
        chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["Giá", "Khối lượng"])
        st.line_chart(chart_data)

    with col_right:
        st.subheader("🤖 Tín hiệu AI Copilot")
        st.info("**Khuyến nghị:** MUA TÍCH LŨY")
        st.write("* **Vùng hỗ trợ:** 27,500")
        st.write("* **Vùng kháng cự:** 30,200")
        st.write("* **Xu hướng:** Tăng ngắn hạn")

with tab2:
    st.subheader("Nghiên cứu định lượng")
    st.file_uploader("Tải lên file dữ liệu (.sav, .dta, .csv)", type=["csv", "sav", "dta"])
