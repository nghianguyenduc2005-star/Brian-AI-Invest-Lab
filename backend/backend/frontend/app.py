import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Brian AI Invest Lab", page_icon="📊", layout="wide")

# Custom CSS phong cách riêng
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stButton>button { background-color: #2563eb; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #1d4ed8; }
    .brand-box { background-color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid #2563eb; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Brian AI Invest Lab — Copilot")
st.caption("Trợ lý Phân tích Chứng khoán Đa Biến Realtime & Nghiên cứu Định lượng Chuyên sâu")

tab1, tab2 = st.tabs(["📈 Phân Tích Chứng Khoán Live", "🔬 Nghiên Cứu Định Lượng (SPSS/Stata)"])

# TAB 1: CHỨNG KHOÁN REALTIME
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Nhập mã cổ phiếu (VD: HPG, SSI, VCB):", value="HPG").upper()
    with col2:
        st.write(" ")
        st.write(" ")
        btn_stock = st.button("Phân Tích Đa Biến")

    if btn_stock and symbol:
        with st.spinner("Đang lấy dữ liệu đa biến DNSE/TCBS & chạy Gemini AI..."):
            try:
                res = requests.get(f"http://127.0.0.1:8000/api/stock/{symbol}")
                if res.status_code == 200:
                    data = res.json()
                    if data.get("prices"):
                        st.subheader(f"📊 Diễn biến giá {symbol} vs VN-Index")
                        df_p = pd.DataFrame(data["prices"])
                        st.line_chart(df_p.set_index("Date")[["Close", "VNINDEX_Close"]])
                    st.subheader("🤖 Nhận Định AI Copilot")
                    st.markdown(f'<div class="brand-box">{data["analysis"]}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error("Không thể kết nối Backend. Vui lòng kiểm tra lại Server.")

# TAB 2: NGHIÊN CỨU ĐỊNH LƯỢNG
with tab2:
    topic = st.text_input("Nhập tên đề tài nghiên cứu:")
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (.sav của SPSS hoặc .arff của Weka)", type=["sav", "arff", "csv"])
    btn_research = st.button("Chạy Báo Cáo Định Lượng")

    if btn_research and uploaded_file and topic:
        with st.spinner("Đang tổng hợp báo cáo..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data_form = {"topic": topic}
                res = requests.post("http://127.0.0.1:8000/api/quant-research", files=files, data=data_form)
                if res.status_code == 200:
                    result = res.json()
                    st.subheader("📄 Báo Cáo Nghiên Cứu Định Lượng (Chuẩn Học Thuật)")
                    st.markdown(f'<div class="brand-box">{result["academic_report"]}</div>', unsafe_allow_html=True)

                    st.subheader("💻 Cú pháp Stata Đề Xuất (.do file)")
                    st.code(result["stata_code"], language="stata")
            except Exception as e:
                st.error("Lỗi khi gửi tệp phân tích.")
