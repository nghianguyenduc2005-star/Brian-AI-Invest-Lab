import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Brian AI Invest Lab", page_icon="⚡", layout="wide")

# CSS giao diện Dark Mode chuẩn Copilot
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Brian AI Invest Lab — Quant & ML Copilot")

# Khởi tạo lịch sử Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào Nghĩa! Bạn muốn chạy mô hình định lượng hay dự báo ML cho mã nào? Cung cấp mã CK, khung thời gian và các biến cho tớ nhé."}
    ]

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Hàm giả lập lấy dữ liệu & chạy mô hình Định lượng / ML
def run_quant_pipeline(symbol, timeframe, dep_var, indep_vars):
    # 1. Giả lập kéo dữ liệu chuỗi thời gian
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    df = pd.DataFrame({
        'Price': np.cumsum(np.random.randn(100)) + 30,
        'Volume': np.random.randint(1000, 5000, 100),
        'RSI': np.random.uniform(30, 70, 100),
        'MACD': np.random.randn(100)
    }, index=dates)
    
    # Tạo biến phụ thuộc dạng Returns nếu cần
    df['Returns'] = df['Price'].pct_change().fillna(0)
    
    # 2. Chạy mô hình Hồi quy OLS (Statsmodels)
    Y = df[dep_var] if dep_var in df.columns else df['Returns']
    X_cols = [c for c in indep_vars if c in df.columns]
    
    if not X_cols:
        X_cols = ['Volume', 'RSI']
        
    X = sm.add_constant(df[X_cols])
    model_ols = sm.OLS(Y, X).fit()
    
    # 3. Chạy Mô hình Học Máy (Random Forest Regressor)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(df[X_cols], Y)
    feature_importance = dict(zip(X_cols, rf.feature_importances_))
    
    return model_ols.summary().as_text(), feature_importance, df

# Xử lý khi người dùng gửi Prompt
if prompt := st.chat_input("Nhập yêu cầu (VD: Chạy OLS HPG từ 2024 đến nay, Y là Returns, X là Volume, RSI)..."):
    # Lưu tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Phân tích Prompt cơ bản & chạy Pipeline
    with st.chat_message("assistant"):
        st.write("🔄 **Đang xử lý dữ liệu và huấn luyện mô hình...**")
        
        # Giả định trích xuất thông tin từ prompt
        summary, importance, data = run_quant_pipeline("HPG", "2024", "Returns", ["Volume", "RSI", "MACD"])
        
        response_text = f"✅ **Đã phân tích xong dữ liệu!**\n\n"
        response_text += f"📊 **Kết quả Hồi quy OLS:**\n```text\n{summary[:800]}...\n```\n"
        response_text += f"🤖 **Độ quan trọng biến trong Mô hình Học Máy (Random Forest):**\n"
        for k, v in importance.items():
            response_text += f"* **{k}**: {v:.4f}\n"

        st.markdown(response_text)
        
        # Biểu đồ minh họa dữ liệu
        st.line_chart(data[['Price']])
        
        # Lưu phản hồi vào session
        st.session_state.messages.append({"role": "assistant", "content": response_text})
