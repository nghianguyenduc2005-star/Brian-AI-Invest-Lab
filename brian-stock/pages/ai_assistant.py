import streamlit as st
from ai.chat import render_chat
from ai.client_message import make_client_message

def render_ai_assistant():
    st.markdown('<div class="section-title">🤖 BRIAN AI Assistant</div>',unsafe_allow_html=True)
    st.caption("Chat, phân tích theo prompt của mày, hoặc soạn tin MUA/BÁN ngắn để gửi khách.")

    st.markdown("### 💬 Chat")
    render_chat()

    st.markdown("---")
    st.markdown("### 📤 Soạn tin cho khách")
    c1,c2,c3=st.columns([1,1,1])
    with c1: action=st.selectbox("Hành động",["MUA","BÁN"])
    with c2: symbol=st.text_input("Mã",value="HPG")
    with c3: price=st.number_input("Giá tham chiếu",min_value=0.0,value=22.2,step=.01)
    extra=st.text_input("Ghi chú thêm (không bắt buộc)")
    if st.button("Soạn tin",type="primary"):
        text,err=make_client_message(action,symbol,price,extra)
        if text: st.success("Đã soạn xong"); st.markdown(f"> {text}")
        else: st.error(err)

    st.markdown("### 📷 Ảnh bảng giá")
    st.file_uploader("Tải ảnh bảng giá để AI đọc ở bước nâng cấp OCR",type=["png","jpg","jpeg"])
    st.caption("Khung upload đã có sẵn; OCR ảnh + trích giá sẽ được tách thành module riêng để nâng cấp.")
