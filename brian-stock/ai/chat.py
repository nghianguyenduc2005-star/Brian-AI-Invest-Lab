import streamlit as st
from ai.client import ask

def render_chat():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages=[]
    for m in st.session_state.chat_messages:
        role = "AI" if m["role"]=="assistant" else "Bạn"
        cls = "chat-bubble" if m["role"]=="assistant" else "chat-bubble chat-user"
        st.markdown(f'<div class="{cls}"><b>{role}</b><br>{m["content"]}</div>', unsafe_allow_html=True)
    prompt=st.chat_input("Hỏi BRIAN AI về cổ phiếu, thị trường, tin tức...")
    if prompt:
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        with st.spinner("BRIAN AI đang xử lý..."):
            text,err=ask(prompt)
        answer=text or err
        st.session_state.chat_messages.append({"role":"assistant","content":answer})
        st.rerun()
