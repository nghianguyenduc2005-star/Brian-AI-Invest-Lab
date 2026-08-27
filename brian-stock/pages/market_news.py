import streamlit as st
from data.news import fetch_market_news
from ai.client import ask

def render_market_news():
    st.markdown('<div class="section-title">📰 Tin tức thị trường</div>',unsafe_allow_html=True)
    if st.button("Lấy tin mới",type="primary"):
        st.cache_data.clear()
    news=fetch_market_news(20)
    for n in news:
        st.markdown(f'<div class="news-card"><div class="news-title">{n["title"]}</div><div class="news-meta">{n["source"]} · {n["published"]}</div></div>',unsafe_allow_html=True)
        if n["link"]: st.markdown(f'[Đọc bài ↗]({n["link"]})')
    st.markdown("### ✨ Tổng hợp bằng AI")
    if st.button("Tóm tắt để gửi khách"):
        text="\n".join(f"- {n['title']} ({n['source']})" for n in news)
        prompt=f"""Tóm tắt các tin dưới đây bằng tiếng Việt.
Chia 2 phần: Việt Nam và Thế giới.
Mỗi phần 3-5 ý. Cuối cùng viết một đoạn ngắn có thể gửi khách hàng.
Không bịa thông tin.
TIN:
{text}"""
        answer,err=ask(prompt)
        if answer: st.markdown(answer)
        else: st.error(err)
