import streamlit as st
from components.cards import metric_card
from data.market import normalize_symbol, display_symbol, load_market_data
from components.charts import price_volume_chart
from data.news import fetch_market_news

def render_dashboard():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">BRIAN STOCK · INVESTMENT INTELLIGENCE</div>
      <h1>Góc nhìn dữ liệu cho nhà đầu tư</h1>
      <p>Dashboard nghiên cứu thị trường, cổ phiếu, tin tức và AI. Dữ liệu được tải khi cần, không dùng dữ liệu random.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Theo dõi nhanh</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for c,label,val,sub in [
        (c1,"VN-INDEX","—","Kết nối nguồn chỉ số"),
        (c2,"Thanh khoản","—","Dữ liệu thị trường"),
        (c3,"Tin tức","LIVE","Google News + nguồn Việt Nam"),
        (c4,"AI","READY","Chỉ gọi khi yêu cầu"),
    ]:
        with c: metric_card(label,val,sub)

    st.markdown('<div class="section-title">📈 Theo dõi cổ phiếu</div>', unsafe_allow_html=True)
    symbol=st.text_input("Mã cổ phiếu",value="HPG",label_visibility="collapsed")
    if st.button("Tải dữ liệu",type="primary"):
        st.session_state.dashboard_symbol=normalize_symbol(symbol)
    active=st.session_state.get("dashboard_symbol","HPG.VN")
    try:
        df=load_market_data(active,"1y")
        last=df.iloc[-1]
        a,b,c,d=st.columns(4)
        a.metric("Giá",f"{last.Close:,.0f}")
        b.metric("1D",f"{last.Return*100:+.2f}%")
        c.metric("RSI",f"{last.RSI:.1f}")
        d.metric("Volume",f"{last.Volume:,.0f}")
        st.plotly_chart(price_volume_chart(df),use_container_width=True,config={"displaylogo":False})
    except Exception as e:
        st.warning(str(e))

    st.markdown('<div class="section-title">📰 Tin mới</div>', unsafe_allow_html=True)
    for n in fetch_market_news(6):
        st.markdown(f'<div class="news-card"><div class="news-title">{n["title"]}</div><div class="news-meta">{n["source"]} · {n["published"]}</div></div>',unsafe_allow_html=True)
