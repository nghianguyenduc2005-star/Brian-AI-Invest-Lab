import streamlit as st
import pandas as pd
from portfolio.storage import empty, COLUMNS
from portfolio.calculator import calculate

def render_portfolio():
    st.markdown('<div class="section-title">💼 Danh mục đầu tư</div>',unsafe_allow_html=True)
    if "portfolio" not in st.session_state:
        st.session_state.portfolio=empty()
    df=st.session_state.portfolio
    edited=st.data_editor(
        df,use_container_width=True,num_rows="dynamic",hide_index=True,
        column_config={
            "Ngày mua":st.column_config.DateColumn("Ngày mua"),
            "Ngày bán":st.column_config.DateColumn("Ngày bán"),
            "Giá mua":st.column_config.NumberColumn("Giá mua",format="%.2f"),
            "Giá hiện tại":st.column_config.NumberColumn("Giá hiện tại",format="%.2f"),
            "Giá bán":st.column_config.NumberColumn("Giá bán",format="%.2f"),
            "Số lượng":st.column_config.NumberColumn("Số lượng",min_value=0,step=1),
        }
    )
    st.session_state.portfolio=edited
    calc=calculate(edited)
    a,b,c,d=st.columns(4)
    a.metric("Tổng vốn",f'{calc["Giá trị mua"].sum():,.0f}')
    b.metric("Giá trị hiện tại",f'{calc["Giá trị hiện tại"].sum():,.0f}')
    c.metric("Lãi/Lỗ",f'{calc["Lãi/Lỗ"].sum():,.0f}')
    invested=calc["Giá trị mua"].sum()
    d.metric("Lãi/Lỗ %",f'{calc["Lãi/Lỗ"].sum()/invested*100:+.2f}%' if invested else "—")
    st.download_button("⬇️ Xuất CSV",calc.to_csv(index=False).encode("utf-8-sig"),"brian_stock_portfolio.csv","text/csv")
