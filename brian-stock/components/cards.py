import streamlit as st

def metric_card(label, value, sub="", cls="neutral"):
    st.markdown(
        f'<div class="card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {cls}">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True
    )
