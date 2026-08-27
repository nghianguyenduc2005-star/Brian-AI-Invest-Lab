import streamlit as st
from portfolio.calculator import calculate

def portfolio_editor(df):
    edited=st.data_editor(df,use_container_width=True,num_rows="dynamic",hide_index=True)
    return calculate(edited)
