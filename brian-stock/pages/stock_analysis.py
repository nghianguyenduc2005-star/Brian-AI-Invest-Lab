import streamlit as st
import pandas as pd
from data.market import normalize_symbol, display_symbol, load_market_data
from components.charts import price_volume_chart
from analysis.quant import run_quant
from analysis.technical import snapshot

def render_stock_analysis():
    st.markdown('<div class="section-title">📊 Phân tích cổ phiếu</div>',unsafe_allow_html=True)
    c1,c2=st.columns([1,1])
    with c1: symbol=st.text_input("Mã cổ phiếu",value="HPG")
    with c2: period=st.selectbox("Khoảng dữ liệu",["3mo","6mo","1y","2y","5y"],index=2)
    try:
        df=load_market_data(normalize_symbol(symbol),period)
    except Exception as e:
        st.error(str(e)); return
    x=snapshot(df)
    a,b,c,d,e=st.columns(5)
    a.metric("Giá",f'{x["close"]:,.0f}')
    b.metric("1D",f'{x["return"]*100:+.2f}%')
    c.metric("RSI",f'{x["rsi"]:.1f}')
    d.metric("MACD",f'{x["macd"]:.3f}')
    e.metric("Volume",f'{x["volume"]:,.0f}')
    st.plotly_chart(price_volume_chart(df),use_container_width=True,config={"displaylogo":False})

    st.markdown("### 📐 Quant — tự chọn biến")
    numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    indep=st.multiselect("Biến độc lập",numeric,default=[c for c in ["RSI","MACD","SMA20","SMA50"] if c in numeric])
    dep=st.selectbox("Biến phụ thuộc",numeric,index=numeric.index("Return") if "Return" in numeric else 0)
    if st.button("Chạy mô hình",type="primary"):
        try:
            result=run_quant(df,indep,dep)
            st.write(f"OLS R²: **{result['ols'].rsquared:.3f}** · RF R² test: **{result['r2']:.3f}** · MAE: **{result['mae']:.5f}**")
            st.dataframe(pd.DataFrame({"Hệ số":result["ols"].params,"p-value":result["ols"].pvalues}),use_container_width=True)
            st.bar_chart(result["importance"])
        except Exception as e:
            st.error(str(e))
