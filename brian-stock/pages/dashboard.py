import streamlit as st

from components.cards import metric_card
from data.market import normalize_symbol, load_market_data
from components.charts import price_volume_chart
from data.news import fetch_market_news


def render_dashboard():
    # =========================
    # HERO
    # =========================
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">BRIAN STOCK · INVESTMENT INTELLIGENCE</div>
          <h1>Góc nhìn dữ liệu cho nhà đầu tư</h1>
          <p>
            Dashboard nghiên cứu thị trường, cổ phiếu, tin tức và AI.
            Dữ liệu được tải khi cần, không dùng dữ liệu random.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================
    # QUICK OVERVIEW
    # =========================
    st.markdown(
        '<div class="section-title">📌 Theo dõi nhanh</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    for c, label, val, sub in [
        (c1, "VN-INDEX", "—", "Kết nối nguồn chỉ số"),
        (c2, "Thanh khoản", "—", "Dữ liệu thị trường"),
        (c3, "Tin tức", "LIVE", "Google News + nguồn Việt Nam"),
        (c4, "AI", "READY", "Chỉ gọi khi yêu cầu"),
    ]:
        with c:
            metric_card(label, val, sub)

    # =========================
    # STOCK DATA
    # =========================
    st.markdown(
        '<div class="section-title">📈 Theo dõi cổ phiếu</div>',
        unsafe_allow_html=True,
    )

    symbol = st.text_input(
        "Mã cổ phiếu",
        value=st.session_state.get("dashboard_input", "HPG"),
        label_visibility="collapsed",
        placeholder="Nhập mã cổ phiếu, ví dụ HPG",
    )

    if st.button("Tải dữ liệu", type="primary"):
        clean_symbol = normalize_symbol(symbol)

        if clean_symbol:
            st.session_state.dashboard_symbol = clean_symbol
            st.session_state.dashboard_input = symbol.upper().strip()
            st.rerun()

    active = st.session_state.get("dashboard_symbol", "HPG.VN")

    try:
        df = load_market_data(active, "1y")

        # Kiểm tra dữ liệu
        if df is None or df.empty:
            st.warning(
                f"Không lấy được dữ liệu cho {active}. "
                "Nguồn dữ liệu có thể đang giới hạn request."
            )
        else:
            last = df.iloc[-1]

            a, b, c, d = st.columns(4)

            # Giá
            if "Close" in df.columns:
                a.metric(
                    "Giá",
                    f"{float(last['Close']):,.0f}"
                )
            else:
                a.metric("Giá", "—")

            # 1D
            if "Return" in df.columns:
                b.metric(
                    "1D",
                    f"{float(last['Return']) * 100:+.2f}%"
                )
            else:
                b.metric("1D", "—")

            # RSI
            if "RSI" in df.columns:
                c.metric(
                    "RSI",
                    f"{float(last['RSI']):.1f}"
                )
            else:
                c.metric("RSI", "—")

            # Volume
            if "Volume" in df.columns:
                d.metric(
                    "Volume",
                    f"{float(last['Volume']):,.0f}"
                )
            else:
                d.metric("Volume", "—")

            # Chart
            try:
                fig = price_volume_chart(df)

                if fig is not None:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"displaylogo": False},
                    )
            except Exception as chart_error:
                st.warning(f"Không thể hiển thị biểu đồ: {chart_error}")

    except Exception as e:
        st.warning(
            f"Không tải được dữ liệu {active}. "
            f"Chi tiết: {e}"
        )

    # =========================
    # NEWS
    # =========================
    st.markdown(
        '<div class="section-title">📰 Tin mới</div>',
        unsafe_allow_html=True,
    )

    try:
        news = fetch_market_news(6)

        if not news:
            st.info("Hiện chưa lấy được tin tức mới.")
        else:
            for n in news:
                title = n.get("title", "Không có tiêu đề")
                source = n.get("source", "Nguồn không xác định")
                published = n.get("published", "")

                st.markdown(
                    f"""
                    <div class="news-card">
                      <div class="news-title">{title}</div>
                      <div class="news-meta">
                        {source} · {published}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.info(f"Chưa thể tải tin tức: {e}")
