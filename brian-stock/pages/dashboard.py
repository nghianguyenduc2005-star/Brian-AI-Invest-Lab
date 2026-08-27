import streamlit as st
import pandas as pd
import yfinance as yf

from components.cards import metric_card
from data.market import normalize_symbol, load_market_data
from components.charts import price_volume_chart
from data.news import fetch_market_news


# =========================================================
# VN-INDEX DATA
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_vnindex_data():
    """
    Lấy dữ liệu VN-INDEX từ Yahoo Finance.

    Thử nhiều symbol vì Yahoo có thể thay đổi cách đặt mã.
    """

    symbols = [
        "^VNINDEX.VN",
        "^VNINDEX",
        "VNINDEX.VN",
    ]

    for symbol in symbols:
        try:
            df = yf.download(
                symbol,
                period="10d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                continue

            # yfinance đôi khi trả MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [
                str(col).strip().title()
                for col in df.columns
            ]

            if "Close" not in df.columns:
                continue

            df["Close"] = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(
                    df["Volume"],
                    errors="coerce"
                )

            df = df.dropna(subset=["Close"])

            if not df.empty:
                return df

        except Exception:
            continue

    return None


# =========================================================
# FORMAT LIQUIDITY
# =========================================================

def format_volume(value):
    """
    Format volume:
    19,500,000 -> 19.50M
    1,200,000,000 -> 1.20B
    """

    if value is None or pd.isna(value):
        return "—"

    value = float(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


# =========================================================
# DASHBOARD
# =========================================================

def render_dashboard():

    # =====================================================
    # HERO
    # =====================================================

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">
            BRIAN STOCK · INVESTMENT INTELLIGENCE
          </div>

          <h1>
            Góc nhìn dữ liệu cho nhà đầu tư
          </h1>

          <p>
            Dashboard nghiên cứu thị trường, cổ phiếu, tin tức và AI.
            Dữ liệu được tải khi cần, không dùng dữ liệu random.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # QUICK OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">📌 Theo dõi nhanh</div>',
        unsafe_allow_html=True,
    )

    # Lấy VN-INDEX
    vnindex_df = load_vnindex_data()

    c1, c2, c3, c4 = st.columns(4)

    # -----------------------------------------------------
    # VN-INDEX
    # -----------------------------------------------------

    with c1:

        if vnindex_df is not None and not vnindex_df.empty:

            latest = vnindex_df.iloc[-1]

            current = float(latest["Close"])

            # Tính thay đổi so với phiên trước
            if len(vnindex_df) >= 2:

                previous = float(
                    vnindex_df.iloc[-2]["Close"]
                )

                change = current - previous

                change_pct = (
                    change / previous * 100
                    if previous != 0
                    else 0
                )

                st.metric(
                    "VN-INDEX",
                    f"{current:,.2f}",
                    f"{change:+,.2f} ({change_pct:+.2f}%)",
                )

            else:

                st.metric(
                    "VN-INDEX",
                    f"{current:,.2f}",
                )

        else:

            metric_card(
                "VN-INDEX",
                "—",
                "Chưa kết nối được nguồn chỉ số",
            )

    # -----------------------------------------------------
    # THANH KHOẢN
    # -----------------------------------------------------

    with c2:

        if vnindex_df is not None and not vnindex_df.empty:

            latest = vnindex_df.iloc[-1]

            volume = latest.get("Volume")

            if (
                volume is not None
                and not pd.isna(volume)
                and float(volume) > 0
            ):

                volume_value = float(volume)

                # So với phiên trước
                delta_text = None

                if len(vnindex_df) >= 2:

                    previous_volume = vnindex_df.iloc[-2].get(
                        "Volume"
                    )

                    if (
                        previous_volume is not None
                        and not pd.isna(previous_volume)
                        and float(previous_volume) > 0
                    ):

                        volume_change = (
                            volume_value / float(previous_volume) - 1
                        ) * 100

                        delta_text = (
                            f"{volume_change:+.1f}% so với phiên trước"
                        )

                st.metric(
                    "Thanh khoản",
                    format_volume(volume_value),
                    delta_text,
                )

            else:

                metric_card(
                    "Thanh khoản",
                    "—",
                    "Nguồn chỉ số không trả Volume",
                )

        else:

            metric_card(
                "Thanh khoản",
                "—",
                "Chưa có dữ liệu thị trường",
            )

    # -----------------------------------------------------
    # NEWS
    # -----------------------------------------------------

    with c3:

        metric_card(
            "Tin tức",
            "LIVE",
            "Google News + nguồn Việt Nam",
        )

    # -----------------------------------------------------
    # AI
    # -----------------------------------------------------

    with c4:

        metric_card(
            "AI",
            "READY",
            "Chỉ gọi khi yêu cầu",
        )

    # =====================================================
    # VN-INDEX DETAIL
    # =====================================================

    if vnindex_df is not None and not vnindex_df.empty:

        st.markdown(
            '<div class="section-title">📊 VN-INDEX</div>',
            unsafe_allow_html=True,
        )

        latest = vnindex_df.iloc[-1]

        index_col1, index_col2, index_col3, index_col4 = st.columns(4)

        current = float(latest["Close"])

        if len(vnindex_df) >= 2:

            previous = float(
                vnindex_df.iloc[-2]["Close"]
            )

            change = current - previous

            change_pct = (
                change / previous * 100
                if previous != 0
                else 0
            )

        else:

            change = 0
            change_pct = 0

        with index_col1:

            st.metric(
                "Điểm",
                f"{current:,.2f}",
            )

        with index_col2:

            st.metric(
                "Thay đổi",
                f"{change:+,.2f}",
            )

        with index_col3:

            st.metric(
                "1D",
                f"{change_pct:+.2f}%",
            )

        with index_col4:

            volume = latest.get("Volume")

            st.metric(
                "Volume",
                format_volume(volume),
            )

    # =====================================================
    # STOCK DATA
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Theo dõi cổ phiếu</div>',
        unsafe_allow_html=True,
    )

    symbol = st.text_input(
        "Mã cổ phiếu",
        value=st.session_state.get(
            "dashboard_input",
            "HPG"
        ),
        label_visibility="collapsed",
        placeholder="Nhập mã cổ phiếu, ví dụ HPG",
    )

    if st.button(
        "Tải dữ liệu",
        type="primary",
    ):

        clean_symbol = normalize_symbol(symbol)

        if clean_symbol:

            st.session_state.dashboard_symbol = clean_symbol

            st.session_state.dashboard_input = (
                symbol.upper().strip()
            )

            st.rerun()

    active = st.session_state.get(
        "dashboard_symbol",
        "HPG.VN",
    )

    # =====================================================
    # LOAD STOCK
    # =====================================================

    try:

        df = load_market_data(
            active,
            "1y",
        )

        if df is None or df.empty:

            st.warning(
                f"Không lấy được dữ liệu cho {active}."
            )

        else:

            last = df.iloc[-1]

            a, b, c, d = st.columns(4)

            # -------------------------------------------------
            # PRICE
            # -------------------------------------------------

            with a:

                if "Close" in df.columns:

                    st.metric(
                        "Giá",
                        f"{float(last['Close']):,.0f}",
                    )

                else:

                    st.metric(
                        "Giá",
                        "—",
                    )

            # -------------------------------------------------
            # 1D
            # -------------------------------------------------

            with b:

                if "Return" in df.columns:

                    st.metric(
                        "1D",
                        f"{float(last['Return']) * 100:+.2f}%",
                    )

                else:

                    st.metric(
                        "1D",
                        "—",
                    )

            # -------------------------------------------------
            # RSI
            # -------------------------------------------------

            with c:

                if "RSI" in df.columns:

                    st.metric(
                        "RSI",
                        f"{float(last['RSI']):.1f}",
                    )

                else:

                    st.metric(
                        "RSI",
                        "—",
                    )

            # -------------------------------------------------
            # VOLUME
            # -------------------------------------------------

            with d:

                if "Volume" in df.columns:

                    st.metric(
                        "Volume",
                        f"{float(last['Volume']):,.0f}",
                    )

                else:

                    st.metric(
                        "Volume",
                        "—",
                    )

            # -------------------------------------------------
            # STOCK CHART
            # -------------------------------------------------

            try:

                fig = price_volume_chart(df)

                if fig is not None:

                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={
                            "displaylogo": False,
                        },
                    )

            except Exception as chart_error:

                st.warning(
                    f"Không thể hiển thị biểu đồ: "
                    f"{chart_error}"
                )

    except Exception as e:

        st.warning(
            f"Không tải được dữ liệu {active}. "
            f"Chi tiết: {e}"
        )

    # =====================================================
    # NEWS
    # =====================================================

    st.markdown(
        '<div class="section-title">📰 Tin mới</div>',
        unsafe_allow_html=True,
    )

    try:

        news = fetch_market_news(6)

        if not news:

            st.info(
                "Hiện chưa lấy được tin tức mới."
            )

        else:

            for n in news:

                title = n.get(
                    "title",
                    "Không có tiêu đề",
                )

                source = n.get(
                    "source",
                    "Nguồn không xác định",
                )

                published = n.get(
                    "published",
                    "",
                )

                link = n.get(
                    "link",
                    "",
                )

                st.markdown(
                    f"""
                    <div class="news-card">

                      <div class="news-title">
                        {title}
                      </div>

                      <div class="news-meta">
                        {source} · {published}
                      </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if link:

                    st.markdown(
                        f"[Đọc bài ↗]({link})"
                    )

    except Exception as e:

        st.info(
            f"Chưa thể tải tin tức: {e}"
        )
