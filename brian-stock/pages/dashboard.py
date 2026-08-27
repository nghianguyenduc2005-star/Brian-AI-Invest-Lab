import streamlit as st

from components.cards import metric_card
from data.market import normalize_symbol, load_market_data
from components.charts import price_volume_chart
from data.news import fetch_market_news


# =========================================================
# VN-INDEX + THANH KHOẢN
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_vnindex():
    """
    Lấy VN-INDEX và thanh khoản thị trường Việt Nam.

    Ưu tiên vnstock.
    Nếu nguồn không trả được volume thì thử các field
    thanh khoản khác.
    """

    try:
        from vnstock import Vnstock

        stock = Vnstock().stock(
            symbol="VNINDEX",
            source="VCI"
        )

        df = stock.quote.history(
            start="2026-01-01",
            end=None,
            interval="1D"
        )

        if df is None or df.empty:
            raise ValueError("vnstock không trả dữ liệu VN-INDEX")

        # Chuẩn hóa tên cột
        df.columns = [
            str(c).strip().lower()
            for c in df.columns
        ]

        last = df.iloc[-1]

        # -------------------------------------------------
        # Điểm VN-INDEX
        # -------------------------------------------------

        price = None

        for col in [
            "close",
            "close_price",
            "index_value",
        ]:
            if col in df.columns:
                value = last[col]

                if value is not None:
                    try:
                        price = float(value)
                        break
                    except Exception:
                        pass

        if price is None:
            raise ValueError(
                f"Không tìm thấy giá VN-INDEX. "
                f"Cột nhận được: {list(df.columns)}"
            )

        # -------------------------------------------------
        # Thay đổi
        # -------------------------------------------------

        change = 0.0

        for col in [
            "change",
            "price_change",
            "change_value",
        ]:
            if col in df.columns:
                try:
                    change = float(last[col])
                    break
                except Exception:
                    pass

        # -------------------------------------------------
        # Volume
        # -------------------------------------------------

        volume = 0.0

        for col in [
            "volume",
            "total_match_volume",
            "total_volume",
            "totalmatchvolume",
        ]:
            if col in df.columns:
                try:
                    value = last[col]

                    if value is not None:
                        volume = float(value)

                        if volume > 0:
                            break
                except Exception:
                    pass

        # -------------------------------------------------
        # Value / GTGD
        # -------------------------------------------------

        value = 0.0

        for col in [
            "value",
            "total_match_value",
            "total_value",
            "totalmatchvalue",
        ]:
            if col in df.columns:
                try:
                    v = last[col]

                    if v is not None:
                        value = float(v)

                        if value > 0:
                            break
                except Exception:
                    pass

        # -------------------------------------------------
        # Nếu có close nhưng không có change,
        # tự tính từ phiên trước.
        # -------------------------------------------------

        if len(df) >= 2:

            try:
                previous = float(
                    df.iloc[-2]["close"]
                )

                if change == 0 and previous != 0:
                    change = price - previous

            except Exception:
                pass

        # -------------------------------------------------
        # % thay đổi
        # -------------------------------------------------

        change_pct = 0.0

        if len(df) >= 2:

            try:
                previous = float(
                    df.iloc[-2]["close"]
                )

                if previous != 0:
                    change_pct = (
                        (price - previous)
                        / previous
                        * 100
                    )

            except Exception:
                pass

        return {
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
            "value": value,
            "columns": list(df.columns),
        }

    except Exception as e:

        return {
            "price": None,
            "change": None,
            "change_pct": None,
            "volume": None,
            "value": None,
            "error": str(e),
        }


def format_number(value):

    if value is None:
        return "—"

    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "—"


def format_volume(value):

    if value is None or value == 0:
        return "—"

    value = float(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.0f}"


def format_value(value):

    if value is None or value == 0:
        return "—"

    value = float(value)

    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f} nghìn tỷ"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu"

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
            Dashboard nghiên cứu thị trường, cổ phiếu,
            tin tức và AI. Dữ liệu được tải khi cần,
            không dùng dữ liệu random.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # LOAD VN-INDEX
    # =====================================================

    vn = load_vnindex()

    # =====================================================
    # QUICK OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">📌 Theo dõi nhanh</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    # VN-INDEX

    if vn.get("price") is not None:

        vn_value = format_number(
            vn["price"]
        )

        vn_sub = (
            f"{vn['change']:+.2f} điểm · "
            f"{vn['change_pct']:+.2f}%"
        )

    else:

        vn_value = "—"
        vn_sub = "Chưa lấy được dữ liệu"

    # Thanh khoản

    if vn.get("value"):
        liquidity_value = format_value(
            vn["value"]
        )

        liquidity_sub = "GTGD thị trường"

    elif vn.get("volume"):
        liquidity_value = format_volume(
            vn["volume"]
        )

        liquidity_sub = "Khối lượng thị trường"

    else:

        liquidity_value = "—"
        liquidity_sub = "Chưa có dữ liệu"

    with c1:
        metric_card(
            "VN-INDEX",
            vn_value,
            vn_sub
        )

    with c2:
        metric_card(
            "Thanh khoản",
            liquidity_value,
            liquidity_sub
        )

    with c3:
        metric_card(
            "Tin tức",
            "LIVE",
            "Google News + nguồn Việt Nam"
        )

    with c4:
        metric_card(
            "AI",
            "READY",
            "Chỉ gọi khi yêu cầu"
        )

    # =====================================================
    # VN-INDEX DETAIL
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 VN-INDEX</div>',
        unsafe_allow_html=True,
    )

    i1, i2, i3, i4 = st.columns(4)

    with i1:

        st.metric(
            "Điểm",
            vn_value
        )

    with i2:

        if vn.get("change") is not None:

            st.metric(
                "Thay đổi",
                f"{vn['change']:+.2f}"
            )

        else:

            st.metric(
                "Thay đổi",
                "—"
            )

    with i3:

        if vn.get("change_pct") is not None:

            st.metric(
                "1D",
                f"{vn['change_pct']:+.2f}%"
            )

        else:

            st.metric(
                "1D",
                "—"
            )

    with i4:

        if vn.get("value"):

            st.metric(
                "GTGD",
                format_value(vn["value"])
            )

        elif vn.get("volume"):

            st.metric(
                "Volume",
                format_volume(vn["volume"])
            )

        else:

            st.metric(
                "Thanh khoản",
                "—"
            )

    # =====================================================
    # DEBUG NHẸ NẾU CHƯA CÓ THANH KHOẢN
    # =====================================================

    if (
        vn.get("price") is not None
        and not vn.get("value")
        and not vn.get("volume")
    ):

        with st.expander(
            "Thông tin nguồn dữ liệu"
        ):

            st.write(
                "VN-INDEX đã lấy được nhưng "
                "nguồn hiện tại không trả trường "
                "thanh khoản."
            )

            if vn.get("columns"):
                st.write(
                    "Các cột nhận được:"
                )

                st.code(
                    ", ".join(vn["columns"])
                )

            if vn.get("error"):
                st.code(
                    vn["error"]
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
        type="primary"
    ):

        clean_symbol = normalize_symbol(
            symbol
        )

        if clean_symbol:

            st.session_state.dashboard_symbol = (
                clean_symbol
            )

            st.session_state.dashboard_input = (
                symbol.upper().strip()
            )

            st.rerun()

    active = st.session_state.get(
        "dashboard_symbol",
        "HPG.VN"
    )

    try:

        df = load_market_data(
            active,
            "1y"
        )

        if df is None or df.empty:

            st.warning(
                f"Không lấy được dữ liệu cho {active}."
            )

        else:

            last = df.iloc[-1]

            a, b, c, d = st.columns(4)

            with a:

                st.metric(
                    "Giá",
                    f"{float(last['Close']):,.0f}"
                )

            with b:

                st.metric(
                    "1D",
                    f"{float(last['Return']) * 100:+.2f}%"
                )

            with c:

                st.metric(
                    "RSI",
                    f"{float(last['RSI']):.1f}"
                )

            with d:

                st.metric(
                    "Volume",
                    f"{float(last['Volume']):,.0f}"
                )

            try:

                fig = price_volume_chart(df)

                if fig is not None:

                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={
                            "displaylogo": False
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
                    "Không có tiêu đề"
                )

                source = n.get(
                    "source",
                    "Nguồn không xác định"
                )

                published = n.get(
                    "published",
                    ""
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

    except Exception as e:

        st.info(
            f"Chưa thể tải tin tức: {e}"
        )
