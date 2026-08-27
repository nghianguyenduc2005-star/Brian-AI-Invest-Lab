from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import metric_card
from components.charts import price_volume_chart
from data.market import (
    display_symbol,
    load_market_data,
    load_vnindex_data,
    market_snapshot,
    normalize_symbol,
)
from data.news import fetch_market_news


def num(value, default=None):
    try:
        value = float(value)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def format_volume(value):
    value = num(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu cổ phiếu"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn cổ phiếu"

    return f"{value:,.0f} cổ phiếu"


def format_value(value):
    value = num(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f} nghìn tỷ đồng"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ đồng"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu đồng"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn đồng"

    return f"{value:,.0f} đồng"


def format_price(value):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng/cổ phiếu"


def format_rsi(value):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:.1f} điểm"


def format_macd(value):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:.3f}"


def format_percent(value):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def find_column(df, names):
    if df is None or df.empty:
        return None

    mapping = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in names:
        key = str(name).strip().lower()

        if key in mapping:
            return mapping[key]

    return None


def get_vnindex():

    try:
        df = load_vnindex_data()

    except Exception as error:
        return {
            "error": str(error)
        }

    if df is None or df.empty:
        return {
            "error": "Không có dữ liệu VN-INDEX."
        }

    df = df.copy()

    price_column = find_column(
        df,
        [
            "Close",
            "close",
            "last",
            "price",
            "index",
        ],
    )

    if price_column is None:
        return {
            "error": "Không tìm thấy cột VN-INDEX."
        }

    df[price_column] = pd.to_numeric(
        df[price_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[price_column]
    )

    if df.empty:
        return {
            "error": "VN-INDEX không có giá hợp lệ."
        }

    current = float(
        df[price_column].iloc[-1]
    )

    if len(df) >= 2:

        previous = float(
            df[price_column].iloc[-2]
        )

        change = current - previous

        if previous != 0:
            change_percent = (
                change / previous
            ) * 100
        else:
            change_percent = 0.0

    else:

        change = None
        change_percent = None

    volume_column = find_column(
        df,
        [
            "Volume",
            "volume",
            "Vol",
            "total_volume",
            "match_volume",
        ],
    )

    volume = None

    if volume_column is not None:

        df[volume_column] = pd.to_numeric(
            df[volume_column],
            errors="coerce",
        )

        volume = num(
            df[volume_column].iloc[-1]
        )

    value_column = find_column(
        df,
        [
            "Value",
            "value",
            "ValueTraded",
            "value_traded",
            "trading_value",
            "traded_value",
            "match_value",
            "turnover",
        ],
    )

    traded_value = None

    if value_column is not None:

        df[value_column] = pd.to_numeric(
            df[value_column],
            errors="coerce",
        )

        traded_value = num(
            df[value_column].iloc[-1]
        )

    # Summary nếu data.market.py đã có
    if traded_value is None or volume is None:

        try:

            import data.market as market_module

            summary_function = getattr(
                market_module,
                "load_vnindex_market_summary",
                None,
            )

            if summary_function is not None:

                summary = summary_function()

                if isinstance(summary, dict):

                    if traded_value is None:

                        traded_value = num(
                            summary.get(
                                "gia_tri"
                            )
                        )

                    if volume is None:

                        volume = num(
                            summary.get(
                                "khoi_luong"
                            )
                        )

        except Exception:
            pass

    return {
        "price": current,
        "change": change,
        "change_percent": change_percent,
        "volume": volume,
        "value": traded_value,
        "df": df,
        "error": None,
    }


def render_dashboard():

    # ========================================================
    # HERO
    # ========================================================

    st.caption(
        "BRIAN STOCK · INVESTMENT INTELLIGENCE"
    )

    st.title(
        "Góc nhìn dữ liệu cho nhà đầu tư"
    )

    st.write(
        "Dashboard nghiên cứu thị trường, "
        "cổ phiếu, tin tức và AI. "
        "Dữ liệu được tải trực tiếp khi cần, "
        "không dùng dữ liệu ngẫu nhiên."
    )

    # ========================================================
    # VN-INDEX
    # ========================================================

    vn = get_vnindex()

    if vn.get("error"):

        vn_price = "—"
        vn_change = "—"
        vn_percent = "—"
        vn_volume = "—"
        vn_value = "—"

    else:

        vn_price = (
            f"{vn['price']:,.2f} điểm"
            if vn.get("price") is not None
            else "—"
        )

        vn_change = (
            f"{vn['change']:+,.2f} điểm"
            if vn.get("change") is not None
            else "—"
        )

        vn_percent = (
            f"{vn['change_percent']:+.2f}%"
            if vn.get("change_percent") is not None
            else "—"
        )

        vn_volume = format_volume(
            vn.get("volume")
        )

        vn_value = format_value(
            vn.get("value")
        )

    # ========================================================
    # THEO DÕI NHANH
    # ========================================================

    st.subheader(
        "📌 Theo dõi nhanh"
    )

    quick_1, quick_2, quick_3, quick_4 = st.columns(4)

    with quick_1:

        metric_card(
            "VN-INDEX",
            vn_price,
            "Điểm chỉ số thị trường",
        )

    with quick_2:

        metric_card(
            "Khối lượng",
            vn_volume,
            "Khối lượng giao dịch toàn thị trường",
        )

    with quick_3:

        metric_card(
            "Giá trị giao dịch",
            vn_value,
            "Tổng giá trị giao dịch",
        )

    with quick_4:

        metric_card(
            "Tin tức",
            "LIVE",
            "Nguồn tin thị trường mới",
        )

    # ========================================================
    # VN-INDEX
    # ========================================================

    st.subheader(
        "📊 VN-INDEX"
    )

    vn_1, vn_2, vn_3, vn_4 = st.columns(4)

    with vn_1:
        st.metric(
            "Điểm",
            vn_price,
        )

    with vn_2:
        st.metric(
            "Thay đổi",
            vn_change,
        )

    with vn_3:
        st.metric(
            "Thay đổi 1D",
            vn_percent,
        )

    with vn_4:
        st.metric(
            "Khối lượng",
            vn_volume,
        )

    if vn.get("error"):

        st.warning(
            "VN-INDEX chưa tải được: "
            + str(
                vn["error"]
            )
        )

    # ========================================================
    # CỔ PHIẾU
    # ========================================================

    st.subheader(
        "📈 Theo dõi cổ phiếu"
    )

    default_symbol = st.session_state.get(
        "dashboard_symbol",
        "HPG",
    )

    symbol_input = st.text_input(
        "Mã cổ phiếu",
        value=default_symbol,
        label_visibility="collapsed",
        placeholder="Ví dụ HPG, VNM, FPT...",
        key="dashboard_stock_input",
    )

    if st.button(
        "Tải dữ liệu",
        type="primary",
        key="dashboard_load_button",
    ):

        clean_symbol = normalize_symbol(
            symbol_input
        )

        if clean_symbol:

            st.session_state[
                "dashboard_symbol"
            ] = clean_symbol

            st.rerun()

        else:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

    current_symbol = normalize_symbol(
        st.session_state.get(
            "dashboard_symbol",
            symbol_input,
        )
    )

    # ========================================================
    # LOAD CỔ PHIẾU
    # ========================================================

    try:

        stock = load_market_data(
            current_symbol,
            "1y",
        )

    except Exception as error:

        st.error(
            f"Không lấy được dữ liệu "
            f"{display_symbol(current_symbol)}: "
            f"{error}"
        )

        stock = None

    if stock is not None and not stock.empty:

        snapshot = market_snapshot(
            stock
        )

        price = num(
            snapshot.get("price")
        )

        change_1d = num(
            snapshot.get("change_1d")
        )

        rsi_value = num(
            snapshot.get("rsi")
        )

        stock_volume = num(
            snapshot.get("volume")
        )

        sma20 = num(
            snapshot.get("sma20")
        )

        sma50 = num(
            snapshot.get("sma50")
        )

        macd = num(
            snapshot.get("macd")
        )

        volatility = num(
            snapshot.get("volatility20")
        )

        last_row = stock.iloc[-1]

        atr14 = num(
            last_row.get("ATR14")
        )

        volume_sma20 = num(
            last_row.get("Volume_SMA20")
        )

        # ====================================================
        # TIÊU ĐỀ
        # ====================================================

        st.subheader(
            f"📈 {display_symbol(current_symbol)}"
        )

        # ====================================================
        # THÔNG TIN
        # ====================================================

        stock_1, stock_2, stock_3, stock_4 = st.columns(4)

        with stock_1:

            st.metric(
                "Giá",
                format_price(price),
            )

        with stock_2:

            st.metric(
                "Thay đổi 1D",
                format_percent(change_1d),
            )

        with stock_3:

            st.metric(
                "RSI",
                format_rsi(rsi_value),
            )

        with stock_4:

            st.metric(
                "Khối lượng",
                format_volume(stock_volume),
            )

        # ====================================================
        # CHỈ BÁO
        # ====================================================

        ind_1, ind_2, ind_3, ind_4 = st.columns(4)

        with ind_1:

            st.metric(
                "Trung bình 20 phiên",
                format_price(sma20),
            )

        with ind_2:

            st.metric(
                "Trung bình 50 phiên",
                format_price(sma50),
            )

        with ind_3:

            st.metric(
                "MACD",
                format_macd(macd),
            )

        with ind_4:

            st.metric(
                "Biến động 20 phiên",
                (
                    f"{volatility:.2f}%"
                    if volatility is not None
                    else "—"
                ),
            )

        # ====================================================
        # CHỈ BÁO BỔ SUNG
        # ====================================================

        extra_1, extra_2, extra_3, extra_4 = st.columns(4)

        with extra_1:

            st.metric(
                "ATR 14 phiên",
                format_price(atr14),
            )

        with extra_2:

            st.metric(
                "Khối lượng TB20",
                format_volume(volume_sma20),
            )

        with extra_3:

            relative_volume = None

            if (
                stock_volume is not None
                and volume_sma20 is not None
                and volume_sma20 != 0
            ):

                relative_volume = (
                    stock_volume
                    / volume_sma20
                )

            st.metric(
                "Khối lượng / TB20",
                (
                    f"{relative_volume:.2f} lần"
                    if relative_volume is not None
                    else "—"
                ),
            )

        with extra_4:

            opening_price = num(
                last_row.get("Open")
            )

            st.metric(
                "Giá mở cửa",
                format_price(opening_price),
            )

        # ====================================================
        # BIỂU ĐỒ
        # ====================================================

        st.subheader(
            "📊 Biểu đồ kỹ thuật"
        )

        try:

            chart = price_volume_chart(
                stock
            )

            if chart is not None:

                st.plotly_chart(
                    chart,
                    width="stretch",
                    config={
                        "displaylogo": False,
                    },
                )

        except Exception as error:

            st.warning(
                "Không thể hiển thị biểu đồ: "
                f"{error}"
            )

    # ========================================================
    # TIN MỚI
    # ========================================================

    st.subheader(
        "📰 Tin mới"
    )

    try:

        news = fetch_market_news(
            6
        )

    except Exception as error:

        st.info(
            f"Chưa thể tải tin tức: {error}"
        )

        news = []

    if not news:

        st.info(
            "Hiện chưa lấy được tin tức mới."
        )

    else:

        for item in news:

            title = str(
                item.get(
                    "title",
                    "Không có tiêu đề",
                )
            ).strip()

            source = str(
                item.get(
                    "source",
                    "Nguồn không xác định",
                )
            ).strip()

            published = str(
                item.get(
                    "published",
                    "",
                )
            ).strip()

            link = str(
                item.get(
                    "link",
                    "",
                )
            ).strip()

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{title}**"
                )

                if source and published:

                    st.caption(
                        f"{source} · {published}"
                    )

                elif source:

                    st.caption(
                        source
                    )

                elif published:

                    st.caption(
                        published
                    )

                if link:

                    st.markdown(
                        f"[Đọc bài ↗]({link})"
                    )
