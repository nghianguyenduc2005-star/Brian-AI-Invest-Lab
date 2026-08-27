from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components.ai import (
    dashboard_prompt,
    render_ai_panel,
)

from components.cards import (
    metric_card,
)

from components.charts import (
    price_volume_chart,
)

from data.market import (
    display_symbol,
    load_market_data,
    load_vnindex_data,
    market_snapshot,
    normalize_symbol,
)

from data.news import (
    fetch_market_news,
)


# ============================================================
# CẤU HÌNH
# ============================================================

DASHBOARD_STOCK_PERIOD = "1y"

VNINDEX_CACHE_TTL = 300
STOCK_CACHE_TTL = 300
NEWS_CACHE_TTL = 300


# ============================================================
# TIỆN ÍCH SỐ
# ============================================================

def num(
    value: Any,
    default=None,
):
    try:
        value = float(value)

        if pd.isna(value):
            return default

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


# ============================================================
# FORMAT VOLUME
# ============================================================

def format_volume(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu cổ phiếu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn cổ phiếu"
        )

    return (
        f"{value:,.0f} cổ phiếu"
    )


# ============================================================
# FORMAT GIÁ TRỊ GIAO DỊCH
# ============================================================

def format_value(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:

        return (
            f"{value / 1_000_000_000_000:.2f} nghìn tỷ đồng"
        )

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ đồng"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu đồng"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn đồng"
        )

    return (
        f"{value:,.0f} đồng"
    )


# ============================================================
# FORMAT PRICE
# ============================================================

def format_price(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:,.0f} đồng"
    )


# ============================================================
# FORMAT PERCENT
# ============================================================

def format_percent(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:+.2f}%"
    )


# ============================================================
# FORMAT RSI
# ============================================================

def format_rsi(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:.1f}"
    )


# ============================================================
# FORMAT MACD
# ============================================================

def format_macd(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:.3f}"
    )


# ============================================================
# TÌM CỘT
# ============================================================

def find_column(
    df: pd.DataFrame,
    names: list[str],
):
    if (
        df is None
        or df.empty
    ):
        return None

    mapping = {
        str(column)
        .strip()
        .lower(): column
        for column in df.columns
    }

    for name in names:

        key = (
            str(name)
            .strip()
            .lower()
        )

        if key in mapping:
            return mapping[key]

    return None


# ============================================================
# VN-INDEX
# ============================================================

@st.cache_data(
    ttl=VNINDEX_CACHE_TTL,
    show_spinner=False,
)
def get_vnindex():

    try:

        df = load_vnindex_data()

    except Exception as error:

        return {
            "error": str(error),
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": None,
        }

    if (
        df is None
        or df.empty
    ):

        return {
            "error": "Không có dữ liệu VN-INDEX.",
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": None,
        }

    df = df.copy()

    # ========================================================
    # GIÁ / ĐIỂM
    # ========================================================

    price_column = find_column(
        df,
        [
            "Close",
            "close",
            "Last",
            "last",
            "Price",
            "price",
            "Index",
            "index",
        ],
    )

    if price_column is None:

        return {
            "error": "Không tìm thấy cột điểm VN-INDEX.",
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": df,
        }

    df[
        price_column
    ] = pd.to_numeric(
        df[
            price_column
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            price_column
        ]
    )

    if df.empty:

        return {
            "error": "VN-INDEX không có điểm hợp lệ.",
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": df,
        }

    current = num(
        df[
            price_column
        ].iloc[-1],
        None,
    )

    # ========================================================
    # THAY ĐỔI
    # ========================================================

    change = None
    change_percent = None

    if len(df) >= 2:

        previous = num(
            df[
                price_column
            ].iloc[-2],
            None,
        )

        if (
            current is not None
            and previous is not None
        ):

            change = (
                current
                - previous
            )

            if previous != 0:

                change_percent = (
                    change
                    / previous
                    * 100
                )

            else:

                change_percent = 0.0

    # ========================================================
    # KHỐI LƯỢNG
    # ========================================================

    volume_column = find_column(
        df,
        [
            "Volume",
            "volume",
            "Vol",
            "vol",
            "total_volume",
            "match_volume",
            "matchvolume",
            "Khối lượng",
        ],
    )

    volume = None

    if volume_column is not None:

        df[
            volume_column
        ] = pd.to_numeric(
            df[
                volume_column
            ],
            errors="coerce",
        )

        volume = num(
            df[
                volume_column
            ].iloc[-1],
            None,
        )

    # ========================================================
    # GIÁ TRỊ GIAO DỊCH
    # ========================================================

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
            "matchvalue",
            "turnover",
            "Turnover",
            "Giá trị",
            "gia_tri",
        ],
    )

    traded_value = None

    if value_column is not None:

        df[
            value_column
        ] = pd.to_numeric(
            df[
                value_column
            ],
            errors="coerce",
        )

        traded_value = num(
            df[
                value_column
            ].iloc[-1],
            None,
        )

    # ========================================================
    # FALLBACK SUMMARY
    # ========================================================

    if (
        traded_value is None
        or volume is None
    ):

        try:

            import data.market as market_module

            summary_function = getattr(
                market_module,
                "load_vnindex_market_summary",
                None,
            )

            if summary_function is not None:

                summary = (
                    summary_function()
                )

                if isinstance(
                    summary,
                    dict,
                ):

                    if traded_value is None:

                        traded_value = num(
                            summary.get(
                                "gia_tri"
                            ),
                            None,
                        )

                    if volume is None:

                        volume = num(
                            summary.get(
                                "khoi_luong"
                            ),
                            None,
                        )

        except Exception:
            pass

    return {
        "error": None,
        "price": current,
        "change": change,
        "change_percent": change_percent,
        "volume": volume,
        "value": traded_value,
        "df": df,
    }


# ============================================================
# STOCK DATA
# ============================================================

@st.cache_data(
    ttl=STOCK_CACHE_TTL,
    show_spinner=False,
)
def get_stock_data(
    symbol: str,
):

    symbol = normalize_symbol(
        symbol
    )

    return load_market_data(
        symbol,
        DASHBOARD_STOCK_PERIOD,
    )


# ============================================================
# NEWS CACHE
#
# Để reload dashboard không gọi news API liên tục.
# ============================================================

@st.cache_data(
    ttl=NEWS_CACHE_TTL,
    show_spinner=False,
)
def get_market_news(
    limit: int = 6,
):

    try:

        news = fetch_market_news(
            limit
        )

        if isinstance(
            news,
            list,
        ):

            return news

        return []

    except Exception:

        return []


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard():

    # ========================================================
    # HEADER
    # ========================================================

    st.caption(
        "BRIAN STOCK · INVESTMENT INTELLIGENCE"
    )

    st.title(
        "Góc nhìn dữ liệu cho nhà đầu tư"
    )

    st.write(
        "Dashboard nghiên cứu thị trường, cổ phiếu, "
        "tin tức và AI. Dữ liệu được lấy từ nguồn thật "
        "và chỉ gọi AI khi người dùng yêu cầu."
    )

    # ========================================================
    # VN-INDEX
    # ========================================================

    vn_index = get_vnindex()

    if vn_index.get(
        "error"
    ):

        vn_price = "—"
        vn_change = "—"
        vn_percent = "—"
        vn_volume = "—"
        vn_value = "—"

    else:

        price = num(
            vn_index.get(
                "price"
            ),
            None,
        )

        change = num(
            vn_index.get(
                "change"
            ),
            None,
        )

        change_percent = num(
            vn_index.get(
                "change_percent"
            ),
            None,
        )

        vn_price = (
            f"{price:,.2f} điểm"
            if price is not None
            else "—"
        )

        vn_change = (
            f"{change:+,.2f} điểm"
            if change is not None
            else "—"
        )

        vn_percent = (
            f"{change_percent:+.2f}%"
            if change_percent is not None
            else "—"
        )

        vn_volume = format_volume(
            vn_index.get(
                "volume"
            )
        )

        vn_value = format_value(
            vn_index.get(
                "value"
            )
        )

    # ========================================================
    # THEO DÕI NHANH
    # ========================================================

    st.subheader(
        "📌 Theo dõi nhanh"
    )

    quick_1, quick_2, quick_3, quick_4 = st.columns(
        4
    )

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
    # VN-INDEX DETAIL
    # ========================================================

    st.subheader(
        "📊 VN-INDEX"
    )

    v1, v2, v3, v4 = st.columns(
        4
    )

    with v1:

        st.metric(
            "Điểm",
            vn_price,
        )

    with v2:

        st.metric(
            "Thay đổi",
            vn_change,
        )

    with v3:

        st.metric(
            "Thay đổi 1D",
            vn_percent,
        )

    with v4:

        st.metric(
            "Khối lượng",
            vn_volume,
        )

    if vn_index.get(
        "error"
    ):

        st.warning(
            "VN-INDEX chưa tải được: "
            + str(
                vn_index.get(
                    "error"
                )
            )
        )

    # ========================================================
    # STOCK WATCH
    # ========================================================

    st.subheader(
        "📈 Theo dõi cổ phiếu"
    )

    default_symbol = (
        st.session_state.get(
            "dashboard_symbol",
            "HPG",
        )
    )

    symbol_input = st.text_input(
        "Mã cổ phiếu",
        value=default_symbol,
        label_visibility="collapsed",
        placeholder="Ví dụ HPG, VNM, FPT...",
        key="dashboard_stock_input",
    )

    load_stock = st.button(
        "🔄 Tải dữ liệu",
        type="primary",
        key="dashboard_load_button",
    )

    if load_stock:

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
    # STOCK DATA
    # ========================================================

    stock = None

    try:

        stock = get_stock_data(
            current_symbol
        )

    except Exception as error:

        st.error(
            f"Không lấy được dữ liệu "
            f"{display_symbol(current_symbol)}."
        )

        st.caption(
            str(error)
        )

    if (
        stock is not None
        and not stock.empty
    ):

        snapshot = market_snapshot(
            stock
        )

        stock_price = num(
            snapshot.get(
                "price"
            ),
            None,
        )

        stock_change = num(
            snapshot.get(
                "change_1d"
            ),
            None,
        )

        stock_rsi = num(
            snapshot.get(
                "rsi"
            ),
            None,
        )

        stock_volume = num(
            snapshot.get(
                "volume"
            ),
            None,
        )

        stock_sma20 = num(
            snapshot.get(
                "sma20"
            ),
            None,
        )

        stock_sma50 = num(
            snapshot.get(
                "sma50"
            ),
            None,
        )

        stock_macd = num(
            snapshot.get(
                "macd"
            ),
            None,
        )

        stock_volatility = num(
            snapshot.get(
                "volatility20"
            ),
            None,
        )

        last_row = stock.iloc[
            -1
        ]

        stock_open = num(
            last_row.get(
                "Open"
            ),
            None,
        )

        atr14 = num(
            last_row.get(
                "ATR14"
            ),
            None,
        )

        volume_sma20 = num(
            last_row.get(
                "Volume_SMA20"
            ),
            None,
        )

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

        # ====================================================
        # STOCK TITLE
        # ====================================================

        st.subheader(
            f"📈 {display_symbol(current_symbol)}"
        )

        # ====================================================
        # MAIN METRICS
        # ====================================================

        s1, s2, s3, s4 = st.columns(
            4
        )

        with s1:

            st.metric(
                "Giá",
                format_price(
                    stock_price
                ),
            )

        with s2:

            st.metric(
                "Thay đổi 1D",
                format_percent(
                    stock_change
                ),
            )

        with s3:

            st.metric(
                "RSI",
                format_rsi(
                    stock_rsi
                ),
            )

        with s4:

            st.metric(
                "Khối lượng",
                format_volume(
                    stock_volume
                ),
            )

        # ====================================================
        # INDICATORS
        # ====================================================

        i1, i2, i3, i4 = st.columns(
            4
        )

        with i1:

            st.metric(
                "Trung bình 20 phiên",
                format_price(
                    stock_sma20
                ),
            )

        with i2:

            st.metric(
                "Trung bình 50 phiên",
                format_price(
                    stock_sma50
                ),
            )

        with i3:

            st.metric(
                "MACD",
                format_macd(
                    stock_macd
                ),
            )

        with i4:

            st.metric(
                "Biến động 20 phiên",
                (
                    f"{stock_volatility:.2f}%"
                    if stock_volatility is not None
                    else "—"
                ),
            )

        # ====================================================
        # EXTRA
        # ====================================================

        x1, x2, x3, x4 = st.columns(
            4
        )

        with x1:

            st.metric(
                "ATR 14",
                format_price(
                    atr14
                ),
            )

        with x2:

            st.metric(
                "Khối lượng TB20",
                format_volume(
                    volume_sma20
                ),
            )

        with x3:

            st.metric(
                "Khối lượng / TB20",
                (
                    f"{relative_volume:.2f} lần"
                    if relative_volume is not None
                    else "—"
                ),
            )

        with x4:

            st.metric(
                "Giá mở cửa",
                format_price(
                    stock_open
                ),
            )

        # ====================================================
        # CHART
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
                "Không thể hiển thị biểu đồ."
            )

            st.caption(
                str(error)
            )

    # ========================================================
    # NEWS
    # ========================================================

    st.subheader(
        "📰 Tin mới"
    )

    news = get_market_news(
        6
    )

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
                    "",
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
                border=True,
            ):

                st.markdown(
                    f"**{title}**"
                )

                metadata = " · ".join(
                    x
                    for x in [
                        source,
                        published,
                    ]
                    if x
                )

                if metadata:

                    st.caption(
                        metadata
                    )

                if link:

                    st.markdown(
                        f"[Đọc bài ↗]({link})"
                    )

    # ========================================================
    # AI MARKET BRIEF
    # ========================================================

    st.divider()

    st.header(
        "🤖 AI Market Brief"
    )

    st.caption(
        "AI chỉ được gọi khi bấm nút. "
        "Reload dashboard hoặc đổi dữ liệu không tự gọi AI."
    )

    # ========================================================
    # AI CONTEXT — luôn tạo trước
    # ========================================================

    ai_vn_index = {
        "diem": vn_index.get(
            "price"
        ),
        "thay_doi": vn_index.get(
            "change"
        ),
        "phan_tram": vn_index.get(
            "change_percent"
        ),
        "khoi_luong": vn_index.get(
            "volume"
        ),
        "gia_tri": vn_index.get(
            "value"
        ),
    }

    ai_stock_symbol = display_symbol(
        current_symbol
    )

    ai_snapshot = {}

    if (
        stock is not None
        and not stock.empty
    ):

        try:

            ai_snapshot = market_snapshot(
                stock
            )

        except Exception:

            ai_snapshot = {}

    ai_news = (
        news
        if isinstance(
            news,
            list,
        )
        else []
    )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    try:

        ai_prompt = dashboard_prompt(
            vn_index=ai_vn_index,
            stock_symbol=ai_stock_symbol,
            stock_snapshot=ai_snapshot,
            news=ai_news,
        )

    except Exception as error:

        ai_prompt = None

        st.warning(
            "Không tạo được AI prompt."
        )

        st.caption(
            str(error)
        )

    # ========================================================
    # AI PANEL
    # ========================================================

    if ai_prompt:

        try:

            render_ai_panel(
                title="🤖 AI Market Brief",
                description=(
                    "AI tổng hợp VN-INDEX, cổ phiếu đang theo dõi "
                    "và tin tức thị trường."
                ),
                prompt=ai_prompt,
                button_label="🤖 Phân tích dashboard bằng AI",
                key="dashboard_ai_analysis",
            )

        except Exception as error:

            st.error(
                "Không thể khởi tạo AI Market Brief."
            )

            st.caption(
                str(error)
            )


# ============================================================
# TƯƠNG THÍCH
# ============================================================

def render():
    render_dashboard()
