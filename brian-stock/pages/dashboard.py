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


# ============================================================
# TIỆN ÍCH
# ============================================================

def to_number(value, default=None):
    try:
        value = float(value)

        if pd.isna(value):
            return default

        return value

    except Exception:
        return default


def find_column(df, names):
    if df is None or df.empty:
        return None

    mapping = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in names:
        key = str(name).strip().lower()

        if key in mapping:
            return mapping[key]

    return None


# ============================================================
# FORMAT
# ============================================================

def fmt_volume(value):
    value = to_number(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu cổ phiếu"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn cổ phiếu"

    return f"{value:,.0f} cổ phiếu"


def fmt_value(value):
    value = to_number(value)

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


def fmt_gia(value):
    value = to_number(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng/cổ phiếu"


def fmt_rsi(value):
    value = to_number(value)

    if value is None:
        return "—"

    return f"{value:.1f} điểm"


def fmt_macd(value):
    value = to_number(value)

    if value is None:
        return "—"

    return f"{value:.3f}"


def fmt_percent(value):
    value = to_number(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


# ============================================================
# LẤY VN-INDEX
# ============================================================

def get_vn_index():

    try:
        df = load_vnindex_data()

    except Exception as error:
        return {
            "loi": str(error)
        }

    if df is None or df.empty:
        return {
            "loi": "Nguồn VN-INDEX trả về dữ liệu rỗng."
        }

    df = df.copy()

    # --------------------------------------------------------
    # Điểm
    # --------------------------------------------------------

    price_col = find_column(
        df,
        [
            "Close",
            "close",
            "last",
            "price",
            "index",
            "Đóng cửa",
            "đóng cửa",
        ],
    )

    if price_col is None:
        return {
            "loi": "Không tìm thấy cột điểm VN-INDEX."
        }

    df[price_col] = pd.to_numeric(
        df[price_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[price_col]
    )

    if df.empty:
        return {
            "loi": "VN-INDEX không có điểm hợp lệ."
        }

    current = float(
        df[price_col].iloc[-1]
    )

    # --------------------------------------------------------
    # Thay đổi
    # --------------------------------------------------------

    if len(df) >= 2:

        previous = float(
            df[price_col].iloc[-2]
        )

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

    else:

        change = None
        change_percent = None

    # --------------------------------------------------------
    # Khối lượng
    # --------------------------------------------------------

    volume_col = find_column(
        df,
        [
            "Volume",
            "volume",
            "Vol",
            "total_volume",
            "match_volume",
            "matchvolume",
            "Khối lượng",
            "khối lượng",
        ],
    )

    volume = None

    if volume_col is not None:

        df[volume_col] = pd.to_numeric(
            df[volume_col],
            errors="coerce",
        )

        volume = to_number(
            df[volume_col].iloc[-1]
        )

    # --------------------------------------------------------
    # Giá trị giao dịch
    # --------------------------------------------------------

    value_col = find_column(
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
            "giá trị",
        ],
    )

    traded_value = None

    if value_col is not None:

        df[value_col] = pd.to_numeric(
            df[value_col],
            errors="coerce",
        )

        traded_value = to_number(
            df[value_col].iloc[-1]
        )

    # --------------------------------------------------------
    # Summary fallback
    #
    # Không import hàm summary ở đầu file để tránh làm
    # Dashboard chết nếu data/market.py chưa có hàm đó.
    # --------------------------------------------------------

    if traded_value is None:

        try:

            import data.market as market_module

            summary_function = getattr(
                market_module,
                "load_vnindex_market_summary",
                None,
            )

            if summary_function is not None:

                summary = summary_function()

                if isinstance(
                    summary,
                    dict,
                ):

                    traded_value = to_number(
                        summary.get(
                            "gia_tri"
                        )
                    )

                    if volume is None:

                        volume = to_number(
                            summary.get(
                                "khoi_luong"
                            )
                        )

        except Exception:
            pass

    return {
        "diem": current,
        "thay_doi": change,
        "phan_tram": change_percent,
        "khoi_luong": volume,
        "gia_tri": traded_value,
        "du_lieu": df,
        "loi": None,
    }


# ============================================================
# DASHBOARD
# ============================================================

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

    vn_index = get_vn_index()

    if vn_index.get("loi"):

        vn_diem = "—"
        vn_change = "—"
        vn_1d = "—"
        vn_volume = "—"
        vn_value = "—"

    else:

        diem = to_number(
            vn_index.get("diem")
        )

        change = to_number(
            vn_index.get("thay_doi")
        )

        change_percent = to_number(
            vn_index.get("phan_tram")
        )

        vn_diem = (
            f"{diem:,.2f} điểm"
            if diem is not None
            else "—"
        )

        vn_change = (
            f"{change:+,.2f} điểm"
            if change is not None
            else "—"
        )

        vn_1d = (
            f"{change_percent:+.2f}%"
            if change_percent is not None
            else "—"
        )

        vn_volume = fmt_volume(
            vn_index.get("khoi_luong")
        )

        vn_value = fmt_value(
            vn_index.get("gia_tri")
        )

    # ========================================================
    # THEO DÕI NHANH
    # ========================================================

    st.subheader(
        "📌 Theo dõi nhanh"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "VN-INDEX",
            vn_diem,
            "Điểm chỉ số thị trường",
        )

    with c2:

        metric_card(
            "Khối lượng",
            vn_volume,
            "Khối lượng giao dịch toàn thị trường",
        )

    with c3:

        metric_card(
            "Giá trị giao dịch",
            vn_value,
            "Tổng giá trị giao dịch",
        )

    with c4:

        metric_card(
            "Tin tức",
            "LIVE",
            "Nguồn tin thị trường mới",
        )

    # ========================================================
    # VN-INDEX CHI TIẾT
    # ========================================================

    st.subheader(
        "📊 VN-INDEX"
    )

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Điểm",
            vn_diem,
        )

    with b:
        st.metric(
            "Thay đổi",
            vn_change,
        )

    with c:
        st.metric(
            "Thay đổi 1D",
            vn_1d,
        )

    with d:
        st.metric(
            "Khối lượng",
            vn_volume,
        )

    if vn_index.get("loi"):

        st.warning(
            "VN-INDEX chưa tải được: "
            + str(
                vn_index["loi"]
            )
        )

    # ========================================================
    # THEO DÕI CỔ PHIẾU
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
        placeholder=(
            "Nhập mã cổ phiếu, ví dụ HPG, MSR, VNM"
        ),
        key="dashboard_stock_input",
    )

    if st.button(
        "Tải dữ liệu",
        type="primary",
        key="dashboard_load_button",
        width="content",
    ):

        clean_symbol = normalize_symbol(
            symbol_input
        )

        if not clean_symbol:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

        else:

            st.session_state[
                "dashboard_symbol"
            ] = clean_symbol

            st.rerun()

    current_symbol = normalize_symbol(
        st.session_state.get(
            "dashboard_symbol",
            symbol_input,
        )
    )

    # ========================================================
    # DỮ LIỆU CỔ PHIẾU
    # ========================================================

    try:

        stock_data = load_market_data(
            current_symbol,
            "1y",
        )

    except Exception as error:

        st.error(
            f"Không lấy được dữ liệu "
            f"{display_symbol(current_symbol)}: "
            f"{error}"
        )

        stock_data = None

    if (
        stock_data is not None
        and not stock_data.empty
    ):

        snapshot = market_snapshot(
            stock_data
        )

        price = to_number(
            snapshot.get("price")
        )

        change_1d = to_number(
            snapshot.get("change_1d")
        )

        rsi_value = to_number(
            snapshot.get("rsi")
        )

        stock_volume = to_number(
            snapshot.get("volume")
        )

        sma20 = to_number(
            snapshot.get("sma20")
        )

        sma50 = to_number(
            snapshot.get("sma50")
        )

        macd_value = to_number(
            snapshot.get("macd")
        )

        volatility = to_number(
            snapshot.get("volatility20")
        )

        last_row = stock_data.iloc[-1]

        atr14 = to_number(
            last_row.get("ATR14")
        )

        volume_sma20 = to_number(
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

        a, b, c, d = st.columns(4)

        with a:

            st.metric(
                "Giá",
                fmt_gia(price),
            )

        with b:

            st.metric(
                "Thay đổi 1D",
                fmt_percent(change_1d),
            )

        with c:

            st.metric(
                "RSI",
                fmt_rsi(rsi_value),
            )

        with d:

            st.metric(
                "Khối lượng",
                fmt_volume(stock_volume),
            )

        # ====================================================
        # CHỈ BÁO
        # ====================================================

        e, f, g, h = st.columns(4)

        with e:

            st.metric(
                "Trung bình 20 phiên",
                fmt_gia(sma20),
            )

        with f:

            st.metric(
                "Trung bình 50 phiên",
                fmt_gia(sma50),
            )

        with g:

            st.metric(
                "MACD",
                fmt_macd(macd_value),
            )

        with h:

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

        i, j, k, l = st.columns(4)

        with i:

            st.metric(
                "ATR 14 phiên",
                fmt_gia(atr14),
            )

        with j:

            st.metric(
                "Khối lượng TB20",
                fmt_volume(volume_sma20),
            )

        with k:

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

        with l:

            open_price = to_number(
                last_row.get("Open")
            )

            st.metric(
                "Giá mở cửa",
                fmt_gia(open_price),
            )

        # ====================================================
        # BIỂU ĐỒ
        # ====================================================

        st.subheader(
            "📊 Biểu đồ kỹ thuật"
        )

        try:

            chart = price_volume_chart(
                stock_data
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

    elif stock_data is not None:

        st.warning(
            f"Không tìm thấy dữ liệu thật cho "
            f"{display_symbol(current_symbol)}."
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
