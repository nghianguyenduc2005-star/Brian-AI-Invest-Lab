from __future__ import annotations

import pandas as pd
import streamlit as st

from components.ai import (
    render_ai_panel,
    stock_analysis_prompt,
)

from components.charts import (
    price_volume_chart,
)

from data.market import (
    build_quant,
    display_symbol,
    load_market_data,
    market_snapshot,
    normalize_symbol,
    run_ols,
    run_random_forest,
)


# ============================================================
# TIỆN ÍCH CHUNG
# ============================================================

def to_number(
    value,
    default=None,
):
    try:
        value = float(value)

        if pd.isna(value):
            return default

        return value

    except Exception:
        return default


def format_price(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "—"

    return f"{value:,.0f} đồng"


def format_percent(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def format_volume(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn"
        )

    return f"{value:,.0f}"


def format_rsi(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "—"

    return f"{value:.1f}"


def format_macd(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "—"

    return f"{value:.3f}"


def format_ratio(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "—"

    return f"{value:.2f}x"


# ============================================================
# TRẠNG THÁI RSI
# ============================================================

def rsi_status(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "Không xác định"

    if value >= 70:
        return "Quá mua"

    if value <= 30:
        return "Quá bán"

    return "Trung tính"


# ============================================================
# TRẠNG THÁI GIÁ / MA
# ============================================================

def price_vs_ma_status(
    price,
    sma20,
    sma50,
):
    price = to_number(
        price,
        None,
    )

    sma20 = to_number(
        sma20,
        None,
    )

    sma50 = to_number(
        sma50,
        None,
    )

    if price is None:
        return "Không xác định"

    if (
        sma20 is not None
        and sma50 is not None
    ):

        if price > sma20 > sma50:
            return "Xu hướng tăng"

        if price < sma20 < sma50:
            return "Xu hướng giảm"

        return "Đang giằng co"

    if sma20 is not None:

        if price > sma20:
            return "Trên MA20"

        if price < sma20:
            return "Dưới MA20"

    return "Không xác định"


# ============================================================
# TRẠNG THÁI MACD
# ============================================================

def macd_status(
    macd_value,
):
    macd_value = to_number(
        macd_value,
        None,
    )

    if macd_value is None:
        return "Không xác định"

    if macd_value > 0:
        return "MACD dương"

    if macd_value < 0:
        return "MACD âm"

    return "MACD trung tính"


# ============================================================
# CHUẨN HÓA DỰ BÁO QUANT
# ============================================================

def quant_prediction_text(
    value,
):
    value = to_number(
        value,
        None,
    )

    if value is None:
        return "Không có dự báo"

    return format_percent(
        value * 100
    )


# ============================================================
# SESSION STATE
# ============================================================

def _init_session():

    if (
        "stock_analysis_symbol"
        not in st.session_state
    ):

        st.session_state[
            "stock_analysis_symbol"
        ] = "HPG"

    if (
        "stock_analysis_loaded_symbol"
        not in st.session_state
    ):

        st.session_state[
            "stock_analysis_loaded_symbol"
        ] = None

    if (
        "stock_analysis_quant"
        not in st.session_state
    ):

        st.session_state[
            "stock_analysis_quant"
        ] = None


# ============================================================
# LOAD DATA CACHE
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def _load_stock(
    symbol,
):
    return load_market_data(
        symbol,
        "1y",
    )


# ============================================================
# RUN QUANT CACHE
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def _run_quant_models(
    data,
):
    """
    Chạy các mô hình định lượng.

    Cache 10 phút để:
    - không rerun model mỗi lần Streamlit rerun
    - không làm page lag khi đổi input/UI
    """

    result = {
        "ols": None,
        "forest": None,
        "quant": None,
    }

    try:
        result["ols"] = run_ols(
            data
        )
    except Exception:
        result["ols"] = None

    try:
        result["forest"] = (
            run_random_forest(
                data
            )
        )
    except Exception:
        result["forest"] = None

    try:
        result["quant"] = build_quant(
            data
        )
    except Exception:
        result["quant"] = None

    return result


# ============================================================
# RENDER PAGE
# ============================================================

def render_stock_analysis():

    _init_session()

    # ========================================================
    # HEADER
    # ========================================================

    st.caption(
        "BRIAN STOCK · STOCK RESEARCH"
    )

    st.title(
        "Phân tích cổ phiếu"
    )

    st.write(
        "Phân tích giá, xu hướng, động lượng, "
        "thanh khoản, biến động và chỉ báo kỹ thuật "
        "từ dữ liệu thị trường thực."
    )

    # ========================================================
    # INPUT
    # ========================================================

    current_symbol = (
        st.session_state.get(
            "stock_analysis_symbol",
            "HPG",
        )
    )

    symbol_input = st.text_input(
        "Mã cổ phiếu",
        value=current_symbol,
        placeholder=(
            "Ví dụ: HPG, FPT, VNM..."
        ),
        key="stock_analysis_symbol_input",
    )

    load_button = st.button(
        "🔄 Tải dữ liệu",
        type="primary",
        key="stock_analysis_load_button",
    )

    if load_button:

        clean_symbol = normalize_symbol(
            symbol_input
        )

        if not clean_symbol:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

            return

        st.session_state[
            "stock_analysis_symbol"
        ] = clean_symbol

        # Xóa model cũ khi đổi mã.
        st.session_state[
            "stock_analysis_quant"
        ] = None

        st.session_state[
            "stock_analysis_loaded_symbol"
        ] = clean_symbol

        st.rerun()

    # ========================================================
    # SYMBOL HIỆN TẠI
    # ========================================================

    symbol = normalize_symbol(
        st.session_state.get(
            "stock_analysis_symbol",
            symbol_input,
        )
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        data = _load_stock(
            symbol
        )

    except Exception as error:

        st.error(
            f"Không thể tải dữ liệu "
            f"{display_symbol(symbol)}."
        )

        st.code(
            str(error)
        )

        return

    if (
        data is None
        or data.empty
    ):

        st.warning(
            f"Không có dữ liệu cho "
            f"{display_symbol(symbol)}."
        )

        return

    # ========================================================
    # SNAPSHOT
    # ========================================================

    snapshot = market_snapshot(
        data
    )

    # ========================================================
    # METRICS
    # ========================================================

    price = to_number(
        snapshot.get(
            "price"
        )
    )

    change_1d = to_number(
        snapshot.get(
            "change_1d"
        )
    )

    rsi_value = to_number(
        snapshot.get(
            "rsi"
        )
    )

    macd_value = to_number(
        snapshot.get(
            "macd"
        )
    )

    sma20 = to_number(
        snapshot.get(
            "sma20"
        )
    )

    sma50 = to_number(
        snapshot.get(
            "sma50"
        )
    )

    volatility20 = to_number(
        snapshot.get(
            "volatility20"
        )
    )

    volume = to_number(
        snapshot.get(
            "volume"
        )
    )

    # --------------------------------------------------------
    # Last row
    # --------------------------------------------------------

    last_row = (
        data.iloc[-1]
    )

    atr14 = to_number(
        last_row.get(
            "ATR14"
        )
    )

    volume_sma20 = to_number(
        last_row.get(
            "Volume_SMA20"
        )
    )

    relative_volume = None

    if (
        volume is not None
        and volume_sma20 is not None
        and volume_sma20 != 0
    ):

        relative_volume = (
            volume
            / volume_sma20
        )

    # ========================================================
    # TITLE
    # ========================================================

    st.subheader(
        f"📈 {display_symbol(symbol)}"
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Giá",
            format_price(
                price
            ),
        )

    with c2:

        st.metric(
            "1D",
            format_percent(
                change_1d
            ),
        )

    with c3:

        st.metric(
            "RSI",
            format_rsi(
                rsi_value
            ),
        )

    with c4:

        st.metric(
            "Khối lượng",
            format_volume(
                volume
            ),
        )

    # ========================================================
    # SECONDARY METRICS
    # ========================================================

    c5, c6, c7, c8 = st.columns(4)

    with c5:

        st.metric(
            "Trung bình 20 phiên",
            format_price(
                sma20
            ),
        )

    with c6:

        st.metric(
            "Trung bình 50 phiên",
            format_price(
                sma50
            ),
        )

    with c7:

        st.metric(
            "MACD",
            format_macd(
                macd_value
            ),
        )

    with c8:

        st.metric(
            "Biến động 20 phiên",
            (
                f"{volatility20:.2f}%"
                if volatility20 is not None
                else "—"
            ),
        )

    # ========================================================
    # TECHNICAL STATUS
    # ========================================================

    st.subheader(
        "🧭 Trạng thái kỹ thuật"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Xu hướng",
            price_vs_ma_status(
                price,
                sma20,
                sma50,
            ),
        )

    with s2:

        st.metric(
            "RSI",
            rsi_status(
                rsi_value
            ),
        )

    with s3:

        st.metric(
            "MACD",
            macd_status(
                macd_value
            ),
        )

    with s4:

        st.metric(
            "Thanh khoản",
            (
                f"{relative_volume:.2f}x TB20"
                if relative_volume is not None
                else "—"
            ),
        )

    # ========================================================
    # ADDITIONAL DATA
    # ========================================================

    st.subheader(
        "📋 Chỉ báo bổ sung"
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "ATR 14",
            format_price(
                atr14
            ),
        )

    with a2:

        st.metric(
            "Khối lượng TB20",
            format_volume(
                volume_sma20
            ),
        )

    with a3:

        open_price = to_number(
            last_row.get(
                "Open"
            )
        )

        st.metric(
            "Giá mở cửa",
            format_price(
                open_price
            ),
        )

    with a4:

        close_price = to_number(
            last_row.get(
                "Close"
            )
        )

        high_price = to_number(
            last_row.get(
                "High"
            )
        )

        low_price = to_number(
            last_row.get(
                "Low"
            )
        )

        if (
            close_price is not None
            and high_price is not None
            and low_price is not None
            and high_price != low_price
        ):

            position = (
                (
                    close_price
                    - low_price
                )
                / (
                    high_price
                    - low_price
                )
                * 100
            )

            st.metric(
                "Vị trí trong biên ngày",
                f"{position:.1f}%",
            )

        else:

            st.metric(
                "Vị trí trong biên ngày",
                "—",
            )

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "📊 Biểu đồ kỹ thuật"
    )

    try:

        chart = price_volume_chart(
            data
        )

        if chart is not None:

            st.plotly_chart(
                chart,
                width="stretch",
                config={
                    "displaylogo": False,
                },
            )

        else:

            st.info(
                "Chưa có biểu đồ kỹ thuật."
            )

    except Exception as error:

        st.warning(
            "Không thể hiển thị biểu đồ."
        )

        st.code(
            str(error)
        )

    # ========================================================
    # QUANT
    # ========================================================
    #
    # KHÔNG tự chạy.
    # Người dùng bấm nút mới chạy.
    # ========================================================

    st.subheader(
        "📐 Phân tích định lượng"
    )

    st.caption(
        "OLS, Random Forest và Quant được chạy khi bạn yêu cầu "
        "để tránh page tải chậm."
    )

    run_quant_button = st.button(
        "🧮 Chạy mô hình Quant",
        key="stock_analysis_run_quant",
    )

    if run_quant_button:

        with st.spinner(
            "Đang chạy mô hình định lượng..."
        ):

            quant_results = _run_quant_models(
                data
            )

        st.session_state[
            "stock_analysis_quant"
        ] = quant_results

    quant_results = st.session_state.get(
        "stock_analysis_quant"
    )

    # ========================================================
    # QUANT RESULT
    # ========================================================

    if quant_results is None:

        st.info(
            "Bấm «Chạy mô hình Quant» để tính OLS, "
            "Random Forest và các chỉ số đánh giá."
        )

    else:

        ols_result = (
            quant_results.get(
                "ols"
            )
        )

        forest_result = (
            quant_results.get(
                "forest"
            )
        )

        quant = (
            quant_results.get(
                "quant"
            )
        )

        q1, q2, q3 = st.columns(3)

        # ----------------------------------------------------
        # OLS
        # ----------------------------------------------------

        with q1:

            if ols_result is not None:

                st.success(
                    "OLS: mô hình đã chạy"
                )

                try:

                    r_squared = float(
                        ols_result.rsquared
                    )

                    st.metric(
                        "R²",
                        f"{r_squared:.3f}",
                    )

                except Exception:

                    st.caption(
                        "Không đọc được R²."
                    )

            else:

                st.info(
                    "OLS chưa đủ dữ liệu."
                )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        with q2:

            if forest_result is not None:

                st.success(
                    "Random Forest: đã chạy"
                )

                prediction = (
                    forest_result.get(
                        "prediction"
                    )
                )

                st.metric(
                    "Dự báo phiên kế",
                    quant_prediction_text(
                        prediction
                    ),
                )

            else:

                st.info(
                    "Random Forest chưa đủ dữ liệu."
                )

        # ----------------------------------------------------
        # Volatility
        # ----------------------------------------------------

        with q3:

            st.metric(
                "Biến động 20 phiên",
                (
                    f"{volatility20:.2f}%"
                    if volatility20 is not None
                    else "—"
                ),
            )

        # ====================================================
        # BUILD QUANT DETAIL
        # ====================================================

        if quant is None:

            st.info(
                "Chưa đủ dữ liệu để xây dựng Quant Model."
            )

        else:

            (
                ols_model,
                rf_model,
                metrics,
                next_prediction,
                importance,
            ) = quant

            st.markdown(
                "#### Kết quả Quant"
            )

            q4, q5, q6 = st.columns(3)

            with q4:

                mae = to_number(
                    metrics.get(
                        "MAE"
                    )
                )

                st.metric(
                    "MAE",
                    (
                        f"{mae:.6f}"
                        if mae is not None
                        else "—"
                    ),
                )

            with q5:

                r2 = to_number(
                    metrics.get(
                        "R2"
                    )
                )

                st.metric(
                    "R²",
                    (
                        f"{r2:.3f}"
                        if r2 is not None
                        else "—"
                    ),
                )

            with q6:

                st.metric(
                    "Dự báo phiên tiếp",
                    quant_prediction_text(
                        next_prediction
                    ),
                )

            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            if (
                importance is not None
                and not importance.empty
            ):

                importance_df = (
                    importance
                    .rename(
                        "Mức quan trọng"
                    )
                    .reset_index()
                )

                importance_df.columns = [
                    "Biến",
                    "Mức quan trọng",
                ]

                importance_df[
                    "Mức quan trọng"
                ] = importance_df[
                    "Mức quan trọng"
                ].round(4)

                st.dataframe(
                    importance_df,
                    width="stretch",
                    hide_index=True,
                )

    # ========================================================
    # AI STOCK ANALYSIS
    # ========================================================
    #
    # AI chỉ chạy khi bấm nút bên render_ai_panel.
    # ========================================================

    ai_prompt = stock_analysis_prompt(
        symbol=display_symbol(
            symbol
        ),

        snapshot={
            "price": price,
            "change_1d": change_1d,
            "RSI": rsi_value,
            "MACD": macd_value,
            "SMA20": sma20,
            "SMA50": sma50,
            "Volatility20": volatility20,
            "Volume": volume,
            "Volume_SMA20": volume_sma20,
            "Relative_Volume": relative_volume,
            "ATR14": atr14,
            "Xu hướng": price_vs_ma_status(
                price,
                sma20,
                sma50,
            ),
            "RSI trạng thái": rsi_status(
                rsi_value
            ),
            "MACD trạng thái": macd_status(
                macd_value
            ),
        },

        latest_row={
            "Date": str(
                data.index[-1]
            ),

            "Open": to_number(
                last_row.get(
                    "Open"
                )
            ),

            "High": to_number(
                last_row.get(
                    "High"
                )
            ),

            "Low": to_number(
                last_row.get(
                    "Low"
                )
            ),

            "Close": to_number(
                last_row.get(
                    "Close"
                )
            ),

            "Volume": to_number(
                last_row.get(
                    "Volume"
                )
            ),

            "RSI": to_number(
                last_row.get(
                    "RSI"
                )
            ),

            "MACD": to_number(
                last_row.get(
                    "MACD"
                )
            ),

            "MACD_Signal": to_number(
                last_row.get(
                    "MACD_Signal"
                )
            ),

            "MACD_Hist": to_number(
                last_row.get(
                    "MACD_Hist"
                )
            ),

            "SMA20": to_number(
                last_row.get(
                    "SMA20"
                )
            ),

            "SMA50": to_number(
                last_row.get(
                    "SMA50"
                )
            ),

            "EMA20": to_number(
                last_row.get(
                    "EMA20"
                )
            ),

            "EMA50": to_number(
                last_row.get(
                    "EMA50"
                )
            ),

            "Volatility20": to_number(
                last_row.get(
                    "Volatility20"
                )
            ),

            "ATR14": to_number(
                last_row.get(
                    "ATR14"
                )
            ),

            "Volume_SMA20": to_number(
                last_row.get(
                    "Volume_SMA20"
                )
            ),

            "Relative_Volume": to_number(
                last_row.get(
                    "Relative_Volume"
                )
            ),
        },
    )

    render_ai_panel(
        title="🤖 AI phân tích cổ phiếu",
        description=(
            "AI đọc dữ liệu kỹ thuật của mã đang xem "
            "và đưa ra bản phân tích có cấu trúc."
        ),
        prompt=ai_prompt,
        button_label="🤖 Phân tích mã này bằng AI",
        key="stock_analysis_ai",
    )


# ============================================================
# TƯƠNG THÍCH CODE CŨ
# ============================================================

def render_analysis():
    render_stock_analysis()
