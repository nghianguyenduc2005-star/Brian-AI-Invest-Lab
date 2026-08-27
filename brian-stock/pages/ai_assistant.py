from __future__ import annotations

import pandas as pd
import streamlit as st

from data.market import (
    build_quant,
    display_symbol,
    load_market_data,
    market_snapshot,
    normalize_symbol,
)

from data.news import fetch_market_news


# ============================================================
# CẤU HÌNH
# ============================================================

SO_TIN_MAC_DINH = 8
KY_MAC_DINH = "1y"


# ============================================================
# TIỆN ÍCH
# ============================================================

def _so(value, mac_dinh=None):
    try:
        value = float(value)

        if pd.isna(value):
            return mac_dinh

        return value

    except Exception:
        return mac_dinh


def _fmt_gia(value):
    value = _so(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng"


def _fmt_pct(value):
    value = _so(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def _fmt_rsi(value):
    value = _so(value)

    if value is None:
        return "—"

    return f"{value:.1f}"


def _fmt_volume(value):
    value = _so(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn"

    return f"{value:,.0f}"


def _fmt_prediction(value):
    value = _so(value)

    if value is None:
        return "—"

    return f"{value * 100:+.2f}%"


# ============================================================
# AI CLIENT
# ============================================================

def _lay_gemini_client():
    try:
        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            "",
        )
    except Exception:
        api_key = ""

    api_key = str(
        api_key or ""
    ).strip()

    if not api_key:
        return None

    try:
        from google import genai

        return genai.Client(
            api_key=api_key
        )

    except Exception:
        return None


# ============================================================
# GỌI GEMINI
# ============================================================

def _goi_gemini(
    prompt,
):
    client = _lay_gemini_client()

    if client is None:

        return (
            None,
            "Chưa cấu hình GEMINI_API_KEY "
            "hoặc package google-genai."
        )

    try:

        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt,
        )

        text = getattr(
            response,
            "text",
            None,
        )

        if not text:

            return (
                None,
                "Gemini không trả về nội dung."
            )

        return text, None

    except Exception as error:

        return (
            None,
            f"Gemini lỗi: {error}"
        )


# ============================================================
# LẤY TIN TỨC AN TOÀN
# ============================================================

def _lay_tin_tuc(
    symbol,
    so_tin,
):
    try:

        news = fetch_market_news(
            so_tin
        )

        if news is None:
            return []

        return list(news)

    except Exception as error:

        st.warning(
            f"Không lấy được tin tức: {error}"
        )

        return []


# ============================================================
# TẠO CONTEXT CHO AI
# ============================================================

def _tao_context(
    symbol,
    df,
    news,
    quant,
):

    latest = df.iloc[-1]

    context = []

    context.append(
        f"Mã cổ phiếu: {display_symbol(symbol)}"
    )

    try:

        ngay_cuoi = df.index[-1]

        context.append(
            "Ngày dữ liệu cuối: "
            + ngay_cuoi.strftime(
                "%d/%m/%Y"
            )
        )

    except Exception:
        pass

    close = _so(
        latest.get("Close")
    )

    ret = _so(
        latest.get("Return")
    )

    rsi = _so(
        latest.get("RSI")
    )

    macd = _so(
        latest.get("MACD")
    )

    macd_signal = _so(
        latest.get("MACD_Signal")
    )

    volatility = _so(
        latest.get("Volatility20")
    )

    sma20 = _so(
        latest.get("SMA20")
    )

    sma50 = _so(
        latest.get("SMA50")
    )

    volume = _so(
        latest.get("Volume")
    )

    atr14 = _so(
        latest.get("ATR14")
    )

    context.append(
        f"Giá đóng cửa: {close}"
    )

    context.append(
        f"Thay đổi 1D: "
        f"{ret * 100:.2f}%"
        if ret is not None
        else "Thay đổi 1D: không có dữ liệu"
    )

    context.append(
        f"RSI: {rsi:.2f}"
        if rsi is not None
        else "RSI: không có dữ liệu"
    )

    context.append(
        f"MACD: {macd:.4f}"
        if macd is not None
        else "MACD: không có dữ liệu"
    )

    context.append(
        f"MACD Signal: {macd_signal:.4f}"
        if macd_signal is not None
        else "MACD Signal: không có dữ liệu"
    )

    context.append(
        f"Volatility 20: {volatility:.2f}%"
        if volatility is not None
        else "Volatility 20: không có dữ liệu"
    )

    context.append(
        f"SMA20: {sma20}"
        if sma20 is not None
        else "SMA20: không có dữ liệu"
    )

    context.append(
        f"SMA50: {sma50}"
        if sma50 is not None
        else "SMA50: không có dữ liệu"
    )

    context.append(
        f"Khối lượng: {volume}"
        if volume is not None
        else "Khối lượng: không có dữ liệu"
    )

    context.append(
        f"ATR14: {atr14}"
        if atr14 is not None
        else "ATR14: không có dữ liệu"
    )

    # --------------------------------------------------------
    # Quant
    # --------------------------------------------------------

    if quant is not None:

        try:

            ols, rf, metrics, next_return, importance = quant

            context.append(
                f"Random Forest dự báo return "
                f"phiên kế tiếp: "
                f"{next_return * 100:.2f}%"
            )

            context.append(
                f"RF MAE: "
                f"{metrics.get('MAE', float('nan')) * 100:.3f}%"
            )

            context.append(
                f"RF R2: "
                f"{metrics.get('R2', float('nan')):.3f}"
            )

            try:

                context.append(
                    f"OLS R2: "
                    f"{ols.rsquared:.3f}"
                )

            except Exception:
                pass

            try:

                top_features = (
                    importance
                    .head(5)
                    .to_dict()
                )

                context.append(
                    f"Feature importance: "
                    f"{top_features}"
                )

            except Exception:
                pass

        except Exception:
            pass

    # --------------------------------------------------------
    # Tin tức
    # --------------------------------------------------------

    context.append("")
    context.append(
        "TIN TỨC THỰC TẾ ĐÃ THU THẬP:"
    )

    if not news:

        context.append(
            "Không có tin tức."
        )

    else:

        for item in news:

            title = str(
                item.get(
                    "title",
                    ""
                )
            ).strip()

            source = str(
                item.get(
                    "source",
                    ""
                )
            ).strip()

            published = str(
                item.get(
                    "published",
                    ""
                )
            ).strip()

            if title:

                context.append(
                    f"- {title} | "
                    f"{source} | "
                    f"{published}"
                )

    return "\n".join(
        context
    )


# ============================================================
# PROMPT
# ============================================================

def _tao_prompt(
    hanh_dong,
    symbol,
    context,
    cau_hoi="",
):

    yeu_cau = {
        "xu_huong": """
Phân tích xu hướng hiện tại.
Tập trung vào giá, SMA20, SMA50, RSI, MACD,
động lượng, volatility và thanh khoản.
Chỉ kết luận dựa trên dữ liệu được cung cấp.
""",
        "rui_ro": """
Phân tích rủi ro.
Tập trung vào RSI, volatility, MACD,
drawdown nếu có dữ liệu, thanh khoản,
tin tức tiêu cực và các yếu tố cần theo dõi.
Không bịa dữ liệu.
""",
        "tong_hop": """
Phân tích tổng hợp toàn bộ.
Kết hợp dữ liệu giá, chỉ báo kỹ thuật,
Quant/ML và tin tức.
Nêu rõ điểm tích cực, rủi ro,
độ tin cậy của dữ liệu và kết luận.
""",
    }.get(
        hanh_dong,
        """
Phân tích cổ phiếu dựa trên dữ liệu được cung cấp.
""",
    )

    prompt = f"""
Bạn là BRIAN AI INVEST LAB,
trợ lý nghiên cứu thị trường chứng khoán Việt Nam.

Hãy trả lời hoàn toàn bằng TIẾNG VIỆT.

Mã cổ phiếu:
{display_symbol(symbol)}

{yeu_cau}

Câu hỏi bổ sung của người dùng:
{cau_hoi or "Không có câu hỏi bổ sung."}

========================
DỮ LIỆU THỰC TẾ
========================

{context}

========================
QUY TẮC
========================

1. Chỉ sử dụng dữ liệu được cung cấp.
2. Không bịa số liệu.
3. Không bịa tin tức.
4. Phân biệt dữ liệu thực tế và nhận định.
5. Nếu thiếu dữ liệu, nói rõ "Chưa đủ dữ liệu để kết luận."
6. Không biến dự báo ML thành cam kết.
7. Không đưa khuyến nghị mua/bán tuyệt đối.
8. Đây là thông tin hỗ trợ nghiên cứu,
   không phải khuyến nghị đầu tư cá nhân.

Hãy viết rõ ràng, thực dụng, dễ đọc.
"""

    return prompt


# ============================================================
# HIỂN THỊ TIN
# ============================================================

def _hien_thi_tin(
    news,
):

    if not news:

        st.info(
            "Chưa có tin tức phù hợp."
        )

        return

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


# ============================================================
# PAGE
# ============================================================

def render_ai_assistant():

    # ========================================================
    # HEADER
    # ========================================================

    st.subheader(
        "🤖 AI Assistant"
    )

    st.caption(
        "Brian Stock · AI Research"
    )

    st.title(
        "Trợ lý phân tích đầu tư"
    )

    st.write(
        "AI đọc dữ liệu thị trường thật, "
        "chỉ báo kỹ thuật, Quant/ML và tin tức "
        "để hỗ trợ nghiên cứu cổ phiếu."
    )

    # ========================================================
    # INPUT
    # ========================================================

    symbol_default = st.session_state.get(
        "ai_assistant_symbol",
        "HPG",
    )

    symbol_input = st.text_input(
        "Mã cổ phiếu",
        value=symbol_default,
        placeholder="Ví dụ HPG, FPT, VCB...",
        key="ai_assistant_symbol_input",
    )

    a, b = st.columns(
        [
            1,
            3,
        ]
    )

    with a:

        so_tin = st.number_input(
            "Số tin",
            min_value=3,
            max_value=15,
            value=SO_TIN_MAC_DINH,
            step=1,
            key="ai_assistant_news_limit",
        )

    with b:

        period = st.selectbox(
            "Khoảng dữ liệu",
            [
                "3mo",
                "6mo",
                "1y",
                "2y",
                "5y",
            ],
            index=2,
            key="ai_assistant_period",
        )

    # ========================================================
    # LOAD
    # ========================================================

    if st.button(
        "Tải dữ liệu phân tích",
        type="primary",
        key="ai_assistant_load_button",
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
                "ai_assistant_symbol"
            ] = clean_symbol

            st.rerun()

    symbol = normalize_symbol(
        st.session_state.get(
            "ai_assistant_symbol",
            symbol_input,
        )
    )

    # ========================================================
    # CHỈ TẢI DỮ LIỆU KHI NGƯỜI DÙNG ĐÃ YÊU CẦU
    # ========================================================

    if not st.session_state.get(
        "ai_assistant_symbol"
    ):

        st.info(
            "Nhập mã cổ phiếu rồi bấm "
            "«Tải dữ liệu phân tích»."
        )

        return

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        with st.spinner(
            f"Đang tải dữ liệu {display_symbol(symbol)}..."
        ):

            df = load_market_data(
                symbol,
                period,
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

    if df is None or df.empty:

        st.warning(
            "Không có dữ liệu thị trường."
        )

        return

    # ========================================================
    # SNAPSHOT
    # ========================================================

    snapshot = market_snapshot(
        df
    )

    price = _so(
        snapshot.get("price")
    )

    change = _so(
        snapshot.get("change_1d")
    )

    rsi = _so(
        snapshot.get("rsi")
    )

    volume = _so(
        snapshot.get("volume")
    )

    # ========================================================
    # STOCK
    # ========================================================

    st.subheader(
        f"📈 {display_symbol(symbol)}"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Giá",
            _fmt_gia(price),
        )

    with m2:

        st.metric(
            "1D",
            _fmt_pct(change),
        )

    with m3:

        st.metric(
            "RSI",
            _fmt_rsi(rsi),
        )

    with m4:

        st.metric(
            "Khối lượng",
            _fmt_volume(volume),
        )

    # ========================================================
    # CÂU HỎI
    # ========================================================

    st.text_area(
        "Câu hỏi cho AI",
        placeholder=(
            "Ví dụ: Hãy đánh giá xu hướng hiện tại "
            "của HPG và những rủi ro cần chú ý."
        ),
        key="ai_assistant_question",
        height=100,
    )

    # ========================================================
    # ACTION BUTTONS
    # ========================================================

    st.subheader(
        "🧠 Phân tích"
    )

    b1, b2, b3 = st.columns(3)

    with b1:

        ask_trend = st.button(
            "📈 Phân tích xu hướng",
            key="ai_action_trend",
            use_container_width=True,
        )

    with b2:

        ask_risk = st.button(
            "⚠️ Phân tích rủi ro",
            key="ai_action_risk",
            use_container_width=True,
        )

    with b3:

        ask_total = st.button(
            "📊 Phân tích tổng hợp",
            key="ai_action_total",
            use_container_width=True,
        )

    # ========================================================
    # NEWS
    # ========================================================

    need_analysis = (
        ask_trend
        or ask_risk
        or ask_total
    )

    news = []

    quant = None

    if need_analysis:

        with st.spinner(
            "Đang thu thập tin tức và chạy Quant/ML..."
        ):

            news = _lay_tin_tuc(
                symbol,
                int(so_tin),
            )

            try:

                quant = build_quant(
                    df
                )

            except Exception:

                quant = None

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = _tao_context(
            symbol,
            df,
            news,
            quant,
        )

        question = st.session_state.get(
            "ai_assistant_question",
            "",
        )

        if ask_trend:

            action = "xu_huong"

        elif ask_risk:

            action = "rui_ro"

        else:

            action = "tong_hop"

        prompt = _tao_prompt(
            action,
            symbol,
            context,
            question,
        )

        with st.spinner(
            "AI đang phân tích..."
        ):

            answer, error = _goi_gemini(
                prompt
            )

        # ----------------------------------------------------
        # Kết quả
        # ----------------------------------------------------

        st.subheader(
            "🧠 Kết quả phân tích"
        )

        if error:

            st.warning(
                error
            )

        else:

            st.markdown(
                answer
            )

    else:

        st.caption(
            "Chọn một kiểu phân tích để AI xử lý dữ liệu."
        )

    # ========================================================
    # DỮ LIỆU AI ĐANG SỬ DỤNG
    # ========================================================

    with st.expander(
        "🔎 Xem dữ liệu AI đang sử dụng"
    ):

        latest = df.iloc[-1]

        data_preview = pd.DataFrame(
            {
                "Chỉ báo": [
                    "Giá đóng cửa",
                    "1D",
                    "RSI",
                    "MACD",
                    "SMA20",
                    "SMA50",
                    "Volatility20",
                    "Khối lượng",
                ],
                "Giá trị": [
                    latest.get("Close"),
                    (
                        latest.get("Return") * 100
                        if pd.notna(
                            latest.get("Return")
                        )
                        else None
                    ),
                    latest.get("RSI"),
                    latest.get("MACD"),
                    latest.get("SMA20"),
                    latest.get("SMA50"),
                    latest.get("Volatility20"),
                    latest.get("Volume"),
                ],
            }
        )

        st.dataframe(
            data_preview,
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # TIN TỨC
    # ========================================================

    if need_analysis:

        st.subheader(
            f"📰 Tin tức {display_symbol(symbol)}"
        )

        _hien_thi_tin(
            news
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.caption(
        "BRIAN AI INVEST LAB · "
        "Dữ liệu dùng để hỗ trợ nghiên cứu, "
        "không phải khuyến nghị mua/bán."
    )


# ============================================================
# TÊN HÀM TƯƠNG THÍCH
# ============================================================

def render_ai():
    render_ai_assistant()
