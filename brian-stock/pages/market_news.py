from __future__ import annotations

from typing import Any

import streamlit as st

from data.news import fetch_market_news


# ============================================================
# CONFIG
# ============================================================

NEWS_CACHE_TTL = 600
AI_CACHE_TTL = 1800

DEFAULT_NEWS_COUNT = 15

# Gemini hiện tại
GEMINI_MODEL = "gemini-3.7-flash"


# ============================================================
# TEXT
# ============================================================

def _text(
    value: Any,
    default: str = "",
) -> str:
    if value is None:
        return default

    text = str(value).strip()

    return text if text else default


# ============================================================
# NEWS NORMALIZE
# ============================================================

def normalize_news(news):
    """
    Chuẩn hóa dữ liệu từ data.news.
    """

    if not isinstance(news, list):
        return []

    result = []
    seen = set()

    for item in news:

        if not isinstance(item, dict):
            continue

        title = _text(
            item.get("title"),
            "",
        )

        if not title:
            continue

        key = " ".join(
            title.lower().split()
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(
            {
                "title": title,

                "source": _text(
                    item.get("source"),
                    "Không rõ nguồn",
                ),

                "published": _text(
                    item.get(
                        "published",
                        item.get(
                            "date",
                            "",
                        ),
                    )
                ),

                "summary": _text(
                    item.get(
                        "summary",
                        item.get(
                            "description",
                            "",
                        ),
                    )
                ),

                "link": _text(
                    item.get(
                        "link",
                        "",
                    )
                ),
            }
        )

    return result


# ============================================================
# LOAD NEWS
# ============================================================

@st.cache_data(
    ttl=NEWS_CACHE_TTL,
    show_spinner=False,
)
def load_news_cached(
    limit: int,
):
    """
    News cache.
    Không gọi API mỗi lần Streamlit rerun.
    """

    try:

        news = fetch_market_news(
            limit
        )

        return normalize_news(
            news
        )

    except Exception:

        return []


# ============================================================
# GEMINI KEY
# ============================================================

def get_gemini_api_key():
    try:

        key = st.secrets.get(
            "GEMINI_API_KEY"
        )

    except Exception:

        key = None

    if not key:
        return None

    return str(
        key
    ).strip()


# ============================================================
# BUILD NEWS INPUT
# ============================================================

def build_news_context(
    news,
):
    """
    Chuyển toàn bộ tin đã tải thành input cho AI.
    """

    blocks = []

    for index, item in enumerate(
        news,
        start=1,
    ):

        title = _text(
            item.get("title"),
            "Không có tiêu đề",
        )

        source = _text(
            item.get("source"),
            "Không rõ nguồn",
        )

        published = _text(
            item.get("published"),
            "",
        )

        summary = _text(
            item.get("summary"),
            "",
        )

        link = _text(
            item.get("link"),
            "",
        )

        parts = [
            f"=== TIN {index} ===",
            f"Tiêu đề: {title}",
            f"Nguồn: {source}",
        ]

        if published:
            parts.append(
                f"Thời gian: {published}"
            )

        if summary:
            parts.append(
                f"Tóm tắt: {summary}"
            )

        if link:
            parts.append(
                f"Link: {link}"
            )

        blocks.append(
            "\n".join(parts)
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# AI PROMPT
# ============================================================

def build_ai_prompt(
    news_context,
):
    return f"""
Bạn là BRIAN AI — hệ thống phân tích tin tức
cho thị trường chứng khoán Việt Nam.

Đọc TOÀN BỘ các tin tức bên dưới và tạo một
Market Intelligence Brief.

QUY TẮC:

- Chỉ sử dụng thông tin có trong dữ liệu.
- Không bịa số liệu.
- Không bịa sự kiện.
- Không tự lấy thêm tin từ trí nhớ.
- Nếu thiếu dữ liệu, nói rõ "Chưa đủ dữ liệu để kết luận."
- Không khẳng định quan hệ nhân quả khi dữ liệu chỉ cho thấy liên hệ.
- Không đưa khuyến nghị mua/bán cá nhân hóa.
- Gom những tin cùng nói về một chủ đề thành một câu chuyện.
- Ưu tiên các yếu tố có thể ảnh hưởng rộng đến thị trường.

FORMAT BẮT BUỘC:

# 🧠 BRIAN AI — MARKET BRIEF

## 1. Tổng quan
Viết 4–6 câu.
Trả lời:
- Thị trường đang được dẫn dắt bởi câu chuyện gì?
- Chủ đề nào nổi bật nhất?
- Thông tin hiện tại nghiêng tích cực, trung tính hay thận trọng?

## 2. 🔥 Tin quan trọng nhất
Chọn tối đa 5 câu chuyện.

Mỗi câu chuyện:
**Tên câu chuyện**
- Thông tin:
- Nguồn:
- Vì sao quan trọng:
- Tác động tiềm năng:

## 3. 🇻🇳 Thị trường Việt Nam
Phân tích:
- VN-INDEX
- VN30
- chứng khoán Việt Nam
- dòng tiền
- chính sách
- kinh tế trong nước

## 4. 🌎 Quốc tế
Chỉ đề cập nếu có dữ liệu:
- Mỹ
- Fed
- lãi suất
- Trung Quốc
- USD
- tỷ giá
- hàng hóa
- MSCI
- nâng hạng

## 5. 💰 Vĩ mô

### Hỗ trợ
Các yếu tố hỗ trợ thị trường.

### Gây áp lực
Các yếu tố gây áp lực.

### Chưa rõ
Những vấn đề chưa đủ dữ liệu.

## 6. 🏢 Doanh nghiệp / ngành
Xác định:
- doanh nghiệp nào được nhắc tới
- ngành nào được nhắc tới
- tác động tiềm năng
- tích cực / tiêu cực / chưa rõ

## 7. 🟢 Yếu tố tích cực
Tối đa 5 điểm.

## 8. 🔴 Yếu tố tiêu cực
Tối đa 5 điểm.

## 9. ⚠️ Rủi ro cần theo dõi
Tối đa 5 điểm.

## 10. 📊 Market Sentiment
Chọn đúng một:

**TÍCH CỰC**
**TRUNG TÍNH**
**THẬN TRỌNG**

Sau đó giải thích ngắn dựa trên tin.

## 11. 🎯 Kết luận
Viết 6–10 câu.

Bắt buộc trả lời:
- Thị trường đang quan tâm gì nhất?
- Yếu tố nào có khả năng ảnh hưởng rộng nhất?
- Điều gì đang hỗ trợ?
- Điều gì đang gây áp lực?
- Cần tiếp tục theo dõi gì?

========================
DỮ LIỆU TIN TỨC
========================

{news_context}
""".strip()


# ============================================================
# GEMINI
# ============================================================

def generate_market_brief(
    news_context,
):
    """
    Chỉ 1 request Gemini.
    Không retry.
    Không AFC.
    Không tools.
    """

    api_key = get_gemini_api_key()

    if not api_key:

        return {
            "ok": False,
            "text": "",
            "error": (
                "Thiếu GEMINI_API_KEY trong Streamlit Secrets."
            ),
        }

    try:

        from google import genai

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "error": (
                "Chưa cài package google-genai. "
                f"{error}"
            ),
        }

    try:

        client = genai.Client(
            api_key=api_key
        )

        prompt = build_ai_prompt(
            news_context
        )

        response = (
            client
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        )

        text = _text(
            getattr(
                response,
                "text",
                "",
            ),
            "",
        )

        if not text:

            return {
                "ok": False,
                "text": "",
                "error": (
                    "Gemini không trả về nội dung."
                ),
            }

        return {
            "ok": True,
            "text": text,
            "error": "",
        }

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "error": str(
                error
            ),
        }


# ============================================================
# CACHE AI
# ============================================================

@st.cache_data(
    ttl=AI_CACHE_TTL,
    show_spinner=False,
)
def generate_market_brief_cached(
    news_context,
):
    return generate_market_brief(
        news_context
    )


# ============================================================
# NEWS CARD
# ============================================================

def render_news_card(
    item,
):
    title = _text(
        item.get("title"),
        "Không có tiêu đề",
    )

    source = _text(
        item.get("source"),
        "Không rõ nguồn",
    )

    published = _text(
        item.get("published"),
        "",
    )

    summary = _text(
        item.get("summary"),
        "",
    )

    link = _text(
        item.get("link"),
        "",
    )

    with st.container(
        border=True
    ):

        st.markdown(
            f"**{title}**"
        )

        metadata = source

        if published:

            metadata += (
                f" · {published}"
            )

        st.caption(
            metadata
        )

        if summary:

            st.write(
                summary
            )

        if link:

            st.markdown(
                f"[Đọc bài ↗]({link})"
            )


# ============================================================
# AI UI
# ============================================================

def render_ai_brief(
    news,
):
    st.subheader(
        "✨ Tổng hợp bằng AI"
    )

    st.caption(
        f"Brian AI sẽ đọc toàn bộ {len(news)} tin hiện có."
    )

    if not news:

        st.info(
            "Không có tin để phân tích."
        )

        return

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    run_ai = st.button(
        "🤖 Tóm tắt & phân tích tin tức",
        type="primary",
        width="stretch",
        key="market_news_ai_button",
    )

    if run_ai:

        context = build_news_context(
            news
        )

        with st.spinner(
            "Brian AI đang tổng hợp..."
        ):

            result = (
                generate_market_brief_cached(
                    context
                )
            )

        st.session_state[
            "market_news_ai_result"
        ] = result

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = st.session_state.get(
        "market_news_ai_result"
    )

    if not result:

        st.info(
            "Bấm nút để AI phân tích toàn bộ tin."
        )

        return

    if not result.get(
        "ok",
        False,
    ):

        st.error(
            "AI không chạy được."
        )

        st.code(
            result.get(
                "error",
                "Unknown error",
            )
        )

        return

    text = result.get(
        "text",
        "",
    )

    if text:

        st.markdown(
            text
        )


# ============================================================
# PAGE
# ============================================================

def render_market_news():

    # ========================================================
    # HEADER
    # ========================================================

    st.caption(
        "BRIAN STOCK · MARKET INTELLIGENCE"
    )

    st.title(
        "📰 Tin tức thị trường"
    )

    st.write(
        "Tổng hợp tin tức thị trường và dùng Brian AI "
        "để tạo Market Intelligence Brief."
    )

    # ========================================================
    # CONTROL
    # ========================================================

    left, right = st.columns(
        [
            1,
            4,
        ]
    )

    with left:

        news_count = st.selectbox(
            "Số tin",
            [
                8,
                12,
                15,
                20,
            ],
            index=2,
            key="market_news_count",
        )

    with right:

        refresh = st.button(
            "🔄 Lấy tin mới",
            key="market_news_refresh",
        )

    # ========================================================
    # REFRESH
    # ========================================================

    if refresh:

        load_news_cached.clear()

        st.session_state.pop(
            "market_news_ai_result",
            None,
        )

        st.rerun()

    # ========================================================
    # NEWS
    # ========================================================

    try:

        news = load_news_cached(
            news_count
        )

    except Exception as error:

        st.error(
            "Không thể tải tin tức."
        )

        st.code(
            str(error)
        )

        return

    if not news:

        st.warning(
            "Hiện chưa lấy được tin tức."
        )

        return

    # ========================================================
    # STATUS
    # ========================================================

    st.success(
        f"Đã tải {len(news)} tin."
    )

    # ========================================================
    # AI
    # ========================================================

    render_ai_brief(
        news
    )

    # ========================================================
    # NEWS SOURCE
    # ========================================================

    st.divider()

    st.subheader(
        "📰 Tin tức nguồn"
    )

    for item in news:

        render_news_card(
            item
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "AI chỉ tổng hợp các tin đã được hệ thống tải. "
        "Kiểm tra nguồn gốc bài viết trước khi sử dụng "
        "cho quyết định đầu tư."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_news():
    render_market_news()


def render_market_news_page():
    render_market_news()
