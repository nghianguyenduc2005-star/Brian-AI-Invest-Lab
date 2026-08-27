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

# Ưu tiên model nhanh trước.
#
# 3.5 Flash-Lite:
# - phù hợp tác vụ tổng hợp
# - throughput cao
# - ưu tiên tốc độ
#
# Nếu không khả dụng -> 3.6 Flash
# Nếu vẫn lỗi -> 3.7 Flash
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]


# ============================================================
# TEXT
# ============================================================

def _text(
    value: Any,
    default: str = "",
) -> str:

    if value is None:
        return default

    value = str(
        value
    ).strip()

    if not value:
        return default

    return value


# ============================================================
# NORMALIZE NEWS
# ============================================================

def normalize_news(
    news,
):
    """
    Chuẩn hóa dữ liệu tin tức.
    """

    if not isinstance(
        news,
        list,
    ):
        return []

    result = []
    seen = set()

    for item in news:

        if not isinstance(
            item,
            dict,
        ):
            continue

        title = _text(
            item.get(
                "title"
            ),
            "",
        )

        if not title:
            continue

        # ----------------------------------------------------
        # Dedupe title
        # ----------------------------------------------------

        key = " ".join(
            title
            .lower()
            .split()
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            {
                "title": title,

                "source": _text(
                    item.get(
                        "source"
                    ),
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
# NEWS CACHE
# ============================================================

@st.cache_data(
    ttl=NEWS_CACHE_TTL,
    show_spinner=False,
)
def load_news_cached(
    limit: int,
):
    """
    Chỉ gọi nguồn news khi cache hết hạn.
    """

    try:

        raw = fetch_market_news(
            limit
        )

        return normalize_news(
            raw
        )

    except Exception as error:

        return {
            "_error": str(
                error
            )
        }


# ============================================================
# API KEY
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
# NEWS CONTEXT
# ============================================================

def build_news_context(
    news,
):
    """
    Chỉ đưa dữ liệu cần thiết cho AI.
    """

    blocks = []

    for index, item in enumerate(
        news,
        start=1,
    ):

        title = _text(
            item.get(
                "title"
            ),
            "Không có tiêu đề",
        )

        source = _text(
            item.get(
                "source"
            ),
            "Không rõ nguồn",
        )

        published = _text(
            item.get(
                "published"
            ),
            "",
        )

        summary = _text(
            item.get(
                "summary"
            ),
            "",
        )

        # ----------------------------------------------------
        # Không cần nhét link vào prompt.
        # Link vẫn hiển thị ở UI.
        # ----------------------------------------------------

        block = [
            f"=== TIN {index} ===",
            f"Tiêu đề: {title}",
            f"Nguồn: {source}",
        ]

        if published:
            block.append(
                f"Thời gian: {published}"
            )

        if summary:
            block.append(
                f"Nội dung: {summary}"
            )

        blocks.append(
            "\n".join(
                block
            )
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
thị trường chứng khoán Việt Nam.

Đọc toàn bộ tin tức dưới đây và tạo một bản
MARKET INTELLIGENCE BRIEF.

QUY TẮC:

- Chỉ sử dụng dữ liệu được cung cấp.
- Không bịa số liệu.
- Không bịa sự kiện.
- Không tự bổ sung dữ liệu từ trí nhớ.
- Không biến tương quan thành quan hệ nhân quả.
- Không đưa khuyến nghị mua/bán cá nhân hóa.
- Khi dữ liệu không đủ, phải ghi:
  "Chưa đủ dữ liệu để kết luận."
- Nếu nhiều tin cùng đề cập một câu chuyện,
  hãy gom thành một chủ đề.

==================================================
TRẢ LỜI THEO ĐÚNG CẤU TRÚC
==================================================

# 🧠 BRIAN AI — MARKET BRIEF

## 1. Tổng quan

4–5 câu.

Trả lời:
- thị trường đang quan tâm gì
- câu chuyện nổi bật nhất
- tâm lý thông tin hiện tại
- yếu tố đáng chú ý

## 2. 🔥 5 tin quan trọng nhất

Mỗi tin:

### Tên câu chuyện

**Thông tin:**
...

**Nguồn:**
...

**Vì sao quan trọng:**
...

**Tác động tiềm năng:**
...

## 3. 🇻🇳 Việt Nam

Phân tích nếu có:
- VN-INDEX
- VN30
- chứng khoán
- dòng tiền
- chính sách
- kinh tế Việt Nam

## 4. 🌎 Quốc tế

Phân tích nếu có:
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

### 🟢 Hỗ trợ

### 🔴 Gây áp lực

### ⚪ Chưa rõ

## 6. 🏢 Doanh nghiệp / ngành

Nêu:
- doanh nghiệp
- ngành
- câu chuyện
- tác động tiềm năng

## 7. 🟢 Yếu tố tích cực

Tối đa 5 điểm.

## 8. 🔴 Yếu tố tiêu cực

Tối đa 5 điểm.

## 9. ⚠️ Rủi ro

Tối đa 5 điểm.

## 10. 📊 Market Sentiment

Chọn đúng một:

**TÍCH CỰC**

**TRUNG TÍNH**

**THẬN TRỌNG**

Sau đó giải thích ngắn.

## 11. 🎯 Kết luận

6–8 câu.

Phải trả lời rõ:

- Thị trường đang quan tâm điều gì nhất?
- Yếu tố nào có khả năng ảnh hưởng rộng nhất?
- Điều gì đang hỗ trợ?
- Điều gì đang gây áp lực?
- Điều gì cần theo dõi tiếp?

==================================================
DỮ LIỆU TIN TỨC
==================================================

{news_context}
""".strip()


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def _is_temporary_model_error(
    error_text,
):
    """
    Xác định lỗi có khả năng chuyển model.
    """

    text = str(
        error_text
    ).lower()

    temporary_tokens = [
        "503",
        "unavailable",
        "service unavailable",
        "high demand",
        "overloaded",
        "temporarily unavailable",
        "resource exhausted",
    ]

    return any(
        token in text
        for token in temporary_tokens
    )


# ============================================================
# GEMINI SINGLE MODEL CALL
# ============================================================

def _call_single_gemini(
    api_key,
    model,
    prompt,
):
    """
    Một request duy nhất cho một model.
    """

    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    text = _text(
        getattr(
            response,
            "text",
            "",
        )
    )

    if not text:

        raise RuntimeError(
            "Gemini trả về nội dung rỗng."
        )

    return text


# ============================================================
# GEMINI FALLBACK
# ============================================================

def generate_market_brief(
    news_context,
):
    """
    Thử từng model đúng một lần.

    Không retry cùng model.

    3.5 Flash-Lite
          ↓ lỗi
    3.6 Flash
          ↓ lỗi
    3.7 Flash
    """

    api_key = get_gemini_api_key()

    if not api_key:

        return {
            "ok": False,
            "text": "",
            "model": None,
            "error": (
                "Thiếu GEMINI_API_KEY "
                "trong Streamlit Secrets."
            ),
        }

    try:

        import google.genai

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "model": None,
            "error": (
                "Không import được google-genai: "
                f"{error}"
            ),
        }

    prompt = build_ai_prompt(
        news_context
    )

    errors = []

    for model in GEMINI_MODELS:

        try:

            text = _call_single_gemini(
                api_key,
                model,
                prompt,
            )

            return {
                "ok": True,
                "text": text,
                "model": model,
                "error": "",
            }

        except Exception as error:

            error_text = str(
                error
            )

            errors.append(
                f"{model}: {error_text}"
            )

            # ------------------------------------------------
            # Chỉ fallback model khi lỗi là kiểu model/service.
            #
            # Với lỗi API key / permission / invalid request
            # không nên bắn thêm request vô ích.
            # ------------------------------------------------

            lower = error_text.lower()

            fatal_tokens = [
                "401",
                "403",
                "invalid api key",
                "api key not valid",
                "permission denied",
                "unauthenticated",
                "quota exceeded",
                "429",
            ]

            is_fatal = any(
                token in lower
                for token in fatal_tokens
            )

            if is_fatal:

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            if not _is_temporary_model_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # lỗi 503 -> thử model kế tiếp

    return {
        "ok": False,
        "text": "",
        "model": None,
        "error": (
            "Tất cả model Gemini hiện tại đều "
            "không khả dụng.\n\n"
            + "\n".join(
                errors
            )
        ),
    }


# ============================================================
# AI CACHE
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
        item.get(
            "title"
        ),
        "Không có tiêu đề",
    )

    source = _text(
        item.get(
            "source"
        ),
        "Không rõ nguồn",
    )

    published = _text(
        item.get(
            "published"
        ),
        "",
    )

    summary = _text(
        item.get(
            "summary"
        ),
        "",
    )

    link = _text(
        item.get(
            "link"
        ),
        "",
    )

    with st.container(
        border=True
    ):

        st.markdown(
            f"**{title}**"
        )

        meta = source

        if published:

            meta += (
                f" · {published}"
            )

        st.caption(
            meta
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
# AI PANEL
# ============================================================

def render_ai_panel(
    news,
):

    st.subheader(
        "✨ Tổng hợp bằng AI"
    )

    st.caption(
        f"Brian AI đọc toàn bộ {len(news)} tin đang có."
    )

    if not news:

        st.info(
            "Chưa có dữ liệu để AI phân tích."
        )

        return

    # --------------------------------------------------------
    # RUN AI
    # --------------------------------------------------------

    if st.button(
        "🤖 Tóm tắt & phân tích tin tức",
        type="primary",
        width="stretch",
        key="market_news_ai_button",
    ):

        context = build_news_context(
            news
        )

        # Hash-like context bằng chính chuỗi context.
        # Cache sẽ tự nhận biết cùng input.
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
            "Bấm nút phía trên để chạy AI."
        )

        return

    if not result.get(
        "ok",
        False,
    ):

        st.error(
            "AI hiện chưa khả dụng."
        )

        error = result.get(
            "error",
            "Unknown error",
        )

        st.code(
            error
        )

        return

    model = result.get(
        "model"
    )

    if model:

        st.caption(
            f"Model: {model}"
        )

    text = _text(
        result.get(
            "text"
        ),
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
        "để biến các tin rời rạc thành một Market Brief."
    )

    # ========================================================
    # CONTROL
    # ========================================================

    c1, c2 = st.columns(
        [
            1,
            4,
        ]
    )

    with c1:

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

    with c2:

        refresh = st.button(
            "🔄 Lấy tin mới",
            key="market_news_refresh",
        )

    # ========================================================
    # REFRESH
    # ========================================================

    if refresh:

        load_news_cached.clear()

        generate_market_brief_cached.clear()

        st.session_state.pop(
            "market_news_ai_result",
            None,
        )

        st.rerun()

    # ========================================================
    # LOAD NEWS
    # ========================================================

    loaded = load_news_cached(
        news_count
    )

    if (
        isinstance(
            loaded,
            dict,
        )
        and "_error" in loaded
    ):

        st.error(
            "Không thể tải tin tức."
        )

        st.code(
            loaded[
                "_error"
            ]
        )

        return

    news = normalize_news(
        loaded
    )

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

    render_ai_panel(
        news
    )

    # ========================================================
    # RAW NEWS
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
        "AI chỉ tổng hợp các tin đã được hệ thống thu thập. "
        "Kiểm tra nguồn trước khi sử dụng cho quyết định đầu tư."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_news():
    render_market_news()


def render_market_news_page():
    render_market_news()
