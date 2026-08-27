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

# Ưu tiên model nhanh và rẻ.
#
# Nếu gặp lỗi 503 / unavailable:
#   3.5 Flash-Lite
#       ↓
#   3.6 Flash
#       ↓
#   3.7 Flash
#
# Mỗi model chỉ gọi 1 lần.
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]


# ============================================================
# TEXT HELPER
# ============================================================

def _text(
    value: Any,
    default: str = "",
) -> str:

    if value is None:
        return default

    try:
        value = str(
            value
        ).strip()

    except Exception:
        return default

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
    Chuẩn hóa tin về cùng một cấu trúc.

    Đồng thời:
    - bỏ phần tử không hợp lệ
    - bỏ tin không có tiêu đề
    - loại trùng tiêu đề
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

        dedupe_key = (
            " ".join(
                title
                .lower()
                .split()
            )
        )

        if dedupe_key in seen:
            continue

        seen.add(
            dedupe_key
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
                        "published"
                    ),
                    item.get(
                        "date",
                        "",
                    ),
                ),

                "summary": _text(
                    item.get(
                        "summary"
                    ),
                    item.get(
                        "description",
                        "",
                    ),
                ),

                "link": _text(
                    item.get(
                        "link"
                    ),
                    "",
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
    Cache dữ liệu tin tức.

    Không reload API mỗi lần Streamlit rerun.
    """

    try:

        raw_news = fetch_market_news(
            limit
        )

        return normalize_news(
            raw_news
        )

    except Exception as error:

        return {
            "_error": str(
                error
            )
        }


# ============================================================
# GEMINI API KEY
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
    Chuyển tin thành context cho Gemini.

    Chỉ gửi:
    - tiêu đề
    - nguồn
    - thời gian
    - nội dung/tóm tắt

    Không gửi link để giảm token.
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
    user_question="",
):
    """
    Có 2 mode:

    1. Không có câu hỏi:
       AI tự tóm tắt thị trường.

    2. Có câu hỏi:
       AI trả lời đúng câu hỏi người dùng.
    """

    user_question = _text(
        user_question,
        "",
    )

    if user_question:

        task = f"""
NGƯỜI DÙNG ĐANG HỎI:

"{user_question}"

Hãy trả lời TRỰC TIẾP câu hỏi trên.

Chỉ sử dụng những tin có liên quan đến câu hỏi.
Không cố nhét những tin không liên quan.
""".strip()

    else:

        task = """
Hãy tự chọn những thông tin quan trọng nhất
và tạo một bản tóm tắt thị trường.

Chỉ giữ lại những câu chuyện thực sự đáng chú ý.
Không cố đưa tất cả tin vào bài.
""".strip()

    return f"""
Bạn là BRIAN AI, trợ lý phân tích tin tức
thị trường chứng khoán Việt Nam.

==================================================
NHIỆM VỤ
==================================================

{task}

==================================================
NGUYÊN TẮC BẮT BUỘC
==================================================

1. Chỉ sử dụng dữ liệu được cung cấp.
2. Không bịa số liệu.
3. Không bịa sự kiện.
4. Không sử dụng thông tin ngoài dữ liệu.
5. Không lặp lại một ý nhiều lần.
6. Gom các tin cùng nói về một câu chuyện.
7. Bỏ qua các tin ít quan trọng.
8. Viết như một chuyên viên phân tích đang nói với
   nhà đầu tư.
9. Câu ngắn, rõ, tự nhiên.
10. Không viết kiểu báo cáo học thuật.
11. Không khuyến nghị mua/bán.
12. Không khẳng định chắc chắn giá sẽ tăng/giảm.
13. Nếu dữ liệu không đủ:
   "Chưa đủ dữ liệu để kết luận."
14. Phân biệt thông tin và nhận định.
15. Không suy diễn quá xa từ một tiêu đề báo.

==================================================
MỨC ĐỘ ƯU TIÊN
==================================================

Ưu tiên:

1. Yếu tố ảnh hưởng rộng tới thị trường.
2. VN-INDEX / VN30.
3. Dòng tiền.
4. Chính sách / vĩ mô.
5. Quốc tế có tác động trực tiếp.
6. Ngành / doanh nghiệp nổi bật.

==================================================
FORMAT
==================================================

# 🧠 BRIAN NEWS

## Thị trường đang chú ý gì?

Viết 2–3 câu.
Nói đúng câu chuyện chính.

## 🔥 Điểm đáng chú ý

Chỉ chọn tối đa 3 ý.

### 1. [Ý chính]

Viết 1–2 câu.

### 2. [Ý chính]

Viết 1–2 câu.

### 3. [Ý chính]

Viết 1–2 câu.

## 🟢 Hỗ trợ

Tối đa 2 ý.

## 🔴 Áp lực

Tối đa 2 ý.

## 🎯 Cần theo dõi

Viết 2–3 câu.

## Kết luận

Đúng 3 câu.

Câu 1:
Thị trường hiện tại đang tích cực,
trung tính hay thận trọng.

Câu 2:
Lý do chính.

Câu 3:
Điều quan trọng nhất cần tiếp tục theo dõi.

==================================================
DỮ LIỆU TIN TỨC
==================================================

{news_context}
""".strip()


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def _is_temporary_model_error(
    error,
):
    text = str(
        error
    ).lower()

    temporary_tokens = [
        "503",
        "unavailable",
        "service unavailable",
        "high demand",
        "overloaded",
        "temporarily unavailable",
        "resource exhausted",
        "internal server error",
    ]

    return any(
        token in text
        for token in temporary_tokens
    )


def _is_fatal_model_error(
    error,
):
    text = str(
        error
    ).lower()

    fatal_tokens = [
        "401",
        "403",
        "invalid api key",
        "api key not valid",
        "permission denied",
        "unauthenticated",
        "429",
        "quota exceeded",
    ]

    return any(
        token in text
        for token in fatal_tokens
    )


# ============================================================
# GEMINI ONE REQUEST
# ============================================================

def _call_gemini_once(
    api_key,
    model,
    prompt,
):
    """
    Một request Gemini.

    Không retry.
    Không AFC.
    Không function calling.
    """

    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    response = (
        client
        .models
        .generate_content(
            model=model,
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

        raise RuntimeError(
            "Gemini không trả về nội dung."
        )

    return text


# ============================================================
# GENERATE AI
# ============================================================

def generate_market_ai(
    news_context,
    user_question="",
):
    """
    Fallback theo model.

    Mỗi model chỉ thử một lần.
    """

    api_key = get_gemini_api_key()

    if not api_key:

        return {
            "ok": False,
            "text": "",
            "model": None,
            "error": (
                "Chưa cấu hình GEMINI_API_KEY "
                "trong Streamlit Secrets."
            ),
        }

    try:

        from google import genai  # noqa: F401

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
        news_context,
        user_question,
    )

    errors = []

    for model in GEMINI_MODELS:

        try:

            text = _call_gemini_once(
                api_key=api_key,
                model=model,
                prompt=prompt,
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
            # API key / quota / permission
            # ------------------------------------------------

            if _is_fatal_model_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # ------------------------------------------------
            # Lỗi không phải lỗi tạm thời
            # ------------------------------------------------

            if not _is_temporary_model_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # ------------------------------------------------
            # 503 -> model tiếp theo.
            # Không retry model hiện tại.
            # ------------------------------------------------

    return {
        "ok": False,
        "text": "",
        "model": None,
        "error": (
            "Không có model Gemini nào khả dụng "
            "ở thời điểm hiện tại.\n\n"
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
def generate_market_ai_cached(
    news_context,
    user_question,
):
    """
    Cache theo:
        news_context
        user_question

    Vì vậy:

        cùng tin + cùng câu hỏi
              ↓
        không gọi Gemini lại
    """

    return generate_market_ai(
        news_context,
        user_question,
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
        f"Brian AI đang có {len(news)} tin nguồn."
    )

    if not news:

        st.info(
            "Chưa có tin để AI phân tích."
        )

        return

    # ========================================================
    # MODE
    # ========================================================

    mode = st.radio(
        "Chế độ",
        [
            "⚡ Tóm tắt nhanh",
            "✍️ Hỏi AI",
        ],
        horizontal=True,
        key="market_news_ai_mode",
    )

    # ========================================================
    # AUTO SUMMARY
    # ========================================================

    if mode == "⚡ Tóm tắt nhanh":

        user_question = ""

        st.info(
            "AI sẽ tự lọc những câu chuyện quan trọng "
            "nhất trong các tin hiện có."
        )

        button_text = (
            "🤖 Tóm tắt thị trường"
        )

    # ========================================================
    # CUSTOM QUESTION
    # ========================================================

    else:

        st.markdown(
            "#### ✍️ Nhập câu hỏi cho AI"
        )

        user_question = st.text_area(
            "Prompt",
            placeholder=(
                "Ví dụ:\n\n"
                "Hôm nay thị trường đang quan tâm điều gì nhất?\n\n"
                "Tin nào ảnh hưởng tới nhóm ngân hàng?\n\n"
                "Tóm tắt riêng câu chuyện nâng hạng MSCI.\n\n"
                "Lọc những tin tiêu cực trong hôm nay.\n\n"
                "VN-Index tăng nhưng thanh khoản giảm nói lên điều gì?"
            ),
            height=150,
            label_visibility="collapsed",
            key="market_news_ai_question",
        ).strip()

        button_text = (
            "🤖 Phân tích câu hỏi"
        )

        if not user_question:

            st.caption(
                "Nhập câu hỏi rồi bấm nút."
            )

    # ========================================================
    # RUN
    # ========================================================

    if st.button(
        button_text,
        type="primary",
        width="stretch",
        key="market_news_ai_run_button",
    ):

        if (
            mode == "✍️ Hỏi AI"
            and not user_question
        ):

            st.warning(
                "Mày chưa nhập câu hỏi."
            )

            return

        context = build_news_context(
            news
        )

        with st.spinner(
            "Brian AI đang đọc tin..."
        ):

            result = (
                generate_market_ai_cached(
                    context,
                    user_question,
                )
            )

        st.session_state[
            "market_news_ai_result"
        ] = result

        st.session_state[
            "market_news_ai_question_used"
        ] = user_question

    # ========================================================
    # RESULT
    # ========================================================

    result = st.session_state.get(
        "market_news_ai_result"
    )

    if not result:

        return

    # ========================================================
    # ERROR
    # ========================================================

    if not result.get(
        "ok",
        False,
    ):

        st.error(
            "AI hiện chưa khả dụng."
        )

        error = result.get(
            "error",
            "Không xác định.",
        )

        st.code(
            error
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    model = result.get(
        "model"
    )

    if model:

        st.caption(
            f"Model: {model}"
        )

    # ========================================================
    # QUESTION USED
    # ========================================================

    question_used = st.session_state.get(
        "market_news_ai_question_used",
        "",
    )

    if question_used:

        with st.container(
            border=True
        ):

            st.caption(
                "Câu hỏi đã gửi"
            )

            st.write(
                question_used
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = _text(
        result.get(
            "text"
        ),
        "",
    )

    if output:

        st.markdown(
            output
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
        "để lọc ra những thông tin thực sự đáng chú ý."
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

        # ----------------------------------------------------
        # Clear news cache
        # ----------------------------------------------------

        load_news_cached.clear()

        # ----------------------------------------------------
        # Clear AI cache
        # ----------------------------------------------------

        generate_market_ai_cached.clear()

        # ----------------------------------------------------
        # Clear old AI result
        # ----------------------------------------------------

        st.session_state.pop(
            "market_news_ai_result",
            None,
        )

        st.session_state.pop(
            "market_news_ai_question_used",
            None,
        )

        st.rerun()

    # ========================================================
    # LOAD NEWS
    # ========================================================

    loaded = load_news_cached(
        news_count
    )

    # ========================================================
    # NEWS LOADING ERROR
    # ========================================================

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

    # ========================================================
    # NORMALIZE
    # ========================================================

    news = normalize_news(
        loaded
    )

    # ========================================================
    # EMPTY
    # ========================================================

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
        "Brian AI chỉ sử dụng các tin đã được hệ thống "
        "thu thập. Hãy kiểm tra bài viết gốc trước khi "
        "sử dụng thông tin cho quyết định đầu tư."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_news():
    render_market_news()


def render_market_news_page():
    render_market_news()
