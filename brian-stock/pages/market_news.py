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

# Ưu tiên model nhanh.
#
# Không retry cùng một model.
# Nếu model đầu gặp 503 / unavailable thì mới thử model kế.
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
    Chuẩn hóa tất cả tin về cùng một format.

    Đồng thời loại bỏ:
    - tin không có tiêu đề
    - tin trùng tiêu đề
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
        # DEDUPE
        # ----------------------------------------------------

        dedupe_key = (
            " ".join(
                title.lower().split()
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
                        "published",
                        item.get(
                            "date",
                            "",
                        ),
                    ),
                    "",
                ),

                "summary": _text(
                    item.get(
                        "summary",
                        item.get(
                            "description",
                            "",
                        ),
                    ),
                    "",
                ),

                "link": _text(
                    item.get(
                        "link",
                        "",
                    ),
                    "",
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
    Cache nguồn tin.

    Cùng số lượng tin:
        -> không gọi API lại cho đến khi cache hết hạn.
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
    Chuyển tin thật thành context cho AI.

    Chỉ gửi:
        tiêu đề
        nguồn
        thời gian
        tóm tắt

    Không cần gửi link vào AI.
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
    Nếu user_question rỗng:
        AI tự tóm tắt.

    Nếu có user_question:
        AI trả lời đúng câu hỏi đó.
    """

    question = _text(
        user_question,
        "",
    )

    # ========================================================
    # MODE 1 — AUTO BRIEF
    # ========================================================

    if not question:

        task = """
Hãy tự chọn những thông tin quan trọng nhất
và tạo một Market Brief ngắn.

Không cần nhắc đến tất cả các tin.
Chỉ giữ lại những câu chuyện có khả năng
ảnh hưởng đáng kể tới thị trường.
""".strip()

    # ========================================================
    # MODE 2 — USER QUESTION
    # ========================================================

    else:

        task = f"""
Người dùng đang hỏi:

"{question}"

Hãy trả lời trực tiếp câu hỏi này.

Chỉ sử dụng những tin thực sự liên quan.
Không cố đưa các tin không liên quan vào câu trả lời.
""".strip()

    return f"""
Bạn là BRIAN AI — trợ lý phân tích tin tức
thị trường chứng khoán Việt Nam.

NHIỆM VỤ:

{task}

==================================================
NGUYÊN TẮC
==================================================

1. Chỉ dùng dữ liệu trong phần TIN TỨC.
2. Không dùng kiến thức ngoài dữ liệu được cung cấp.
3. Không bịa số liệu.
4. Không bịa sự kiện.
5. Không lặp lại một ý nhiều lần.
6. Gom các tin cùng một câu chuyện.
7. Bỏ qua tin ít quan trọng hoặc không liên quan.
8. Viết như một chuyên viên phân tích đang nói
   chuyện với nhà đầu tư.
9. Văn phong tự nhiên, dễ đọc.
10. Không viết kiểu báo cáo học thuật.
11. Không đưa khuyến nghị mua/bán cá nhân hóa.
12. Không khẳng định chắc chắn giá sẽ tăng/giảm.
13. Nếu không đủ dữ liệu:
   "Chưa đủ dữ liệu để kết luận."

==================================================
ƯU TIÊN
==================================================

Ưu tiên theo thứ tự:

1. Yếu tố ảnh hưởng rộng tới thị trường.
2. VN-INDEX / VN30.
3. Dòng tiền.
4. Chính sách / vĩ mô.
5. Tin quốc tế có liên quan trực tiếp.
6. Ngành / doanh nghiệp nổi bật.

Không cần nói về một chủ đề
nếu không có thông tin đáng kể.

==================================================
ĐẦU RA
==================================================

# 🧠 BRIAN NEWS

## Thị trường đang chú ý gì?

2–3 câu.

## 🔥 Điểm đáng chú ý

Chọn tối đa 3 ý.

### 1. [Ý chính]

1–2 câu giải thích.

### 2. [Ý chính]

1–2 câu giải thích.

### 3. [Ý chính]

1–2 câu giải thích.

## 🟢 Hỗ trợ

Tối đa 2 ý.

## 🔴 Áp lực

Tối đa 2 ý.

## 🎯 Cần theo dõi

2–3 câu.

## Kết luận

Đúng 3 câu.

Trong phần kết luận phải nói:
- bức tranh hiện tại tích cực / trung tính / thận trọng
- nguyên nhân chính
- điều quan trọng nhất cần theo dõi

==================================================
DỮ LIỆU TIN TỨC
==================================================

{news_context}
""".strip()


# ============================================================
# MODEL ERROR
# ============================================================

def is_temporary_error(
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
    ]

    return any(
        token in text
        for token in temporary_tokens
    )


def is_fatal_error(
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
# GEMINI SINGLE CALL
# ============================================================

def call_gemini_once(
    api_key,
    model,
    prompt,
):
    """
    Một request duy nhất.

    Không retry.
    Không AFC.
    Không tools.
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
    Gọi Gemini.

    Thứ tự:
        3.5 Flash-Lite
        -> 3.6 Flash
        -> 3.7 Flash

    Chỉ fallback khi model/service tạm thời không khả dụng.
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
                "Không cài được google-genai: "
                f"{error}"
            ),
        }

    prompt = build_ai_prompt(
        news_context,
        user_question=user_question,
    )

    errors = []

    for model in GEMINI_MODELS:

        try:

            text = call_gemini_once(
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
            # Lỗi xác thực / quota:
            # dừng ngay.
            # ------------------------------------------------

            if is_fatal_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # ------------------------------------------------
            # Lỗi không phải tạm thời:
            # dừng ngay.
            # ------------------------------------------------

            if not is_temporary_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # ------------------------------------------------
            # 503 -> thử model kế tiếp.
            # Không retry model hiện tại.
            # ------------------------------------------------

    return {
        "ok": False,
        "text": "",
        "model": None,
        "error": (
            "Các model Gemini hiện tại đều "
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
def generate_market_ai_cached(
    news_context,
    user_question,
):
    """
    Cache theo cả:

        news_context
        user_question

    Nghĩa là:

    cùng bộ tin + cùng câu hỏi
        -> không gọi Gemini lần nữa.
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
        "✨ BRIAN AI"
    )

    st.caption(
        f"AI đang có {len(news)} tin nguồn. "
        "Chọn tóm tắt nhanh hoặc tự đặt câu hỏi."
    )

    if not news:

        st.info(
            "Chưa có tin để phân tích."
        )

        return

    # ========================================================
    # MODE
    # ========================================================

    mode = st.radio(
        "Chế độ phân tích",
        [
            "⚡ Tóm tắt nhanh",
            "✍️ Hỏi AI",
        ],
        horizontal=True,
        key="market_news_ai_mode",
    )

    # ========================================================
    # QUESTION
    # ========================================================

    if mode == "✍️ Hỏi AI":

        user_question = st.text_area(
            "Mày muốn AI phân tích gì?",
            placeholder=(
                "Ví dụ:\n"
                "• Hôm nay thị trường đang lo điều gì nhất?\n"
                "• Tóm tắt riêng câu chuyện nâng hạng MSCI.\n"
                "• Tin nào ảnh hưởng tới nhóm ngân hàng?\n"
                "• Lọc riêng những tin tiêu cực.\n"
                "• Vì sao VN-Index tăng nhưng dòng tiền lại thận trọng?"
            ),
            height=130,
            key="market_news_ai_question",
        ).strip()

        button_label = (
            "🤖 Phân tích câu hỏi"
        )

    else:

        user_question = ""

        st.info(
            "AI sẽ tự chọn 3–4 câu chuyện quan trọng nhất "
            "thay vì nhồi toàn bộ tin vào báo cáo."
        )

        button_label = (
            "🤖 Tóm tắt thị trường"
        )

    # ========================================================
    # RUN
    # ========================================================

    if st.button(
        button_label,
        type="primary",
        width="stretch",
        key="market_news_ai_run",
    ):

        if (
            mode == "✍️ Hỏi AI"
            and not user_question
        ):

            st.warning(
                "Nhập câu hỏi trước."
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
    # QUESTION
    # ========================================================

    question_used = st.session_state.get(
        "market_news_ai_question_used",
        "",
    )

    if question_used:

        st.markdown(
            f"**Câu hỏi:** {question_used}"
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
        "Tin tức thị trường được tổng hợp từ nguồn thật. "
        "Brian AI giúp lọc ra những câu chuyện đáng chú ý "
        "hoặc trả lời đúng câu hỏi mà mày đặt ra."
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

        generate_market_ai_cached.clear()

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
    # LOAD
    # ========================================================

    loaded = load_news_cached(
        news_count
    )

    # --------------------------------------------------------
    # NEWS API ERROR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

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
    # SOURCE NEWS
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
        "Brian AI chỉ phân tích các tin đã được hệ thống "
        "tải về. Nội dung AI không thay thế việc kiểm tra "
        "nguồn gốc bài viết."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_news():
    render_market_news()


def render_market_news_page():
    render_market_news()
