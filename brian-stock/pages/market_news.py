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

# Ưu tiên model Flash.
GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

MAX_IMAGE_BYTES = 10 * 1024 * 1024


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
        value = str(value).strip()

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
    Chuẩn hóa toàn bộ tin về format thống nhất.
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

        key = " ".join(
            title.lower().split()
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
    Chỉ gọi news API khi cache hết hạn.
    """

    try:

        data = fetch_market_news(
            limit
        )

        return normalize_news(
            data
        )

    except Exception as error:

        return {
            "_error": str(
                error
            )
        }


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
# NEWS CONTEXT
# ============================================================

def build_news_context(
    news,
):
    """
    Chuyển tin nguồn thành context cho AI.
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
    has_image=False,
):
    """
    Prompt dùng cho:
    - text only
    - text + ảnh
    """

    user_question = _text(
        user_question,
        "",
    )

    if user_question:

        task = f"""
Người dùng đang yêu cầu:

"{user_question}"

Hãy trả lời trực tiếp yêu cầu này.
Chỉ sử dụng thông tin từ tin tức và hình ảnh được cung cấp.
Nếu câu hỏi không liên quan đến dữ liệu được cung cấp,
hãy nói rõ là chưa đủ dữ liệu.
""".strip()

    elif has_image:

        task = """
Hãy phân tích hình ảnh được người dùng gửi.

Xác định:
- hình ảnh đang thể hiện điều gì
- các con số/thông tin quan trọng
- tín hiệu đáng chú ý
- mối liên hệ với các tin tức đang có

Không đoán những chi tiết không nhìn thấy rõ.
""".strip()

    else:

        task = """
Hãy tự tổng hợp những tin quan trọng nhất
thành một bản Market Brief ngắn gọn.
""".strip()

    image_instruction = ""

    if has_image:

        image_instruction = """
ẢNH ĐƯỢC GỬI KÈM:

Bạn PHẢI quan sát hình ảnh và sử dụng các thông tin
nhìn thấy trong ảnh khi chúng liên quan đến yêu cầu.

Không được giả vờ nhìn thấy thông tin mà ảnh không thể hiện.
""".strip()

    return f"""
Bạn là BRIAN AI — trợ lý phân tích thị trường
chứng khoán Việt Nam.

==================================================
NHIỆM VỤ
==================================================

{task}

{image_instruction}

==================================================
NGUYÊN TẮC
==================================================

- Chỉ sử dụng dữ liệu được cung cấp.
- Không bịa số liệu.
- Không bịa sự kiện.
- Không tự bổ sung dữ liệu ngoài nguồn.
- Không lặp lại cùng một ý.
- Bỏ qua thông tin không liên quan.
- Gộp những tin nói về cùng một câu chuyện.
- Viết tự nhiên như chuyên viên phân tích.
- Không viết quá dài.
- Không viết kiểu báo cáo học thuật.
- Không đưa khuyến nghị mua/bán cá nhân hóa.
- Không khẳng định chắc chắn thị trường sẽ tăng hoặc giảm.
- Phân biệt rõ dữ liệu và nhận định.
- Nếu không đủ dữ liệu, nói:
  "Chưa đủ dữ liệu để kết luận."

==================================================
ƯU TIÊN
==================================================

1. Yếu tố ảnh hưởng rộng tới thị trường.
2. VN-INDEX / VN30.
3. Dòng tiền.
4. Chính sách và vĩ mô.
5. Tin quốc tế quan trọng.
6. Ngành / doanh nghiệp nổi bật.
7. Thông tin xuất hiện trong hình ảnh nếu có.

==================================================
FORMAT
==================================================

# 🧠 BRIAN NEWS

## Thị trường đang chú ý gì?

2–3 câu.

## 🔥 Điểm đáng chú ý

Chỉ tối đa 3 ý.

### 1. [Ý chính]
1–2 câu.

### 2. [Ý chính]
1–2 câu.

### 3. [Ý chính]
1–2 câu.

## 🟢 Hỗ trợ

Tối đa 2 ý.

## 🔴 Áp lực

Tối đa 2 ý.

## 🎯 Cần theo dõi

2–3 câu.

## Kết luận

Đúng 3 câu:
- trạng thái thị trường
- nguyên nhân chính
- điều cần theo dõi nhất

==================================================
TIN TỨC NGUỒN
==================================================

{news_context}
""".strip()


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def _is_temporary_error(
    error,
):
    text = str(
        error
    ).lower()

    tokens = [
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
        for token in tokens
    )


def _is_fatal_error(
    error,
):
    text = str(
        error
    ).lower()

    tokens = [
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
        for token in tokens
    )


# ============================================================
# GEMINI SINGLE REQUEST
# ============================================================

def _call_gemini_once(
    api_key,
    model,
    prompt,
    image_bytes=None,
    image_mime=None,
):
    """
    Một request duy nhất.

    Không function calling.
    Không tools.
    """

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key
    )

    contents = []

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if (
        image_bytes
        and image_mime
    ):

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=image_mime,
        )

        contents.append(
            image_part
        )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    contents.append(
        prompt
    )

    response = (
        client
        .models
        .generate_content(
            model=model,
            contents=contents,
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
# AI GENERATION
# ============================================================

def generate_market_ai(
    news_context,
    user_question="",
    image_bytes=None,
    image_mime=None,
):
    """
    Gửi text và có thể kèm ảnh.

    Mỗi model chỉ gọi 1 lần.
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
        user_question=user_question,
        has_image=bool(
            image_bytes
        ),
    )

    errors = []

    for model in GEMINI_MODELS:

        try:

            text = _call_gemini_once(
                api_key=api_key,
                model=model,
                prompt=prompt,
                image_bytes=image_bytes,
                image_mime=image_mime,
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
            # API KEY / QUOTA
            # ------------------------------------------------

            if _is_fatal_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # ------------------------------------------------
            # Không phải lỗi temporary
            # ------------------------------------------------

            if not _is_temporary_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # ------------------------------------------------
            # 503 -> model tiếp
            # ------------------------------------------------

    return {
        "ok": False,
        "text": "",
        "model": None,
        "error": (
            "Các model Gemini hiện không khả dụng.\n\n"
            + "\n".join(
                errors
            )
        ),
    }


# ============================================================
# AI CACHE — TEXT ONLY
# ============================================================

@st.cache_data(
    ttl=AI_CACHE_TTL,
    show_spinner=False,
)
def generate_market_ai_text_cached(
    news_context,
    user_question,
):
    """
    Cache cho trường hợp không có ảnh.
    """

    return generate_market_ai(
        news_context=news_context,
        user_question=user_question,
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
        f"AI đang có {len(news)} tin nguồn."
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
        "Chế độ",
        [
            "⚡ Tóm tắt nhanh",
            "✍️ Hỏi AI",
            "📷 Phân tích ảnh",
        ],
        horizontal=True,
        key="market_news_ai_mode",
    )

    # ========================================================
    # QUESTION
    # ========================================================

    user_question = ""

    if mode == "✍️ Hỏi AI":

        st.markdown(
            "#### ✍️ Nhập câu hỏi"
        )

        user_question = st.text_area(
            "Prompt",
            placeholder=(
                "Ví dụ:\n"
                "Hôm nay thị trường đang quan tâm điều gì nhất?\n"
                "Tin nào ảnh hưởng nhóm ngân hàng?\n"
                "Tóm tắt riêng câu chuyện nâng hạng MSCI.\n"
                "Lọc các tin tiêu cực.\n"
                "VN-Index tăng nhưng thanh khoản nói lên điều gì?"
            ),
            height=140,
            label_visibility="collapsed",
            key="market_news_ai_question",
        ).strip()

        button_text = (
            "🤖 Phân tích câu hỏi"
        )

    elif mode == "📷 Phân tích ảnh":

        st.markdown(
            "#### 📷 Gửi ảnh cho AI"
        )

        image_file = st.file_uploader(
            "Chọn ảnh",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            key="market_news_ai_image",
            help=(
                "Upload screenshot bảng giá, biểu đồ, "
                "báo cáo hoặc hình ảnh tin tức."
            ),
        )

        if image_file is not None:

            image_size = len(
                image_file.getvalue()
            )

            if image_size > MAX_IMAGE_BYTES:

                st.error(
                    "Ảnh quá lớn. Vui lòng dùng ảnh dưới 10 MB."
                )

                return

            st.image(
                image_file,
                caption="Ảnh đã chọn",
                width="stretch",
            )

        user_question = st.text_area(
            "Prompt cho ảnh",
            placeholder=(
                "Ví dụ:\n"
                "Phân tích biểu đồ này.\n"
                "Đọc các số liệu trong ảnh.\n"
                "Tín hiệu kỹ thuật đáng chú ý là gì?\n"
                "Kết hợp ảnh này với tin tức hôm nay."
            ),
            height=120,
            label_visibility="collapsed",
            key="market_news_ai_image_question",
        ).strip()

        button_text = (
            "📷 Phân tích ảnh"
        )

    else:

        button_text = (
            "🤖 Tóm tắt thị trường"
        )

        st.info(
            "AI sẽ tự chọn những câu chuyện quan trọng nhất "
            "từ các tin hiện có."
        )

        image_file = None

    # ========================================================
    # RUN
    # ========================================================

    if st.button(
        button_text,
        type="primary",
        width="stretch",
        key="market_news_ai_run",
    ):

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if (
            mode == "✍️ Hỏi AI"
            and not user_question
        ):

            st.warning(
                "Nhập câu hỏi trước."
            )

            return

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if (
            mode == "📷 Phân tích ảnh"
            and image_file is None
        ):

            st.warning(
                "Hãy upload một ảnh trước."
            )

            return

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = build_news_context(
            news
        )

        # ----------------------------------------------------
        # IMAGE BYTES
        # ----------------------------------------------------

        image_bytes = None
        image_mime = None

        if (
            mode == "📷 Phân tích ảnh"
            and image_file is not None
        ):

            image_bytes = (
                image_file.getvalue()
            )

            image_mime = _text(
                image_file.type,
                "image/jpeg",
            )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        with st.spinner(
            "Brian AI đang đọc dữ liệu..."
        ):

            # ------------------------------------------------
            # TEXT ONLY
            # Cache.
            # ------------------------------------------------

            if image_bytes is None:

                result = (
                    generate_market_ai_text_cached(
                        context,
                        user_question,
                    )
                )

            # ------------------------------------------------
            # IMAGE
            #
            # Không cache bytes phức tạp.
            # Chỉ gọi một request.
            # ------------------------------------------------

            else:

                result = generate_market_ai(
                    news_context=context,
                    user_question=user_question,
                    image_bytes=image_bytes,
                    image_mime=image_mime,
                )

        st.session_state[
            "market_news_ai_result"
        ] = result

        st.session_state[
            "market_news_ai_question_used"
        ] = user_question

        st.session_state[
            "market_news_ai_has_image"
        ] = bool(
            image_bytes
        )

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
    # QUESTION
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
                "Câu hỏi"
            )

            st.write(
                question_used
            )

    # ========================================================
    # IMAGE STATUS
    # ========================================================

    has_image = st.session_state.get(
        "market_news_ai_has_image",
        False,
    )

    if has_image:

        st.caption(
            "📷 AI đã phân tích ảnh được upload."
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

        with st.container(
            border=True
        ):

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
        "Tin tức thị trường được thu thập từ nguồn thật. "
        "Brian AI có thể tự tổng hợp, trả lời câu hỏi "
        "hoặc phân tích trực tiếp ảnh bạn gửi."
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

        generate_market_ai_text_cached.clear()

        st.session_state.pop(
            "market_news_ai_result",
            None,
        )

        st.session_state.pop(
            "market_news_ai_question_used",
            None,
        )

        st.session_state.pop(
            "market_news_ai_has_image",
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
    # ERROR
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
        "Brian AI chỉ phân tích dữ liệu được cung cấp "
        "và không thay thế việc kiểm tra nguồn gốc bài viết."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_news():
    render_market_news()


def render_market_news_page():
    render_market_news()
