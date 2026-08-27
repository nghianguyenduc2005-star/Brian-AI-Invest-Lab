from __future__ import annotations

import io
from typing import Any

import streamlit as st

from data.news import fetch_market_news


# ============================================================
# OPTIONAL CLIPBOARD IMAGE COMPONENT
# ============================================================

try:
    from streamlit_paste_button import paste_image_button

    PASTE_BUTTON_AVAILABLE = True

except Exception:
    paste_image_button = None
    PASTE_BUTTON_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

NEWS_CACHE_TTL = 600
AI_CACHE_TTL = 1800

DEFAULT_NEWS_COUNT = 15

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = (
    MAX_IMAGE_SIZE_MB
    * 1024
    * 1024
)

# Gemini 3 stable models.
#
# Ưu tiên nhanh trước.
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
        text = str(
            value
        ).strip()

    except Exception:
        return default

    if not text:
        return default

    return text


# ============================================================
# NORMALIZE NEWS
# ============================================================

def normalize_news(
    news,
):
    """
    Chuẩn hóa dữ liệu tin tức.

    Kết quả:
        [
            {
                title,
                source,
                published,
                summary,
                link,
            }
        ]
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

        if not published:

            published = _text(
                item.get(
                    "date"
                ),
                "",
            )

        summary = _text(
            item.get(
                "summary"
            ),
            "",
        )

        if not summary:

            summary = _text(
                item.get(
                    "description"
                ),
                "",
            )

        link = _text(
            item.get(
                "link"
            ),
            "",
        )

        result.append(
            {
                "title": title,
                "source": source,
                "published": published,
                "summary": summary,
                "link": link,
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
    Cache tin tức.

    Streamlit rerun không gọi API lại
    nếu cache còn hạn.
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
# GEMINI KEY
# ============================================================

def get_gemini_api_key():
    try:

        value = st.secrets.get(
            "GEMINI_API_KEY"
        )

    except Exception:

        value = None

    value = _text(
        value,
        "",
    )

    return value or None


# ============================================================
# NEWS CONTEXT
# ============================================================

def build_news_context(
    news,
):
    """
    Chuyển tin thật thành context cho Gemini.

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
# GENERAL AI PROMPT
# ============================================================

def build_ai_prompt(
    news_context,
    user_question="",
):
    """
    Prompt cho:
    - Tóm tắt nhanh
    - Hỏi AI
    """

    user_question = _text(
        user_question,
        "",
    )

    if user_question:

        task = f"""
NGƯỜI DÙNG ĐANG HỎI:

{user_question}

Hãy trả lời thẳng câu hỏi này.

Chỉ sử dụng các tin có liên quan.
Không cố nhét toàn bộ tin tức vào câu trả lời.
""".strip()

    else:

        task = """
Hãy tự chọn những câu chuyện quan trọng nhất
trong số các tin được cung cấp.

Mục tiêu là để người đọc hiểu nhanh thị trường
đang quan tâm điều gì.
""".strip()

    return f"""
Bạn là BRIAN AI, một chuyên viên phân tích
tin tức thị trường chứng khoán Việt Nam.

==================================================
NHIỆM VỤ
==================================================

{task}

==================================================
NGUYÊN TẮC
==================================================

- Chỉ sử dụng dữ liệu trong phần tin tức.
- Không dùng kiến thức bên ngoài.
- Không bịa số liệu.
- Không bịa sự kiện.
- Không lặp ý.
- Gom tin trùng thành một câu chuyện.
- Bỏ qua tin ít quan trọng.
- Không cố liệt kê tất cả các tin.
- Viết tự nhiên, giống một chuyên viên phân tích.
- Không dùng văn phong học thuật.
- Không viết quá dài.
- Không khuyến nghị mua/bán.
- Không khẳng định chắc chắn giá sẽ tăng/giảm.
- Khi thiếu dữ liệu, nói:
  "Chưa đủ dữ liệu để kết luận."

==================================================
ƯU TIÊN
==================================================

1. VN-INDEX / VN30
2. Dòng tiền
3. Chính sách / vĩ mô
4. Nâng hạng / dòng vốn ngoại
5. Tin quốc tế có ảnh hưởng
6. Ngành / doanh nghiệp nổi bật

==================================================
FORMAT
==================================================

# 🧠 BRIAN NEWS

## Thị trường đang chú ý gì?

2–3 câu.

## 🔥 3 điểm đáng chú ý

Chỉ chọn tối đa 3 ý.

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
- trạng thái hiện tại
- lý do chính
- điều quan trọng nhất cần theo dõi

==================================================
TIN TỨC
==================================================

{news_context}
""".strip()


# ============================================================
# IMAGE AI PROMPT
# ============================================================

def build_image_prompt(
    news_context,
    user_question,
):
    user_question = _text(
        user_question,
        "",
    )

    if not user_question:

        user_question = (
            "Phân tích những điểm quan trọng nhất "
            "trong hình ảnh này."
        )

    return f"""
Bạn là BRIAN AI, chuyên viên phân tích
thị trường chứng khoán Việt Nam.

Người dùng gửi một hình ảnh và yêu cầu:

"{user_question}"

==================================================
CÁCH PHÂN TÍCH
==================================================

1. Quan sát trực tiếp hình ảnh.
2. Xác định ảnh đang thể hiện gì.
3. Đọc các số liệu hoặc thông tin nhìn thấy rõ.
4. Chỉ sử dụng thông tin thực sự nhìn thấy.
5. Không đoán phần ảnh không rõ.
6. Kết hợp với tin tức nguồn nếu thực sự liên quan.
7. Trả lời đúng câu hỏi.
8. Không lan sang những chủ đề không liên quan.
9. Không khuyến nghị mua/bán.
10. Không khẳng định chắc chắn thị trường sẽ tăng/giảm.

==================================================
FORMAT
==================================================

# 📷 BRIAN AI — PHÂN TÍCH ẢNH

## 👀 Ảnh đang cho thấy gì?

Nói ngắn gọn.

## 🔎 Điểm đáng chú ý

Tối đa 3 điểm.

## 📰 Liên hệ với tin tức

Chỉ nói nếu có liên quan.

## 🎯 Kết luận

2–4 câu trả lời trực tiếp câu hỏi.

Nếu ảnh không đủ rõ:
"Chưa đủ dữ liệu để kết luận."

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

    return any(
        token in text
        for token in [
            "503",
            "unavailable",
            "service unavailable",
            "high demand",
            "overloaded",
            "temporarily unavailable",
            "resource exhausted",
            "internal server error",
        ]
    )


def _is_fatal_error(
    error,
):
    text = str(
        error
    ).lower()

    return any(
        token in text
        for token in [
            "401",
            "403",
            "invalid api key",
            "api key not valid",
            "permission denied",
            "unauthenticated",
            "quota exceeded",
            "429",
        ]
    )


# ============================================================
# GEMINI TEXT REQUEST
# ============================================================

def _call_gemini_text_once(
    api_key,
    model,
    prompt,
):
    """
    Một request text-only.
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
            "Gemini trả về nội dung rỗng."
        )

    return text


# ============================================================
# GEMINI IMAGE REQUEST
# ============================================================

def _call_gemini_image_once(
    api_key,
    model,
    prompt,
    image_bytes,
    image_mime,
):
    """
    Một request multimodal:

        image + text

    Gemini hỗ trợ gửi Part ảnh cùng prompt.
    """

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key
    )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=image_mime,
    )

    response = (
        client
        .models
        .generate_content(
            model=model,
            contents=[
                image_part,
                prompt,
            ],
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
# AI TEXT
# ============================================================

def generate_text_ai(
    news_context,
    user_question="",
):
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

    prompt = build_ai_prompt(
        news_context,
        user_question,
    )

    errors = []

    for model in GEMINI_MODELS:

        try:

            text = _call_gemini_text_once(
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

            if _is_fatal_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            if not _is_temporary_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # 503 => thử model tiếp

    return {
        "ok": False,
        "text": "",
        "model": None,
        "error": (
            "Các model Gemini hiện chưa khả dụng.\n\n"
            + "\n".join(
                errors
            )
        ),
    }


# ============================================================
# AI IMAGE
# ============================================================

def generate_image_ai(
    news_context,
    user_question,
    image_bytes,
    image_mime,
):
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

    prompt = build_image_prompt(
        news_context,
        user_question,
    )

    errors = []

    for model in GEMINI_MODELS:

        try:

            text = _call_gemini_image_once(
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

            if _is_fatal_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            if not _is_temporary_error(
                error_text
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

    return {
        "ok": False,
        "text": "",
        "model": None,
        "error": (
            "Không có model Gemini khả dụng.\n\n"
            + "\n".join(
                errors
            )
        ),
    }


# ============================================================
# TEXT AI CACHE
# ============================================================

@st.cache_data(
    ttl=AI_CACHE_TTL,
    show_spinner=False,
)
def generate_text_ai_cached(
    news_context,
    user_question,
):
    return generate_text_ai(
        news_context=news_context,
        user_question=user_question,
    )


# ============================================================
# READ UPLOADED IMAGE
# ============================================================

def read_uploaded_image(
    uploaded_file,
):
    if uploaded_file is None:

        return None

    try:

        data = uploaded_file.getvalue()

    except Exception:

        return None

    if not data:

        return None

    if len(data) > MAX_IMAGE_SIZE_BYTES:

        st.error(
            f"Ảnh vượt quá {MAX_IMAGE_SIZE_MB} MB."
        )

        return None

    mime = _text(
        getattr(
            uploaded_file,
            "type",
            None,
        ),
        "image/jpeg",
    )

    return {
        "bytes": data,
        "mime_type": mime,
        "source": "file",
        "name": _text(
            getattr(
                uploaded_file,
                "name",
                None,
            ),
            "image",
        ),
    }


# ============================================================
# PASTE IMAGE
# ============================================================

def read_pasted_image():
    """
    Lấy ảnh clipboard từ streamlit-paste-button.

    Component có thể trả PIL Image.
    Chuyển thành PNG bytes.
    """

    if not PASTE_BUTTON_AVAILABLE:
        return None

    try:

        result = paste_image_button(
            label="📋 Paste ảnh từ clipboard",
            text_color="#ffffff",
            background_color="#ff4b4b",
            hover_background_color="#d93636",
            key="market_news_clipboard",
            errors="raise",
        )

    except Exception:

        return None

    image = getattr(
        result,
        "image_data",
        None,
    )

    if image is None:

        return None

    try:

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG",
        )

        data = buffer.getvalue()

    except Exception:

        return None

    if len(data) > MAX_IMAGE_SIZE_BYTES:

        st.error(
            f"Ảnh clipboard vượt quá "
            f"{MAX_IMAGE_SIZE_MB} MB."
        )

        return None

    return {
        "bytes": data,
        "mime_type": "image/png",
        "source": "clipboard",
        "name": "clipboard.png",
    }


# ============================================================
# IMAGE INPUT
# ============================================================

def render_image_input():
    """
    Hỗ trợ:

    1. Chọn file
    2. Kéo-thả file
    3. Paste clipboard
    """

    st.markdown(
        "#### 📷 Gửi ảnh cho AI"
    )

    st.caption(
        "Chọn ảnh, kéo-thả ảnh hoặc paste ảnh từ clipboard."
    )

    # --------------------------------------------------------
    # FILE / DRAG DROP
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Chọn hoặc kéo-thả ảnh",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        accept_multiple_files=False,
        key="market_news_image_upload",
        help=(
            f"PNG / JPG / JPEG / WEBP · tối đa "
            f"{MAX_IMAGE_SIZE_MB} MB"
        ),
    )

    file_image = read_uploaded_image(
        uploaded_file
    )

    # --------------------------------------------------------
    # PASTE
    # --------------------------------------------------------

    pasted_image = read_pasted_image()

    # --------------------------------------------------------
    # PREFER CLIPBOARD IF AVAILABLE
    # --------------------------------------------------------

    image_data = (
        pasted_image
        if pasted_image is not None
        else file_image
    )

    # --------------------------------------------------------
    # SAVE TO SESSION
    # --------------------------------------------------------

    if image_data is not None:

        st.session_state[
            "market_news_selected_image"
        ] = image_data

    else:

        image_data = st.session_state.get(
            "market_news_selected_image"
        )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    if image_data is not None:

        st.success(
            f"Đã nhận ảnh: {image_data['name']}"
        )

        st.image(
            image_data["bytes"],
            caption=(
                "Nguồn: "
                + image_data["source"]
            ),
            width="stretch",
        )

    return image_data


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
            "Chưa có tin để phân tích."
        )

        return

    # ========================================================
    # MODE
    # ========================================================

    mode = st.radio(
        "Chế độ AI",
        [
            "⚡ Tóm tắt nhanh",
            "✍️ Hỏi AI",
            "📷 Phân tích ảnh",
        ],
        horizontal=True,
        key="market_news_ai_mode",
    )

    # ========================================================
    # TEXT MODE
    # ========================================================

    if mode == "⚡ Tóm tắt nhanh":

        st.info(
            "AI sẽ tự chọn 3 câu chuyện quan trọng nhất."
        )

        user_question = ""

        image_data = None

        button_text = (
            "🤖 Tóm tắt thị trường"
        )

    elif mode == "✍️ Hỏi AI":

        st.markdown(
            "#### ✍️ Nhập câu hỏi"
        )

        user_question = st.text_area(
            "Prompt",
            placeholder=(
                "Ví dụ:\n"
                "Hôm nay thị trường đang quan tâm điều gì?\n"
                "Tin nào ảnh hưởng nhóm ngân hàng?\n"
                "Tóm tắt riêng câu chuyện nâng hạng MSCI.\n"
                "Có tin gì tiêu cực đáng chú ý?"
            ),
            height=140,
            label_visibility="collapsed",
            key="market_news_ai_question",
        ).strip()

        image_data = None

        button_text = (
            "🤖 Phân tích câu hỏi"
        )

    # ========================================================
    # IMAGE MODE
    # ========================================================

    else:

        image_data = render_image_input()

        user_question = st.text_area(
            "Prompt ảnh",
            placeholder=(
                "Ví dụ:\n"
                "Phân tích biểu đồ này.\n"
                "Đọc các số liệu trong ảnh.\n"
                "Tín hiệu kỹ thuật nào đáng chú ý?\n"
                "Kết hợp ảnh này với tin tức hôm nay."
            ),
            height=130,
            label_visibility="collapsed",
            key="market_news_image_question",
        ).strip()

        button_text = (
            "📷 Phân tích ảnh"
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

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if (
            mode == "✍️ Hỏi AI"
            and not user_question
        ):

            st.warning(
                "Nhập câu hỏi trước."
            )

            return

        if (
            mode == "📷 Phân tích ảnh"
            and image_data is None
        ):

            st.warning(
                "Chọn hoặc paste ảnh trước."
            )

            return

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = build_news_context(
            news
        )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        with st.spinner(
            "Brian AI đang phân tích..."
        ):

            if mode == "📷 Phân tích ảnh":

                result = generate_image_ai(
                    news_context=context,
                    user_question=user_question,
                    image_bytes=image_data[
                        "bytes"
                    ],
                    image_mime=image_data[
                        "mime_type"
                    ],
                )

            else:

                result = generate_text_ai_cached(
                    context,
                    user_question,
                )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        st.session_state[
            "market_news_ai_result"
        ] = result

        st.session_state[
            "market_news_ai_question_used"
        ] = user_question

        st.session_state[
            "market_news_ai_mode_used"
        ] = mode

    # ========================================================
    # RESULT
    # ========================================================

    result = st.session_state.get(
        "market_news_ai_result"
    )

    if not result:

        return

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if not result.get(
        "ok",
        False,
    ):

        st.error(
            "AI hiện chưa khả dụng."
        )

        st.code(
            result.get(
                "error",
                "Không xác định.",
            )
        )

        return

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = result.get(
        "model"
    )

    if model:

        st.caption(
            f"Model: {model}"
        )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question_used = st.session_state.get(
        "market_news_ai_question_used",
        "",
    )

    if question_used:

        with st.container(
            border=True
        ):

            st.caption(
                "Prompt"
            )

            st.write(
                question_used
            )

    # --------------------------------------------------------
    # IMAGE STATUS
    # --------------------------------------------------------

    used_mode = st.session_state.get(
        "market_news_ai_mode_used",
        "",
    )

    if used_mode == "📷 Phân tích ảnh":

        st.caption(
            "📷 Kết quả có sử dụng ảnh người dùng gửi."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

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

        load_news_cached.clear()

        generate_text_ai_cached.clear()

        # Xóa AI result.
        st.session_state.pop(
            "market_news_ai_result",
            None,
        )

        st.session_state.pop(
            "market_news_ai_question_used",
            None,
        )

        st.session_state.pop(
            "market_news_ai_mode_used",
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
    # LOAD ERROR
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
