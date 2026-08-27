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

GEMINI_MODEL = "gemini-3.7-flash"


# ============================================================
# TEXT HELPER
# ============================================================

def _text(
    value: Any,
    default: str = "",
) -> str:
    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


# ============================================================
# NORMALIZE NEWS
# ============================================================

def normalize_news(news):
    """
    Chuẩn hóa danh sách tin.
    Loại tin trùng tiêu đề.
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
    Chỉ gọi nguồn tin khi cache hết hạn
    hoặc khi người dùng bấm Lấy tin mới.
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
            "_error": str(error)
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
# BUILD NEWS CONTEXT
# ============================================================

def build_news_context(
    news,
):
    """
    Chuyển toàn bộ tin thành một input duy nhất cho Gemini.
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

        lines = [
            f"=== TIN {index} ===",
            f"Tiêu đề: {title}",
            f"Nguồn: {source}",
        ]

        if published:
            lines.append(
                f"Thời gian: {published}"
            )

        if summary:
            lines.append(
                f"Nội dung: {summary}"
            )

        if link:
            lines.append(
                f"Link: {link}"
            )

        blocks.append(
            "\n".join(lines)
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
Bạn là BRIAN AI — chuyên gia tổng hợp và phân tích
tin tức thị trường chứng khoán Việt Nam.

Bạn được cung cấp TOÀN BỘ tin tức thật đã được hệ thống
thu thập.

MỤC TIÊU:
Biến các tin rời rạc thành một bản Market Intelligence
Brief để người đọc hiểu ngay thị trường đang quan tâm gì.

QUY TẮC BẮT BUỘC:

1. Chỉ sử dụng thông tin trong dữ liệu được cung cấp.
2. Không bịa số liệu.
3. Không bịa sự kiện.
4. Không tự bổ sung thông tin ngoài dữ liệu.
5. Nếu dữ liệu không đủ, phải ghi:
   "Chưa đủ dữ liệu để kết luận."
6. Không đưa khuyến nghị mua/bán cá nhân hóa.
7. Không biến tương quan thành quan hệ nhân quả.
8. Nếu nhiều bài viết cùng nói về một câu chuyện,
   phải gom chúng thành một nhóm.
9. Phải phân biệt:
   - Thông tin từ nguồn
   - Nhận định tổng hợp
10. Ưu tiên các yếu tố có thể ảnh hưởng rộng đến
    thị trường chứng khoán Việt Nam.

============================================================
FORMAT TRẢ LỜI
============================================================

# 🧠 BRIAN AI — MARKET INTELLIGENCE BRIEF

## 1. TÓM TẮT NHANH

Viết 4–6 câu.

Bắt buộc trả lời:
- Thị trường đang quan tâm điều gì?
- Câu chuyện nào nổi bật nhất?
- Tâm lý thông tin đang nghiêng về đâu?
- Có yếu tố nào đáng chú ý trong ngắn hạn?

## 2. 🔥 5 CÂU CHUYỆN QUAN TRỌNG NHẤT

Chọn tối đa 5 câu chuyện.

Mỗi câu chuyện:

### [Tên câu chuyện]

**Thông tin:**
...

**Nguồn:**
...

**Vì sao quan trọng:**
...

**Tác động tiềm năng:**
...

## 3. 🇻🇳 THỊ TRƯỜNG VIỆT NAM

Phân tích nếu có dữ liệu:

- VN-INDEX
- VN30
- chứng khoán Việt Nam
- dòng tiền
- chính sách
- kinh tế Việt Nam
- tâm lý nhà đầu tư

## 4. 🌎 QUỐC TẾ

Chỉ phân tích nếu dữ liệu có đề cập:

- Mỹ
- Fed
- lãi suất
- Trung Quốc
- USD
- tỷ giá
- hàng hóa
- MSCI
- nâng hạng

## 5. 💰 VĨ MÔ

### 🟢 Hỗ trợ

Các yếu tố có khả năng hỗ trợ thị trường.

### 🔴 Gây áp lực

Các yếu tố có khả năng gây áp lực.

### ⚪ Chưa rõ

Các yếu tố cần thêm dữ liệu.

## 6. 🏢 DOANH NGHIỆP / NGÀNH

Nêu:

- doanh nghiệp được nhắc tới
- ngành được nhắc tới
- câu chuyện chính
- tác động tiềm năng
- tích cực / tiêu cực / chưa rõ

Không được tự tạo số liệu tài chính.

## 7. 🟢 YẾU TỐ TÍCH CỰC

Tối đa 5 yếu tố.

Mỗi yếu tố phải có bằng chứng từ dữ liệu.

## 8. 🔴 YẾU TỐ TIÊU CỰC

Tối đa 5 yếu tố.

Mỗi yếu tố phải có bằng chứng từ dữ liệu.

## 9. ⚠️ RỦI RO CẦN THEO DÕI

Tối đa 5 điểm.

## 10. 📊 MARKET SENTIMENT

Chọn đúng một:

**TÍCH CỰC**

hoặc

**TRUNG TÍNH**

hoặc

**THẬN TRỌNG**

Sau đó giải thích 3–5 câu.

## 11. 🎯 KẾT LUẬN

Viết 6–10 câu.

Phải trả lời rõ:

- Thị trường đang tập trung vào điều gì nhất?
- Yếu tố nào có khả năng ảnh hưởng rộng nhất?
- Điều gì đang hỗ trợ?
- Điều gì đang gây áp lực?
- Điều gì cần theo dõi tiếp?

============================================================
DỮ LIỆU TIN TỨC
============================================================

{news_context}
""".strip()


# ============================================================
# CALL GEMINI
# ============================================================

def _call_gemini(
    prompt,
):
    """
    Một request duy nhất.
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
                "Chưa cấu hình GEMINI_API_KEY "
                "trong Streamlit Secrets."
            ),
        }

    try:

        from google import genai

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "error": (
                "Không import được google-genai: "
                f"{error}"
            ),
        }

    try:

        client = genai.Client(
            api_key=api_key
        )

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "error": (
                f"Không khởi tạo Gemini: {error}"
            ),
        }

    try:

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
# AI CACHE
# ============================================================

@st.cache_data(
    ttl=AI_CACHE_TTL,
    show_spinner=False,
)
def generate_market_brief_cached(
    news_context,
):
    prompt = build_ai_prompt(
        news_context
    )

    return _call_gemini(
        prompt
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
# AI PANEL
# ============================================================

def render_ai_brief(
    news,
):

    st.subheader(
        "✨ Tổng hợp bằng AI"
    )

    st.caption(
        f"Brian AI sẽ đọc toàn bộ {len(news)} tin "
        "đang hiển thị."
    )

    if st.button(
        "🤖 Phân tích toàn bộ tin",
        type="primary",
        width="stretch",
        key="market_news_ai_button",
    ):

        news_context = build_news_context(
            news
        )

        with st.spinner(
            "Brian AI đang tổng hợp tin tức..."
        ):

            result = (
                generate_market_brief_cached(
                    news_context
                )
            )

        st.session_state[
            "market_news_ai_result"
        ] = result

        st.session_state[
            "market_news_ai_news_count"
        ] = len(news)

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
            "AI không chạy được."
        )

        st.code(
            result.get(
                "error",
                "Unknown error",
            )
        )

        return

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
# MAIN PAGE
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
        "Tổng hợp tin tức thị trường và sử dụng "
        "Brian AI để tạo Market Intelligence Brief."
    )

    # ========================================================
    # CONTROLS
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
    # REFRESH NEWS
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

    loaded_news = load_news_cached(
        news_count
    )

    # Error returned by loader
    if (
        isinstance(
            loaded_news,
            dict,
        )
        and "_error" in loaded_news
    ):

        st.error(
            "Không thể tải tin tức."
        )

        st.code(
            loaded_news[
                "_error"
            ]
        )

        return

    news = normalize_news(
        loaded_news
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

    render_ai_brief(
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
        "Brian AI chỉ tổng hợp thông tin từ các tin đã "
        "được hệ thống thu thập. Hãy kiểm tra nguồn gốc "
        "bài viết trước khi sử dụng cho quyết định đầu tư."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_news():
    render_market_news()


def render_market_news_page():
    render_market_news()
