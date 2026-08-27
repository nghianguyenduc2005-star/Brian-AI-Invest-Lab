from __future__ import annotations

import time
from typing import Any

import streamlit as st

from data.news import fetch_market_news


# ============================================================
# CONFIG
# ============================================================

NEWS_CACHE_TTL = 600
AI_CACHE_TTL = 900

DEFAULT_NEWS_COUNT = 15

# Model Gemini hiện tại
GEMINI_MODEL = "gemini-3.7-flash"

# Retry khi lỗi tạm thời
AI_MAX_RETRIES = 2
AI_RETRY_SECONDS = 2


# ============================================================
# TEXT HELPERS
# ============================================================

def _text(
    value: Any,
    default: str = "",
) -> str:

    if value is None:
        return default

    text = str(value).strip()

    return (
        text
        if text
        else default
    )


# ============================================================
# NEWS NORMALIZATION
# ============================================================

def normalize_news(
    news,
):
    """
    Chuẩn hóa dữ liệu tin về một format duy nhất.
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
            item.get("title"),
            "",
        )

        if not title:
            continue

        # Deduplicate theo title
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
                    "Nguồn không xác định",
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
# LOAD NEWS
# ============================================================

@st.cache_data(
    ttl=NEWS_CACHE_TTL,
    show_spinner=False,
)
def load_news_cached(
    limit: int,
):
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

def _get_gemini_api_key():

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
    Chuyển toàn bộ tin thật thành context cho AI.
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

        link = _text(
            item.get(
                "link"
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
                f"Tóm tắt: {summary}"
            )

        if link:

            block.append(
                f"Link: {link}"
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
Bạn là BRIAN AI — hệ thống phân tích và tổng hợp tin tức
thị trường chứng khoán Việt Nam.

Bạn được cung cấp một tập tin tức THẬT đã được hệ thống
thu thập.

NHIỆM VỤ:
Đọc TOÀN BỘ tập tin và tạo một bản Market Intelligence Brief
ngắn gọn nhưng có chiều sâu.

QUY TẮC BẮT BUỘC:

1. Chỉ sử dụng dữ liệu trong các tin được cung cấp.
2. Không bịa số liệu.
3. Không bịa sự kiện.
4. Không tự bổ sung tin từ trí nhớ.
5. Nếu thông tin chưa đủ, phải nói rõ:
   "Chưa đủ dữ liệu để kết luận."
6. Phân biệt "thông tin từ nguồn" và "nhận định tổng hợp".
7. Không đưa khuyến nghị mua/bán cá nhân hóa.
8. Không được coi tương quan tin tức là quan hệ nhân quả.
9. Nếu nhiều tin cùng nói về một chủ đề,
   hãy gom thành một câu chuyện chung.
10. Ưu tiên những thông tin có thể ảnh hưởng rộng tới thị trường.

========================
OUTPUT
========================

# 🧠 BRIAN AI — MARKET INTELLIGENCE BRIEF

## 1. TÓM TẮT NHANH

Viết 4–6 câu:

- Thị trường hiện đang được dẫn dắt bởi câu chuyện gì?
- Chủ đề nào xuất hiện nổi bật nhất?
- Tâm lý thông tin hiện tại nghiêng tích cực, trung tính hay thận trọng?
- Có yếu tố nào đáng chú ý cho phiên tiếp theo?

## 2. 🔥 5 CÂU CHUYỆN QUAN TRỌNG NHẤT

Chọn tối đa 5 câu chuyện.

Mỗi câu chuyện gồm:

**Tên câu chuyện**
- Thông tin:
- Nguồn:
- Vì sao quan trọng:
- Tác động tiềm năng:

## 3. 🇻🇳 THỊ TRƯỜNG VIỆT NAM

Phân tích các thông tin liên quan:

- VN-INDEX
- VN30
- dòng tiền
- chính sách
- kinh tế Việt Nam
- thị trường chứng khoán
- tâm lý nhà đầu tư

## 4. 🌎 QUỐC TẾ

Chỉ phân tích những yếu tố thực sự xuất hiện:

- Mỹ
- Fed
- lãi suất
- Trung Quốc
- kinh tế toàn cầu
- hàng hóa
- USD / tỷ giá
- MSCI / nâng hạng

## 5. 💰 VĨ MÔ

Nhóm các tác động thành:

### Hỗ trợ
Các thông tin có khả năng hỗ trợ thị trường.

### Gây áp lực
Các thông tin có khả năng gây áp lực.

### Chưa rõ
Các yếu tố cần thêm dữ liệu.

## 6. 🏢 DOANH NGHIỆP / NGÀNH

Nêu:

- Doanh nghiệp được nhắc tới.
- Ngành được nhắc tới.
- Tin nào có tác động tiềm năng.
- Tác động tích cực / tiêu cực / chưa rõ.

Không được tự suy diễn số liệu tài chính không có trong nguồn.

## 7. 🟢 YẾU TỐ TÍCH CỰC

Tối đa 5 điểm.

Mỗi điểm phải có:
- yếu tố
- bằng chứng từ tin

## 8. 🔴 YẾU TỐ TIÊU CỰC

Tối đa 5 điểm.

Mỗi điểm phải có:
- yếu tố
- bằng chứng từ tin

## 9. ⚠️ RỦI RO CẦN THEO DÕI

Tối đa 5 điểm.

Chỉ dựa trên thông tin hiện có.

## 10. 📊 MARKET SENTIMENT

Chỉ chọn MỘT:

**TÍCH CỰC**

hoặc

**TRUNG TÍNH**

hoặc

**THẬN TRỌNG**

Sau đó giải thích 3–5 câu dựa trên tin tức.

## 11. 🎯 KẾT LUẬN CHO NHÀ ĐẦU TƯ

Viết 6–10 câu.

Phải trả lời:

- Hiện thị trường đang quan tâm điều gì nhất?
- Yếu tố nào có khả năng tác động rộng nhất?
- Yếu tố nào đang hỗ trợ?
- Yếu tố nào đang gây áp lực?
- Nhà đầu tư cần tiếp tục theo dõi gì?

Không được viết câu kiểu:
"chắc chắn VN-INDEX sẽ tăng"
hoặc
"nên mua cổ phiếu X".

========================
DỮ LIỆU TIN TỨC
========================

{news_context}
""".strip()


# ============================================================
# AI GENERATION
# ============================================================

def _generate_with_gemini(
    prompt,
):
    """
    Gọi Gemini bằng generate_content thuần text.
    Không dùng function calling / tools.
    """

    api_key = _get_gemini_api_key()

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
                "Chưa cài google-genai. "
                f"Chi tiết: {error}"
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
                f"Không khởi tạo Gemini client: {error}"
            ),
        }

    last_error = None

    for attempt in range(
        AI_MAX_RETRIES + 1
    ):

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

                last_error = (
                    "Gemini không trả về nội dung."
                )

            else:

                return {
                    "ok": True,
                    "text": text,
                    "error": "",
                }

        except Exception as error:

            last_error = error

            if attempt < AI_MAX_RETRIES:

                time.sleep(
                    AI_RETRY_SECONDS
                    * (
                        attempt + 1
                    )
                )

    return {
        "ok": False,
        "text": "",
        "error": (
            f"Gemini lỗi sau "
            f"{AI_MAX_RETRIES + 1} lần thử: "
            f"{last_error}"
        ),
    }


# ============================================================
# CACHED AI
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

    return _generate_with_gemini(
        prompt
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
        "Nguồn không xác định",
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
        f"Brian AI sẽ đọc toàn bộ "
        f"{len(news)} tin hiện có và tạo Market Brief."
    )

    if not news:

        st.info(
            "Chưa có tin để AI phân tích."
        )

        return

    if st.button(
        "🤖 Tóm tắt & phân tích tin tức",
        type="primary",
        key="market_news_ai_run",
        width="stretch",
    ):

        context = build_news_context(
            news
        )

        with st.spinner(
            "Brian AI đang đọc toàn bộ tin tức..."
        ):

            result = (
                generate_market_brief_cached(
                    context
                )
            )

        st.session_state[
            "market_news_ai_result"
        ] = result

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
            result.get(
                "error",
                "Không thể chạy AI.",
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
        "để phân tích thành một bản Market Intelligence Brief."
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

        if st.button(
            "🔄 Lấy tin mới",
            key="market_news_reload",
        ):

            load_news_cached.clear()

            st.session_state.pop(
                "market_news_ai_result",
                None,
            )

            st.rerun()

    # ========================================================
    # LOAD
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

    # ========================================================
    # STATUS
    # ========================================================

    if not news:

        st.warning(
            "Hiện chưa lấy được tin tức."
        )

        return

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
        "Nguồn dữ liệu được hệ thống thu thập từ các nguồn tin "
        "thị trường. AI chỉ tổng hợp dữ liệu đã tải và không "
        "thay thế việc kiểm tra nguồn gốc bài viết."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_news():
    render_market_news()


def render_market_news_page():
    render_market_news()
