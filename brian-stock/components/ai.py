from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_ai_client():

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
# GỌI AI
# ============================================================

def ask_ai(
    prompt: str,
):

    client = get_ai_client()

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
                "AI không trả về nội dung."
            )

        return text, None

    except Exception as error:

        return (
            None,
            f"AI lỗi: {error}"
        )


# ============================================================
# DATAFRAME -> TEXT
# ============================================================

def dataframe_to_context(
    df: pd.DataFrame,
    max_rows: int = 30,
):
    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return "Không có dữ liệu."

    work = df.copy()

    work = work.head(
        max_rows
    )

    try:
        return work.to_string()
    except Exception:
        return str(work)


# ============================================================
# AI PANEL CHUNG
# ============================================================

def render_ai_panel(
    *,
    title: str,
    description: str,
    prompt: str,
    button_label: str = "🤖 Phân tích bằng AI",
    key: str,
):

    st.subheader(
        title
    )

    st.caption(
        description
    )

    if st.button(
        button_label,
        type="primary",
        key=key,
    ):

        with st.spinner(
            "AI đang phân tích..."
        ):

            answer, error = ask_ai(
                prompt
            )

        if error:

            st.warning(
                error
            )

        else:

            st.markdown(
                answer
            )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_prompt(
    *,
    vn_index: dict,
    stock_symbol: str,
    stock_snapshot: dict,
    news: list,
):
    return f"""
Bạn là BRIAN STOCK AI.

Phân tích nhanh tình trạng thị trường và cổ phiếu
từ dữ liệu thực tế được cung cấp.

VN-INDEX:
{vn_index}

CỔ PHIẾU:
{stock_symbol}

SNAPSHOT:
{stock_snapshot}

TIN MỚI:
{news}

Hãy trả lời bằng tiếng Việt.

Cấu trúc:
1. Tổng quan thị trường
2. Tình trạng cổ phiếu
3. Điểm tích cực
4. Rủi ro đáng chú ý
5. Điều cần theo dõi trong phiên tiếp theo

Quy tắc:
- Chỉ dùng dữ liệu được cung cấp.
- Không bịa số liệu.
- Không bịa tin tức.
- Không cam kết giá tương lai.
- Không đưa lệnh mua/bán tuyệt đối.
"""


# ============================================================
# STOCK ANALYSIS
# ============================================================

def stock_analysis_prompt(
    *,
    symbol: str,
    snapshot: dict,
    latest_row: dict,
):
    return f"""
Bạn là BRIAN STOCK AI chuyên phân tích kỹ thuật.

Mã:
{symbol}

SNAPSHOT:
{snapshot}

DỮ LIỆU PHIÊN GẦN NHẤT:
{latest_row}

Hãy phân tích bằng tiếng Việt.

Cấu trúc:
1. Xu hướng
2. Động lượng
3. RSI
4. MACD
5. SMA20 / SMA50
6. Thanh khoản
7. Biến động
8. Rủi ro
9. Kết luận

Chỉ dùng dữ liệu thực tế đã cung cấp.
Không bịa dữ liệu.
Không biến phân tích thành khuyến nghị mua/bán tuyệt đối.
"""


# ============================================================
# MARKET NEWS
# ============================================================

def market_news_prompt(
    news: list,
):
    return f"""
Bạn là BRIAN STOCK AI chuyên đọc tin tài chính.

Danh sách tin thực tế:
{news}

Hãy:

1. Tóm tắt các tin quan trọng nhất.
2. Phân loại:
   - Tích cực
   - Tiêu cực
   - Trung tính
3. Nêu tác động có thể có tới thị trường Việt Nam.
4. Chỉ ra các chủ đề nổi bật đang xuất hiện.
5. Nêu các tin cần theo dõi tiếp.

Trả lời bằng tiếng Việt.

Không bịa tin.
Không thêm thông tin không có trong dữ liệu.
Không coi tin tức là bằng chứng chắc chắn về giá cổ phiếu.
"""


# ============================================================
# PORTFOLIO
# ============================================================

def portfolio_prompt(
    portfolio_data: dict,
):
    return f"""
Bạn là BRIAN STOCK AI chuyên phân tích danh mục đầu tư.

DỮ LIỆU DANH MỤC:
{portfolio_data}

Hãy phân tích:

1. Mức độ đa dạng hóa
2. Mức độ tập trung
3. Rủi ro lớn nhất
4. Các vị thế cần theo dõi
5. Nhóm ngành bị tập trung
6. Điểm mạnh của danh mục
7. Điểm yếu của danh mục
8. Các câu hỏi nhà đầu tư nên kiểm tra tiếp

Trả lời bằng tiếng Việt.

Không tự bịa giá trị tài sản.
Không tự bịa tỷ trọng.
Không đưa lệnh mua/bán tuyệt đối.
"""


# ============================================================
# MARKET OVERVIEW
# ============================================================

def market_prompt(
    *,
    stats: dict,
    top_up,
    top_down,
    top_volume,
    sectors,
):
    return f"""
Bạn là BRIAN STOCK AI chuyên phân tích toàn thị trường.

THỐNG KÊ:
{stats}

TOP TĂNG:
{top_up}

TOP GIẢM:
{top_down}

TOP KHỐI LƯỢNG:
{top_volume}

NHÓM NGÀNH:
{sectors}

Hãy phân tích bằng tiếng Việt:

1. Độ rộng thị trường
2. Tâm lý
3. Thanh khoản
4. Nhóm ngành nổi bật
5. Cổ phiếu dẫn dắt
6. Rủi ro thị trường
7. Điều cần theo dõi

Chỉ dùng dữ liệu được cung cấp.
Không bịa dữ liệu.
Không đưa dự báo chắc chắn.
"""
