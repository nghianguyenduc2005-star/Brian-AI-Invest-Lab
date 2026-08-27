from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

AI_CACHE_TTL = 1800

GEMINI_MODEL = "gemini-3.7-flash"

MAX_ROWS_PER_TABLE = 15


# ============================================================
# TEXT
# ============================================================

def _text(
    value: Any,
    default: str = "",
) -> str:
    if value is None:
        return default

    try:
        text = str(value).strip()
    except Exception:
        return default

    return text if text else default


def _num(
    value,
    default=np.nan,
):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


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

    key = _text(
        key,
        "",
    )

    return key or None


# ============================================================
# DATAFRAME -> TEXT
# ============================================================

def dataframe_to_text(
    df,
    columns=None,
    limit=MAX_ROWS_PER_TABLE,
):
    """
    Chuyển DataFrame thành text gọn để AI đọc.

    Không đẩy toàn bộ DataFrame khổng lồ lên model.
    """

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

    if columns:

        columns = [
            column
            for column in columns
            if column in work.columns
        ]

        if columns:
            work = work[columns]

    work = work.head(
        limit
    ).copy()

    # Làm tròn số để giảm token.
    for column in work.columns:

        if pd.api.types.is_numeric_dtype(
            work[column]
        ):

            work[column] = (
                pd.to_numeric(
                    work[column],
                    errors="coerce",
                )
                .round(6)
            )

    return work.to_string(
        index=False
    )


# ============================================================
# TEST FORMAT
# ============================================================

def format_tests(
    tests,
):
    if not isinstance(
        tests,
        dict,
    ):
        return "Không có kiểm định."

    lines = []

    # ADF
    adf = tests.get(
        "ADF"
    )

    if isinstance(
        adf,
        dict,
    ):

        lines.append(
            "ADF: "
            f"stat={_num(adf.get('statistic')):.5f}, "
            f"p={_num(adf.get('p_value')):.6f}"
        )

    # Jarque-Bera
    jb = tests.get(
        "Jarque-Bera"
    )

    if isinstance(
        jb,
        dict,
    ):

        lines.append(
            "Jarque-Bera: "
            f"stat={_num(jb.get('statistic')):.5f}, "
            f"p={_num(jb.get('p_value')):.6f}"
        )

    # Breusch-Pagan
    bp = tests.get(
        "Breusch-Pagan"
    )

    if isinstance(
        bp,
        dict,
    ):

        lines.append(
            "Breusch-Pagan: "
            f"stat={_num(bp.get('statistic')):.5f}, "
            f"p={_num(bp.get('p_value')):.6f}"
        )

    # White
    white = tests.get(
        "White"
    )

    if isinstance(
        white,
        dict,
    ):

        lines.append(
            "White: "
            f"stat={_num(white.get('statistic')):.5f}, "
            f"p={_num(white.get('p_value')):.6f}"
        )

    # DW
    dw = tests.get(
        "Durbin-Watson"
    )

    if isinstance(
        dw,
        dict,
    ):

        lines.append(
            "Durbin-Watson: "
            f"{_num(dw.get('statistic')):.5f}"
        )

    # Ljung-Box
    lb = tests.get(
        "Ljung-Box"
    )

    if isinstance(
        lb,
        dict,
    ):

        lines.append(
            "Ljung-Box: "
            f"lag={lb.get('lag', '—')}, "
            f"stat={_num(lb.get('statistic')):.5f}, "
            f"p={_num(lb.get('p_value')):.6f}"
        )

    if not lines:
        return "Không có kiểm định."

    return "\n".join(
        lines
    )


# ============================================================
# BUILD RESEARCH CONTEXT
# ============================================================

def build_research_ai_context(
    result,
    symbol="",
    start_date=None,
    end_date=None,
):
    """
    Tạo một báo cáo dữ liệu cô đọng để AI đọc.

    AI nhận:
    - số quan sát
    - toàn bộ danh sách biến
    - top factor
    - correlation
    - standardized OLS
    - permutation importance
    - VIF
    - model performance
    - statistical tests
    - 3 horizon nếu có
    """

    if not isinstance(
        result,
        dict,
    ):
        return "Không có kết quả nghiên cứu."

    blocks = []

    # ========================================================
    # HEADER
    # ========================================================

    blocks.append(
        "=== NGHIÊN CỨU ĐỊNH LƯỢNG ==="
    )

    blocks.append(
        f"Mã cổ phiếu: {_text(symbol, 'Không xác định')}"
    )

    if start_date is not None:
        blocks.append(
            f"Từ ngày: {_text(start_date)}"
        )

    if end_date is not None:
        blocks.append(
            f"Đến ngày: {_text(end_date)}"
        )

    blocks.append(
        f"Tổng biến: {len(result.get('features', []))}"
    )

    blocks.append(
        f"Biến sử dụng: {len(result.get('usable_features', []))}"
    )

    blocks.append(
        f"Biến variance = 0: {len(result.get('zero_variance', []))}"
    )

    # ========================================================
    # VARIABLE GROUPS
    # ========================================================

    features = result.get(
        "features",
        [],
    )

    if features:

        blocks.append(
            "\nDANH SÁCH BIẾN:"
        )

        blocks.append(
            ", ".join(
                map(
                    str,
                    features,
                )
            )
        )

    # ========================================================
    # EACH HORIZON
    # ========================================================

    horizons = result.get(
        "horizons",
        {},
    )

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = horizons.get(
            horizon
        )

        if not isinstance(
            item,
            dict,
        ):
            continue

        blocks.append(
            "\n"
            + "=" * 70
        )

        blocks.append(
            f"HORIZON {horizon}"
        )

        blocks.append(
            "=" * 70
        )

        blocks.append(
            f"Quan sát: {item.get('observations', 0)}"
        )

        blocks.append(
            f"Train: {item.get('train', 0)}"
        )

        blocks.append(
            f"Test: {item.get('test', 0)}"
        )

        # ----------------------------------------------------
        # FACTOR RANKING
        # ----------------------------------------------------

        ranking = item.get(
            "ranking"
        )

        blocks.append(
            "\nTOP YẾU TỐ:"
        )

        blocks.append(
            dataframe_to_text(
                ranking,
                columns=[
                    "Biến",
                    "Score",
                    "Quan hệ",
                    "Pearson",
                    "Spearman",
                    "Beta",
                    "p-value",
                    "Permutation",
                    "Ý nghĩa",
                ],
                limit=15,
            )
        )

        # ----------------------------------------------------
        # OLS
        # ----------------------------------------------------

        ols = item.get(
            "ols"
        )

        blocks.append(
            "\nOLS:"
        )

        if isinstance(
            ols,
            dict,
        ):

            blocks.append(
                f"R2 = {_num(ols.get('r2')):.6f}"
            )

            blocks.append(
                f"Adjusted R2 = {_num(ols.get('adj_r2')):.6f}"
            )

        else:

            blocks.append(
                "OLS không có kết quả."
            )

        # ----------------------------------------------------
        # OLS TABLE
        # ----------------------------------------------------

        blocks.append(
            "\nTOP HỆ SỐ OLS:"
        )

        blocks.append(
            dataframe_to_text(
                item.get(
                    "ols_table"
                ),
                columns=[
                    "Biến",
                    "Beta",
                    "p-value",
                    "Std Error",
                ],
                limit=15,
            )
        )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        models = item.get(
            "models"
        )

        blocks.append(
            "\nSO SÁNH MÔ HÌNH:"
        )

        blocks.append(
            dataframe_to_text(
                models,
                columns=[
                    "Mô hình",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R²",
                ],
                limit=10,
            )
        )

        # ----------------------------------------------------
        # VIF
        # ----------------------------------------------------

        vif = item.get(
            "vif"
        )

        blocks.append(
            "\nVIF CAO NHẤT:"
        )

        blocks.append(
            dataframe_to_text(
                vif,
                columns=[
                    "Biến",
                    "VIF",
                ],
                limit=10,
            )
        )

        # ----------------------------------------------------
        # PERMUTATION
        # ----------------------------------------------------

        permutation = item.get(
            "permutation"
        )

        blocks.append(
            "\nPERMUTATION IMPORTANCE:"
        )

        blocks.append(
            dataframe_to_text(
                permutation,
                columns=[
                    "Biến",
                    "Permutation",
                    "Permutation_STD",
                ],
                limit=15,
            )
        )

        # ----------------------------------------------------
        # TESTS
        # ----------------------------------------------------

        blocks.append(
            "\nKIỂM ĐỊNH:"
        )

        blocks.append(
            format_tests(
                item.get(
                    "tests",
                    {},
                )
            )
        )

    return "\n".join(
        blocks
    )


# ============================================================
# PROMPT
# ============================================================

def build_research_ai_prompt(
    research_context,
    user_question="",
):
    user_question = _text(
        user_question,
        "",
    )

    if user_question:

        question_block = f"""
CÂU HỎI CỦA NGƯỜI DÙNG:

{user_question}

Hãy trả lời thẳng câu hỏi trên bằng số liệu nghiên cứu.
""".strip()

    else:

        question_block = """
Không có câu hỏi riêng.

Hãy tự đọc nghiên cứu và giải thích:
- yếu tố nào nổi bật nhất
- ảnh hưởng theo chiều nào
- bằng chứng thống kê mạnh đến đâu
- mô hình nào đáng tin hơn
- kết luận thực tế cho nhà đầu tư là gì
""".strip()

    return f"""
Bạn là BRIAN AI — chuyên viên đọc và giải thích
nghiên cứu định lượng chứng khoán.

Nhiệm vụ của bạn KHÔNG phải chạy lại mô hình.

Bạn phải ĐỌC KẾT QUẢ NGHIÊN CỨU đã có và biến nó
thành một kết luận dễ hiểu cho con người.

==================================================
{question_block}
==================================================

QUY TẮC RẤT QUAN TRỌNG
==================================================

1. Chỉ sử dụng số liệu trong nghiên cứu.
2. Không tự bịa số liệu.
3. Không tự thêm biến không xuất hiện.
4. Không nhầm correlation với causality.
5. Không nói "biến X gây ra giá tăng" nếu nghiên cứu
   chỉ cho thấy quan hệ thống kê.
6. Phân biệt:
   - liên hệ
   - ý nghĩa thống kê
   - khả năng dự báo
7. Nếu p-value > 0.05 thì không được nói biến đó
   có bằng chứng mạnh ở mức 5%.
8. Nếu R² âm trên test thì phải nói rõ mô hình chưa
   vượt benchmark trung bình.
9. Nếu VIF rất cao thì cảnh báo hệ số OLS khó diễn giải.
10. Nếu Breusch-Pagan / White có p < 0.05 thì nói rằng
    có dấu hiệu heteroskedasticity.
11. Nếu Durbin-Watson hoặc Ljung-Box cho thấy tự tương quan
    thì phải nói rõ.
12. Không chọn một biến chỉ vì nó có score cao.
13. Ưu tiên yếu tố đồng thời có:
    - correlation đáng kể
    - beta đáng kể
    - p-value tốt
    - permutation importance
    - xuất hiện nhất quán ở nhiều horizon.
14. Không dài dòng.
15. Viết như một chuyên viên phân tích đang giải thích
    cho nhà đầu tư.
16. Không khuyến nghị mua/bán cá nhân hóa.

==================================================
ĐIỀU CẦN TRẢ LỜI
==================================================

# 🧠 BRIAN AI — ĐỌC NGHIÊN CỨU

## 1. Kết luận chính

Viết 3–5 câu.

Bắt buộc trả lời:
- Yếu tố nào nổi bật nhất?
- Theo chiều nào?
- Bằng chứng mạnh hay yếu?
- Kết luận này có nhất quán không?

## 2. 🎯 Giá/lợi suất đang phụ thuộc vào gì?

Chọn tối đa 5 yếu tố quan trọng nhất.

Với mỗi yếu tố:
- Tên biến
- Quan hệ cùng chiều/ngược chiều
- Spearman hoặc Pearson
- Beta nếu có
- p-value nếu có
- Ý nghĩa thực tế

## 3. 📈 Theo từng horizon

Nếu có 1D, 5D, 20D:

Nói ngắn gọn:
- Yếu tố nào mạnh ở 1D?
- Yếu tố nào mạnh ở 5D?
- Yếu tố nào mạnh ở 20D?
- Yếu tố nào xuất hiện ổn định nhất?

## 4. 🤖 Mô hình dự báo

Nêu:
- mô hình tốt nhất
- RMSE
- R² test
- mô hình có thực sự có sức dự báo đáng kể không

## 5. ⚠️ Chất lượng nghiên cứu

Chỉ nêu các vấn đề thực sự xuất hiện:
- đa cộng tuyến
- heteroskedasticity
- tự tương quan
- nghiệm đơn vị
- mẫu nhỏ
- R² thấp/âm

## 6. 🎯 Nói như người

Viết 3–5 câu, không dùng thuật ngữ
nặng nếu không cần.

Phải trả lời:

"Nói đơn giản thì nghiên cứu này đang nói
cổ phiếu bị ảnh hưởng bởi cái gì nhiều nhất?"

==================================================
DỮ LIỆU NGHIÊN CỨU
==================================================

{research_context}
""".strip()


# ============================================================
# GEMINI
# ============================================================

def _call_gemini(
    prompt,
):
    try:

        from google import genai

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "model": None,
            "error": (
                "Thiếu package google-genai: "
                f"{error}"
            ),
        }

    api_key = get_gemini_api_key()

    if not api_key:

        return {
            "ok": False,
            "text": "",
            "model": None,
            "error": (
                "Chưa cấu hình GEMINI_API_KEY."
            ),
        }

    try:

        client = genai.Client(
            api_key=api_key
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
                "model": GEMINI_MODEL,
                "error": (
                    "Gemini không trả về nội dung."
                ),
            }

        return {
            "ok": True,
            "text": text,
            "model": GEMINI_MODEL,
            "error": "",
        }

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "model": GEMINI_MODEL,
            "error": str(
                error
            ),
        }


# ============================================================
# CACHED GENERATION
# ============================================================

@st.cache_data(
    ttl=AI_CACHE_TTL,
    show_spinner=False,
)
def generate_research_ai_cached(
    research_context,
    user_question,
):
    prompt = build_research_ai_prompt(
        research_context,
        user_question,
    )

    return _call_gemini(
        prompt
    )


# ============================================================
# RENDER AI
# ============================================================

def render_research_ai(
    result,
    symbol="",
    start_date=None,
    end_date=None,
):
    """
    UI AI đọc kết quả nghiên cứu.

    Chỉ gọi API khi user bấm nút.
    """

    st.divider()

    st.header(
        "🧠 BRIAN AI — Đọc nghiên cứu"
    )

    st.caption(
        "AI đọc kết quả định lượng đã chạy và giải thích "
        "yếu tố nào đang liên hệ mạnh nhất với lợi suất."
    )

    if (
        not isinstance(
            result,
            dict,
        )
        or not result.get(
            "ok",
            False,
        )
    ):

        st.info(
            "Chưa có kết quả nghiên cứu để AI đọc."
        )

        return

    # ========================================================
    # OPTIONAL QUESTION
    # ========================================================

    question = st.text_area(
        "Câu hỏi cho AI",
        placeholder=(
            "Ví dụ:\n"
            "Biến nào ảnh hưởng giá mạnh nhất?\n"
            "Tại sao kết luận này đáng tin hay không?\n"
            "1D, 5D hay 20D yếu tố nào quan trọng nhất?\n"
            "Nói đơn giản nghiên cứu này đang chỉ ra điều gì?"
        ),
        height=110,
        key="research_ai_question",
    ).strip()

    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.markdown(
        "#### Hỏi nhanh"
    )

    q1, q2, q3 = st.columns(
        3
    )

    with q1:

        if st.button(
            "🎯 Yếu tố ảnh hưởng mạnh nhất",
            key="research_ai_quick_factor",
            width="stretch",
        ):

            question = (
                "Yếu tố nào ảnh hưởng mạnh nhất "
                "đến lợi suất cổ phiếu trong nghiên cứu này?"
            )

    with q2:

        if st.button(
            "📈 Vì sao giá tăng/giảm?",
            key="research_ai_quick_direction",
            width="stretch",
        ):

            question = (
                "Nghiên cứu cho thấy lợi suất tương lai "
                "tăng hoặc giảm phụ thuộc vào những yếu tố nào "
                "và theo chiều nào?"
            )

    with q3:

        if st.button(
            "🔬 Đánh giá độ tin cậy",
            key="research_ai_quick_quality",
            width="stretch",
        ):

            question = (
                "Nghiên cứu này đáng tin đến mức nào? "
                "Có vấn đề gì về thống kê, mô hình hoặc dữ liệu?"
            )

    # ========================================================
    # RUN
    # ========================================================

    if st.button(
        "🤖 AI đọc toàn bộ nghiên cứu",
        type="primary",
        width="stretch",
        key="research_ai_run",
    ):

        context = build_research_ai_context(
            result=result,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        with st.spinner(
            "Brian AI đang đọc kết quả nghiên cứu..."
        ):

            ai_result = (
                generate_research_ai_cached(
                    context,
                    question,
                )
            )

        st.session_state[
            "research_ai_result"
        ] = ai_result

        st.session_state[
            "research_ai_question_used"
        ] = question

    # ========================================================
    # RESULT
    # ========================================================

    ai_result = st.session_state.get(
        "research_ai_result"
    )

    if not ai_result:

        return

    if not ai_result.get(
        "ok",
        False,
    ):

        st.error(
            "AI chưa thể đọc nghiên cứu."
        )

        st.code(
            ai_result.get(
                "error",
                "Không xác định.",
            )
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    model = ai_result.get(
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
        "research_ai_question_used",
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
    # OUTPUT
    # ========================================================

    text = _text(
        ai_result.get(
            "text"
        ),
        "",
    )

    if text:

        with st.container(
            border=True
        ):

            st.markdown(
                text
            )
