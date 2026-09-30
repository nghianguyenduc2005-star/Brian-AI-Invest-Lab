from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


AI_CACHE_TTL = 1800
DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-2.5-flash"


# ============================================================
# BASIC HELPERS
# ============================================================

def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        text = str(value).strip()
    except Exception:
        return default
    return text if text else default


def _num(value, default=np.nan):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _safe_round(value, digits=6):
    value = _num(value)
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def get_gemini_api_key() -> str | None:
    try:
        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        key = ""
    return key or None


# ============================================================
# DATA -> COMPACT RESEARCH CONTEXT
# ============================================================

def _df_text(df: pd.DataFrame | None, columns=None, limit=40) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "Không có dữ liệu."

    work = df.copy()
    if columns:
        cols = [c for c in columns if c in work.columns]
        if cols:
            work = work[cols]

    work = work.head(limit).copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce").round(6)

    return work.to_string(index=False)


def _describe_research_data(data: pd.DataFrame, max_columns=90) -> str:
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return "Không có dữ liệu gốc."

    numeric = [
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and not str(c).startswith("Target_")
    ]
    numeric = numeric[:max_columns]

    rows = []
    last_row = data.iloc[-1]
    for c in numeric:
        s = pd.to_numeric(data[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "Biến": c,
            "N": int(s.size),
            "Mean": _safe_round(s.mean()),
            "Std": _safe_round(s.std()),
            "Min": _safe_round(s.min()),
            "Max": _safe_round(s.max()),
            "Mới nhất": _safe_round(last_row.get(c)),
        })

    if not rows:
        return "Không có thống kê mô tả số."

    return pd.DataFrame(rows).to_string(index=False)


def _format_tests(tests: dict[str, Any] | None) -> str:
    if not isinstance(tests, dict) or not tests:
        return "Không có kiểm định."

    lines = []
    for name, item in tests.items():
        if not isinstance(item, dict):
            continue
        pieces = []
        if "statistic" in item:
            pieces.append(f"stat={_safe_round(item.get('statistic'), 5)}")
        if "p_value" in item:
            pieces.append(f"p={_safe_round(item.get('p_value'), 6)}")
        if "lag" in item:
            pieces.append(f"lag={item.get('lag')}")
        if pieces:
            lines.append(f"{name}: " + ", ".join(pieces))
    return "\n".join(lines) if lines else "Không có kiểm định."


def _consistency_table(horizons: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for horizon, item in (horizons or {}).items():
        ranking = item.get("ranking") if isinstance(item, dict) else None
        if not isinstance(ranking, pd.DataFrame) or ranking.empty:
            continue
        for rank, (_, row) in enumerate(ranking.iterrows(), start=1):
            rows.append({
                "Horizon": horizon,
                "Biến": row.get("Biến"),
                "Nhóm": row.get("Nhóm"),
                "Rank": rank,
                "Score": _num(row.get("Score")),
                "Quan hệ": row.get("Quan hệ"),
            })

    if not rows:
        return pd.DataFrame()

    work = pd.DataFrame(rows)
    stable = (
        work.groupby(["Biến", "Nhóm"], as_index=False)
        .agg(
            So_horizon=("Horizon", "nunique"),
            Rank_TB=("Rank", "mean"),
            Score_TB=("Score", "mean"),
        )
        .sort_values(
            ["So_horizon", "Score_TB", "Rank_TB"],
            ascending=[False, False, True],
        )
    )
    stable["Rank_TB"] = stable["Rank_TB"].round(2)
    stable["Score_TB"] = stable["Score_TB"].round(2)
    return stable.head(40)


def build_research_ai_context(result, symbol="", start_date=None, end_date=None) -> str:
    """Đóng gói toàn bộ kết quả nghiên cứu đã tính để Gemini đọc.

    Không gửi hàng nghìn dòng OHLCV thô. Thay vào đó gửi:
    - toàn bộ universe biến;
    - thống kê mô tả toàn bộ biến số;
    - toàn bộ kết quả quan trọng của từng horizon;
    - xếp hạng yếu tố;
    - OLS / ML / VIF / kiểm định / permutation / forecast;
    - tính nhất quán giữa 1D / 5D / 20D.
    """
    if not isinstance(result, dict) or not result.get("ok", False):
        return "Không có kết quả nghiên cứu hợp lệ."

    data = result.get("data")
    horizons = result.get("horizons", {})
    all_features = result.get("all_features", [])
    groups = result.get("groups", {})

    blocks: list[str] = []
    blocks.append("=== BRIAN STOCK — NGHIÊN CỨU ĐỊNH LƯỢNG ===")
    blocks.append(f"Mã: {_text(symbol, 'Không xác định')}")
    blocks.append(f"Từ: {_text(start_date)}")
    blocks.append(f"Đến: {_text(end_date)}")
    blocks.append(f"Số quan sát thực tế: {len(data) if isinstance(data, pd.DataFrame) else 0}")
    blocks.append(f"Tổng universe biến: {len(all_features)}")
    blocks.append("Nhóm biến: " + "; ".join(f"{k}={len(v)}" for k, v in (groups or {}).items()))

    blocks.append("\n=== THỐNG KÊ MÔ TẢ TOÀN BỘ BIẾN ===")
    blocks.append(_describe_research_data(data))

    if all_features:
        blocks.append("\n=== TOÀN BỘ TÊN BIẾN ===")
        blocks.append(", ".join(map(str, all_features)))

    consistency = _consistency_table(horizons)
    blocks.append("\n=== YẾU TỐ NHẤT QUÁN QUA 1D / 5D / 20D ===")
    blocks.append(_df_text(consistency, limit=40))

    for horizon in ("1D", "5D", "20D"):
        item = horizons.get(horizon)
        if not isinstance(item, dict):
            continue

        blocks.append("\n" + "=" * 78)
        blocks.append(f"HORIZON {horizon}")
        blocks.append("=" * 78)
        blocks.append(f"Quan sát={item.get('observations', 0)} | Train={item.get('train', 0)} | Test={item.get('test', 0)}")
        blocks.append(f"Model tốt nhất hiện tại={item.get('best_model', 'Không xác định')}")

        forecast = item.get("forecast", {})
        blocks.append(
            "Dự báo model: "
            f"return={_safe_round(forecast.get('predicted_return'), 6)}, "
            f"price={_safe_round(forecast.get('predicted_price'), 2)}, "
            f"current={_safe_round(forecast.get('current_price'), 2)}"
        )

        ranking = item.get("ranking")
        blocks.append("\n--- XẾP HẠNG YẾU TỐ ---")
        blocks.append(
            _df_text(
                ranking,
                columns=[
                    "Biến", "Nhóm", "Score", "Quan hệ",
                    "Pearson", "Spearman", "Beta", "p-value",
                    "Permutation", "TreeImportance", "Ý nghĩa",
                ],
                limit=80,
            )
        )

        models = item.get("models")
        blocks.append("\n--- SO SÁNH MÔ HÌNH ---")
        blocks.append(_df_text(models, limit=20))

        ols = item.get("ols")
        blocks.append("\n--- OLS ---")
        if isinstance(ols, dict):
            blocks.append(
                f"R²={_safe_round(ols.get('r2'))}; "
                f"Adjusted R²={_safe_round(ols.get('adj_r2'))}; "
                f"Features={len(ols.get('features', []))}"
            )
        else:
            blocks.append("OLS không có kết quả.")

        blocks.append("--- BẢNG HỆ SỐ OLS ---")
        blocks.append(_df_text(item.get("ols_table"), limit=25))

        blocks.append("\n--- VIF ---")
        blocks.append(_df_text(item.get("vif"), limit=20))

        blocks.append("\n--- PERMUTATION IMPORTANCE ---")
        blocks.append(_df_text(item.get("permutation"), limit=35))

        blocks.append("\n--- KIỂM ĐỊNH ---")
        blocks.append(_format_tests(item.get("tests", {})))

        quality = item.get("quality")
        if quality is not None:
            blocks.append("\n--- QUALITY FLAGS ---")
            blocks.append(str(quality))

    return "\n".join(blocks)


# ============================================================
# AI SCHEMA / PROMPT
# ============================================================

RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "array", "items": {"type": "string"}},
        "key_drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "group": {"type": "string"},
                    "direction": {"type": "string"},
                    "evidence": {"type": "string"},
                    "interpretation": {"type": "string"},
                },
                "required": ["feature", "group", "direction", "evidence", "interpretation"],
            },
        },
        "horizon_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "stable_factors": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["horizon", "conclusion", "stable_factors"],
            },
        },
        "model_assessment": {
            "type": "object",
            "properties": {
                "best_model": {"type": "string"},
                "rmse": {"type": "string"},
                "r2": {"type": "string"},
                "benchmark": {"type": "string"},
                "assessment": {"type": "string"},
            },
            "required": ["best_model", "rmse", "r2", "benchmark", "assessment"],
        },
        "statistical_quality": {"type": "array", "items": {"type": "string"}},
        "plain_language": {"type": "string"},
        "important_cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "executive_summary",
        "key_drivers",
        "horizon_analysis",
        "model_assessment",
        "statistical_quality",
        "plain_language",
        "important_cautions",
    ],
}


def build_research_ai_prompt(research_context: str, user_question: str = "") -> str:
    question = _text(user_question, "")
    question_block = (
        f"CÂU HỎI ƯU TIÊN CỦA NGƯỜI DÙNG:\n{question}"
        if question
        else "Không có câu hỏi riêng. Hãy tự tổng hợp toàn bộ nghiên cứu."
    )

    return f"""
Bạn là BRIAN AI — chuyên viên phân tích nghiên cứu định lượng chứng khoán.

MỤC TIÊU
- Đọc TOÀN BỘ kết quả nghiên cứu được cung cấp.
- Viết kết luận chuyên nghiệp nhưng người mới cũng hiểu được.
- Không chạy lại mô hình và không tự bịa dữ liệu.
- Mọi con số phải xuất phát từ dữ liệu nghiên cứu.

{question_block}

NGUYÊN TẮC PHÂN TÍCH
1. Phân biệt rõ tương quan, ý nghĩa thống kê và khả năng dự báo.
2. Không dùng ngôn ngữ nhân quả như "gây ra" nếu nghiên cứu chỉ cho thấy quan hệ thống kê.
3. Không chọn yếu tố chỉ vì một Score cao; xem đồng thời hướng quan hệ, Pearson/Spearman, Beta/p-value, permutation/tree importance và tính nhất quán qua các horizon.
4. Nếu p-value > 0.05, nói rõ bằng chứng chưa mạnh ở mức 5%.
5. Nếu R² test âm, nói rõ mô hình không vượt benchmark trên test.
6. Nếu VIF cao, cảnh báo rằng hệ số OLS khó diễn giải riêng lẻ.
7. Nếu Breusch-Pagan/White có p < 0.05, nêu dấu hiệu phương sai thay đổi.
8. Nếu Durbin-Watson/Ljung-Box cho thấy tự tương quan, nêu rõ.
9. Không biến dự báo mô hình thành cam kết giá tương lai.
10. Không đưa khuyến nghị MUA/BÁN cá nhân hóa.
11. Khi dữ liệu không đủ để kết luận, phải nói "chưa đủ bằng chứng".
12. Giải thích thuật ngữ lần đầu xuất hiện bằng tiếng Việt đơn giản.

YÊU CẦU ĐẦU RA
- Tóm tắt điều hành 3–6 ý.
- Tối đa 5 yếu tố nổi bật, giải thích bằng số liệu.
- Phân tích riêng 1D / 5D / 20D và yếu tố nào nhất quán.
- Đánh giá model bằng RMSE, R² test và benchmark nếu có.
- Nêu các vấn đề thống kê/dữ liệu thật sự xuất hiện.
- Kết thúc bằng một đoạn "Nói đơn giản" cho người mới.
- Không dùng markdown heading dài dòng; dữ liệu phải nằm trong JSON schema.

DỮ LIỆU NGHIÊN CỨU:
{research_context}
""".strip()


# ============================================================
# GEMINI CALL
# ============================================================


def _call_gemini(prompt: str):
    try:
        from google import genai
        from google.genai import types
    except Exception as error:
        return {"ok": False, "text": "", "json": None, "model": None, "error": f"Thiếu google-genai: {error}"}

    api_key = get_gemini_api_key()
    if not api_key:
        return {"ok": False, "text": "", "json": None, "model": None, "error": "Chưa cấu hình GEMINI_API_KEY."}

    requested_model = _text(st.session_state.get("ai_model"), DEFAULT_MODEL)
    models_to_try = [requested_model]
    if FALLBACK_MODEL not in models_to_try:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None
    for model_name in models_to_try:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=RESEARCH_SCHEMA,
                    max_output_tokens=5000,
                ),
            )
            text = _text(getattr(response, "text", ""), "")
            if not text:
                last_error = "Gemini không trả về nội dung."
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as error:
                last_error = f"JSON AI không hợp lệ: {error}"
                continue
            return {"ok": True, "text": text, "json": parsed, "model": model_name, "error": ""}
        except Exception as error:
            last_error = str(error)

    return {"ok": False, "text": "", "json": None, "model": requested_model, "error": last_error or "AI lỗi không xác định."}


@st.cache_data(ttl=AI_CACHE_TTL, show_spinner=False)
def generate_research_ai_cached(research_context: str, user_question: str):
    return _call_gemini(build_research_ai_prompt(research_context, user_question))


# ============================================================
# RENDER
# ============================================================


def _render_json_result(ai_result: dict[str, Any]):
    data = ai_result.get("json") or {}

    st.subheader("🧾 Kết luận điều hành")
    for item in data.get("executive_summary", []):
        st.markdown(f"• {item}")

    drivers = data.get("key_drivers", [])
    if drivers:
        st.subheader("🎯 Các yếu tố nổi bật")
        st.dataframe(pd.DataFrame(drivers), width="stretch", hide_index=True)

    horizons = data.get("horizon_analysis", [])
    if horizons:
        st.subheader("📈 Phân tích theo thời gian dự báo")
        for item in horizons:
            with st.expander(str(item.get("horizon", "Horizon")), expanded=True):
                st.write(item.get("conclusion", ""))
                factors = item.get("stable_factors", [])
                if factors:
                    st.markdown("**Yếu tố chính:** " + ", ".join(map(str, factors)))

    assessment = data.get("model_assessment", {})
    if assessment:
        st.subheader("🤖 Đánh giá mô hình")
        a, b, c, d = st.columns(4)
        with a:
            st.metric("Model", assessment.get("best_model", "—"))
        with b:
            st.metric("RMSE", assessment.get("rmse", "—"))
        with c:
            st.metric("R² test", assessment.get("r2", "—"))
        with d:
            st.metric("Benchmark", assessment.get("benchmark", "—"))
        st.write(assessment.get("assessment", ""))

    quality = data.get("statistical_quality", [])
    if quality:
        st.subheader("🔬 Chất lượng nghiên cứu")
        for item in quality:
            st.markdown(f"• {item}")

    plain = _text(data.get("plain_language"), "")
    if plain:
        st.subheader("🗣️ Nói đơn giản cho người mới")
        st.info(plain)

    cautions = data.get("important_cautions", [])
    if cautions:
        st.subheader("⚠️ Lưu ý")
        for item in cautions:
            st.markdown(f"• {item}")


def render_research_ai(result, symbol="", start_date=None, end_date=None):
    """AI đọc toàn bộ kết quả nghiên cứu định lượng sau khi model đã chạy."""
    st.divider()
    st.header("🧠 BRIAN AI — Đọc toàn bộ nghiên cứu")
    st.caption(
        "AI đọc dữ liệu thống kê, các yếu tố, OLS, Machine Learning, VIF, kiểm định và kết quả 1D/5D/20D; "
        "sau đó giải thích lại theo ngôn ngữ dễ hiểu."
    )

    if not isinstance(result, dict) or not result.get("ok", False):
        st.info("Chưa có kết quả nghiên cứu để AI đọc.")
        return

    question = st.text_area(
        "Đặt câu hỏi cho AI",
        placeholder=(
            "Ví dụ: Vì sao model có R² thấp?\n"
            "Yếu tố nào nhất quán nhất 1D/5D/20D?\n"
            "Nói toàn bộ nghiên cứu này cho người mới hiểu."
        ),
        height=100,
        key="research_ai_question",
    ).strip()

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("🎯 Yếu tố cốt lõi", key="research_ai_quick_factor", width="stretch"):
            question = "Yếu tố nào có bằng chứng đáng chú ý nhất khi xét đồng thời tương quan, p-value, beta, importance và sự nhất quán qua các horizon?"
    with q2:
        if st.button("📚 Giải thích toàn bộ", key="research_ai_quick_full", width="stretch"):
            question = "Hãy giải thích toàn bộ nghiên cứu từ đầu đến cuối cho một người mới, nhưng vẫn giữ chính xác số liệu và thuật ngữ cần thiết."
    with q3:
        if st.button("🤖 Đánh giá model", key="research_ai_quick_model", width="stretch"):
            question = "Đánh giá khả năng dự báo của các model bằng RMSE, R² test và benchmark; giải thích vì sao model đạt hoặc chưa đạt."
    with q4:
        if st.button("🔬 Kiểm tra độ tin cậy", key="research_ai_quick_quality", width="stretch"):
            question = "Kiểm tra toàn bộ vấn đề thống kê và dữ liệu có thể làm kết luận yếu đi: VIF, p-value, heteroskedasticity, autocorrelation, stationarity, mẫu và leakage nếu có."

    if st.button("🤖 AI đọc toàn bộ nghiên cứu", type="primary", width="stretch", key="research_ai_run"):
        context = build_research_ai_context(result, symbol, start_date, end_date)
        with st.spinner("Brian AI đang đọc toàn bộ nghiên cứu..."):
            ai_result = generate_research_ai_cached(context, question)
        st.session_state["research_ai_result"] = ai_result
        st.session_state["research_ai_question_used"] = question

    ai_result = st.session_state.get("research_ai_result")
    if not ai_result:
        return

    if not ai_result.get("ok", False):
        st.error("AI chưa thể đọc nghiên cứu.")
        st.code(ai_result.get("error", "Không xác định."))
        return

    model_name = ai_result.get("model")
    if model_name:
        st.caption(f"Model: {model_name}")

    q_used = st.session_state.get("research_ai_question_used", "")
    if q_used:
        with st.container(border=True):
            st.caption("Câu hỏi đã dùng")
            st.write(q_used)

    _render_json_result(ai_result)
