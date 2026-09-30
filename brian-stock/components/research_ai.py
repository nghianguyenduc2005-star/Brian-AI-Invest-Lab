from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

AI_CACHE_TTL = 1800
DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.7-flash"]
SUPPORTED_RESEARCH_MODEL = "gemini-3.5-flash"

DEFAULT_AUTO_QUESTION = (
    "Hãy đọc TOÀN BỘ nghiên cứu của cổ phiếu và giải thích như đang hướng dẫn một người mới. "
    "Trước tiên nói nghiên cứu đang hỏi câu gì. Sau đó nói nó tìm thấy gì, điều gì đáng tin, "
    "điều gì chưa đủ bằng chứng, và model có thực sự dự báo tốt trên tập test hay không. "
    "Mọi kết luận phải bám số liệu. Với mỗi thuật ngữ kỹ thuật quan trọng, giải thích ngay bằng "
    "một câu tiếng Việt đơn giản. Luôn phân biệt rõ: có liên hệ, có ý nghĩa thống kê, và dự báo tốt. "
    "Không dùng ngôn ngữ nhân quả nếu nghiên cứu không chứng minh nhân quả."
)


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


def _num(value: Any, default: float | None = None):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _safe_round(value: Any, digits: int = 6):
    value = _num(value)
    if value is None:
        return None
    return round(float(value), digits)


def _fmt_pct(value: Any, digits: int = 2) -> str:
    value = _num(value)
    if value is None:
        return "—"
    return f"{value * 100:+.{digits}f}%"


def _fmt_num(value: Any, digits: int = 4) -> str:
    value = _num(value)
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def get_gemini_api_key() -> str | None:
    """Ưu tiên Streamlit Secrets, fallback sang environment variable."""
    key = ""
    try:
        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass
    if not key:
        key = str(os.getenv("GEMINI_API_KEY", "")).strip()
    return key or None


def _parse_json_text(text: str):
    text = _text(text)
    if not text:
        raise ValueError("Gemini không trả về nội dung.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


# ============================================================
# DATA -> COMPACT RESEARCH CONTEXT
# ============================================================


def _df_text(df: pd.DataFrame | None, columns=None, limit: int = 40) -> str:
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


def _describe_research_data(data: pd.DataFrame, max_columns: int = 120) -> str:
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return "Không có dữ liệu gốc."

    numeric = [
        c
        for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and not str(c).lower().startswith("target_")
    ]
    numeric = numeric[:max_columns]

    rows = []
    last_row = data.iloc[-1]
    for col in numeric:
        s = pd.to_numeric(data[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "Biến": col,
                "N": int(s.size),
                "Mean": _safe_round(s.mean()),
                "Std": _safe_round(s.std()),
                "Min": _safe_round(s.min()),
                "Max": _safe_round(s.max()),
                "Mới nhất": _safe_round(last_row.get(col)),
            }
        )

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
        for rank, (_, row) in enumerate(ranking.head(30).iterrows(), start=1):
            rows.append(
                {
                    "Horizon": horizon,
                    "Biến": row.get("Biến"),
                    "Nhóm": row.get("Nhóm"),
                    "Rank": rank,
                    "Score": _num(row.get("Score")),
                    "Quan hệ": row.get("Quan hệ"),
                }
            )

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


def _model_table_for_context(models: pd.DataFrame | None) -> str:
    if not isinstance(models, pd.DataFrame) or models.empty:
        return "Không có kết quả model."

    cols = [
        c
        for c in ["Mô hình", "MAE", "RMSE", "R²", "Baseline_RMSE", "So_baseline"]
        if c in models.columns
    ]
    work = models[cols].copy()
    for c in ["MAE", "RMSE", "R²", "Baseline_RMSE"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce").round(8)
    return work.to_string(index=False)


def _horizon_headline(item: dict[str, Any]) -> str:
    horizon = _text(item.get("horizon"), "Horizon")
    models = item.get("models")
    best_name = _text(item.get("best_model"), "Không xác định")
    best_rmse = None
    best_r2 = None
    baseline = None

    if isinstance(models, pd.DataFrame) and not models.empty:
        row = models.iloc[0]
        best_rmse = _num(row.get("RMSE"))
        best_r2 = _num(row.get("R²"))
        baseline = _num(row.get("Baseline_RMSE"))

    if best_r2 is not None:
        if best_r2 < 0:
            model_result = "R² test âm: model chưa vượt benchmark về khả năng dự báo ngoài mẫu."
        else:
            model_result = "R² test không âm: cần xem thêm RMSE và so sánh với baseline."
    else:
        model_result = "Chưa đủ R² test để đánh giá."

    rmse_text = _fmt_num(best_rmse, 6)
    base_text = _fmt_num(baseline, 6)
    return (
        f"{horizon}: model tốt nhất trong nhóm thử là {best_name}; "
        f"RMSE={rmse_text}, baseline RMSE={base_text}, R² test={_fmt_num(best_r2, 6)}. "
        f"{model_result}"
    )


def build_research_ai_context(result, symbol="", start_date=None, end_date=None) -> str:
    """Đóng gói toàn bộ kết quả nghiên cứu để AI đọc và giải thích."""
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
    blocks.append(
        f"Số quan sát thực tế: {len(data) if isinstance(data, pd.DataFrame) else 0}"
    )
    blocks.append(f"Tổng universe biến: {len(all_features)}")
    blocks.append(
        "Nhóm biến: "
        + "; ".join(f"{k}={len(v)}" for k, v in (groups or {}).items())
    )

    blocks.append("\n=== CÁCH HIỂU NGHIÊN CỨU ===")
    blocks.append(
        "Mục tiêu: dùng các biến đầu vào để xem chúng có liên hệ với lợi suất tương lai "
        "của cổ phiếu ở 1D, 5D và 20D hay không, và kiểm tra xem model có dự báo tốt trên tập test hay không."
    )
    blocks.append(
        "Quy tắc đọc model: RMSE thấp hơn là sai số nhỏ hơn trên cùng horizon; R² test âm "
        "có nghĩa model kém hơn benchmark theo thước đo R²; luôn so sánh model với Baseline_RMSE."
    )

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
        blocks.append(
            f"Quan sát={item.get('observations', 0)} | "
            f"Train={item.get('train', 0)} | Test={item.get('test', 0)}"
        )
        blocks.append(_horizon_headline(item))

        forecast = item.get("forecast", {})
        blocks.append(
            "Forecast model: "
            f"current_price={_safe_round(forecast.get('current_price'), 2)}, "
            f"predicted_return={_safe_round(forecast.get('predicted_return'), 6)}, "
            f"predicted_price={_safe_round(forecast.get('predicted_price'), 2)}"
        )

        ranking = item.get("ranking")
        blocks.append("\n--- XẾP HẠNG YẾU TỐ ---")
        blocks.append(
            _df_text(
                ranking,
                columns=[
                    "Biến", "Nhóm", "Score", "Quan hệ", "Pearson", "Spearman",
                    "Beta", "p-value", "Permutation", "TreeImportance", "Ý nghĩa",
                ],
                limit=100,
            )
        )

        blocks.append("\n--- SO SÁNH MÔ HÌNH ---")
        blocks.append(_model_table_for_context(item.get("models")))

        ols = item.get("ols")
        blocks.append("\n--- OLS ---")
        if isinstance(ols, dict):
            blocks.append(
                f"R² train={_safe_round(ols.get('r2'))}; "
                f"Adjusted R² train={_safe_round(ols.get('adj_r2'))}; "
                f"Features={len(ols.get('features', []))}"
            )
            blocks.append("Lưu ý: đây là R² của OLS trên train, không phải R² test của ML.")
        else:
            blocks.append("OLS không có kết quả.")

        blocks.append("--- BẢNG HỆ SỐ OLS ---")
        blocks.append(_df_text(item.get("ols_table"), limit=30))

        blocks.append("\n--- VIF ---")
        blocks.append(_df_text(item.get("vif"), limit=20))

        blocks.append("\n--- PERMUTATION IMPORTANCE ---")
        blocks.append(_df_text(item.get("permutation"), limit=40))

        blocks.append("\n--- TREE IMPORTANCE ---")
        blocks.append(_df_text(item.get("tree"), limit=40))

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
        "study_objective": {"type": "string"},
        "study_setup": {"type": "string"},
        "main_takeaway": {"type": "string"},
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
                    "simple_explanation": {"type": "string"},
                    "what_it_does_not_mean": {"type": "string"},
                },
                "required": [
                    "feature", "group", "direction", "evidence",
                    "simple_explanation", "what_it_does_not_mean",
                ],
            },
        },
        "horizon_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "string"},
                    "what_was_tested": {"type": "string"},
                    "model_result": {"type": "string"},
                    "beginner_translation": {"type": "string"},
                    "stable_factors": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "horizon", "what_was_tested", "model_result",
                    "beginner_translation", "stable_factors",
                ],
            },
        },
        "model_assessment": {
            "type": "object",
            "properties": {
                "best_model": {"type": "string"},
                "best_model_is_actually_good": {"type": "string"},
                "rmse_comparison": {"type": "string"},
                "r2_test_explanation": {"type": "string"},
                "benchmark_explanation": {"type": "string"},
                "assessment": {"type": "string"},
            },
            "required": [
                "best_model", "best_model_is_actually_good", "rmse_comparison",
                "r2_test_explanation", "benchmark_explanation", "assessment",
            ],
        },
        "statistical_quality": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test": {"type": "string"},
                    "result": {"type": "string"},
                    "simple_explanation": {"type": "string"},
                    "impact_on_reading": {"type": "string"},
                },
                "required": ["test", "result", "simple_explanation", "impact_on_reading"],
            },
        },
        "plain_language": {"type": "string"},
        "important_cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "study_objective",
        "study_setup",
        "main_takeaway",
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
    question = _text(user_question, DEFAULT_AUTO_QUESTION)

    return f"""
Bạn là BRIAN AI — chuyên viên phân tích nghiên cứu định lượng chứng khoán.

MỤC TIÊU CỐT LÕI
Bạn không chỉ tóm tắt báo cáo. Bạn phải 'dịch' nghiên cứu thành cách hiểu mà một nhà đầu tư mới có thể nắm được.

CÂU HỎI PHÂN TÍCH
{question}

NGUYÊN TẮC BẮT BUỘC
1. Chỉ sử dụng số liệu thật có trong dữ liệu được cung cấp. Không tự bịa số.
2. Luôn trả lời theo thứ tự:
   (a) nghiên cứu đang hỏi gì;
   (b) dữ liệu đang làm gì;
   (c) tìm thấy gì;
   (d) model có dự báo tốt ngoài mẫu hay không;
   (e) người mới nên hiểu kết quả như thế nào.
3. Phải phân biệt rất rõ 3 khái niệm:
   - 'Có liên hệ': biến có tương quan/importance với target trong mẫu.
   - 'Có ý nghĩa thống kê': bằng chứng OLS/p-value đủ mạnh theo ngưỡng đang dùng.
   - 'Dự báo tốt': model phải thể hiện tốt trên tập test và phải so với baseline.
   Một biến có liên hệ không đồng nghĩa model dự báo tốt.
4. Khi nói 'model tốt nhất', phải thêm caveat 'tốt nhất trong các model được thử' nếu model vẫn thua baseline.
5. R² test âm phải được giải thích thẳng: model dự báo kém hơn benchmark theo thước đo R²; không được hiểu là giá cổ phiếu sẽ giảm.
6. RMSE chỉ so sánh trực tiếp giữa các model trên cùng horizon. Không nói 1D RMSE thấp hơn 20D là vì 1D dễ hơn nếu chưa có bằng chứng.
7. Nếu OLS R² cao nhưng ML test yếu, phải nói rõ OLS R² đang là train/in-sample và không dùng nó để quảng bá khả năng dự báo ngoài mẫu.
8. VIF cao: giải thích rằng nhiều biến cung cấp thông tin giống nhau, nên hệ số OLS riêng lẻ có thể khó diễn giải.
9. ADF chỉ nói về tính dừng; không được biến thành kết luận rằng model dự báo tốt.
10. Breusch-Pagan: nếu p nhỏ, giải thích là dấu hiệu phương sai phần dư không đồng đều.
11. Durbin-Watson/Ljung-Box: nếu có tín hiệu, giải thích là phần dư còn có cấu trúc theo thời gian.
12. Không dùng từ 'gây ra' nếu nghiên cứu không chứng minh nhân quả.
13. Không biến forecast thành cam kết giá tương lai.
14. Nếu dữ liệu không đủ, nói rõ 'chưa đủ bằng chứng'.
15. Với từng yếu tố, phải có một câu 'nó có nghĩa gì' và một câu 'nó KHÔNG có nghĩa là gì'.
16. Không viết kiểu giáo trình. Dùng ví dụ số liệu ngay trong câu khi có thể.
17. Nếu có một kết luận quan trọng, ưu tiên nói bằng tiếng Việt đời thường trước rồi mới thêm thuật ngữ trong ngoặc.

YÊU CẦU ĐẦU RA
- study_objective: 1 đoạn rất dễ hiểu, mô tả câu hỏi nghiên cứu.
- study_setup: giải thích 773 quan sát, 161 biến, train/test và 1D/5D/20D nếu dữ liệu cho phép.
- main_takeaway: 1 đoạn kết luận lớn nhất, tối đa 5 câu.
- executive_summary: 3–6 ý, ưu tiên ý nghĩa hơn thuật ngữ.
- key_drivers: tối đa 5 yếu tố. Với mỗi yếu tố phải nói bằng chứng số liệu, giải thích đơn giản, và nói rõ nó KHÔNG có nghĩa là gì.
- horizon_analysis: 1D, 5D, 20D. Mỗi horizon phải có: đã test cái gì, model ra sao, dịch sang ngôn ngữ người mới, yếu tố nhất quán.
- model_assessment: phải trả lời trực tiếp 'model tốt nhất có thực sự tốt không?'.
- statistical_quality: chỉ nêu các kiểm định thực sự có dữ liệu, kèm ý nghĩa đơn giản và tác động tới việc đọc kết quả.
- plain_language: một đoạn đủ dài để người mới đọc xong có thể tự nói lại nghiên cứu bằng lời của mình.
- important_cautions: những điều không nên suy diễn.

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
        return {
            "ok": False,
            "text": "",
            "json": None,
            "model": None,
            "error": f"Thiếu google-genai: {error}",
        }

    api_key = get_gemini_api_key()
    if not api_key:
        return {
            "ok": False,
            "text": "",
            "json": None,
            "model": None,
            "error": "Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets.",
        }

    requested_model = SUPPORTED_RESEARCH_MODEL
    if st.session_state.get("ai_model") != SUPPORTED_RESEARCH_MODEL:
        st.session_state["ai_model"] = SUPPORTED_RESEARCH_MODEL

    models_to_try = [requested_model]
    for model_name in FALLBACK_MODELS:
        if model_name not in models_to_try:
            models_to_try.append(model_name)

    try:
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 45000},
        )
    except Exception as error:
        return {
            "ok": False,
            "text": "",
            "json": None,
            "model": requested_model,
            "error": f"Không khởi tạo được Gemini client: {error}",
        }

    last_error = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESEARCH_SCHEMA,
                    max_output_tokens=6000,
                ),
            )

            text = _text(getattr(response, "text", ""), "")
            if not text:
                last_error = f"{model_name}: Gemini không trả về nội dung."
                continue

            try:
                parsed = _parse_json_text(text)
            except Exception as error:
                last_error = f"{model_name}: JSON AI không hợp lệ: {error}"
                continue

            if not isinstance(parsed, dict):
                last_error = f"{model_name}: AI trả về JSON không phải object."
                continue

            return {
                "ok": True,
                "text": text,
                "json": parsed,
                "model": model_name,
                "error": "",
            }
        except Exception as error:
            last_error = f"{model_name}: {error}"

    error_text = last_error or "AI lỗi không xác định."
    if "503" in error_text or "UNAVAILABLE" in error_text or "high demand" in error_text.lower():
        error_text = (
            "Gemini đang tạm thời quá tải (503). "
            "Hệ thống đã thử các model được cấu hình. Phần nghiên cứu vẫn giữ nguyên; "
            "hãy thử lại Brian AI sau ít phút."
        )

    return {
        "ok": False,
        "text": "",
        "json": None,
        "model": requested_model,
        "error": error_text,
    }


@st.cache_data(ttl=AI_CACHE_TTL, show_spinner=False)
def generate_research_ai_cached(research_context: str, user_question: str):
    prompt = build_research_ai_prompt(research_context, user_question)
    return _call_gemini(prompt)


# ============================================================
# RESEARCH FINGERPRINT
# ============================================================


def _research_signature(result, symbol="", start_date=None, end_date=None) -> str:
    payload: dict[str, Any] = {
        "symbol": _text(symbol),
        "start": _text(start_date),
        "end": _text(end_date),
        "all_features": list(result.get("all_features", [])) if isinstance(result, dict) else [],
        "horizons": {},
    }

    if isinstance(result, dict):
        horizons = result.get("horizons", {}) or {}
        for horizon in ("1D", "5D", "20D"):
            item = horizons.get(horizon)
            if not isinstance(item, dict):
                continue

            entry: dict[str, Any] = {
                "observations": item.get("observations"),
                "train": item.get("train"),
                "test": item.get("test"),
                "best_model": item.get("best_model"),
                "quality": item.get("quality"),
                "forecast": item.get("forecast", {}),
                "models": [],
                "top_factors": [],
            }

            models = item.get("models")
            if isinstance(models, pd.DataFrame) and not models.empty:
                for _, row in models.head(10).iterrows():
                    entry["models"].append(
                        {
                            "model": _text(row.get("Mô hình")),
                            "mae": _safe_round(row.get("MAE"), 8),
                            "rmse": _safe_round(row.get("RMSE"), 8),
                            "r2": _safe_round(row.get("R²"), 8),
                            "baseline": _safe_round(row.get("Baseline_RMSE"), 8),
                        }
                    )

            ranking = item.get("ranking")
            if isinstance(ranking, pd.DataFrame) and not ranking.empty:
                cols = [
                    c
                    for c in [
                        "Biến", "Nhóm", "Score", "Quan hệ", "Spearman",
                        "Beta", "p-value", "Permutation", "TreeImportance",
                    ]
                    if c in ranking.columns
                ]
                for _, row in ranking.head(15).iterrows():
                    entry["top_factors"].append(
                        {
                            c: (
                                _safe_round(row.get(c), 8)
                                if c not in {"Biến", "Nhóm", "Quan hệ"}
                                else _text(row.get(c))
                            )
                            for c in cols
                        }
                    )

            payload["horizons"][horizon] = entry

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ============================================================
# RENDER HELPERS
# ============================================================


def _render_json_result(ai_result: dict[str, Any]):
    data = ai_result.get("json") or {}

    study_objective = _text(data.get("study_objective"))
    if study_objective:
        st.subheader("🔎 1. Nghiên cứu này đang hỏi gì?")
        st.info(study_objective)

    study_setup = _text(data.get("study_setup"))
    if study_setup:
        st.subheader("🧪 2. Nghiên cứu được thực hiện như thế nào?")
        st.write(study_setup)

    main_takeaway = _text(data.get("main_takeaway"))
    if main_takeaway:
        st.subheader("🎯 3. Kết luận lớn nhất")
        st.success(main_takeaway)

    summary = data.get("executive_summary", [])
    if summary:
        st.subheader("🧾 4. Tóm tắt điều hành")
        for item in summary:
            st.markdown(f"• {item}")

    drivers = data.get("key_drivers", [])
    if drivers:
        st.subheader("🔍 5. Các yếu tố nổi bật — hiểu như người mới")
        for idx, item in enumerate(drivers[:5], start=1):
            feature = _text(item.get("feature"), "Yếu tố")
            group = _text(item.get("group"), "")
            direction = _text(item.get("direction"), "")
            evidence = _text(item.get("evidence"), "")
            simple = _text(item.get("simple_explanation"), "")
            not_mean = _text(item.get("what_it_does_not_mean"), "")

            with st.container(border=True):
                st.markdown(f"### {idx}. {feature}")
                if group:
                    st.caption(f"Nhóm: {group} · Quan hệ: {direction}")
                if evidence:
                    st.markdown(f"**Bằng chứng từ nghiên cứu:** {evidence}")
                if simple:
                    st.markdown(f"**Nói đơn giản:** {simple}")
                if not_mean:
                    st.markdown(f"**Không có nghĩa là:** {not_mean}")

    horizons = data.get("horizon_analysis", [])
    if horizons:
        st.subheader("📈 6. 1D / 5D / 20D — đọc thế nào?")
        for item in horizons:
            horizon = _text(item.get("horizon"), "Horizon")
            tested = _text(item.get("what_was_tested"), "")
            result = _text(item.get("model_result"), "")
            beginner = _text(item.get("beginner_translation"), "")
            factors = item.get("stable_factors", [])

            with st.expander(horizon, expanded=True):
                if tested:
                    st.markdown(f"**Đã kiểm tra:** {tested}")
                if result:
                    st.markdown(f"**Kết quả model:** {result}")
                if beginner:
                    st.info(f"**Nói đơn giản:** {beginner}")
                if factors:
                    st.markdown(
                        "**Yếu tố chính:** " + ", ".join(map(str, factors))
                    )

    assessment = data.get("model_assessment", {})
    if assessment:
        st.subheader("🤖 7. Model có thực sự tốt không?")
        best = _text(assessment.get("best_model"), "—")
        good = _text(assessment.get("best_model_is_actually_good"), "")
        rmse_cmp = _text(assessment.get("rmse_comparison"), "")
        r2_expl = _text(assessment.get("r2_test_explanation"), "")
        benchmark = _text(assessment.get("benchmark_explanation"), "")
        assess = _text(assessment.get("assessment"), "")

        a, b = st.columns(2)
        with a:
            st.metric("Model tốt nhất trong nhóm thử", best)
        with b:
            st.metric("Có thực sự tốt?", good or "—")

        if rmse_cmp:
            st.markdown(f"**So sánh sai số:** {rmse_cmp}")
        if r2_expl:
            st.markdown(f"**R² test:** {r2_expl}")
        if benchmark:
            st.markdown(f"**Benchmark:** {benchmark}")
        if assess:
            st.info(assess)

    quality = data.get("statistical_quality", [])
    if quality:
        st.subheader("🔬 8. Các kiểm định — thực sự nói gì?")
        for item in quality:
            test = _text(item.get("test"), "Kiểm định")
            result = _text(item.get("result"), "")
            simple = _text(item.get("simple_explanation"), "")
            impact = _text(item.get("impact_on_reading"), "")
            with st.container(border=True):
                st.markdown(f"**{test}**")
                if result:
                    st.markdown(f"Kết quả: {result}")
                if simple:
                    st.markdown(f"**Nói đơn giản:** {simple}")
                if impact:
                    st.markdown(f"**Ảnh hưởng đến cách đọc nghiên cứu:** {impact}")

    plain = _text(data.get("plain_language"), "")
    if plain:
        st.subheader("🗣️ 9. Nói toàn bộ nghiên cứu bằng ngôn ngữ đời thường")
        st.info(plain)

    cautions = data.get("important_cautions", [])
    if cautions:
        st.subheader("⚠️ 10. Những điều không nên suy diễn")
        for item in cautions:
            st.markdown(f"• {item}")


# ============================================================
# MAIN INTEGRATION
# ============================================================


def render_research_ai(result, symbol="", start_date=None, end_date=None):
    """AI an toàn sau nghiên cứu; không tự gọi khi trang render."""
    st.divider()
    st.header("🧠 BRIAN AI — Phân tích toàn bộ nghiên cứu")
    st.caption(
        "AI không chỉ tóm tắt số liệu. AI sẽ giải thích nghiên cứu đang hỏi gì, "
        "đang tìm thấy gì, model có dự báo tốt hay không và mỗi kết quả có ý nghĩa gì với người mới."
    )

    if not isinstance(result, dict) or not result.get("ok", False):
        st.info("Chưa có kết quả nghiên cứu hợp lệ để AI phân tích.")
        return

    signature = _research_signature(
        result,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    stored_signature = st.session_state.get("research_ai_signature")
    stored_result = st.session_state.get("research_ai_result")

    if stored_signature != signature:
        stored_result = None
        st.session_state.pop("research_ai_result", None)
        st.session_state.pop("research_ai_signature", None)
        st.session_state.pop("research_ai_followup_result", None)
        st.session_state.pop("research_ai_followup_question_used", None)

    if stored_result is None:
        st.info(
            "Nghiên cứu đã hoàn tất. Bấm nút để Brian AI chuyển toàn bộ kết quả kỹ thuật "
            "thành lời giải thích dễ hiểu."
        )

        if st.button(
            "🤖 Brian AI — Giải thích toàn bộ nghiên cứu",
            type="primary",
            width="stretch",
            key="research_ai_run",
        ):
            context = build_research_ai_context(
                result,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

            with st.spinner("🧠 Brian AI đang đọc nghiên cứu và diễn giải..."):
                ai_result = generate_research_ai_cached(
                    context,
                    DEFAULT_AUTO_QUESTION,
                )

            st.session_state["research_ai_result"] = ai_result
            st.session_state["research_ai_signature"] = signature
            st.session_state["research_ai_question_used"] = DEFAULT_AUTO_QUESTION
            st.rerun()

        return

    ai_result = stored_result

    if not ai_result.get("ok", False):
        st.error("🧠 BRIAN AI chưa thể phân tích nghiên cứu.")
        st.code(ai_result.get("error", "Không xác định."))
        st.caption("Phần nghiên cứu định lượng vẫn được giữ nguyên.")
        if st.button("🔄 Thử lại Brian AI", key="research_ai_retry"):
            st.session_state.pop("research_ai_result", None)
            st.session_state["research_ai_signature"] = signature
            st.rerun()
        return

    model_name = ai_result.get("model")
    if model_name:
        st.caption(f"Brian AI sử dụng: {model_name}")

    _render_json_result(ai_result)

    with st.expander("💬 Hỏi thêm Brian AI về nghiên cứu này", expanded=False):
        followup = st.text_area(
            "Câu hỏi bổ sung",
            placeholder=(
                "Ví dụ: Vì sao R² âm?\n"
                "VIF > 10 ảnh hưởng thế nào?\n"
                "EMA50 có thực sự là yếu tố quan trọng nhất không?"
            ),
            height=100,
            key="research_ai_followup_question",
        ).strip()

        if st.button(
            "🔎 Phân tích câu hỏi này",
            type="primary",
            width="stretch",
            key="research_ai_followup_run",
        ):
            if not followup:
                st.warning("Hãy nhập câu hỏi trước.")
            else:
                context = build_research_ai_context(
                    result,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                with st.spinner("Brian AI đang phân tích câu hỏi..."):
                    followup_result = generate_research_ai_cached(
                        context,
                        followup,
                    )
                st.session_state["research_ai_followup_result"] = followup_result
                st.session_state["research_ai_followup_question_used"] = followup
                st.rerun()

        followup_result = st.session_state.get("research_ai_followup_result")
        if followup_result:
            if followup_result.get("ok", False):
                st.caption(
                    f"Phân tích bổ sung bằng {followup_result.get('model', 'Gemini')}"
                )
                _render_json_result(followup_result)
            else:
                st.error("AI chưa trả lời được câu hỏi bổ sung.")
                st.code(followup_result.get("error", "Không xác định."))


__all__ = [
    "build_research_ai_context",
    "build_research_ai_prompt",
    "generate_research_ai_cached",
    "render_research_ai",
]

