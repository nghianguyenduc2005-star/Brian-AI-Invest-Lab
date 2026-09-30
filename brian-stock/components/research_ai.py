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
DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = []
SUPPORTED_RESEARCH_MODEL = "gemini-3.8-flash"
DEFAULT_AUTO_QUESTION = (
    "Hãy tự đọc TOÀN BỘ nghiên cứu của cổ phiếu này và viết một báo cáo phân tích định lượng "
    "chuyên nghiệp, chi tiết nhưng người mới cũng hiểu được. Hãy tổng hợp dữ liệu mẫu, "
    "tương quan, ý nghĩa thống kê, OLS, Machine Learning, importance, VIF, kiểm định, "
    "kết quả 1D/5D/20D và forecast. Chỉ kết luận từ số liệu thực tế được cung cấp; "
    "phân biệt rõ tương quan, ý nghĩa thống kê và khả năng dự báo; không dùng ngôn ngữ nhân quả "
    "khi dữ liệu không cho phép. Cuối cùng, nói đơn giản xem toàn bộ nghiên cứu đang cho thấy điều gì."
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


def get_gemini_api_key() -> str | None:
    """Ưu tiên Streamlit Secrets, fallback sang environment variable để chạy local."""
    key = ""
    try:
        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass
    if not key:
        key = str(os.getenv("GEMINI_API_KEY", "")).strip()
    return key or None


def _parse_json_text(text: str):
    """Parse JSON ngay cả khi model lỡ bọc trong ```json ... ``` hoặc thêm text."""
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
            candidate = cleaned[start : end + 1]
            return json.loads(candidate)
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
        for rank, (_, row) in enumerate(ranking.head(40).iterrows(), start=1):
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
    return stable.head(50)


def build_research_ai_context(result, symbol="", start_date=None, end_date=None) -> str:
    """Đóng gói toàn bộ kết quả nghiên cứu đã tính để Gemini đọc."""
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

    blocks.append("\n=== THỐNG KÊ MÔ TẢ TOÀN BỘ BIẾN ===")
    blocks.append(_describe_research_data(data))

    if all_features:
        blocks.append("\n=== TOÀN BỘ TÊN BIẾN ===")
        blocks.append(", ".join(map(str, all_features)))

    consistency = _consistency_table(horizons)
    blocks.append("\n=== YẾU TỐ NHẤT QUÁN QUA 1D / 5D / 20D ===")
    blocks.append(_df_text(consistency, limit=50))

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
        blocks.append(
            f"Model tốt nhất hiện tại={item.get('best_model', 'Không xác định')}"
        )

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
                    "Biến",
                    "Nhóm",
                    "Score",
                    "Quan hệ",
                    "Pearson",
                    "Spearman",
                    "Beta",
                    "p-value",
                    "Permutation",
                    "TreeImportance",
                    "Ý nghĩa",
                ],
                limit=100,
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

        blocks.append("\n--- MODEL FEATURES ---")
        blocks.append(", ".join(map(str, item.get("model_features", []))))

        blocks.append("\n--- OLS FEATURES ---")
        blocks.append(", ".join(map(str, item.get("ols_features", []))))

    return "\n".join(blocks)


# ============================================================
# AI SCHEMA / PROMPT
# ============================================================

RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "array",
            "items": {"type": "string"},
        },
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
                "required": [
                    "feature",
                    "group",
                    "direction",
                    "evidence",
                    "interpretation",
                ],
            },
        },
        "horizon_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "stable_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "horizon",
                    "conclusion",
                    "stable_factors",
                ],
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
            "required": [
                "best_model",
                "rmse",
                "r2",
                "benchmark",
                "assessment",
            ],
        },
        "statistical_quality": {
            "type": "array",
            "items": {"type": "string"},
        },
        "plain_language": {"type": "string"},
        "important_cautions": {
            "type": "array",
            "items": {"type": "string"},
        },
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


def build_research_ai_prompt(
    research_context: str,
    user_question: str = "",
) -> str:
    question = _text(user_question, DEFAULT_AUTO_QUESTION)

    return f"""
Bạn là BRIAN AI — chuyên viên phân tích nghiên cứu định lượng chứng khoán.

NHIỆM VỤ
Đọc toàn bộ dữ liệu nghiên cứu được cung cấp và viết báo cáo phân tích dựa hoàn toàn trên dữ liệu đó.
Báo cáo phải chuyên nghiệp, chi tiết và dễ hiểu đối với người mới.

CÂU HỎI / MỤC TIÊU PHÂN TÍCH
{question}

NGUYÊN TẮC BẮT BUỘC
1. Chỉ sử dụng số liệu xuất hiện trong dữ liệu nghiên cứu. Không tự bịa số.
2. Phân biệt rõ:
   - tương quan = hai biến cùng biến động như thế nào;
   - ý nghĩa thống kê = bằng chứng thống kê có đủ mạnh hay chưa;
   - khả năng dự báo = model dự báo tốt đến đâu trên tập test.
3. Không nói "gây ra" hoặc khẳng định nhân quả khi nghiên cứu chỉ cho thấy quan hệ thống kê.
4. Không đánh giá một yếu tố chỉ vì Score cao; đối chiếu Spearman/Pearson, Beta, p-value,
   permutation/tree importance và sự nhất quán giữa 1D/5D/20D.
5. Nếu p-value > 0.05, phải nói rõ rằng bằng chứng chưa mạnh ở mức 5%.
6. Nếu R² test thấp hoặc âm, nói rõ điều đó và đối chiếu benchmark.
7. RMSE càng thấp thì sai số dự báo trên cùng thang đo càng nhỏ; không so sánh RMSE
   giữa các horizon khác nhau như thể cùng một mục tiêu.
8. Nếu VIF cao, cảnh báo đa cộng tuyến và hạn chế diễn giải riêng từng hệ số OLS.
9. Nếu ADF, Breusch-Pagan, Durbin-Watson hoặc Ljung-Box có tín hiệu đáng chú ý,
   phải giải thích bằng ngôn ngữ đơn giản và nêu giới hạn của kiểm định.
10. Không biến forecast thành cam kết giá tương lai.
11. Không đưa khuyến nghị mua/bán cá nhân hóa.
12. Nếu dữ liệu chưa đủ để kết luận, phải nói "chưa đủ bằng chứng".
13. Khi một kết luận chỉ đúng ở một horizon, phải nói rõ; không được biến thành kết luận chung.
14. Tập test là phần quan trọng nhất để đánh giá khả năng dự báo thực tế; đừng dùng R² train
    để quảng bá model.
15. Giải thích thuật ngữ lần đầu xuất hiện bằng tiếng Việt đơn giản.

YÊU CẦU NỘI DUNG
- Tóm tắt điều hành 3–6 ý.
- Tối đa 5 yếu tố nổi bật và phải giải thích bằng bằng chứng số liệu.
- Phân tích riêng 1D, 5D, 20D.
- Chỉ ra yếu tố nào lặp lại hoặc nhất quán giữa các horizon, nếu có.
- Đánh giá model bằng RMSE, R² test và benchmark.
- Đánh giá chất lượng thống kê / dữ liệu chỉ dựa trên các kiểm định thực sự có trong dữ liệu.
- Giải thích forecast theo ngôn ngữ xác suất / mô hình, không coi đó là chắc chắn.
- Kết thúc bằng phần "Nói đơn giản cho người mới".

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
            "error": (
                "Chưa cấu hình GEMINI_API_KEY. Hãy thêm GEMINI_API_KEY vào "
                "Streamlit Secrets hoặc biến môi trường."
            ),
        }

    # Nghiên cứu định lượng dùng cố định Gemini 3.8 Flash.
    # Không để model cũ trong session_state (ví dụ gemini-2.5-flash)
    # ghi đè cấu hình mới.
    requested_model = SUPPORTED_RESEARCH_MODEL
    if st.session_state.get("ai_model") != SUPPORTED_RESEARCH_MODEL:
        st.session_state["ai_model"] = SUPPORTED_RESEARCH_MODEL

    models_to_try = [requested_model]

    last_error = None
    client = None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as error:
        return {
            "ok": False,
            "text": "",
            "json": None,
            "model": requested_model,
            "error": f"Không khởi tạo được Gemini client: {error}",
        }

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESEARCH_SCHEMA,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="low",
                    ),
                    max_output_tokens=5000,
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

    return {
        "ok": False,
        "text": "",
        "json": None,
        "model": requested_model,
        "error": last_error or "AI lỗi không xác định.",
    }


@st.cache_data(ttl=AI_CACHE_TTL, show_spinner=False)
def generate_research_ai_cached(
    research_context: str,
    user_question: str,
):
    prompt = build_research_ai_prompt(research_context, user_question)
    return _call_gemini(prompt)


# ============================================================
# RESEARCH FINGERPRINT
# ============================================================


def _research_signature(result, symbol="", start_date=None, end_date=None) -> str:
    """Tạo khóa ổn định cho một lần nghiên cứu, tránh gọi Gemini lặp lại."""
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
                        "Biến",
                        "Nhóm",
                        "Score",
                        "Quan hệ",
                        "Spearman",
                        "Beta",
                        "p-value",
                        "Permutation",
                        "TreeImportance",
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
# RENDER AI RESULT
# ============================================================


def _render_json_result(ai_result: dict[str, Any]):
    data = ai_result.get("json") or {}

    st.subheader("🧾 Kết luận điều hành")
    for item in data.get("executive_summary", []):
        st.markdown(f"• {item}")

    drivers = data.get("key_drivers", [])
    if drivers:
        st.subheader("🎯 Các yếu tố nổi bật")
        st.dataframe(
            pd.DataFrame(drivers),
            width="stretch",
            hide_index=True,
        )

    horizons = data.get("horizon_analysis", [])
    if horizons:
        st.subheader("📈 Phân tích theo thời gian dự báo")
        for item in horizons:
            horizon = str(item.get("horizon", "Horizon"))
            with st.expander(horizon, expanded=True):
                st.write(item.get("conclusion", ""))
                factors = item.get("stable_factors", [])
                if factors:
                    st.markdown(
                        "**Yếu tố chính:** "
                        + ", ".join(map(str, factors))
                    )

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


# ============================================================
# MAIN INTEGRATION
# ============================================================


def render_research_ai(result, symbol="", start_date=None, end_date=None):
    """Tự động phân tích AI ngay sau khi nghiên cứu định lượng hoàn tất.

    Không cần người dùng bấm nút để chạy AI lần đầu.
    AI chỉ chạy lại khi nội dung nghiên cứu / mã / khoảng thời gian thay đổi.
    """
    st.divider()
    st.header("🧠 BRIAN AI — Phân tích toàn bộ nghiên cứu")
    st.caption(
        "AI tự đọc kết quả nghiên cứu vừa chạy: thống kê mô tả, tương quan, OLS, Machine Learning, "
        "importance, VIF, kiểm định, 1D/5D/20D và forecast; sau đó giải thích bằng ngôn ngữ dễ hiểu."
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

    # Khi có một nghiên cứu mới, bỏ kết quả hỏi thêm của nghiên cứu cũ.
    if stored_signature != signature:
        st.session_state.pop("research_ai_followup_result", None)
        st.session_state.pop("research_ai_followup_question_used", None)

    # --------------------------------------------------------
    # AUTO AI: chạy ngay sau khi nghiên cứu hoàn tất
    # --------------------------------------------------------
    if stored_signature != signature or not stored_result:
        context = build_research_ai_context(
            result,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        with st.spinner("🧠 Brian AI đang đọc toàn bộ nghiên cứu và viết kết luận..."):
            ai_result = generate_research_ai_cached(
                context,
                DEFAULT_AUTO_QUESTION,
            )

        st.session_state["research_ai_result"] = ai_result
        st.session_state["research_ai_signature"] = signature
        st.session_state["research_ai_question_used"] = DEFAULT_AUTO_QUESTION
        stored_result = ai_result

    ai_result = stored_result
    if not ai_result:
        st.warning("AI chưa tạo được kết quả.")
        return

    # --------------------------------------------------------
    # ERROR DISPLAY
    # --------------------------------------------------------
    if not ai_result.get("ok", False):
        st.error("🧠 BRIAN AI chưa thể phân tích nghiên cứu.")
        st.code(ai_result.get("error", "Không xác định."))
        st.caption(
            "Kiểm tra GEMINI_API_KEY trong Streamlit Secrets và kết nối mạng của ứng dụng."
        )
        return

    model_name = ai_result.get("model")
    if model_name:
        st.caption(f"Brian AI sử dụng: {model_name}")

    # --------------------------------------------------------
    # AUTO REPORT
    # --------------------------------------------------------
    _render_json_result(ai_result)

    # --------------------------------------------------------
    # OPTIONAL FOLLOW-UP — không ảnh hưởng auto analysis
    # --------------------------------------------------------
    with st.expander("💬 Hỏi thêm Brian AI về nghiên cứu này", expanded=False):
        followup = st.text_area(
            "Câu hỏi bổ sung",
            placeholder=(
                "Ví dụ: Vì sao R² thấp?\n"
                "Yếu tố nào nhất quán nhất 1D / 5D / 20D?\n"
                "Giải thích VIF cho người mới."
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
