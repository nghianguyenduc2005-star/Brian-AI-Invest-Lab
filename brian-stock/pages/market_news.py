from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from data.news import fetch_market_news


# ============================================================
# CONFIG
# ============================================================

NEWS_CACHE_TTL = 600
AI_CACHE_TTL = 900

DEFAULT_NEWS_COUNT = 15


# ============================================================
# UTIL
# ============================================================

def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    text = str(value).strip()

    return text if text else default


def _published_text(item: dict) -> str:
    return _text(
        item.get("published"),
        item.get("date", ""),
    )


def _source_text(item: dict) -> str:
    return _text(
        item.get("source"),
        item.get("source_group", "Nguồn không xác định"),
    )


def _title_text(item: dict) -> str:
    return _text(
        item.get("title"),
        "Không có tiêu đề",
    )


def _link_text(item: dict) -> str:
    return _text(
        item.get("link"),
        "",
    )


def _summary_text(item: dict) -> str:
    return _text(
        item.get("summary"),
        item.get("description", ""),
    )


# ============================================================
# NEWS NORMALIZATION
# ============================================================

def normalize_news(news):
    """
    Chuẩn hóa mọi kiểu dữ liệu news thành:

    [
        {
            "title": ...,
            "source": ...,
            "published": ...,
            "summary": ...,
            "link": ...
        }
    ]
    """

    if not isinstance(news, list):
        return []

    result = []
    seen = set()

    for item in news:

        if not isinstance(item, dict):
            continue

        title = _title_text(item)

        if not title:
            continue

        # Dedupe title
        key = " ".join(
            title.lower().split()
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(
            {
                "title": title,
                "source": _source_text(item),
                "published": _published_text(item),
                "summary": _summary_text(item),
                "link": _link_text(item),
            }
        )

    return result


# ============================================================
# NEWS LOADER
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
    except Exception:
        return []

    return normalize_news(
        news
    )


# ============================================================
# BUILD AI INPUT
# ============================================================

def build_news_context(
    news,
):
    lines = []

    for index, item in enumerate(
        news,
        start=1,
    ):

        title = _title_text(item)
        source = _source_text(item)
        published = _published_text(item)
        summary = _summary_text(item)
        link = _link_text(item)

        lines.append(
            f"""
TIN {index}
Tiêu đề: {title}
Nguồn: {source}
Thời gian: {published}
Tóm tắt: {summary}
Link: {link}
""".strip()
        )

    return "\n\n".join(
        lines
    )


# ============================================================
# AI KEY
# ============================================================

def get_gemini_key():
    try:
        return st.secrets.get(
            "GEMINI_API_KEY"
        )
    except Exception:
        return None


# ============================================================
# AI ENGINE
# ============================================================

@st.cache_data(
    ttl=AI_CACHE_TTL,
    show_spinner=False,
)
def generate_market_news_ai(
    news_text: str,
):
    """
    AI chỉ nhận tin thật đã thu thập.
    Không tự tìm tin ngoài.
    Không tự bịa số liệu.
    """

    api_key = get_gemini_key()

    if not api_key:

        return {
            "ok": False,
            "error": (
                "Chưa cấu hình GEMINI_API_KEY."
            ),
            "text": "",
        }

    if not news_text.strip():

        return {
            "ok": False,
            "error": (
                "Không có dữ liệu tin tức để phân tích."
            ),
            "text": "",
        }

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
Bạn là BRIAN AI — hệ thống phân tích và tổng hợp tin tức
cho nhà đầu tư chứng khoán Việt Nam.

NHIỆM VỤ:
Đọc TOÀN BỘ các tin tức được cung cấp bên dưới và tạo
một bản Market Brief bằng
