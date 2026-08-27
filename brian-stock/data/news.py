from __future__ import annotations

import html
import re
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
import pandas as pd
import streamlit as st


# ============================================================
# CẤU HÌNH
# ============================================================

CACHE_SECONDS = 300

RSS_FEEDS = [
    (
        "DNSE",
        "https://www.dnse.com.vn/senses/tin-tuc.rss",
    ),
    (
        "Vnstock",
        "https://vnstock.site/rss",
    ),
    (
        "Google News",
        "https://news.google.com/rss/search?q=ch%E1%BB%A9ng+kho%C3%A1n+Vi%E1%BB%87t+Nam&hl=vi&gl=VN&ceid=VN:vi",
    ),
]


# ============================================================
# LÀM SẠCH HTML
# ============================================================

def _lam_sach_html(
    gia_tri,
):
    if gia_tri is None:
        return ""

    text = str(
        gia_tri
    )

    # Giải mã entity HTML trước.
    text = html.unescape(
        text
    )

    # Loại toàn bộ tag HTML.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Chuẩn hóa khoảng trắng.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# LÀM SẠCH URL
# ============================================================

def _lam_sach_url(
    gia_tri,
):
    if gia_tri is None:
        return ""

    url = str(
        gia_tri
    ).strip()

    # Không cho javascript/data HTML lọt vào.
    if url.lower().startswith(
        (
            "javascript:",
            "data:",
        )
    ):
        return ""

    return url


# ============================================================
# NGÀY GIỜ
# ============================================================

def _dinh_dang_thoi_gian(
    entry,
):
    """
    Chuyển thời gian RSS thành dạng tiếng Việt:

    26/08/2026 16:07
    """

    gia_tri = (
        entry.get(
            "published",
            "",
        )
        or entry.get(
            "updated",
            "",
        )
    )

    gia_tri = str(
        gia_tri
    ).strip()

    if not gia_tri:
        return ""

    # --------------------------------------------------------
    # RSS chuẩn RFC 2822
    # --------------------------------------------------------

    try:

        dt = parsedate_to_datetime(
            gia_tri
        )

        if dt.tzinfo is not None:

            dt = dt.astimezone(
                pd.Timestamp.now(
                    tz="Asia/Ho_Chi_Minh"
                ).tz
            )

        return dt.strftime(
            "%d/%m/%Y %H:%M"
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # ISO
    # --------------------------------------------------------

    try:

        dt = pd.to_datetime(
            gia_tri,
            errors="coerce",
        )

        if pd.notna(dt):

            if getattr(
                dt,
                "tzinfo",
                None,
            ) is not None:

                dt = dt.tz_convert(
                    "Asia/Ho_Chi_Minh"
                )

            return dt.strftime(
                "%d/%m/%Y %H:%M"
            )

    except Exception:
        pass

    return _lam_sach_html(
        gia_tri
    )


# ============================================================
# ĐỌC RSS
# ============================================================

def _doc_rss(
    ten_nguon,
    url,
    gioi_han=20,
):
    try:

        feed = feedparser.parse(
            url
        )

    except Exception:
        return []

    ket_qua = []

    entries = getattr(
        feed,
        "entries",
        [],
    )

    for entry in entries[
        :gioi_han
    ]:

        tieu_de = _lam_sach_html(
            entry.get(
                "title",
                "",
            )
        )

        lien_ket = _lam_sach_url(
            entry.get(
                "link",
                "",
            )
        )

        mo_ta = _lam_sach_html(
            entry.get(
                "summary",
                "",
            )
        )

        thoi_gian = _dinh_dang_thoi_gian(
            entry
        )

        if not tieu_de:
            continue

        if not lien_ket:
            continue

        ket_qua.append(
            {
                "title": tieu_de,
                "source": _lam_sach_html(
                    ten_nguon
                ),
                "published": thoi_gian,
                "link": lien_ket,
                "summary": mo_ta,
            }
        )

    return ket_qua


# ============================================================
# KHỬ TRÙNG
# ============================================================

def _khung_trung(
    tin_tuc,
):
    da_co = set()

    ket_qua = []

    for tin in tin_tuc:

        tieu_de = (
            str(
                tin.get(
                    "title",
                    "",
                )
            )
            .strip()
            .lower()
        )

        link = (
            str(
                tin.get(
                    "link",
                    "",
                )
            )
            .strip()
            .lower()
        )

        khoa = (
            link
            or tieu_de
        )

        if not khoa:
            continue

        if khoa in da_co:
            continue

        da_co.add(
            khoa
        )

        ket_qua.append(
            tin
        )

    return ket_qua


# ============================================================
# SẮP XẾP
# ============================================================

def _gia_tri_thoi_gian(
    tin,
):
    text = str(
        tin.get(
            "published",
            "",
        )
    ).strip()

    if not text:
        return pd.Timestamp.min

    try:

        return pd.to_datetime(
            text,
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

    except Exception:

        return pd.Timestamp.min


# ============================================================
# LẤY TIN THỊ TRƯỜNG
# ============================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False,
)
def fetch_market_news(
    limit=6,
):
    """
    Trả về danh sách tin sạch:

    {
        "title": str,
        "source": str,
        "published": str,
        "link": str,
        "summary": str,
    }

    Tuyệt đối không trả HTML trong source/title/date.
    """

    try:

        limit = int(
            limit
        )

    except Exception:

        limit = 6

    limit = max(
        1,
        min(
            limit,
            50,
        ),
    )

    tat_ca_tin = []

    # ========================================================
    # ĐỌC TẤT CẢ NGUỒN
    # ========================================================

    for ten_nguon, url in RSS_FEEDS:

        tin = _doc_rss(
            ten_nguon,
            url,
            gioi_han=max(
                20,
                limit * 3,
            ),
        )

        tat_ca_tin.extend(
            tin
        )

    # ========================================================
    # KHÔNG CÓ RSS
    # ========================================================

    if not tat_ca_tin:
        return []

    # ========================================================
    # KHỬ TRÙNG
    # ========================================================

    tat_ca_tin = _khung_trung(
        tat_ca_tin
    )

    # ========================================================
    # SẮP XẾP MỚI NHẤT
    # ========================================================

    tat_ca_tin.sort(
        key=_gia_tri_thoi_gian,
        reverse=True,
    )

    # ========================================================
    # LÀM SẠCH LẦN CUỐI
    # ========================================================

    ket_qua = []

    for tin in tat_ca_tin:

        tieu_de = _lam_sach_html(
            tin.get(
                "title",
                "",
            )
        )

        nguon = _lam_sach_html(
            tin.get(
                "source",
                "",
            )
        )

        published = _lam_sach_html(
            tin.get(
                "published",
                "",
            )
        )

        summary = _lam_sach_html(
            tin.get(
                "summary",
                "",
            )
        )

        link = _lam_sach_url(
            tin.get(
                "link",
                "",
            )
        )

        if not tieu_de:
            continue

        ket_qua.append(
            {
                "title": tieu_de,
                "source": nguon,
                "published": published,
                "link": link,
                "summary": summary,
            }
        )

        if len(
            ket_qua
        ) >= limit:

            break

    return ket_qua


# ============================================================
# HÀM PHỤ TƯƠNG THÍCH
# ============================================================

def get_market_news(
    limit=6,
):
    return fetch_market_news(
        limit
    )


def load_market_news(
    limit=6,
):
    return fetch_market_news(
        limit
    )
