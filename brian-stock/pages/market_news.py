from __future__ import annotations

import io
from typing import Any

import streamlit as st

from data.news import fetch_market_news

try:
    from streamlit_paste_button import (
        paste_image_button,
    )
    PASTE_BUTTON_AVAILABLE = True
except Exception:
    paste_image_button = None
    PASTE_BUTTON_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = (
    MAX_IMAGE_SIZE_MB * 1024 * 1024
)


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
        value = str(
            value
        ).strip()
    except Exception:
        return default

    return value or default


# ============================================================
# READ UPLOADED IMAGE
# ============================================================

def _read_uploaded_image(
    uploaded_file,
):
    """
    Đọc ảnh từ st.file_uploader.

    Có thể nhận:
    - PNG
    - JPG
    - JPEG
    - WEBP
    """

    if uploaded_file is None:
        return None

    try:

        image_bytes = uploaded_file.getvalue()

    except Exception:

        return None

    if not image_bytes:
        return None

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:

        st.error(
            f"Ảnh quá lớn. "
            f"Giới hạn {MAX_IMAGE_SIZE_MB} MB."
        )

        return None

    mime_type = _text(
        getattr(
            uploaded_file,
            "type",
            None,
        ),
        "image/jpeg",
    )

    return {
        "bytes": image_bytes,
        "mime_type": mime_type,
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
# READ PASTED IMAGE
# ============================================================

def _read_pasted_image():
    """
    Lấy ảnh từ clipboard qua streamlit-paste-button.

    Component trả về PIL Image.
    Chuyển lại thành PNG bytes để gửi Gemini.
    """

    if not PASTE_BUTTON_AVAILABLE:

        return None

    try:

        result = paste_image_button(
            label="📋 Paste ảnh từ clipboard",
            text_color="#ffffff",
            background_color="#ff4b4b",
            hover_background_color="#d93636",
            key="market_news_paste_image",
            errors="raise",
        )

    except Exception as error:

        st.warning(
            f"Không thể mở clipboard: {error}"
        )

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

        image_bytes = buffer.getvalue()

    except Exception as error:

        st.error(
            f"Không thể đọc ảnh clipboard: {error}"
        )

        return None

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:

        st.error(
            f"Ảnh quá lớn. "
            f"Giới hạn {MAX_IMAGE_SIZE_MB} MB."
        )

        return None

    return {
        "bytes": image_bytes,
        "mime_type": "image/png",
        "source": "clipboard",
        "name": "clipboard.png",
    }


# ============================================================
# IMAGE INPUT
# ============================================================

def render_image_input():
    """
    Trả về:

        {
            "bytes": ...,
            "mime_type": ...,
            "source": ...,
            "name": ...
        }

    hoặc None.
    """

    st.markdown(
        "### 📷 Gửi ảnh cho AI"
    )

    st.caption(
        "Kéo-thả ảnh vào ô bên dưới, chọn file, "
        "hoặc paste ảnh từ clipboard."
    )

    # ========================================================
    # FILE UPLOADER
    # ========================================================

    uploaded_file = st.file_uploader(
        "Chọn hoặc kéo-thả ảnh",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        accept_multiple_files=False,
        key="market_news_image_uploader",
        max_upload_size=MAX_IMAGE_SIZE_MB,
        help=(
            "Hỗ trợ PNG, JPG, JPEG, WEBP. "
            f"Tối đa {MAX_IMAGE_SIZE_MB} MB."
        ),
    )

    uploaded_image = _read_uploaded_image(
        uploaded_file
    )

    # ========================================================
    # CLIPBOARD
    # ========================================================

    pasted_image = _read_pasted_image()

    # ========================================================
    # CHỌN INPUT
    #
    # Nếu cả hai có:
    # ưu tiên ảnh paste mới nhất.
    # ========================================================

    image_data = (
        pasted_image
        if pasted_image is not None
        else uploaded_image
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    if image_data is not None:

        st.success(
            "Đã nhận ảnh."
        )

        st.image(
            image_data["bytes"],
            caption=(
                "Nguồn: "
                + image_data["source"]
            ),
            width="stretch",
        )

        st.session_state[
            "market_news_selected_image"
        ] = image_data

    return image_data
