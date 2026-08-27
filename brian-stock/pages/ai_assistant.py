from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from data.market import (
    normalize_symbol,
    display_symbol,
    load_market_data,
    market_snapshot,
)


# ============================================================
# CẤU HÌNH AI
# ============================================================

TEN_MO_HINH = "gemini-3.7-flash"


# ============================================================
# LẤY KHÓA AI
# ============================================================

def _lay_khoa_ai():

    try:
        khoa = st.secrets.get(
            "GEMINI_API_KEY"
        )
    except Exception:
        khoa = None

    if not khoa:
        khoa = st.session_state.get(
            "GEMINI_API_KEY"
        )

    if not khoa:
        return None

    return str(khoa).strip()


# ============================================================
# TẠO MÁY AI
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def _tao_ai():

    khoa = _lay_khoa_ai()

    if not khoa:
        return None

    try:

        from google import genai

        return genai.Client(
            api_key=khoa
        )

    except Exception:
        return None


# ============================================================
# LẤY SỐ AN TOÀN
# ============================================================

def _so(
    gia_tri,
    mac_dinh=None,
):

    try:

        gia_tri = float(
            gia_tri
        )

        if pd.isna(
            gia_tri
        ):
            return mac_dinh

        return gia_tri

    except Exception:
        return mac_dinh


# ============================================================
# ĐỊNH DẠNG KHỐI LƯỢNG
# ============================================================

def _dinh_dang_khoi_luong(
    gia_tri,
):

    gia_tri = _so(
        gia_tri,
        None,
    )

    if gia_tri is None:
        return "—"

    if gia_tri >= 1_000_000_000:
        return (
            f"{gia_tri / 1_000_000_000:.2f} tỷ"
        )

    if gia_tri >= 1_000_000:
        return (
            f"{gia_tri / 1_000_000:.2f} triệu"
        )

    if gia_tri >= 1_000:
        return (
            f"{gia_tri / 1_000:.2f} nghìn"
        )

    return f"{gia_tri:,.0f}"


# ============================================================
# CHUẨN BỊ DỮ LIỆU CHO AI
# ============================================================

def _tao_thong_tin_phan_tich(
    ma,
    du_lieu,
):

    anh_chup = market_snapshot(
        du_lieu
    )

    dong_cuoi = (
        du_lieu.iloc[-1]
    )

    def _lay(
        ten_cot,
        mac_dinh=None,
    ):

        try:

            gia_tri = dong_cuoi.get(
                ten_cot
            )

            if pd.isna(
                gia_tri
            ):
                return mac_dinh

            return float(
                gia_tri
            )

        except Exception:
            return mac_dinh

    gia = _so(
        anh_chup.get(
            "price"
        ),
        None,
    )

    thay_doi = _so(
        anh_chup.get(
            "change_1d"
        ),
        None,
    )

    rsi = _so(
        anh_chup.get(
            "rsi"
        ),
        None,
    )

    macd = _so(
        anh_chup.get(
            "macd"
        ),
        None,
    )

    sma20 = _so(
        anh_chup.get(
            "sma20"
        ),
        None,
    )

    sma50 = _so(
        anh_chup.get(
            "sma50"
        ),
        None,
    )

    bien_dong = _so(
        anh_chup.get(
            "volatility20"
        ),
        None,
    )

    khoi_luong = _so(
        anh_chup.get(
            "volume"
        ),
        None,
    )

    macd_tin_hieu = _lay(
        "MACD_Signal"
    )

    macd_hist = _lay(
        "MACD_Hist"
    )

    ema20 = _lay(
        "EMA20"
    )

    ema50 = _lay(
        "EMA50"
    )

    bollinger_tren = _lay(
        "Bollinger_Upper"
    )

    bollinger_duoi = _lay(
        "Bollinger_Lower"
    )

    atr14 = _lay(
        "ATR14"
    )

    momentum5 = _lay(
        "Momentum5"
    )

    momentum20 = _lay(
        "Momentum20"
    )

    volume_tb20 = _lay(
        "Volume_SMA20"
    )

    high20 = _lay(
        "High20"
    )

    low20 = _lay(
        "Low20"
    )

    high252 = _lay(
        "High252"
    )

    low252 = _lay(
        "Low252"
    )

    duong = []

    if (
        gia is not None
        and sma20 is not None
    ):

        duong.append(
            "Giá trên SMA20"
            if gia > sma20
            else "Giá dưới SMA20"
        )

    if (
        gia is not None
        and sma50 is not None
    ):

        duong.append(
            "Giá trên SMA50"
            if gia > sma50
            else "Giá dưới SMA50"
        )

    if (
        gia is not None
        and ema20 is not None
    ):

        duong.append(
            "Giá trên EMA20"
            if gia > ema20
            else "Giá dưới EMA20"
        )

    if (
        gia is not None
        and ema50 is not None
    ):

        duong.append(
            "Giá trên EMA50"
            if gia > ema50
            else "Giá dưới EMA50"
        )

    if rsi is not None:

        if rsi >= 70:
            duong.append(
                "RSI quá mua"
            )

        elif rsi <= 30:
            duong.append(
                "RSI quá bán"
            )

        elif rsi >= 50:
            duong.append(
                "RSI trên 50"
            )

        else:
            duong.append(
                "RSI dưới 50"
            )

    if (
        macd is not None
        and macd_tin_hieu is not None
    ):

        duong.append(
            "MACD trên tín hiệu"
            if macd > macd_tin_hieu
            else "MACD dưới tín hiệu"
        )

    if (
        khoi_luong is not None
        and volume_tb20 is not None
        and volume_tb20 > 0
    ):

        ty_le = (
            khoi_luong
            / volume_tb20
        )

        duong.append(
            f"Khối lượng bằng {ty_le:.2f} lần trung bình 20 phiên"
        )

    thong_tin = f"""
MÃ CỔ PHIẾU: {display_symbol(ma)}

DỮ LIỆU MỚI NHẤT:
- Giá đóng cửa: {gia}
- Thay đổi 1 ngày: {thay_doi}%
- RSI: {rsi}
- MACD: {macd}
- Tín hiệu MACD: {macd_tin_hieu}
- MACD Histogram: {macd_hist}

TRUNG BÌNH:
- SMA20: {sma20}
- SMA50: {sma50}
- EMA20: {ema20}
- EMA50: {ema50}

BIẾN ĐỘNG:
- Biến động 20 phiên: {bien_dong}%
- ATR14: {atr14}
- Động lượng 5 phiên: {momentum5}
- Động lượng 20 phiên: {momentum20}

KHỐI LƯỢNG:
- Khối lượng hiện tại: {khoi_luong}
- Trung bình 20 phiên: {volume_tb20}

BOLLINGER:
- Dải trên: {bollinger_tren}
- Dải dưới: {bollinger_duoi}

ĐỈNH / ĐÁY:
- Cao nhất 20 phiên: {high20}
- Thấp nhất 20 phiên: {low20}
- Cao nhất 252 phiên: {high252}
- Thấp nhất 252 phiên: {low252}

TÍN HIỆU TỰ ĐỘNG:
{chr(10).join("- " + x for x in duong)}
"""

    return thong_tin


# ============================================================
# GỌI AI
# ============================================================

def _hoi_ai(
    cau_hoi,
    thong_tin,
):

    may_ai = _tao_ai()

    if may_ai is None:

        return (
            "Chưa cấu hình GEMINI_API_KEY "
            "trong Streamlit Secrets."
        )

    chi_dan = f"""
Bạn là trợ lý nghiên cứu chứng khoán Việt Nam của BRIAN STOCK.

Nhiệm vụ:
- Phân tích dữ liệu được cung cấp.
- Không bịa số liệu.
- Không tự tạo dữ liệu thị trường.
- Phân biệt rõ dữ liệu quan sát và nhận định.
- Không cam kết lợi nhuận.
- Không đưa ra khẳng định chắc chắn về giá tương lai.
- Trả lời bằng tiếng Việt.
- Ưu tiên phân tích xu hướng, động lượng,
  thanh khoản, biến động, vùng hỗ trợ/kháng cự
  và rủi ro.

DỮ LIỆU:
{thong_tin}

CÂU HỎI CỦA NHÀ ĐẦU TƯ:
{cau_hoi}

Hãy trả lời có cấu trúc:
1. Kết luận chính
2. Dữ liệu đáng chú ý
3. Tín hiệu tích cực
4. Tín hiệu tiêu cực
5. Rủi ro cần theo dõi
6. Kịch bản có thể xảy ra
"""

    try:

        phan_hoi = (
            may_ai.models.generate_content(
                model=TEN_MO_HINH,
                contents=chi_dan,
            )
        )

        van_ban = getattr(
            phan_hoi,
            "text",
            None,
        )

        if not van_ban:

            return (
                "AI không trả về nội dung."
            )

        return str(
            van_ban
        ).strip()

    except Exception as loi:

        return (
            "Không thể gọi AI: "
            f"{loi}"
        )


# ============================================================
# RENDER
# ============================================================

def render_ai_assistant():

    st.markdown(
        '<div class="section-title">🤖 AI Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                BRIAN STOCK · AI RESEARCH
            </div>

            <h1>
                Trợ lý phân tích đầu tư
            </h1>

            <p>
                AI đọc dữ liệu thị trường thật và
                hỗ trợ diễn giải các tín hiệu kỹ thuật
                và định lượng.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # CHỌN MÃ
    # ========================================================

    ma_mac_dinh = (
        st.session_state.get(
            "ai_symbol",
            "HPG",
        )
    )

    ma_nhap = st.text_input(
        "Mã cổ phiếu",
        value=ma_mac_dinh,
        placeholder="Ví dụ HPG, MSR, VNM",
        key="ai_stock_input",
    )

    if st.button(
        "Tải dữ liệu phân tích",
        type="primary",
        key="ai_load_data",
    ):

        ma_sach = normalize_symbol(
            ma_nhap
        )

        st.session_state[
            "ai_symbol"
        ] = ma_sach

        st.rerun()

    ma = normalize_symbol(
        st.session_state.get(
            "ai_symbol",
            ma_nhap,
        )
    )

    # ========================================================
    # LẤY DỮ LIỆU
    # ========================================================

    try:

        du_lieu = load_market_data(
            ma,
            "1y",
        )

    except Exception as loi:

        st.error(
            f"Không tải được dữ liệu {display_symbol(ma)}: {loi}"
        )

        return

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        st.error(
            "Không có dữ liệu để phân tích."
        )

        return

    # ========================================================
    # SNAPSHOT
    # ========================================================

    anh_chup = market_snapshot(
        du_lieu
    )

    st.markdown(
        f"""
        <div class="section-title">
            📈 {html.escape(display_symbol(ma))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:

        gia = _so(
            anh_chup.get(
                "price"
            ),
            None,
        )

        st.metric(
            "Giá",
            (
                f"{gia:,.0f}"
                if gia is not None
                else "—"
            ),
        )

    with b:

        thay_doi = _so(
            anh_chup.get(
                "change_1d"
            ),
            None,
        )

        st.metric(
            "1D",
            (
                f"{thay_doi:+.2f}%"
                if thay_doi is not None
                else "—"
            ),
        )

    with c:

        gia_tri_rsi = _so(
            anh_chup.get(
                "rsi"
            ),
            None,
        )

        st.metric(
            "RSI",
            (
                f"{gia_tri_rsi:.1f}"
                if gia_tri_rsi is not None
                else "—"
            ),
        )

    with d:

        khoi_luong = _so(
            anh_chup.get(
                "volume"
            ),
            None,
        )

        st.metric(
            "Khối lượng",
            _dinh_dang_khoi_luong(
                khoi_luong
            ),
        )

    # ========================================================
    # CÂU HỎI
    # ========================================================

    cau_hoi = st.text_area(
        "Câu hỏi cho AI",
        placeholder=(
            "Ví dụ: Hãy đánh giá xu hướng hiện tại "
            "của MSR và những rủi ro cần chú ý."
        ),
        height=130,
        key="ai_question",
    )

    # ========================================================
    # PHÂN TÍCH NHANH
    # ========================================================

    col_1, col_2, col_3 = st.columns(3)

    with col_1:

        if st.button(
            "Phân tích xu hướng",
            key="ai_trend",
            width="stretch",
        ):

            cau_hoi_hien_tai = (
                "Hãy phân tích xu hướng "
                "ngắn hạn và trung hạn."
            )

            thong_tin = (
                _tao_thong_tin_phan_tich(
                    ma,
                    du_lieu,
                )
            )

            with st.spinner(
                "AI đang phân tích..."
            ):

                ket_qua = _hoi_ai(
                    cau_hoi_hien_tai,
                    thong_tin,
                )

            st.session_state[
                "ai_result"
            ] = ket_qua

    with col_2:

        if st.button(
            "Phân tích rủi ro",
            key="ai_risk",
            width="stretch",
        ):

            cau_hoi_hien_tai = (
                "Hãy phân tích các rủi ro "
                "kỹ thuật và biến động cần chú ý."
            )

            thong_tin = (
                _tao_thong_tin_phan_tich(
                    ma,
                    du_lieu,
                )
            )

            with st.spinner(
                "AI đang phân tích..."
            ):

                ket_qua = _hoi_ai(
                    cau_hoi_hien_tai,
                    thong_tin,
                )

            st.session_state[
                "ai_result"
            ] = ket_qua

    with col_3:

        if st.button(
            "Phân tích tổng hợp",
            key="ai_full",
            width="stretch",
        ):

            cau_hoi_hien_tai = (
                cau_hoi.strip()
                if cau_hoi.strip()
                else (
                    "Hãy phân tích tổng hợp "
                    "cổ phiếu này."
                )
            )

            thong_tin = (
                _tao_thong_tin_phan_tich(
                    ma,
                    du_lieu,
                )
            )

            with st.spinner(
                "AI đang phân tích..."
            ):

                ket_qua = _hoi_ai(
                    cau_hoi_hien_tai,
                    thong_tin,
                )

            st.session_state[
                "ai_result"
            ] = ket_qua

    # ========================================================
    # KẾT QUẢ
    # ========================================================

    ket_qua = st.session_state.get(
        "ai_result"
    )

    if ket_qua:

        st.markdown(
            '<div class="section-title">🧠 Kết quả AI</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            ket_qua
        )

    # ========================================================
    # DỮ LIỆU DÙNG ĐỂ PHÂN TÍCH
    # ========================================================

    with st.expander(
        "📊 Xem dữ liệu AI đang sử dụng",
        expanded=False,
    ):

        st.code(
            _tao_thong_tin_phan_tich(
                ma,
                du_lieu,
            ),
            language="text",
        )
