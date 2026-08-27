from __future__ import annotations

import html
import json
import os
import re

import pandas as pd
import streamlit as st

from components.charts import price_volume_chart
from data.market import (
    normalize_symbol,
    display_symbol,
    load_market_data,
    market_snapshot,
)

from analysis.quant import (
    ten_bien_tieng_viet,
    lay_bien_theo_nhom,
    lay_danh_sach_bien,
    run_quant,
)


# ============================================================
# TÊN BIẾN HIỂN THỊ
# ============================================================

TEN_BIEN_HIEN_THI = {
    "Open": "Giá mở cửa",
    "High": "Giá cao nhất",
    "Low": "Giá thấp nhất",
    "Close": "Giá đóng cửa",
    "Volume": "Khối lượng",

    "Return": "Lợi suất",
    "ReturnPct": "Lợi suất (%)",

    "SMA5": "Trung bình 5 phiên",
    "SMA10": "Trung bình 10 phiên",
    "SMA20": "Trung bình 20 phiên",
    "SMA50": "Trung bình 50 phiên",
    "SMA100": "Trung bình 100 phiên",
    "SMA200": "Trung bình 200 phiên",

    "EMA9": "Trung bình lũy thừa 9 phiên",
    "EMA12": "Trung bình lũy thừa 12 phiên",
    "EMA20": "Trung bình lũy thừa 20 phiên",
    "EMA26": "Trung bình lũy thừa 26 phiên",
    "EMA50": "Trung bình lũy thừa 50 phiên",

    "RSI": "Sức mạnh tương đối RSI",
    "MACD": "MACD",
    "MACD_Signal": "Tín hiệu MACD",
    "MACD_Hist": "Động lượng MACD",

    "Volatility5": "Biến động 5 phiên",
    "Volatility20": "Biến động 20 phiên",
    "Volatility60": "Biến động 60 phiên",

    "Volume_SMA5": "Khối lượng trung bình 5 phiên",
    "Volume_SMA20": "Khối lượng trung bình 20 phiên",
    "Volume_SMA50": "Khối lượng trung bình 50 phiên",
    "Volume_Change": "Thay đổi khối lượng",
    "Relative_Volume": "Khối lượng tương đối",

    "Range": "Biên độ giá",
    "Range_Percent": "Biên độ giá (%)",
    "ATR14": "ATR 14 phiên",

    "Momentum5": "Động lượng 5 phiên",
    "Momentum10": "Động lượng 10 phiên",
    "Momentum20": "Động lượng 20 phiên",

    "Bollinger_Mid": "Đường giữa Bollinger",
    "Bollinger_Upper": "Dải trên Bollinger",
    "Bollinger_Lower": "Dải dưới Bollinger",
    "Bollinger_Width": "Độ rộng Bollinger (%)",

    "High20": "Đỉnh 20 phiên",
    "Low20": "Đáy 20 phiên",
    "High50": "Đỉnh 50 phiên",
    "Low50": "Đáy 50 phiên",
    "High252": "Đỉnh 252 phiên",
    "Low252": "Đáy 252 phiên",

    "Distance_From_High20": "Khoảng cách tới đỉnh 20 phiên",
    "Distance_From_Low20": "Khoảng cách tới đáy 20 phiên",
    "Distance_From_High252": "Khoảng cách tới đỉnh 252 phiên",
    "Distance_From_Low252": "Khoảng cách tới đáy 252 phiên",

    "Gap_Open_Pct": "Khoảng trống giá (%)",
    "Gap_Up": "Khoảng trống tăng",
    "Gap_Down": "Khoảng trống giảm",

    "Khoang_Cach_SMA20": "Khoảng cách tới SMA20 (%)",
    "Khoang_Cach_SMA50": "Khoảng cách tới SMA50 (%)",
    "Khoang_Cach_SMA200": "Khoảng cách tới SMA200 (%)",
    "Khoang_Cach_EMA20": "Khoảng cách tới EMA20 (%)",
    "Khoang_Cach_EMA50": "Khoảng cách tới EMA50 (%)",

    "Bien_Dong_Gia_2": "Thay đổi giá 2 phiên",
    "Bien_Dong_Gia_3": "Thay đổi giá 3 phiên",
    "Bien_Dong_Gia_5": "Thay đổi giá 5 phiên",
    "Bien_Dong_Gia_10": "Thay đổi giá 10 phiên",
    "Bien_Dong_Gia_20": "Thay đổi giá 20 phiên",
    "Bien_Dong_Gia_60": "Thay đổi giá 60 phiên",
    "Bien_Dong_Gia_120": "Thay đổi giá 120 phiên",
    "Bien_Dong_Gia_252": "Thay đổi giá 252 phiên",

    "Range_Real": "Biên độ thực",
    "Range_Real_Pct": "Biên độ thực (%)",

    "Vi_Tri_Vung_20": "Vị trí trong vùng 20 phiên",
    "Vi_Tri_Vung_50": "Vị trí trong vùng 50 phiên",
    "Vi_Tri_Vung_252": "Vị trí trong vùng 252 phiên",

    "Bien_Do_Ngay": "Biên độ trong ngày (%)",
    "Ty_Le_Khoi_Luong_TB20": "Tỷ lệ khối lượng / trung bình 20 phiên",

    "Thu_Trong_Tuan": "Thứ trong tuần",
    "Ngay_Trong_Thang": "Ngày trong tháng",
    "Thang_Trong_Nam": "Tháng trong năm",
}


def _ten_hien_thi(ten_bien):
    return TEN_BIEN_HIEN_THI.get(
        ten_bien,
        ten_bien_tieng_viet(ten_bien),
    )


# ============================================================
# TIỆN ÍCH
# ============================================================

def _so(
    gia_tri,
    mac_dinh=None,
):
    try:
        gia_tri = float(gia_tri)

        if pd.isna(gia_tri):
            return mac_dinh

        return gia_tri

    except Exception:
        return mac_dinh


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
        return f"{gia_tri / 1_000_000_000:.2f} tỷ"

    if gia_tri >= 1_000_000:
        return f"{gia_tri / 1_000_000:.2f} triệu"

    if gia_tri >= 1_000:
        return f"{gia_tri / 1_000:.2f} nghìn"

    return f"{gia_tri:,.0f}"


# ============================================================
# LẤY API KEY GOOGLE
# ============================================================

def _lay_google_api_key():

    try:

        if "GOOGLE_API_KEY" in st.secrets:

            key = st.secrets[
                "GOOGLE_API_KEY"
            ]

            if key:
                return str(key).strip()

    except Exception:
        pass

    key = os.getenv(
        "GOOGLE_API_KEY",
        "",
    )

    return str(
        key or ""
    ).strip()


# ============================================================
# AI PHÂN TÍCH PROMPT
# ============================================================

def _ai_chon_bien(
    prompt,
    danh_sach_bien,
):

    prompt = str(
        prompt or ""
    ).strip()

    if not prompt:
        return []

    api_key = _lay_google_api_key()

    if not api_key:
        return []

    try:

        from google import genai

    except Exception:
        return []

    # --------------------------------------------------------
    # Danh sách biến cho AI
    # --------------------------------------------------------

    danh_sach_text = "\n".join(
        [
            f"- {bien}: {_ten_hien_thi(bien)}"
            for bien
            in danh_sach_bien
        ]
    )

    yeu_cau = f"""
Bạn là trợ lý phân tích định lượng cổ phiếu.

Người dùng yêu cầu:
{prompt}

Danh sách biến có thể sử dụng:
{danh_sach_text}

Hãy chọn những biến phù hợp nhất với yêu cầu.

Chỉ được chọn những biến xuất hiện chính xác
trong danh sách ở trên.

Trả về JSON hợp lệ theo đúng cấu trúc:

{{
  "bien": [
    "ten_bien_1",
    "ten_bien_2"
  ],
  "ly_do": "Giải thích ngắn gọn"
}}

Không thêm markdown.
Không thêm biến ngoài danh sách.
"""

    try:

        client = genai.Client(
            api_key=api_key
        )

        phan_hoi = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=yeu_cau,
        )

        noi_dung = str(
            getattr(
                phan_hoi,
                "text",
                "",
            )
            or ""
        ).strip()

        if not noi_dung:
            return []

        # ----------------------------------------------------
        # Làm sạch JSON
        # ----------------------------------------------------

        noi_dung = re.sub(
            r"^```json\s*",
            "",
            noi_dung,
            flags=re.IGNORECASE,
        )

        noi_dung = re.sub(
            r"\s*```$",
            "",
            noi_dung,
        )

        du_lieu_ai = json.loads(
            noi_dung
        )

        bien_ai = (
            du_lieu_ai.get(
                "bien",
                [],
            )
        )

        if not isinstance(
            bien_ai,
            list,
        ):
            return []

        # ----------------------------------------------------
        # Chỉ nhận biến hợp lệ
        # ----------------------------------------------------

        ket_qua = []

        for bien in bien_ai:

            bien = str(
                bien
            ).strip()

            if (
                bien in danh_sach_bien
                and bien not in ket_qua
            ):

                ket_qua.append(
                    bien
                )

        # Lưu lý do
        st.session_state[
            "quant_ai_reason"
        ] = str(
            du_lieu_ai.get(
                "ly_do",
                "",
            )
        )

        return ket_qua

    except Exception as loi:

        st.session_state[
            "quant_ai_error"
        ] = str(
            loi
        )

        return []


# ============================================================
# RENDER
# ============================================================

def render_stock_analysis():

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                BRIAN STOCK · STOCK RESEARCH
            </div>

            <h1>
                Phân tích cổ phiếu
            </h1>

            <p>
                Phân tích giá, xu hướng, động lượng,
                thanh khoản, biến động và định lượng
                từ dữ liệu thị trường thực.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # NHẬP MÃ
    # ========================================================

    ma_mac_dinh = st.session_state.get(
        "stock_analysis_symbol",
        "HPG",
    )

    ma_nhap = st.text_input(
        "Mã cổ phiếu",
        value=ma_mac_dinh,
        placeholder="Ví dụ HPG, MSR, VNM",
        key="stock_analysis_input",
    )

    if st.button(
        "Tải dữ liệu",
        type="primary",
        key="stock_analysis_load",
    ):

        ma_sach = normalize_symbol(
            ma_nhap
        )

        if not ma_sach:

            st.error(
                "Vui lòng nhập mã cổ phiếu."
            )

        else:

            st.session_state[
                "stock_analysis_symbol"
            ] = ma_sach

            st.session_state.pop(
                "quant_result",
                None,
            )

            st.session_state.pop(
                "quant_ai_reason",
                None,
            )

            st.rerun()

    ma = normalize_symbol(
        st.session_state.get(
            "stock_analysis_symbol",
            ma_nhap,
        )
    )

    # ========================================================
    # TẢI DỮ LIỆU
    # ========================================================

    try:

        du_lieu = load_market_data(
            ma,
            "2y",
        )

    except Exception as loi:

        st.error(
            f"Không lấy được dữ liệu "
            f"{display_symbol(ma)}: {loi}"
        )

        return

    if (
        du_lieu is None
        or du_lieu.empty
    ):

        st.error(
            f"Không có dữ liệu cho "
            f"{display_symbol(ma)}."
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

    gia = _so(
        anh_chup.get("price")
    )

    thay_doi = _so(
        anh_chup.get("change_1d")
    )

    rsi = _so(
        anh_chup.get("rsi")
    )

    volume = _so(
        anh_chup.get("volume")
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Giá",
            (
                f"{gia:,.0f}"
                if gia is not None
                else "—"
            ),
        )

    with b:

        st.metric(
            "1D",
            (
                f"{thay_doi:+.2f}%"
                if thay_doi is not None
                else "—"
            ),
        )

    with c:

        st.metric(
            "RSI",
            (
                f"{rsi:.1f}"
                if rsi is not None
                else "—"
            ),
        )

    with d:

        st.metric(
            "Khối lượng",
            _dinh_dang_khoi_luong(
                volume
            ),
        )

    # ========================================================
    # CHỈ SỐ BỔ SUNG
    # ========================================================

    e, f, g, h = st.columns(4)

    sma20 = _so(
        anh_chup.get("sma20")
    )

    sma50 = _so(
        anh_chup.get("sma50")
    )

    macd = _so(
        anh_chup.get("macd")
    )

    bien_dong = _so(
        anh_chup.get(
            "volatility20"
        )
    )

    with e:

        st.metric(
            "Trung bình 20 phiên",
            (
                f"{sma20:,.0f}"
                if sma20 is not None
                else "—"
            ),
        )

    with f:

        st.metric(
            "Trung bình 50 phiên",
            (
                f"{sma50:,.0f}"
                if sma50 is not None
                else "—"
            ),
        )

    with g:

        st.metric(
            "MACD",
            (
                f"{macd:,.3f}"
                if macd is not None
                else "—"
            ),
        )

    with h:

        st.metric(
            "Biến động 20 phiên",
            (
                f"{bien_dong:.2f}%"
                if bien_dong is not None
                else "—"
            ),
        )

    # ========================================================
    # BIỂU ĐỒ
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Biểu đồ kỹ thuật</div>',
        unsafe_allow_html=True,
    )

    try:

        bieu_do = price_volume_chart(
            du_lieu
        )

        if bieu_do is not None:

            st.plotly_chart(
                bieu_do,
                width="stretch",
                config={
                    "displaylogo": False,
                },
            )

    except Exception as loi:

        st.warning(
            f"Không thể hiển thị biểu đồ: {loi}"
        )

    # ========================================================
    # QUANT
    # ========================================================

    st.markdown(
        '<div class="section-title">🧮 Phân tích định lượng</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Có thể tự chọn biến hoặc nhập yêu cầu "
        "bằng ngôn ngữ tự nhiên để AI lựa chọn biến."
    )

    # ========================================================
    # DANH SÁCH BIẾN
    # ========================================================

    try:

        danh_sach_bien = lay_danh_sach_bien(
            du_lieu,
            "Return",
        )

    except Exception as loi:

        st.error(
            f"Không thể tạo danh sách biến: {loi}"
        )

        return

    if not danh_sach_bien:

        st.warning(
            "Không có đủ biến để phân tích."
        )

        return

    # ========================================================
    # AI PROMPT
    # ========================================================

    st.markdown(
        "#### 🤖 AI tự chọn biến"
    )

    st.caption(
        "Ví dụ: "
        "\"Tìm các biến giải thích khả năng tăng giá ngắn hạn, "
        "ưu tiên động lượng và thanh khoản.\""
    )

    prompt_ai = st.text_area(
        "Yêu cầu cho AI",
        placeholder=(
            "Ví dụ: Hãy chọn biến để phân tích "
            "khả năng tăng giá trong 20 phiên tới, "
            "ưu tiên xu hướng, động lượng và khối lượng."
        ),
        height=110,
        key="quant_ai_prompt",
    )

    col_ai_1, col_ai_2 = st.columns(
        [1, 1]
    )

    with col_ai_1:

        if st.button(
            "🤖 AI chọn biến",
            type="secondary",
            key="quant_ai_choose",
        ):

            if not prompt_ai.strip():

                st.warning(
                    "Hãy nhập yêu cầu cho AI."
                )

            else:

                with st.spinner(
                    "AI đang phân tích yêu cầu..."
                ):

                    bien_ai = _ai_chon_bien(
                        prompt_ai,
                        danh_sach_bien,
                    )

                if bien_ai:

                    st.session_state[
                        "quant_ai_selected"
                    ] = bien_ai

                    # Đồng bộ với multiselect
                    st.session_state[
                        "quant_bien_giai_thich"
                    ] = [
                        _ten_hien_thi(x)
                        for x
                        in bien_ai
                    ]

                    st.success(
                        "AI đã chọn "
                        f"{len(bien_ai)} biến."
                    )

                    st.rerun()

                else:

                    api_key = _lay_google_api_key()

                    if not api_key:

                        st.warning(
                            "Chưa cấu hình GOOGLE_API_KEY. "
                            "Không có API key thì vẫn có thể "
                            "chọn biến thủ công."
                        )

                    else:

                        st.error(
                            "AI không chọn được biến. "
                            "Kiểm tra yêu cầu hoặc API key."
                        )

    with col_ai_2:

        if st.button(
            "🧹 Xóa lựa chọn AI",
            key="quant_ai_clear",
        ):

            st.session_state.pop(
                "quant_ai_selected",
                None,
            )

            st.session_state.pop(
                "quant_ai_reason",
                None,
            )

            st.session_state.pop(
                "quant_ai_error",
                None,
            )

            st.rerun()

    # ========================================================
    # LÝ DO AI
    # ========================================================

    ly_do_ai = st.session_state.get(
        "quant_ai_reason",
        "",
    )

    if ly_do_ai:

        st.info(
            f"AI giải thích: {ly_do_ai}"
        )

    # ========================================================
    # NHÓM BIẾN
    # ========================================================

    try:

        nhom_bien = lay_bien_theo_nhom(
            du_lieu,
            "Return",
        )

    except Exception:

        nhom_bien = {}

    with st.expander(
        "📚 Xem toàn bộ biến",
        expanded=False,
    ):

        for ten_nhom, cac_bien in nhom_bien.items():

            st.markdown(
                f"**{ten_nhom}**"
            )

            for bien in cac_bien:

                if bien in danh_sach_bien:

                    st.write(
                        f"• {_ten_hien_thi(bien)}"
                    )

    # ========================================================
    # ÁNH XẠ BIẾN
    # ========================================================

    anh_xa = {}

    for bien in danh_sach_bien:

        ten_hien_thi = _ten_hien_thi(
            bien
        )

        if ten_hien_thi in anh_xa:

            ten_hien_thi = (
                f"{ten_hien_thi} [{bien}]"
            )

        anh_xa[
            ten_hien_thi
        ] = bien

    danh_sach_hien_thi = list(
        anh_xa.keys()
    )

    # ========================================================
    # GIÁ TRỊ MẶC ĐỊNH
    # ========================================================

    bien_ai_da_chon = (
        st.session_state.get(
            "quant_ai_selected",
            [],
        )
    )

    if bien_ai_da_chon:

        mac_dinh_hien_thi = [
            _ten_hien_thi(
                bien
            )
            for bien
            in bien_ai_da_chon
            if bien in danh_sach_bien
        ]

    else:

        bien_mac_dinh = [
            "RSI",
            "MACD",
            "SMA20",
            "SMA50",
            "Volatility20",
            "Volume_Change",
            "Momentum20",
        ]

        mac_dinh_hien_thi = [
            _ten_hien_thi(
                bien
            )
            for bien
            in bien_mac_dinh
            if bien in danh_sach_bien
        ]

    # ========================================================
    # CHỌN BIẾN THỦ CÔNG
    # ========================================================

    bien_giai_thich_hien_thi = (
        st.multiselect(
            "Biến giải thích",
            options=danh_sach_hien_thi,
            default=mac_dinh_hien_thi,
            key="quant_bien_giai_thich",
        )
    )

    bien_giai_thich = [
        anh_xa[
            ten
        ]
        for ten
        in bien_giai_thich_hien_thi
        if ten in anh_xa
    ]

    # ========================================================
    # BIẾN MỤC TIÊU
    # ========================================================

    cac_muc_tieu = []

    if "Return" in du_lieu.columns:

        cac_muc_tieu.append(
            "Return"
        )

    if "ReturnPct" in du_lieu.columns:

        cac_muc_tieu.append(
            "ReturnPct"
        )

    if not cac_muc_tieu:

        st.error(
            "Không có biến mục tiêu."
        )

        return

    anh_xa_muc_tieu = {
        _ten_hien_thi(
            bien
        ): bien
        for bien
        in cac_muc_tieu
    }

    muc_tieu_hien_thi = st.selectbox(
        "Biến mục tiêu",
        options=list(
            anh_xa_muc_tieu.keys()
        ),
        key="quant_bien_muc_tieu",
    )

    bien_phu_thuoc = (
        anh_xa_muc_tieu[
            muc_tieu_hien_thi
        ]
    )

    # ========================================================
    # TÓM TẮT
    # ========================================================

    if bien_giai_thich_hien_thi:

        st.info(
            "Đã chọn "
            f"{len(bien_giai_thich_hien_thi)} biến: "
            + ", ".join(
                bien_giai_thich_hien_thi
            )
        )

    else:

        st.warning(
            "Chưa chọn biến giải thích."
        )

    # ========================================================
    # CHẠY MÔ HÌNH
    # ========================================================

    if st.button(
        "🚀 Chạy mô hình",
        type="primary",
        key="quant_run_model",
    ):

        if not bien_giai_thich:

            st.error(
                "Hãy chọn ít nhất một biến."
            )

        else:

            try:

                with st.spinner(
                    "Đang huấn luyện mô hình..."
                ):

                    ket_qua = run_quant(
                        du_lieu,
                        bien_phu_thuoc=(
                            bien_phu_thuoc
                        ),
                        bien_giai_thich=(
                            bien_giai_thich
                        ),
                    )

                if ket_qua is None:

                    st.error(
                        "Không đủ dữ liệu hợp lệ "
                        "để chạy mô hình."
                    )

                else:

                    st.session_state[
                        "quant_result"
                    ] = ket_qua

                    st.success(
                        "Mô hình đã chạy xong."
                    )

            except Exception as loi:

                st.error(
                    f"Lỗi mô hình: {loi}"
                )

    # ========================================================
    # KẾT QUẢ
    # ========================================================

    ket_qua = st.session_state.get(
        "quant_result"
    )

    if not ket_qua:
        return

    st.markdown(
        "#### 📌 Kết quả mô hình"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Số biến",
            ket_qua.get(
                "so_bien",
                0,
            ),
        )

    with b:

        st.metric(
            "Số quan sát",
            ket_qua.get(
                "so_dong",
                0,
            ),
        )

    with c:

        st.metric(
            "Huấn luyện",
            ket_qua.get(
                "so_huan_luyen",
                0,
            ),
        )

    with d:

        st.metric(
            "Kiểm tra",
            ket_qua.get(
                "so_kiem_tra",
                0,
            ),
        )

    # ========================================================
    # DỰ BÁO
    # ========================================================

    du_bao = _so(
        ket_qua.get(
            "du_bao_tiep_theo"
        )
    )

    if du_bao is not None:

        st.metric(
            "Dự báo lợi suất phiên kế tiếp",
            f"{du_bao * 100:+.2f}%",
        )

    # ========================================================
    # BIẾN SỬ DỤNG
    # ========================================================

    st.markdown(
        "#### 🔎 Biến mô hình sử dụng"
    )

    cac_bien = ket_qua.get(
        "danh_sach_bien",
        [],
    )

    bang_bien = pd.DataFrame(
        {
            "Biến": [
                _ten_hien_thi(
                    bien
                )
                for bien
                in cac_bien
            ],
            "Mã kỹ thuật": cac_bien,
        }
    )

    st.dataframe(
        bang_bien,
        width="stretch",
        hide_index=True,
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    ket_qua_rf = (
        ket_qua.get(
            "rung_ngau_nhien"
        )
    )

    if ket_qua_rf:

        st.markdown(
            "#### 🌲 Rừng ngẫu nhiên"
        )

        a, b, c = st.columns(3)

        with a:

            r2 = _so(
                ket_qua_rf.get(
                    "r2"
                )
            )

            st.metric(
                "R² kiểm tra",
                (
                    f"{r2:.3f}"
                    if r2 is not None
                    else "—"
                ),
            )

        with b:

            mae = _so(
                ket_qua_rf.get(
                    "mae"
                )
            )

            st.metric(
                "MAE",
                (
                    f"{mae * 100:.3f}%"
                    if mae is not None
                    else "—"
                ),
            )

        with c:

            rmse = _so(
                ket_qua_rf.get(
                    "rmse"
                )
            )

            st.metric(
                "RMSE",
                (
                    f"{rmse * 100:.3f}%"
                    if rmse is not None
                    else "—"
                ),
            )

        tam_quan_trong = (
            ket_qua_rf.get(
                "tam_quan_trong"
            )
        )

        if (
            tam_quan_trong is not None
            and isinstance(
                tam_quan_trong,
                pd.Series,
            )
        ):

            bang = (
                tam_quan_trong
                .sort_values(
                    ascending=False
                )
                .head(15)
                .rename(
                    index=_ten_hien_thi
                )
                .to_frame(
                    "Mức độ quan trọng (%)"
                )
            )

            bang[
                "Mức độ quan trọng (%)"
            ] *= 100

            st.bar_chart(
                bang
            )

    # ========================================================
    # OLS
    # ========================================================

    ket_qua_ols = (
        ket_qua.get(
            "hoi_quy"
        )
    )

    if ket_qua_ols:

        st.markdown(
            "#### 📐 Hồi quy tuyến tính"
        )

        a, b, c = st.columns(3)

        with a:

            r2 = _so(
                ket_qua_ols.get(
                    "r2"
                )
            )

            st.metric(
                "R²",
                (
                    f"{r2:.3f}"
                    if r2 is not None
                    else "—"
                ),
            )

        with b:

            r2_hieu_chinh = _so(
                ket_qua_ols.get(
                    "r2_hieu_chinh"
                )
            )

            st.metric(
                "R² hiệu chỉnh",
                (
                    f"{r2_hieu_chinh:.3f}"
                    if r2_hieu_chinh is not None
                    else "—"
                ),
            )

        with c:

            mae = _so(
                ket_qua_ols.get(
                    "mae"
                )
            )

            st.metric(
                "MAE",
                (
                    f"{mae * 100:.3f}%"
                    if mae is not None
                    else "—"
                ),
            )

        he_so = (
            ket_qua_ols.get(
                "he_so",
                {},
            )
        )

        p_value = (
            ket_qua_ols.get(
                "p_value",
                {},
            )
        )

        if he_so:

            bang_he_so = []

            for bien, gia_tri in he_so.items():

                bang_he_so.append(
                    {
                        "Biến": (
                            "Hằng số"
                            if bien == "const"
                            else _ten_hien_thi(
                                bien
                            )
                        ),
                        "Hệ số": gia_tri,
                        "P-value": p_value.get(
                            bien
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    bang_he_so
                ),
                width="stretch",
                hide_index=True,
            )
