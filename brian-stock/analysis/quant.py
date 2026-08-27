# ============================================================
# QUANT - TỰ CHỌN BIẾN
# ============================================================

st.markdown(
    "### 🧮 Định lượng – tự chọn biến"
)

ten_bien_tieng_viet = {
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

    "RSI": "Sức mạnh tương đối (RSI)",
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
}


# ------------------------------------------------------------
# Lấy toàn bộ biến hợp lệ
# ------------------------------------------------------------

cac_bien = []

for ten_bien in du_lieu.columns:

    if ten_bien == "Target":
        continue

    try:

        so_luong_hop_le = (
            pd.to_numeric(
                du_lieu[ten_bien],
                errors="coerce",
            )
            .notna()
            .sum()
        )

    except Exception:
        continue

    if so_luong_hop_le >= 40:

        if ten_bien not in cac_bien:
            cac_bien.append(
                ten_bien
            )


# ------------------------------------------------------------
# Tạo tên hiển thị tiếng Việt
# ------------------------------------------------------------

ten_hien_thi_to_code = {
    ten_bien_tieng_viet.get(
        ten_bien,
        ten_bien,
    ): ten_bien
    for ten_bien in cac_bien
}

danh_sach_hien_thi = list(
    ten_hien_thi_to_code.keys()
)


# ------------------------------------------------------------
# Biến giải thích
# ------------------------------------------------------------

bien_giai_thich_hien_thi = st.multiselect(
    "Biến giải thích",
    options=danh_sach_hien_thi,
    default=[
        ten_bien_tieng_viet.get(
            x,
            x,
        )
        for x in [
            "RSI",
            "MACD",
            "SMA20",
            "SMA50",
        ]
        if x in cac_bien
    ],
    key="quant_bien_giai_thich",
)

bien_giai_thich = [
    ten_hien_thi_to_code[x]
    for x in bien_giai_thich_hien_thi
    if x in ten_hien_thi_to_code
]


# ------------------------------------------------------------
# Biến phụ thuộc
# ------------------------------------------------------------

bien_phu_thuoc_hien_thi = st.selectbox(
    "Biến phụ thuộc",
    options=[
        ten_bien_tieng_viet.get(
            "Return",
            "Lợi suất",
        )
    ],
    index=0,
    key="quant_bien_phu_thuoc",
)

bien_phu_thuoc = "Return"


# ------------------------------------------------------------
# Hiển thị lựa chọn
# ------------------------------------------------------------

if bien_giai_thich:

    st.caption(
        "Đang chọn: "
        + ", ".join(
            bien_giai_thich_hien_thi
        )
    )

else:

    st.warning(
        "Hãy chọn ít nhất một biến giải thích."
    )


# ------------------------------------------------------------
# CHẠY MÔ HÌNH
# ------------------------------------------------------------

if st.button(
    "🚀 Chạy mô hình",
    type="primary",
    key="quant_chay_mo_hinh",
):

    if not bien_giai_thich:

        st.error(
            "Chưa chọn biến giải thích."
        )

    else:

        try:

            from analysis.quant import run_quant

            ket_qua_quant = run_quant(
                du_lieu,
                bien_phu_thuoc=bien_phu_thuoc,
            )

            if ket_qua_quant is None:

                st.warning(
                    "Không đủ dữ liệu hợp lệ để xây dựng mô hình."
                )

            else:

                st.success(
                    "Mô hình đã chạy thành công."
                )

                # ============================================
                # THỐNG KÊ
                # ============================================

                a, b, c, d = st.columns(4)

                with a:

                    st.metric(
                        "Số biến",
                        ket_qua_quant.get(
                            "so_bien",
                            0,
                        ),
                    )

                with b:

                    st.metric(
                        "Số quan sát",
                        ket_qua_quant.get(
                            "so_dong",
                            0,
                        ),
                    )

                with c:

                    st.metric(
                        "Huấn luyện",
                        ket_qua_quant.get(
                            "so_huan_luyen",
                            0,
                        ),
                    )

                with d:

                    st.metric(
                        "Kiểm tra",
                        ket_qua_quant.get(
                            "so_kiem_tra",
                            0,
                        ),
                    )

                # ============================================
                # DỰ BÁO
                # ============================================

                du_bao = ket_qua_quant.get(
                    "du_bao_tiep_theo"
                )

                if du_bao is not None:

                    st.metric(
                        "Dự báo lợi suất kế tiếp",
                        f"{du_bao * 100:+.2f}%",
                    )

                # ============================================
                # DANH SÁCH BIẾN
                # ============================================

                st.markdown(
                    "#### Các biến được mô hình sử dụng"
                )

                danh_sach_bien_vn = (
                    ket_qua_quant.get(
                        "danh_sach_bien_tieng_viet",
                        [],
                    )
                )

                for bien in danh_sach_bien_vn:

                    st.write(
                        f"• {bien}"
                    )

                # ============================================
                # RANDOM FOREST
                # ============================================

                ket_qua_rung = (
                    ket_qua_quant.get(
                        "rung_ngau_nhien"
                    )
                )

                if ket_qua_rung:

                    st.markdown(
                        "#### 🌲 Rừng ngẫu nhiên"
                    )

                    r2 = ket_qua_rung.get(
                        "r2"
                    )

                    mae = ket_qua_rung.get(
                        "mae"
                    )

                    x, y = st.columns(2)

                    with x:

                        st.metric(
                            "R² kiểm tra",
                            (
                                f"{r2:.3f}"
                                if r2 is not None
                                else "—"
                            ),
                        )

                    with y:

                        st.metric(
                            "MAE",
                            (
                                f"{mae * 100:.3f}%"
                                if mae is not None
                                else "—"
                            ),
                        )

                    tam_quan_trong = (
                        ket_qua_rung.get(
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
                            .head(15)
                            .sort_values(
                                ascending=True
                            )
                            .rename(
                                index=ten_bien_tieng_viet
                            )
                            .to_frame(
                                "Mức độ quan trọng"
                            )
                        )

                        bang[
                            "Mức độ quan trọng"
                        ] *= 100

                        st.bar_chart(
                            bang
                        )

                # ============================================
                # HỒI QUY
                # ============================================

                ket_qua_hoi_quy = (
                    ket_qua_quant.get(
                        "hoi_quy"
                    )
                )

                if ket_qua_hoi_quy:

                    st.markdown(
                        "#### 📐 Hồi quy"
                    )

                    r2_ols = (
                        ket_qua_hoi_quy.get(
                            "r2"
                        )
                    )

                    r2_dieu_chinh = (
                        ket_qua_hoi_quy.get(
                            "r2_hieu_chinh"
                        )
                    )

                    x, y = st.columns(2)

                    with x:

                        st.metric(
                            "R²",
                            (
                                f"{r2_ols:.3f}"
                                if r2_ols is not None
                                else "—"
                            ),
                        )

                    with y:

                        st.metric(
                            "R² hiệu chỉnh",
                            (
                                f"{r2_dieu_chinh:.3f}"
                                if r2_dieu_chinh is not None
                                else "—"
                            ),
                        )

        except Exception as loi:

            st.error(
                f"Lỗi mô hình định lượng: {loi}"
            )
