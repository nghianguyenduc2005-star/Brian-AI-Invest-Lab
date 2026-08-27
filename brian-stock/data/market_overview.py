
def lay_market_overview(
    force_reload: bool = False,
):
    """
    Trả về toàn bộ dữ liệu cần cho trang Thị trường.

    Dữ liệu được dùng chung để page không phải
    gọi lại nguồn nhiều lần.
    """

    bang_gia = lay_bang_gia_toan_thi_truong(
        force_reload=force_reload
    )

    thong_ke = thong_ke_thi_truong(
        bang_gia
    )

    diem_tam_ly = tinh_diem_tam_ly(
        bang_gia
    )

    return {
        "bang_gia": bang_gia,

        "universe": lay_universe(
            force_reload=force_reload
        ),

        "thong_ke": thong_ke,

        "tam_ly": diem_tam_ly,

        "nhan_tam_ly": nhan_diem_tam_ly(
            diem_tam_ly
        ),

        "theo_san": thong_ke_theo_san(
            bang_gia
        ),

        "theo_nganh": thong_ke_theo_nganh(
            bang_gia
        ),

        "top_tang": top_tang(
            bang_gia
        ),

        "top_giam": top_giam(
            bang_gia
        ),

        "top_khoi_luong": top_khoi_luong(
            bang_gia
        ),

        "top_gia_tri": top_gia_tri_giao_dich(
            bang_gia
        ),

        "nuoc_ngoai": thong_ke_nuoc_ngoai(
            bang_gia
        ),

        "nguon": thong_tin_nguon(
            bang_gia
        ),
    }
