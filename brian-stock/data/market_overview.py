
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
def loc_bang_gia(
    bang_gia,
    san="Tất cả",
    tu_khoa="",
    nganh="Tất cả",
    huong="Tất cả",
):
    """
    Lọc bảng giá toàn thị trường.

    san:
        Tất cả / HOSE / HNX / UPCOM

    tu_khoa:
        Mã cổ phiếu hoặc tên doanh nghiệp

    nganh:
        Tên ngành

    huong:
        Tất cả / Tăng / Đứng giá / Giảm
    """

    if (
        bang_gia is None
        or not isinstance(
            bang_gia,
            pd.DataFrame,
        )
        or bang_gia.empty
    ):
        return pd.DataFrame()

    df = bang_gia.copy()

    # --------------------------------------------------------
    # Lọc sàn
    # --------------------------------------------------------

    if (
        san
        and san != "Tất cả"
        and "san" in df.columns
    ):

        df = df[
            df["san"]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq(
                str(san).upper()
            )
        ].copy()

    # --------------------------------------------------------
    # Lọc ngành
    # --------------------------------------------------------

    if (
        nganh
        and nganh != "Tất cả"
        and "ten_nganh" in df.columns
    ):

        df = df[
            df["ten_nganh"]
            .fillna("")
            .astype(str)
            .eq(
                str(nganh)
            )
        ].copy()

    # --------------------------------------------------------
    # Tìm mã / tên doanh nghiệp
    # --------------------------------------------------------

    tu_khoa = str(
        tu_khoa or ""
    ).strip()

    if tu_khoa:

        tu_khoa_upper = (
            tu_khoa.upper()
        )

        mask_ma = pd.Series(
            False,
            index=df.index,
        )

        mask_ten = pd.Series(
            False,
            index=df.index,
        )

        if "ma" in df.columns:

            mask_ma = (
                df["ma"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.contains(
                    tu_khoa_upper,
                    regex=False,
                    na=False,
                )
            )

        if (
            "ten_doanh_nghiep"
            in df.columns
        ):

            mask_ten = (
                df[
                    "ten_doanh_nghiep"
                ]
                .fillna("")
                .astype(str)
                .str.contains(
                    tu_khoa,
                    case=False,
                    regex=False,
                    na=False,
                )
            )

        df = df[
            mask_ma | mask_ten
        ].copy()

    # --------------------------------------------------------
    # Lọc tăng / giảm
    # --------------------------------------------------------

    if (
        huong
        and huong != "Tất cả"
        and "trang_thai" in df.columns
    ):

        df = df[
            df["trang_thai"]
            .fillna("")
            .astype(str)
            .eq(
                str(huong)
            )
        ].copy()

    return df.reset_index(
        drop=True
    )
