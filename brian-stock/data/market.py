import pandas as pd
import numpy as np

try:
    from vnstock import Market, Reference
    _CO_VNSTOCK_HIEN_DAI = True
except Exception:
    Market = None
    Reference = None
    _CO_VNSTOCK_HIEN_DAI = False


# ============================================================
# CHUẨN HÓA MÃ CỔ PHIẾU
# ============================================================

def chuan_hoa_ma(ma):
    if ma is None:
        return ""

    ma = str(ma).strip().upper()
    ma = "".join(ky_tu for ky_tu in ma if ky_tu.isalnum())

    return ma


def hien_thi_ma(ma):
    return chuan_hoa_ma(ma)


# Giữ tên hàm cũ để các tệp khác không bị lỗi nhập hàm.
normalize_symbol = chuan_hoa_ma
display_symbol = hien_thi_ma


# ============================================================
# CHUẨN HÓA DỮ LIỆU GIÁ
# ============================================================

def _tim_cot(du_lieu, danh_sach_ten):
    if du_lieu is None or du_lieu.empty:
        return None

    ban_do = {
        str(cot).strip().lower(): cot
        for cot in du_lieu.columns
    }

    for ten in danh_sach_ten:
        if ten.lower() in ban_do:
            return ban_do[ten.lower()]

    return None


def _chuan_hoa_ohlcv(du_lieu):
    if du_lieu is None or du_lieu.empty:
        return pd.DataFrame()

    du_lieu = du_lieu.copy()

    cot_ngay = _tim_cot(
        du_lieu,
        [
            "time",
            "date",
            "datetime",
            "timestamp",
            "ngày",
            "thời gian",
        ],
    )

    cot_mo = _tim_cot(
        du_lieu,
        [
            "open",
            "giá mở cửa",
            "mở cửa",
        ],
    )

    cot_cao = _tim_cot(
        du_lieu,
        [
            "high",
            "giá cao nhất",
            "cao nhất",
        ],
    )

    cot_thap = _tim_cot(
        du_lieu,
        [
            "low",
            "giá thấp nhất",
            "thấp nhất",
        ],
    )

    cot_dong = _tim_cot(
        du_lieu,
        [
            "close",
            "giá đóng cửa",
            "đóng cửa",
        ],
    )

    cot_khoi_luong = _tim_cot(
        du_lieu,
        [
            "volume",
            "khối lượng",
            "khoiluong",
        ],
    )

    cot_gia_tri = _tim_cot(
        du_lieu,
        [
            "value",
            "trading_value",
            "giá trị",
            "giá trị giao dịch",
        ],
    )

    cot_dieu_chinh = _tim_cot(
        du_lieu,
        [
            "adjust",
            "adjusted",
            "giá điều chỉnh",
        ],
    )

    du_lieu_chuan = pd.DataFrame(index=du_lieu.index)

    if cot_ngay is not None:
        du_lieu_chuan["ngày"] = pd.to_datetime(
            du_lieu[cot_ngay],
            errors="coerce",
        )
    else:
        du_lieu_chuan["ngày"] = pd.NaT

    if cot_mo is not None:
        du_lieu_chuan["mở_cửa"] = pd.to_numeric(
            du_lieu[cot_mo],
            errors="coerce",
        )

    if cot_cao is not None:
        du_lieu_chuan["cao_nhất"] = pd.to_numeric(
            du_lieu[cot_cao],
            errors="coerce",
        )

    if cot_thap is not None:
        du_lieu_chuan["thấp_nhất"] = pd.to_numeric(
            du_lieu[cot_thap],
            errors="coerce",
        )

    if cot_dong is not None:
        du_lieu_chuan["đóng_cửa"] = pd.to_numeric(
            du_lieu[cot_dong],
            errors="coerce",
        )

    if cot_khoi_luong is not None:
        du_lieu_chuan["khối_lượng"] = pd.to_numeric(
            du_lieu[cot_khoi_luong],
            errors="coerce",
        )

    if cot_gia_tri is not None:
        du_lieu_chuan["giá_trị"] = pd.to_numeric(
            du_lieu[cot_gia_tri],
            errors="coerce",
        )

    if cot_dieu_chinh is not None:
        du_lieu_chuan["giá_điều_chỉnh"] = pd.to_numeric(
            du_lieu[cot_dieu_chinh],
            errors="coerce",
        )

    du_lieu_chuan = du_lieu_chuan.dropna(
        subset=["đóng_cửa"],
        how="all",
    )

    du_lieu_chuan = du_lieu_chuan.sort_values(
        "ngày",
        ignore_index=True,
    )

    return du_lieu_chuan


# ============================================================
# LẤY TOÀN BỘ DANH SÁCH MÃ
# ============================================================

def lay_danh_sach_ma():
    """
    Lấy toàn bộ mã cổ phiếu đang được hệ thống dữ liệu cung cấp.
    Không sử dụng danh sách mã viết tay.
    """

    if not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame(
            columns=[
                "mã",
                "tên_doanh_nghiệp",
                "sàn",
            ]
        )

    try:
        tham_chieu = Reference()
        du_lieu = tham_chieu.equity.list()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu).copy()

        cot_ma = _tim_cot(
            du_lieu,
            [
                "symbol",
                "ticker",
                "code",
                "mã",
            ],
        )

        cot_ten = _tim_cot(
            du_lieu,
            [
                "organ_name",
                "company_name",
                "companyname",
                "tên doanh nghiệp",
                "tên công ty",
            ],
        )

        cot_san = _tim_cot(
            du_lieu,
            [
                "exchange",
                "sàn",
                "board",
            ],
        )

        ket_qua = pd.DataFrame()

        if cot_ma is not None:
            ket_qua["mã"] = (
                du_lieu[cot_ma]
                .astype(str)
                .str.upper()
                .str.strip()
            )

        if cot_ten is not None:
            ket_qua["tên_doanh_nghiệp"] = (
                du_lieu[cot_ten]
                .astype(str)
                .str.strip()
            )
        else:
            ket_qua["tên_doanh_nghiệp"] = ""

        if cot_san is not None:
            ket_qua["sàn"] = (
                du_lieu[cot_san]
                .astype(str)
                .str.upper()
                .str.strip()
            )
        else:
            ket_qua["sàn"] = ""

        if "mã" in ket_qua.columns:
            ket_qua = ket_qua[
                ket_qua["mã"].str.match(
                    r"^[A-Z0-9]{2,10}$",
                    na=False,
                )
            ]

            ket_qua = ket_qua.drop_duplicates(
                subset=["mã"]
            )

        return ket_qua.reset_index(drop=True)

    except Exception:
        return pd.DataFrame(
            columns=[
                "mã",
                "tên_doanh_nghiệp",
                "sàn",
            ]
        )


def lay_toan_bo_ma():
    return lay_danh_sach_ma()


# ============================================================
# LẤY DỮ LIỆU LỊCH SỬ
# ============================================================

def lay_du_lieu_lich_su(
    ma,
    ngay_bat_dau=None,
    ngay_ket_thuc=None,
    khung_thoi_gian="1D",
):
    ma = chuan_hoa_ma(ma)

    if not ma:
        return pd.DataFrame()

    if not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        doi_tuong = thi_truong.equity(ma)

        tham_so = {}

        if ngay_bat_dau:
            tham_so["start"] = ngay_bat_dau

        if ngay_ket_thuc:
            tham_so["end"] = ngay_ket_thuc

        if khung_thoi_gian:
            tham_so["interval"] = khung_thoi_gian

        du_lieu = doi_tuong.ohlcv(**tham_so)

        du_lieu = _chuan_hoa_ohlcv(du_lieu)

        if du_lieu.empty:
            return du_lieu

        du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


def load_market_data(
    symbol,
    start=None,
    end=None,
    interval="1D",
):
    return lay_du_lieu_lich_su(
        ma=symbol,
        ngay_bat_dau=start,
        ngay_ket_thuc=end,
        khung_thoi_gian=interval,
    )


# ============================================================
# GIÁ HIỆN TẠI
# ============================================================

def lay_gia_hien_tai(ma):
    ma = chuan_hoa_ma(ma)

    if not ma:
        return pd.DataFrame()

    if not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).quote()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if du_lieu.empty:
            return du_lieu

        du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


def lay_bang_gia(ma):
    return lay_gia_hien_tai(ma)


# ============================================================
# TỔNG HỢP DỮ LIỆU CỔ PHIẾU
# ============================================================

def lay_tong_hop(ma):
    ma = chuan_hoa_ma(ma)

    if not ma:
        return pd.DataFrame()

    if not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).summary()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# GIAO DỊCH TRONG PHIÊN
# ============================================================

def lay_giao_dich_trong_phien(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).trades()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# SỔ LỆNH
# ============================================================

def lay_so_lenh(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).order_book()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# THỐNG KÊ PHIÊN
# ============================================================

def lay_thong_ke_phien(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).session_stats()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# DÒNG TIỀN NƯỚC NGOÀI
# ============================================================

def lay_dong_tien_nuoc_ngoai(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).foreign_flow()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# DÒNG TIỀN TỰ DOANH
# ============================================================

def lay_dong_tien_tu_doanh(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).proprietary_flow()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# GIAO DỊCH THỎA THUẬN
# ============================================================

def lay_giao_dich_thoa_thuan(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).block_trades()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# GIAO DỊCH LÔ LẺ
# ============================================================

def lay_giao_dich_lo_le(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).odd_lot()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# PHÂN BỐ KHỐI LƯỢNG THEO GIÁ
# ============================================================

def lay_phan_bo_khoi_luong(ma):
    ma = chuan_hoa_ma(ma)

    if not ma or not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.equity(ma).volume_profile()

        if du_lieu is None:
            return pd.DataFrame()

        du_lieu = pd.DataFrame(du_lieu)

        if not du_lieu.empty:
            du_lieu["mã"] = ma

        return du_lieu

    except Exception:
        return pd.DataFrame()


# ============================================================
# CHỈ SỐ THỊ TRƯỜNG
# ============================================================

def lay_du_lieu_chi_so(ma_chi_so="VNINDEX"):
    ma_chi_so = chuan_hoa_ma(ma_chi_so)

    if not _CO_VNSTOCK_HIEN_DAI:
        return pd.DataFrame()

    try:
        thi_truong = Market()

        du_lieu = thi_truong.index(ma_chi_so).ohlcv()

        if du_lieu is None:
            return pd.DataFrame()

        return _chuan_hoa_ohlcv(du_lieu)

    except Exception:
        return pd.DataFrame()


# ============================================================
# KIỂM TRA KẾT NỐI
# ============================================================

def kiem_tra_nguon_du_lieu():
    try:
        danh_sach = lay_danh_sach_ma()

        if danh_sach.empty:
            return {
                "trạng_thái": False,
                "thông_báo": "Không lấy được danh sách mã cổ phiếu.",
            }

        return {
            "trạng_thái": True,
            "thông_báo": (
                f"Lấy được {len(danh_sach):,} mã "
                "từ nguồn dữ liệu."
            ),
        }

    except Exception as loi:
        return {
            "trạng_thái": False,
            "thông_báo": str(loi),
        }


# ============================================================
# TƯƠNG THÍCH VỚI CÁC TỆP CŨ
# ============================================================

def lay_lich_su(
    ma,
    ngay_bat_dau=None,
    ngay_ket_thuc=None,
    khung_thoi_gian="1D",
):
    return lay_du_lieu_lich_su(
        ma=ma,
        ngay_bat_dau=ngay_bat_dau,
        ngay_ket_thuc=ngay_ket_thuc,
        khung_thoi_gian=khung_thoi_gian,
    )


def fetch_market_data(
    ma,
    ngay_bat_dau=None,
    ngay_ket_thuc=None,
):
    return lay_du_lieu_lich_su(
        ma=ma,
        ngay_bat_dau=ngay_bat_dau,
        ngay_ket_thuc=ngay_ket_thuc,
        khung_thoi_gian="1D",
    )
