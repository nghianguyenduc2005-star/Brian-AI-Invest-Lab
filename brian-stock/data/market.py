import pandas as pd
from datetime import datetime, timedelta

from vnstock import Market, Reference


# ============================================================
# KHỞI TẠO NGUỒN DỮ LIỆU
# ============================================================

thi_truong = Market()
tham_chieu = Reference()


# ============================================================
# CHUẨN HÓA MÃ CỔ PHIẾU
# ============================================================

def normalize_symbol(ma_co_phieu):
    if ma_co_phieu is None:
        return ""

    ma_co_phieu = str(ma_co_phieu).strip().upper()

    ky_tu_hop_le = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    ma_co_phieu = "".join(
        ky_tu
        for ky_tu in ma_co_phieu
        if ky_tu in ky_tu_hop_le
    )

    return ma_co_phieu


def display_symbol(ma_co_phieu):
    ma_co_phieu = normalize_symbol(ma_co_phieu)

    if not ma_co_phieu:
        return "—"

    return ma_co_phieu


# ============================================================
# LẤY DANH SÁCH TOÀN BỘ MÃ CỔ PHIẾU
# ============================================================

def load_symbol_list():

    try:
        du_lieu = tham_chieu.listing.all_symbols()

        if du_lieu is None:
            return []

        if not isinstance(du_lieu, pd.DataFrame):
            du_lieu = pd.DataFrame(du_lieu)

        if du_lieu.empty:
            return []

        if "symbol" not in du_lieu.columns:
            return []

        danh_sach = (
            du_lieu["symbol"]
            .dropna()
            .astype(str)
            .map(normalize_symbol)
            .tolist()
        )

        danh_sach = [
            ma
            for ma in danh_sach
            if ma
        ]

        return sorted(set(danh_sach))

    except Exception:
        return []


# ============================================================
# KIỂM TRA MÃ
# ============================================================

def is_valid_symbol(ma_co_phieu):

    ma_co_phieu = normalize_symbol(ma_co_phieu)

    if not ma_co_phieu:
        return False

    danh_sach = load_symbol_list()

    if not danh_sach:
        return True

    return ma_co_phieu in danh_sach


# ============================================================
# CHUẨN HÓA TÊN CỘT
# ============================================================

def _normalize_columns(du_lieu):

    if du_lieu is None:
        return pd.DataFrame()

    if not isinstance(du_lieu, pd.DataFrame):
        du_lieu = pd.DataFrame(du_lieu)

    if du_lieu.empty:
        return du_lieu

    anh_xa = {}

    for ten_cot in du_lieu.columns:

        ten = str(ten_cot).strip().lower()

        if ten in ["time", "datetime", "date", "timestamp"]:
            anh_xa[ten_cot] = "Thời gian"

        elif ten in ["open", "open_price"]:
            anh_xa[ten_cot] = "Mở cửa"

        elif ten in ["high", "high_price"]:
            anh_xa[ten_cot] = "Cao nhất"

        elif ten in ["low", "low_price"]:
            anh_xa[ten_cot] = "Thấp nhất"

        elif ten in ["close", "close_price"]:
            anh_xa[ten_cot] = "Đóng cửa"

        elif ten in ["volume", "total_volume"]:
            anh_xa[ten_cot] = "Khối lượng"

        elif ten in ["value", "value_traded"]:
            anh_xa[ten_cot] = "Giá trị giao dịch"

        elif ten in ["change", "price_change"]:
            anh_xa[ten_cot] = "Thay đổi"

        elif ten in ["change_percent", "percent_change"]:
            anh_xa[ten_cot] = "Phần trăm thay đổi"

    du_lieu = du_lieu.rename(
        columns=anh_xa
    )

    return du_lieu


# ============================================================
# CHUẨN HÓA GIÁ
# ============================================================

def _normalize_price(du_lieu):

    if du_lieu.empty:
        return du_lieu

    cac_cot_gia = [
        "Mở cửa",
        "Cao nhất",
        "Thấp nhất",
        "Đóng cửa",
    ]

    for ten_cot in cac_cot_gia:

        if ten_cot not in du_lieu.columns:
            continue

        try:

            gia_tri = pd.to_numeric(
                du_lieu[ten_cot],
                errors="coerce"
            )

            gia_trung_binh = gia_tri.median()

            if (
                pd.notna(gia_trung_binh)
                and gia_trung_binh < 1000
            ):
                du_lieu[ten_cot] = gia_tri * 1000
            else:
                du_lieu[ten_cot] = gia_tri

        except Exception:
            continue

    return du_lieu


# ============================================================
# THÊM CHỈ BÁO PHÂN TÍCH
# ============================================================

def add_indicators(du_lieu):

    if du_lieu is None or du_lieu.empty:
        return pd.DataFrame()

    du_lieu = du_lieu.copy()

    if "Đóng cửa" not in du_lieu.columns:
        return du_lieu

    gia_dong_cua = pd.to_numeric(
        du_lieu["Đóng cửa"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # TRUNG BÌNH ĐỘNG
    # --------------------------------------------------------

    du_lieu["Trung bình động 5"] = (
        gia_dong_cua.rolling(5).mean()
    )

    du_lieu["Trung bình động 10"] = (
        gia_dong_cua.rolling(10).mean()
    )

    du_lieu["Trung bình động 20"] = (
        gia_dong_cua.rolling(20).mean()
    )

    du_lieu["Trung bình động 50"] = (
        gia_dong_cua.rolling(50).mean()
    )

    du_lieu["Trung bình động 100"] = (
        gia_dong_cua.rolling(100).mean()
    )

    du_lieu["Trung bình động 200"] = (
        gia_dong_cua.rolling(200).mean()
    )

    # --------------------------------------------------------
    # THAY ĐỔI GIÁ
    # --------------------------------------------------------

    du_lieu["Thay đổi giá"] = (
        gia_dong_cua.diff()
    )

    du_lieu["Phần trăm thay đổi"] = (
        gia_dong_cua.pct_change() * 100
    )

    # --------------------------------------------------------
    # BIẾN ĐỘNG
    # --------------------------------------------------------

    du_lieu["Biến động 20 ngày"] = (
        du_lieu["Phần trăm thay đổi"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # CAO NHẤT / THẤP NHẤT
    # --------------------------------------------------------

    du_lieu["Cao nhất 20 ngày"] = (
        gia_dong_cua
        .rolling(20)
        .max()
    )

    du_lieu["Thấp nhất 20 ngày"] = (
        gia_dong_cua
        .rolling(20)
        .min()
    )

    du_lieu["Cao nhất 52 tuần"] = (
        gia_dong_cua
        .rolling(252)
        .max()
    )

    du_lieu["Thấp nhất 52 tuần"] = (
        gia_dong_cua
        .rolling(252)
        .min()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    thay_doi = gia_dong_cua.diff()

    muc_tang = thay_doi.clip(lower=0)

    muc_giam = -thay_doi.clip(upper=0)

    tang_trung_binh = (
        muc_tang
        .rolling(14)
        .mean()
    )

    giam_trung_binh = (
        muc_giam
        .rolling(14)
        .mean()
    )

    ty_le_tang_giam = (
        tang_trung_binh /
        giam_trung_binh.replace(0, pd.NA)
    )

    du_lieu["RSI"] = (
        100 -
        (
            100 /
            (1 + ty_le_tang_giam)
        )
    )

    # --------------------------------------------------------
    # KHỐI LƯỢNG
    # --------------------------------------------------------

    if "Khối lượng" in du_lieu.columns:

        khoi_luong = pd.to_numeric(
            du_lieu["Khối lượng"],
            errors="coerce"
        )

        du_lieu["Khối lượng trung bình 20"] = (
            khoi_luong
            .rolling(20)
            .mean()
        )

        trung_binh_khoi_luong = (
            khoi_luong
            .rolling(20)
            .mean()
        )

        du_lieu["Tỷ lệ khối lượng"] = (
            khoi_luong /
            trung_binh_khoi_luong.replace(
                0,
                pd.NA
            )
        )

    # --------------------------------------------------------
    # KHOẢNG DAO ĐỘNG TRONG NGÀY
    # --------------------------------------------------------

    if (
        "Cao nhất" in du_lieu.columns
        and "Thấp nhất" in du_lieu.columns
    ):

        cao_nhat = pd.to_numeric(
            du_lieu["Cao nhất"],
            errors="coerce"
        )

        thap_nhat = pd.to_numeric(
            du_lieu["Thấp nhất"],
            errors="coerce"
        )

        du_lieu["Biên độ trong ngày"] = (
            cao_nhat - thap_nhat
        )

        du_lieu["Phần trăm biên độ"] = (
            (
                cao_nhat - thap_nhat
            )
            / gia_dong_cua.replace(
                0,
                pd.NA
            )
        ) * 100

    return du_lieu


# ============================================================
# LẤY LỊCH SỬ GIÁ
# ============================================================

def load_market_data(
    ma_co_phieu,
    so_ngay=365
):

    ma_co_phieu = normalize_symbol(
        ma_co_phieu
    )

    if not ma_co_phieu:
        raise ValueError(
            "Mã cổ phiếu không hợp lệ."
        )

    ngay_ket_thuc = datetime.now()

    ngay_bat_dau = (
        ngay_ket_thuc -
        timedelta(days=so_ngay)
    )

    loi_cuoi = None

    try:

        du_lieu = (
            thi_truong.equity.ohlcv(
                symbol=ma_co_phieu,
                start=ngay_bat_dau.strftime(
                    "%Y-%m-%d"
                ),
                end=ngay_ket_thuc.strftime(
                    "%Y-%m-%d"
                ),
                interval="1D"
            )
        )

        if du_lieu is None:
            raise RuntimeError(
                "Nguồn dữ liệu trả về rỗng."
            )

        du_lieu = _normalize_columns(
            du_lieu
        )

        if du_lieu.empty:
            raise RuntimeError(
                f"Không có dữ liệu lịch sử cho {ma_co_phieu}."
            )

        du_lieu = _normalize_price(
            du_lieu
        )

        du_lieu = add_indicators(
            du_lieu
        )

        return du_lieu

    except Exception as loi:

        loi_cuoi = loi

    raise RuntimeError(
        f"Không lấy được dữ liệu {ma_co_phieu}: {loi_cuoi}"
    )


# ============================================================
# LẤY GIÁ HIỆN TẠI
# ============================================================

def load_current_quote(ma_co_phieu):

    ma_co_phieu = normalize_symbol(
        ma_co_phieu
    )

    if not ma_co_phieu:
        return {}

    try:

        du_lieu = (
            thi_truong.equity.quote(
                symbol=ma_co_phieu
            )
        )

        if du_lieu is None:
            return {}

        if not isinstance(
            du_lieu,
            pd.DataFrame
        ):
            du_lieu = pd.DataFrame(
                du_lieu
            )

        if du_lieu.empty:
            return {}

        du_lieu = _normalize_columns(
            du_lieu
        )

        dong_cuoi = du_lieu.iloc[-1]

        ket_qua = {}

        for ten_cot in du_lieu.columns:

            gia_tri = dong_cuoi[ten_cot]

            if pd.notna(gia_tri):
                ket_qua[ten_cot] = gia_tri

        return ket_qua

    except Exception:
        return {}


# ============================================================
# LẤY VN-INDEX
# ============================================================

def load_vnindex_data():

    try:

        du_lieu = (
            thi_truong.index.ohlcv(
                symbol="VNINDEX",
                start=(
                    datetime.now() -
                    timedelta(days=60)
                ).strftime(
                    "%Y-%m-%d"
                ),
                end=datetime.now().strftime(
                    "%Y-%m-%d"
                ),
                interval="1D"
            )
        )

        if du_lieu is None:
            return pd.DataFrame()

        if not isinstance(
            du_lieu,
            pd.DataFrame
        ):
            du_lieu = pd.DataFrame(
                du_lieu
            )

        du_lieu = _normalize_columns(
            du_lieu
        )

        du_lieu = _normalize_price(
            du_lieu
        )

        du_lieu = add_indicators(
            du_lieu
        )

        return du_lieu

    except Exception:
        return pd.DataFrame()
