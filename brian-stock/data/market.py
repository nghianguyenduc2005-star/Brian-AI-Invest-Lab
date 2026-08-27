```python
import pandas as pd
from datetime import datetime, timedelta

from vnstock_data import Market, Reference


# ============================================================
# KHỞI TẠO NGUỒN DỮ LIỆU
# ============================================================

_thi_truong = Market()
_tham_chieu = Reference()


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
# LẤY TOÀN BỘ DANH SÁCH CỔ PHIẾU
# ============================================================

def load_symbol_list():
    try:
        du_lieu = _tham_chieu.equity.list()

        if du_lieu is None:
            return []

        if not isinstance(du_lieu, pd.DataFrame):
            du_lieu = pd.DataFrame(du_lieu)

        if du_lieu.empty:
            return []

        cot_ma = None

        for ten_cot in du_lieu.columns:
            ten_cot_chuan = str(ten_cot).strip().lower()

            if ten_cot_chuan in [
                "symbol",
                "ticker",
                "code",
                "ma",
                "mã",
                "mã ck",
                "ma_ck",
            ]:
                cot_ma = ten_cot
                break

        if cot_ma is None:
            return []

        danh_sach = (
            du_lieu[cot_ma]
            .dropna()
            .astype(str)
            .map(normalize_symbol)
        )

        danh_sach = [
            ma
            for ma in danh_sach.tolist()
            if ma
        ]

        return sorted(list(set(danh_sach)))

    except Exception:
        return []


# ============================================================
# KIỂM TRA MÃ CỔ PHIẾU
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
# CHUYỂN TÊN CỘT VỀ CHUẨN TIẾNG VIỆT
# ============================================================

def _normalize_columns(du_lieu):

    if du_lieu is None:
        return pd.DataFrame()

    if not isinstance(du_lieu, pd.DataFrame):
        du_lieu = pd.DataFrame(du_lieu)

    if du_lieu.empty:
        return du_lieu

    anh_xa = {}

    for cot in du_lieu.columns:

        ten = str(cot).strip().lower()

        if ten in ["time", "datetime", "date", "timestamp"]:
            anh_xa[cot] = "Thời gian"

        elif ten in ["open", "open_price"]:
            anh_xa[cot] = "Mở cửa"

        elif ten in ["high", "high_price"]:
            anh_xa[cot] = "Cao nhất"

        elif ten in ["low", "low_price"]:
            anh_xa[cot] = "Thấp nhất"

        elif ten in ["close", "close_price"]:
            anh_xa[cot] = "Đóng cửa"

        elif ten in ["volume", "total_volume"]:
            anh_xa[cot] = "Khối lượng"

        elif ten in ["value", "value_traded"]:
            anh_xa[cot] = "Giá trị giao dịch"

        elif ten in ["change", "price_change"]:
            anh_xa[cot] = "Thay đổi"

        elif ten in ["change_percent", "percent_change"]:
            anh_xa[cot] = "Phần trăm thay đổi"

        elif ten in ["symbol", "ticker"]:
            anh_xa[cot] = "Mã cổ phiếu"

    du_lieu = du_lieu.rename(columns=anh_xa)

    return du_lieu


# ============================================================
# CHUẨN HÓA ĐƠN VỊ GIÁ
# ============================================================

def _normalize_price(du_lieu):

    if du_lieu.empty:
        return du_lieu

    for cot in [
        "Mở cửa",
        "Cao nhất",
        "Thấp nhất",
        "Đóng cửa",
    ]:

        if cot not in du_lieu.columns:
            continue

        try:

            gia_trung_binh = pd.to_numeric(
                du_lieu[cot],
                errors="coerce"
            ).median()

            if pd.notna(gia_trung_binh) and gia_trung_binh < 1000:

                du_lieu[cot] = pd.to_numeric(
                    du_lieu[cot],
                    errors="coerce"
                ) * 1000

        except Exception:
            pass

    return du_lieu


# ============================================================
# THÊM CÁC CHỈ BÁO CƠ BẢN
# ============================================================

def add_indicators(du_lieu):

    if du_lieu is None or du_lieu.empty:
        return pd.DataFrame()

    du_lieu = du_lieu.copy()

    cot_gia = None

    for ten_cot in [
        "Đóng cửa",
        "close",
        "Close",
    ]:

        if ten_cot in du_lieu.columns:
            cot_gia = ten_cot
            break

    if cot_gia is None:
        return du_lieu

    gia = pd.to_numeric(
        du_lieu[cot_gia],
        errors="coerce"
    )

    # --------------------------------------------------------
    # ĐƯỜNG TRUNG BÌNH 5
    # --------------------------------------------------------

    du_lieu["Trung bình động 5"] = gia.rolling(
        window=5
    ).mean()

    # --------------------------------------------------------
    # ĐƯỜNG TRUNG BÌNH 10
    # --------------------------------------------------------

    du_lieu["Trung bình động 10"] = gia.rolling(
        window=10
    ).mean()

    # --------------------------------------------------------
    # ĐƯỜNG TRUNG BÌNH 20
    # --------------------------------------------------------

    du_lieu["Trung bình động 20"] = gia.rolling(
        window=20
    ).mean()

    # --------------------------------------------------------
    # ĐƯỜNG TRUNG BÌNH 50
    # --------------------------------------------------------

    du_lieu["Trung bình động 50"] = gia.rolling(
        window=50
    ).mean()

    # --------------------------------------------------------
    # ĐƯỜNG TRUNG BÌNH 100
    # --------------------------------------------------------

    du_lieu["Trung bình động 100"] = gia.rolling(
        window=100
    ).mean()

    # --------------------------------------------------------
    # ĐƯỜNG TRUNG BÌNH 200
    # --------------------------------------------------------

    du_lieu["Trung bình động 200"] = gia.rolling(
        window=200
    ).mean()

    # --------------------------------------------------------
    # THAY ĐỔI GIÁ
    # --------------------------------------------------------

    du_lieu["Thay đổi giá"] = gia.diff()

    # --------------------------------------------------------
    # PHẦN TRĂM THAY ĐỔI
    # --------------------------------------------------------

    du_lieu["Phần trăm thay đổi"] = gia.pct_change() * 100

    # --------------------------------------------------------
    # BIẾN ĐỘNG
    # --------------------------------------------------------

    du_lieu["Biến động 20 ngày"] = (
        du_lieu["Phần trăm thay đổi"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # GIÁ CAO NHẤT 20 NGÀY
    # --------------------------------------------------------

    du_lieu["Cao nhất 20 ngày"] = gia.rolling(
        20
    ).max()

    # --------------------------------------------------------
    # GIÁ THẤP NHẤT 20 NGÀY
    # --------------------------------------------------------

    du_lieu["Thấp nhất 20 ngày"] = gia.rolling(
        20
    ).min()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    thay_doi = gia.diff()

    tang = thay_doi.clip(lower=0)

    giam = -thay_doi.clip(upper=0)

    trung_binh_tang = tang.rolling(
        14
    ).mean()

    trung_binh_giam = giam.rolling(
        14
    ).mean()

    ti_le_tang_giam = (
        trung_binh_tang /
        trung_binh_giam.replace(0, pd.NA)
    )

    du_lieu["RSI"] = (
        100 -
        (
            100 /
            (1 + ti_le_tang_giam)
        )
    )

    # --------------------------------------------------------
    # KHỐI LƯỢNG TRUNG BÌNH
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

        du_lieu["Tỷ lệ khối lượng"] = (
            khoi_luong /
            du_luong_trung_binh_an_toan(
                khoi_luong,
                20
            )
        )

    return du_lieu


def du_luong_trung_binh_an_toan(
    du_lieu,
    so_phien
):

    return (
        du_lieu
        .rolling(so_phien)
        .mean()
        .replace(0, pd.NA)
    )


# ============================================================
# LẤY DỮ LIỆU LỊCH SỬ
# ============================================================

def load_market_data(
    ma_co_phieu,
    so_ngay=365
):

    ma_co_phieu = normalize_symbol(ma_co_phieu)

    if not ma_co_phieu:
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    ngay_ket_thuc = datetime.now()

    ngay_bat_dau = (
        ngay_ket_thuc -
        timedelta(days=so_ngay)
    )

    loi_cuoi = None

    # --------------------------------------------------------
    # ƯU TIÊN KBS
    # --------------------------------------------------------

    for nguon_du_lieu in [
        "kbs",
        "vci",
    ]:

        try:

            du_lieu = (
                _thi_truong
                .equity(ma_co_phieu)
                .ohlcv(
                    start=ngay_bat_dau.strftime("%Y-%m-%d"),
                    end=ngay_ket_thuc.strftime("%Y-%m-%d"),
                )
            )

            if du_lieu is None:
                continue

            du_lieu = _normalize_columns(
                du_lieu
            )

            if du_lieu.empty:
                continue

            du_lieu = _normalize_price(
                du_lieu
            )

            du_lieu = add_indicators(
                du_lieu
            )

            return du_lieu

        except Exception as loi:

            loi_cuoi = loi
            continue

    if loi_cuoi is not None:
        raise RuntimeError(
            f"Không lấy được dữ liệu {ma_co_phieu}: {loi_cuoi}"
        )

    raise RuntimeError(
        f"Không lấy được dữ liệu {ma_co_phieu}."
    )


# ============================================================
# GIÁ HIỆN TẠI
# ============================================================

def load_current_quote(ma_co_phieu):

    ma_co_phieu = normalize_symbol(
        ma_co_phieu
    )

    if not ma_co_phieu:
        return {}

    try:

        du_lieu = (
            _thi_truong
            .equity(ma_co_phieu)
            .quote()
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

        for cot in du_lieu.columns:

            gia_tri = dong_cuoi[cot]

            if pd.notna(gia_tri):
                ket_qua[cot] = gia_tri

        return ket_qua

    except Exception:
        return {}


# ============================================================
# LẤY TỔNG HỢP CỔ PHIẾU
# ============================================================

def load_stock_summary(ma_co_phieu):

    ma_co_phieu = normalize_symbol(
        ma_co_phieu
    )

    try:

        du_lieu = (
            _thi_truong
            .equity(ma_co_phieu)
            .summary()
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

        return _normalize_columns(
            du_lieu
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# DỮ LIỆU CHỈ SỐ VN-INDEX
# ============================================================

def load_vnindex_data():

    try:

        du_lieu = (
            _thi_truong
            .index("VNINDEX")
            .ohlcv(
                start=(
                    datetime.now() -
                    timedelta(days=30)
                ).strftime("%Y-%m-%d"),
                end=datetime.now().strftime(
                    "%Y-%m-%d"
                ),
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

        return _normalize_columns(
            du_lieu
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# LẤY TOÀN BỘ DỮ LIỆU NGHIÊN CỨU
# ============================================================

def load_full_stock_data(
    ma_co_phieu
):

    du_lieu_lich_su = load_market_data(
        ma_co_phieu
    )

    du_lieu_hien_tai = (
        load_current_quote(
            ma_co_phieu
        )
    )

    du_lieu_tong_hop = (
        load_stock_summary(
            ma_co_phieu
        )
    )

    return {
        "ma_co_phieu": normalize_symbol(
            ma_co_phieu
        ),
        "lich_su": du_lieu_lich_su,
        "hien_tai": du_lieu_hien_tai,
        "tong_hop": du_lieu_tong_hop,
    }
```
