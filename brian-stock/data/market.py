# data/market.py

import re
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# NGUỒN DỮ LIỆU
# ============================================================
# File này KHÔNG dùng vnstock.
# Không dùng danh sách mã cổ phiếu cố định.
#
# DNSE hiện không có API công khai đơn giản kiểu:
# get_history("HPG")
# nếu chưa có thông tin xác thực/API endpoint của tài khoản.
#
# Vì vậy phần lấy dữ liệu được tách riêng để sau này thay nguồn
# mà không phải sửa dashboard / phân tích kỹ thuật.
# ============================================================


def normalize_symbol(ma):
    """
    Chuẩn hóa mã cổ phiếu.

    HPG       -> HPG
    hpg       -> HPG
    HPG.VN    -> HPG
    HPG:HOSE  -> HPG
    """

    ma = str(ma or "").strip().upper()

    ma = ma.replace(" ", "")
    ma = ma.replace(".VN", "")
    ma = ma.replace(":HOSE", "")
    ma = ma.replace(":HNX", "")
    ma = ma.replace(":UPCOM", "")

    ma = re.sub(r"[^A-Z0-9]", "", ma)

    return ma


def display_symbol(ma):
    """
    Mã hiển thị trên giao diện.
    """

    return normalize_symbol(ma)


# ============================================================
# CHỈ BÁO KỸ THUẬT
# ============================================================

def tinh_rsi(gia_dong_cua, chu_ky=14):

    thay_doi = gia_dong_cua.diff()

    tang = thay_doi.clip(lower=0)
    giam = -thay_doi.clip(upper=0)

    trung_binh_tang = tang.ewm(
        alpha=1 / chu_ky,
        adjust=False
    ).mean()

    trung_binh_giam = giam.ewm(
        alpha=1 / chu_ky,
        adjust=False
    ).mean()

    ty_le = trung_binh_tang / trung_binh_giam.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + ty_le)
    )

    return rsi


def them_chi_bao(df):

    df = df.copy()

    # --------------------------------------------------------
    # Xử lý tên cột
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            cot[0] if isinstance(cot, tuple) else cot
            for cot in df.columns
        ]

    df.columns = [
        str(cot).strip().lower()
        for cot in df.columns
    ]

    anh_xa = {
        "open": "Mở cửa",
        "high": "Cao nhất",
        "low": "Thấp nhất",
        "close": "Đóng cửa",
        "adj close": "Giá điều chỉnh",
        "volume": "Khối lượng",
    }

    df = df.rename(
        columns={
            cot: anh_xa[cot]
            for cot in df.columns
            if cot in anh_xa
        }
    )

    # --------------------------------------------------------
    # Nếu nguồn trả tên cột tiếng Việt sẵn
    # --------------------------------------------------------

    cot_bat_buoc = [
        "Mở cửa",
        "Cao nhất",
        "Thấp nhất",
        "Đóng cửa",
        "Khối lượng",
    ]

    thieu = [
        cot
        for cot in cot_bat_buoc
        if cot not in df.columns
    ]

    if thieu:

        # Hỗ trợ trường hợp nguồn trả tên tiếng Anh viết hoa
        anh_xa_lai = {}

        for cot in df.columns:

            ten = str(cot).lower().strip()

            if ten == "open":
                anh_xa_lai[cot] = "Mở cửa"

            elif ten == "high":
                anh_xa_lai[cot] = "Cao nhất"

            elif ten == "low":
                anh_xa_lai[cot] = "Thấp nhất"

            elif ten == "close":
                anh_xa_lai[cot] = "Đóng cửa"

            elif ten == "volume":
                anh_xa_lai[cot] = "Khối lượng"

        df = df.rename(columns=anh_xa_lai)

    thieu = [
        cot
        for cot in cot_bat_buoc
        if cot not in df.columns
    ]

    if thieu:
        raise ValueError(
            "Nguồn dữ liệu thiếu cột: "
            + ", ".join(thieu)
        )

    # --------------------------------------------------------
    # Ép kiểu số
    # --------------------------------------------------------

    for cot in cot_bat_buoc:

        df[cot] = pd.to_numeric(
            df[cot],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Mở cửa",
            "Cao nhất",
            "Thấp nhất",
            "Đóng cửa",
        ]
    )

    if len(df) < 60:
        raise ValueError(
            f"Dữ liệu chỉ có {len(df)} phiên, "
            "không đủ để tính đầy đủ chỉ báo."
        )

    # ========================================================
    # BIẾN GIÁ
    # ========================================================

    df["Thay đổi"] = (
        df["Đóng cửa"].diff()
    )

    df["Thay đổi phần trăm"] = (
        df["Đóng cửa"].pct_change() * 100
    )

    df["Biên độ trong ngày"] = (
        (
            df["Cao nhất"]
            - df["Thấp nhất"]
        )
        / df["Đóng cửa"].shift(1)
        * 100
    )

    df["Khoảng cách mở cửa"] = (
        (
            df["Mở cửa"]
            - df["Đóng cửa"].shift(1)
        )
        / df["Đóng cửa"].shift(1)
        * 100
    )

    # ========================================================
    # ĐƯỜNG TRUNG BÌNH
    # ========================================================

    df["Trung bình động 5"] = (
        df["Đóng cửa"].rolling(5).mean()
    )

    df["Trung bình động 10"] = (
        df["Đóng cửa"].rolling(10).mean()
    )

    df["Trung bình động 20"] = (
        df["Đóng cửa"].rolling(20).mean()
    )

    df["Trung bình động 50"] = (
        df["Đóng cửa"].rolling(50).mean()
    )

    df["Trung bình động 100"] = (
        df["Đóng cửa"].rolling(100).mean()
    )

    df["Trung bình động 200"] = (
        df["Đóng cửa"].rolling(200).mean()
    )

    # ========================================================
    # EMA
    # ========================================================

    df["EMA 9"] = (
        df["Đóng cửa"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["EMA 12"] = (
        df["Đóng cửa"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    df["EMA 20"] = (
        df["Đóng cửa"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA 26"] = (
        df["Đóng cửa"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["EMA 50"] = (
        df["Đóng cửa"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    # ========================================================
    # RSI
    # ========================================================

    df["RSI 14"] = tinh_rsi(
        df["Đóng cửa"],
        14
    )

    # Giữ tên cũ để các file hiện tại không lỗi
    df["RSI"] = df["RSI 14"]

    # ========================================================
    # MACD
    # ========================================================

    df["MACD"] = (
        df["EMA 12"]
        - df["EMA 26"]
    )

    df["MACD tín hiệu"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MACD chênh lệch"] = (
        df["MACD"]
        - df["MACD tín hiệu"]
    )

    # Giữ tên cũ
    df["MACD_Signal"] = df["MACD tín hiệu"]

    # ========================================================
    # DẢI BOLLINGER
    # ========================================================

    df["Độ lệch chuẩn 20"] = (
        df["Đóng cửa"]
        .rolling(20)
        .std()
    )

    df["Bollinger giữa"] = (
        df["Trung bình động 20"]
    )

    df["Bollinger trên"] = (
        df["Bollinger giữa"]
        + 2 * df["Độ lệch chuẩn 20"]
    )

    df["Bollinger dưới"] = (
        df["Bollinger giữa"]
        - 2 * df["Độ lệch chuẩn 20"]
    )

    df["Bollinger độ rộng"] = (
        (
            df["Bollinger trên"]
            - df["Bollinger dưới"]
        )
        / df["Bollinger giữa"]
        * 100
    )

    # ========================================================
    # KHỐI LƯỢNG
    # ========================================================

    df["Khối lượng trung bình 5"] = (
        df["Khối lượng"].rolling(5).mean()
    )

    df["Khối lượng trung bình 20"] = (
        df["Khối lượng"].rolling(20).mean()
    )

    df["Khối lượng trung bình 50"] = (
        df["Khối lượng"].rolling(50).mean()
    )

    df["Khối lượng so với trung bình 20"] = (
        df["Khối lượng"]
        / df["Khối lượng trung bình 20"]
    )

    df["Thay đổi khối lượng"] = (
        df["Khối lượng"].pct_change() * 100
    )

    # ========================================================
    # BIẾN ĐỘNG
    # ========================================================

    df["Biến động 5"] = (
        df["Thay đổi phần trăm"]
        .rolling(5)
        .std()
    )

    df["Biến động 20"] = (
        df["Thay đổi phần trăm"]
        .rolling(20)
        .std()
    )

    df["Biến động năm hóa"] = (
        df["Thay đổi phần trăm"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # Giữ tên cũ
    df["Volatility20"] = df["Biến động năm hóa"]

    # ========================================================
    # ĐỈNH / ĐÁY
    # ========================================================

    df["Đỉnh 20"] = (
        df["Cao nhất"]
        .rolling(20)
        .max()
    )

    df["Đáy 20"] = (
        df["Thấp nhất"]
        .rolling(20)
        .min()
    )

    df["Đỉnh 50"] = (
        df["Cao nhất"]
        .rolling(50)
        .max()
    )

    df["Đáy 50"] = (
        df["Thấp nhất"]
        .rolling(50)
        .min()
    )

    df["Đỉnh 52 tuần"] = (
        df["Cao nhất"]
        .rolling(252)
        .max()
    )

    df["Đáy 52 tuần"] = (
        df["Thấp nhất"]
        .rolling(252)
        .min()
    )

    # ========================================================
    # KHOẢNG CÁCH ĐẾN ĐỈNH / ĐÁY
    # ========================================================

    df["Khoảng cách đỉnh 20"] = (
        (
            df["Đóng cửa"]
            / df["Đỉnh 20"]
            - 1
        )
        * 100
    )

    df["Khoảng cách đáy 20"] = (
        (
            df["Đóng cửa"]
            / df["Đáy 20"]
            - 1
        )
        * 100
    )

    df["Khoảng cách đỉnh 52 tuần"] = (
        (
            df["Đóng cửa"]
            / df["Đỉnh 52 tuần"]
            - 1
        )
        * 100
    )

    df["Khoảng cách đáy 52 tuần"] = (
        (
            df["Đóng cửa"]
            / df["Đáy 52 tuần"]
            - 1
        )
        * 100
    )

    # ========================================================
    # STOCHASTIC
    # ========================================================

    đáy_14 = (
        df["Thấp nhất"]
        .rolling(14)
        .min()
    )

    đỉnh_14 = (
        df["Cao nhất"]
        .rolling(14)
        .max()
    )

    df["Stochastic K"] = (
        (
            df["Đóng cửa"] - đáy_14
        )
        / (
            đỉnh_14 - đáy_14
        )
        * 100
    )

    df["Stochastic D"] = (
        df["Stochastic K"]
        .rolling(3)
        .mean()
    )

    # ========================================================
    # ATR
    # ========================================================

    biên_độ_1 = (
        df["Cao nhất"]
        - df["Thấp nhất"]
    )

    biên_độ_2 = (
        df["Cao nhất"]
        - df["Đóng cửa"].shift(1)
    ).abs()

    biên_độ_3 = (
        df["Thấp nhất"]
        - df["Đóng cửa"].shift(1)
    ).abs()

    biên_độ_thật = pd.concat(
        [
            biên_độ_1,
            biên_độ_2,
            biên_độ_3,
        ],
        axis=1
    ).max(axis=1)

    df["ATR 14"] = (
        biên_độ_thật
        .rolling(14)
        .mean()
    )

    # ========================================================
    # DIỄN BIẾN TĂNG GIẢM
    # ========================================================

    df["Tăng"] = (
        df["Đóng cửa"]
        > df["Đóng cửa"].shift(1)
    )

    df["Giảm"] = (
        df["Đóng cửa"]
        < df["Đóng cửa"].shift(1)
    )

    df["Thân nến"] = (
        df["Đóng cửa"]
        - df["Mở cửa"]
    )

    df["Tỷ lệ thân nến"] = (
        (
            df["Đóng cửa"]
            - df["Mở cửa"]
        )
        / df["Mở cửa"]
        * 100
    )

    # ========================================================
    # TÍN HIỆU XU HƯỚNG
    # ========================================================

    df["Giá trên MA20"] = (
        df["Đóng cửa"]
        > df["Trung bình động 20"]
    )

    df["Giá trên MA50"] = (
        df["Đóng cửa"]
        > df["Trung bình động 50"]
    )

    df["MA20 trên MA50"] = (
        df["Trung bình động 20"]
        > df["Trung bình động 50"]
    )

    df["MA50 trên MA200"] = (
        df["Trung bình động 50"]
        > df["Trung bình động 200"]
    )

    # ========================================================
    # LOẠI BỎ DÒNG CHƯA ĐỦ CHỈ BÁO
    # ========================================================

    cot_chi_bao = [
        "Thay đổi phần trăm",
        "RSI",
        "MACD",
        "MACD tín hiệu",
        "Trung bình động 20",
        "Trung bình động 50",
        "Khối lượng trung bình 20",
        "Biến động 20",
        "Bollinger trên",
        "Bollinger dưới",
        "Stochastic K",
        "Stochastic D",
        "ATR 14",
    ]

    df = df.dropna(
        subset=cot_chi_bao
    )

    return df


# ============================================================
# HÀM ĐỔI CỘT VỀ TÊN CŨ
# ĐỂ CÁC FILE CŨ KHÔNG BỊ VỠ
# ============================================================

def tao_cot_tuong_thich(df):

    df = df.copy()

    if "Đóng cửa" in df.columns:
        df["Close"] = df["Đóng cửa"]

    if "Mở cửa" in df.columns:
        df["Open"] = df["Mở cửa"]

    if "Cao nhất" in df.columns:
        df["High"] = df["Cao nhất"]

    if "Thấp nhất" in df.columns:
        df["Low"] = df["Thấp nhất"]

    if "Khối lượng" in df.columns:
        df["Volume"] = df["Khối lượng"]

    if "Thay đổi phần trăm" in df.columns:
        df["Return"] = (
            df["Thay đổi phần trăm"] / 100
        )

    if "RSI" not in df.columns and "RSI 14" in df.columns:
        df["RSI"] = df["RSI 14"]

    if (
        "MACD_Signal" not in df.columns
        and "MACD tín hiệu" in df.columns
    ):
        df["MACD_Signal"] = df["MACD tín hiệu"]

    if (
        "SMA20" not in df.columns
        and "Trung bình động 20" in df.columns
    ):
        df["SMA20"] = df["Trung bình động 20"]

    if (
        "SMA50" not in df.columns
        and "Trung bình động 50" in df.columns
    ):
        df["SMA50"] = df["Trung bình động 50"]

    if (
        "Volatility20" not in df.columns
        and "Biến động năm hóa" in df.columns
    ):
        df["Volatility20"] = df["Biến động năm hóa"]

    return df


# ============================================================
# HÀM NHẬN DỮ LIỆU TỪ NGUỒN
# ============================================================

def _chuan_hoa_du_lieu_nguon(df):

    if df is None:
        raise ValueError(
            "Nguồn dữ liệu trả về rỗng."
        )

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if df.empty:
        raise ValueError(
            "Nguồn dữ liệu không có dữ liệu."
        )

    # Nếu ngày đang nằm trong cột
    cac_cot_ngay = [
        "time",
        "timestamp",
        "date",
        "datetime",
        "ngày",
    ]

    cot_ngay = None

    for cot in df.columns:

        if str(cot).lower().strip() in cac_cot_ngay:
            cot_ngay = cot
            break

    if cot_ngay is not None:

        df[cot_ngay] = pd.to_datetime(
            df[cot_ngay],
            errors="coerce"
        )

        df = df.set_index(cot_ngay)

    # Chuẩn hóa chỉ số thời gian
    try:
        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )
    except Exception:
        pass

    df = df.sort_index()

    return df


# ============================================================
# HÀM CHÍNH
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_market_data(ma, khoang_thoi_gian="1y"):

    ma = normalize_symbol(ma)

    if not ma:
        raise ValueError(
            "Chưa nhập mã cổ phiếu."
        )

    # --------------------------------------------------------
    # QUAN TRỌNG:
    #
    # Không gọi vnstock.
    # Không gọi Yahoo.
    #
    # Để kết nối DNSE thật sự, chỉ cần thay phần
    # lay_du_lieu_dnse() bằng API DNSE của tài khoản.
    # --------------------------------------------------------

    df = lay_du_lieu_dnse(
        ma,
        khoang_thoi_gian
    )

    df = _chuan_hoa_du_lieu_nguon(df)

    df = them_chi_bao(df)

    df = tao_cot_tuong_thich(df)

    if df.empty:
        raise ValueError(
            f"Không có dữ liệu sau khi xử lý {ma}."
        )

    return df


# ============================================================
# ĐIỂM KẾT NỐI DNSE
# ============================================================

def lay_du_lieu_dnse(ma, khoang_thoi_gian="1y"):
    """
    Đây là điểm duy nhất cần nối API DNSE.

    Không tự ý dùng vnstock.
    Không tự ý dùng Yahoo.

    Đọc cấu hình từ:
        st.secrets["DNSE_API_URL"]
        st.secrets["DNSE_API_KEY"]

    Ví dụ secrets:

        DNSE_API_URL = "..."
        DNSE_API_KEY = "..."

    Khi có endpoint DNSE thật, hàm này gọi trực tiếp endpoint đó.
    """

    try:
        api_url = st.secrets.get(
            "DNSE_API_URL",
            ""
        )

        api_key = st.secrets.get(
            "DNSE_API_KEY",
            ""
        )

    except Exception:
        api_url = ""
        api_key = ""

    if not api_url:

        raise ValueError(
            "Chưa cấu hình nguồn dữ liệu DNSE. "
            "Hãy thêm DNSE_API_URL và DNSE_API_KEY "
            "vào Streamlit Secrets."
        )

    # --------------------------------------------------------
    # Không đoán endpoint DNSE.
    #
    # Vì nếu tự đoán endpoint/API của DNSE sẽ lại dẫn tới
    # lỗi dữ liệu giả hoặc API sai.
    #
    # Phần này chờ đúng endpoint DNSE mà tài khoản đang dùng.
    # --------------------------------------------------------

    import requests

    headers = {
        "Authorization": (
            f"Bearer {api_key}"
            if api_key
            else ""
        ),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    params = {
        "symbol": ma,
        "period": khoang_thoi_gian,
    }

    try:

        phan_hoi = requests.get(
            api_url,
            headers=headers,
            params=params,
            timeout=20
        )

        phan_hoi.raise_for_status()

    except Exception as e:

        raise ValueError(
            f"Không kết nối được dữ liệu DNSE cho {ma}: {e}"
        )

    try:

        du_lieu = phan_hoi.json()

    except Exception:

        raise ValueError(
            "DNSE trả về dữ liệu không phải JSON."
        )

    # --------------------------------------------------------
    # Một số API trả:
    #
    # {"data": [...]}
    #
    # hoặc:
    #
    # {"data": {"items": [...]}}
    # --------------------------------------------------------

    if isinstance(du_lieu, dict):

        if isinstance(
            du_lieu.get("data"),
            list
        ):
            du_lieu = du_lieu["data"]

        elif isinstance(
            du_lieu.get("data"),
            dict
        ):

            data = du_lieu["data"]

            for ten in [
                "items",
                "data",
                "rows",
                "results",
                "candles",
            ]:

                if isinstance(
                    data.get(ten),
                    list
                ):
                    du_lieu = data[ten]
                    break

    df = pd.DataFrame(
        du_lieu
    )

    if df.empty:

        raise ValueError(
            f"DNSE không trả dữ liệu cho mã {ma}."
        )

    return df
