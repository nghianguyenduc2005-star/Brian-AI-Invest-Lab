from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CẤU HÌNH BIẾN
# ============================================================

BIEN_DU_BAO_MAC_DINH = "Return"

BIEN_SO_DAY_DU = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

CAC_BIEN_GIA = [
    "Open",
    "High",
    "Low",
    "Close",
]

CAC_BIEN_XU_HUONG = [
    "SMA5",
    "SMA10",
    "SMA20",
    "SMA50",
    "SMA100",
    "SMA200",
    "EMA9",
    "EMA12",
    "EMA20",
    "EMA26",
    "EMA50",
]

CAC_BIEN_DONG_LUONG = [
    "Return",
    "ReturnPct",
    "Momentum5",
    "Momentum10",
    "Momentum20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
]

CAC_BIEN_BIEN_DONG = [
    "Volatility5",
    "Volatility20",
    "Volatility60",
    "ATR14",
    "Range",
    "Range_Percent",
]

CAC_BIEN_KHOI_LUONG = [
    "Volume",
    "Volume_SMA5",
    "Volume_SMA20",
    "Volume_SMA50",
    "Volume_Change",
    "Relative_Volume",
]

CAC_BIEN_BOLLINGER = [
    "Bollinger_Mid",
    "Bollinger_Upper",
    "Bollinger_Lower",
    "Bollinger_Width",
]

CAC_BIEN_DINH_CAO_THAP = [
    "High20",
    "Low20",
    "High50",
    "Low50",
    "High252",
    "Low252",
    "Distance_From_High20",
    "Distance_From_Low20",
    "Distance_From_High252",
    "Distance_From_Low252",
]

CAC_BIEN_GAP = [
    "Gap_Open_Pct",
    "Gap_Up",
    "Gap_Down",
]

CAC_BIEN_DUOC_SU_DUNG = (
    CAC_BIEN_XU_HUONG
    + CAC_BIEN_DONG_LUONG
    + CAC_BIEN_BIEN_DONG
    + CAC_BIEN_KHOI_LUONG
    + CAC_BIEN_BOLLINGER
    + CAC_BIEN_DINH_CAO_THAP
    + CAC_BIEN_GAP
)


# ============================================================
# ÁNH XẠ TÊN TIẾNG VIỆT
# ============================================================

TEN_BIEN_TIEN_VIET = {
    "Open": "Giá mở cửa",
    "High": "Giá cao nhất",
    "Low": "Giá thấp nhất",
    "Close": "Giá đóng cửa",
    "Volume": "Khối lượng",

    "Return": "Lợi suất",
    "ReturnPct": "Lợi suất phần trăm",

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

    "RSI": "Sức mạnh tương đối",
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
    "Range_Percent": "Biên độ giá phần trăm",

    "ATR14": "ATR 14 phiên",

    "Momentum5": "Động lượng 5 phiên",
    "Momentum10": "Động lượng 10 phiên",
    "Momentum20": "Động lượng 20 phiên",

    "Bollinger_Mid": "Đường giữa Bollinger",
    "Bollinger_Upper": "Dải trên Bollinger",
    "Bollinger_Lower": "Dải dưới Bollinger",
    "Bollinger_Width": "Độ rộng Bollinger",

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

    "Gap_Open_Pct": "Khoảng trống giá",
    "Gap_Up": "Khoảng trống tăng",
    "Gap_Down": "Khoảng trống giảm",
}


# ============================================================
# TIỆN ÍCH
# ============================================================

def _lay_so(
    gia_tri: Any,
    mac_dinh: float | None = None,
) -> float | None:
    try:
        gia_tri = float(gia_tri)

        if pd.isna(gia_tri):
            return mac_dinh

        return gia_tri

    except Exception:
        return mac_dinh


def ten_bien_tieng_viet(
    ten_bien: str,
) -> str:
    return TEN_BIEN_TIEN_VIET.get(
        str(ten_bien),
        str(ten_bien),
    )


# ============================================================
# BỔ SUNG TOÀN BỘ BIẾN PHÁI SINH
# ============================================================

def _bo_sung_bien_phai_sinh(
    du_lieu: pd.DataFrame,
) -> pd.DataFrame:

    du_lieu = du_lieu.copy()

    # --------------------------------------------------------
    # Kiểm tra giá
    # --------------------------------------------------------

    for ten_cot in BIEN_SO_DAY_DU:

        if ten_cot not in du_lieu.columns:
            continue

        du_lieu[ten_cot] = pd.to_numeric(
            du_lieu[ten_cot],
            errors="coerce",
        )

    if "Close" not in du_lieu.columns:
        return du_lieu

    gia_dong_cua = du_lieu["Close"]

    # --------------------------------------------------------
    # Lợi suất
    # --------------------------------------------------------

    if "Return" not in du_lieu.columns:
        du_lieu["Return"] = (
            gia_dong_cua.pct_change()
        )

    if "ReturnPct" not in du_lieu.columns:
        du_lieu["ReturnPct"] = (
            du_lieu["Return"] * 100
        )

    # --------------------------------------------------------
    # Khoảng trống giá
    # --------------------------------------------------------

    if "Open" in du_lieu.columns:

        gia_dong_cua_phien_truoc = (
            gia_dong_cua.shift(1)
        )

        du_lieu["Gap_Open_Pct"] = (
            (
                du_lieu["Open"]
                / gia_dong_cua_phien_truoc
                - 1
            )
            * 100
        )

    else:

        du_lieu["Gap_Open_Pct"] = np.nan

    if "High" in du_lieu.columns:

        cao_phien_truoc = (
            du_lieu["High"].shift(1)
        )

        if "Open" in du_lieu.columns:

            du_lieu["Gap_Up"] = (
                du_lieu["Open"]
                > cao_phien_truoc
            )

        else:

            du_lieu["Gap_Up"] = False

    else:

        du_lieu["Gap_Up"] = False

    if "Low" in du_lieu.columns:

        thap_phien_truoc = (
            du_lieu["Low"].shift(1)
        )

        if "Open" in du_lieu.columns:

            du_lieu["Gap_Down"] = (
                du_lieu["Open"]
                < thap_phien_truoc
            )

        else:

            du_lieu["Gap_Down"] = False

    else:

        du_lieu["Gap_Down"] = False

    # --------------------------------------------------------
    # Chuyển cờ thành số
    # --------------------------------------------------------

    du_lieu["Gap_Up"] = (
        du_lieu["Gap_Up"]
        .astype(float)
    )

    du_lieu["Gap_Down"] = (
        du_lieu["Gap_Down"]
        .astype(float)
    )

    # --------------------------------------------------------
    # Bổ sung khoảng cách tới trung bình
    # --------------------------------------------------------

    for ten_trung_binh in [
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA20",
        "EMA50",
    ]:

        if ten_trung_binh in du_lieu.columns:

            ten_khoang_cach = (
                f"Khoang_Cach_{ten_trung_binh}"
            )

            du_lieu[
                ten_khoang_cach
            ] = (
                (
                    gia_dong_cua
                    / du_lieu[
                        ten_trung_binh
                    ]
                    - 1
                )
                * 100
            )

    # --------------------------------------------------------
    # Chênh lệch cao thấp trong ngày
    # --------------------------------------------------------

    if {
        "High",
        "Low",
    }.issubset(
        du_lieu.columns
    ):

        du_lieu["Range_Real"] = (
            du_lieu["High"]
            - du_lieu["Low"]
        )

        du_lieu["Range_Real_Pct"] = (
            du_lieu["Range_Real"]
            / gia_dong_cua
            * 100
        )

    # --------------------------------------------------------
    # Tỷ lệ giá trong vùng 20 / 50 / 252 phiên
    # --------------------------------------------------------

    if "High20" in du_lieu.columns and \
       "Low20" in du_lieu.columns:

        chieu_rong_20 = (
            du_lieu["High20"]
            - du_lieu["Low20"]
        )

        du_lieu["Vi_Tri_Vung_20"] = (
            (
                gia_dong_cua
                - du_lieu["Low20"]
            )
            / chieu_rong_20
        )

    if "High50" in du_lieu.columns and \
       "Low50" in du_lieu.columns:

        chieu_rong_50 = (
            du_lieu["High50"]
            - du_lieu["Low50"]
        )

        du_lieu["Vi_Tri_Vung_50"] = (
            (
                gia_dong_cua
                - du_lieu["Low50"]
            )
            / chieu_rong_50
        )

    if "High252" in du_lieu.columns and \
       "Low252" in du_lieu.columns:

        chieu_rong_252 = (
            du_lieu["High252"]
            - du_lieu["Low252"]
        )

        du_lieu["Vi_Tri_Vung_252"] = (
            (
                gia_dong_cua
                - du_lieu["Low252"]
            )
            / chieu_rong_252
        )

    # --------------------------------------------------------
    # Thay đổi giá nhiều phiên
    # --------------------------------------------------------

    for so_phien in [
        2,
        3,
        5,
        10,
        20,
        60,
        120,
        252,
    ]:

        ten_cot = (
            f"Bien_Dong_Gia_{so_phien}"
        )

        du_lieu[ten_cot] = (
            gia_dong_cua
            / gia_dong_cua.shift(so_phien)
            - 1
        )

    # --------------------------------------------------------
    # Biến động thực tế ngắn hạn
    # --------------------------------------------------------

    if {
        "High",
        "Low",
    }.issubset(
        du_lieu.columns
    ):

        du_lieu["Bien_Do_Ngay"] = (
            (
                du_lieu["High"]
                - du_lieu["Low"]
            )
            / gia_dong_cua
            * 100
        )

    # --------------------------------------------------------
    # Chỉ báo khối lượng nâng cao
    # --------------------------------------------------------

    if "Volume" in du_lieu.columns:

        for so_phien in [
            5,
            10,
            20,
            50,
            100,
        ]:

            ten_tb = (
                f"Volume_SMA{so_phien}"
            )

            if ten_tb not in du_lieu.columns:

                du_lieu[ten_tb] = (
                    du_lieu["Volume"]
                    .rolling(
                        so_phien
                    )
                    .mean()
                )

        if "Volume_SMA20" in du_lieu.columns:

            du_lieu[
                "Ty_Le_Khoi_Luong_TB20"
            ] = (
                du_lieu["Volume"]
                / du_lieu[
                    "Volume_SMA20"
                ]
            )

    # --------------------------------------------------------
    # Mã hóa ngày trong tuần
    # --------------------------------------------------------

    try:

        chi_so_thoi_gian = pd.DatetimeIndex(
            du_lieu.index
        )

        du_lieu[
            "Thu_Trong_Tuan"
        ] = (
            chi_so_thoi_gian.weekday
            + 1
        )

        du_lieu[
            "Ngay_Trong_Thang"
        ] = (
            chi_so_thoi_gian.day
        )

        du_lieu[
            "Thang_Trong_Nam"
        ] = (
            chi_so_thoi_gian.month
        )

    except Exception:
        pass

    du_lieu = du_lieu.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return du_lieu


# ============================================================
# LẤY DANH SÁCH BIẾN
# ============================================================

def lay_danh_sach_bien(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_DU_BAO_MAC_DINH,
) -> list[str]:

    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return []

    du_lieu = _bo_sung_bien_phai_sinh(
        du_lieu
    )

    danh_sach = []

    for ten_bien in CAC_BIEN_DUOC_SU_DUNG:

        if ten_bien not in du_lieu.columns:
            continue

        if ten_bien == bien_phu_thuoc:
            continue

        so_luong_hop_le = (
            pd.to_numeric(
                du_lieu[ten_bien],
                errors="coerce",
            )
            .notna()
            .sum()
        )

        if so_luong_hop_le < 40:
            continue

        do_lech = (
            pd.to_numeric(
                du_lieu[ten_bien],
                errors="coerce",
            )
            .std()
        )

        if (
            pd.isna(do_lech)
            or do_lech == 0
        ):
            continue

        danh_sach.append(
            ten_bien
        )

    # --------------------------------------------------------
    # Biến phái sinh tự tạo
    # --------------------------------------------------------

    bien_phai_sinh = [
        ten_cot
        for ten_cot in du_lieu.columns
        if (
            str(ten_cot).startswith(
                "Khoang_Cach_"
            )
            or str(ten_cot).startswith(
                "Bien_Dong_Gia_"
            )
            or str(ten_cot)
            in {
                "Range_Real",
                "Range_Real_Pct",
                "Vi_Tri_Vung_20",
                "Vi_Tri_Vung_50",
                "Vi_Tri_Vung_252",
                "Bien_Do_Ngay",
                "Ty_Le_Khoi_Luong_TB20",
                "Thu_Trong_Tuan",
                "Ngay_Trong_Thang",
                "Thang_Trong_Nam",
            }
        )
    ]

    for ten_bien in bien_phai_sinh:

        if ten_bien == bien_phu_thuoc:
            continue

        if ten_bien in danh_sach:
            continue

        gia_tri = pd.to_numeric(
            du_lieu[ten_bien],
            errors="coerce",
        )

        if gia_tri.notna().sum() < 40:
            continue

        do_lech = gia_tri.std()

        if (
            pd.isna(do_lech)
            or do_lech == 0
        ):
            continue

        danh_sach.append(
            ten_bien
        )

    return danh_sach


# ============================================================
# LOẠI BIẾN TƯƠNG QUAN QUÁ CAO
# ============================================================

def _loc_tuong_quan(
    bang_du_lieu: pd.DataFrame,
    danh_sach_bien: list[str],
    nguong: float = 0.95,
) -> list[str]:

    if (
        bang_du_lieu is None
        or bang_du_lieu.empty
        or not danh_sach_bien
    ):
        return []

    bang = (
        bang_du_lieu[
            danh_sach_bien
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if bang.empty:
        return danh_sach_bien

    if len(bang.columns) <= 1:
        return list(
            bang.columns
        )

    ma_tran_tuong_quan = (
        bang.corr()
        .abs()
    )

    bien_bi_loai = set()

    for vi_tri, ten_bien in enumerate(
        ma_tran_tuong_quan.columns
    ):

        if ten_bien in bien_bi_loai:
            continue

        for bien_khac in (
            ma_tran_tuong_quan.columns[
                vi_tri + 1:
            ]
        ):

            if bien_khac in bien_bi_loai:
                continue

            try:

                he_so = float(
                    ma_tran_tuong_quan.loc[
                        ten_bien,
                        bien_khac,
                    ]
                )

            except Exception:
                continue

            if he_so >= nguong:

                # Giữ biến đứng trước,
                # loại biến đứng sau
                bien_bi_loai.add(
                    bien_khac
                )

    return [
        ten_bien
        for ten_bien
        in danh_sach_bien
        if ten_bien
        not in bien_bi_loai
    ]


# ============================================================
# TẠO BỘ DỮ LIỆU MÔ HÌNH
# ============================================================

def chuan_bi_du_lieu_mo_hinh(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_DU_BAO_MAC_DINH,
):

    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return None

    du_lieu = _bo_sung_bien_phai_sinh(
        du_lieu
    )

    if (
        bien_phu_thuoc
        not in du_lieu.columns
    ):
        return None

    danh_sach_bien = lay_danh_sach_bien(
        du_lieu,
        bien_phu_thuoc,
    )

    if not danh_sach_bien:
        return None

    bang = du_lieu[
        danh_sach_bien
        + [
            bien_phu_thuoc
        ]
    ].copy()

    for ten_cot in bang.columns:

        bang[ten_cot] = (
            pd.to_numeric(
                bang[ten_cot],
                errors="coerce",
            )
        )

    bang = bang.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    bang = bang.dropna()

    if len(bang) < 80:
        return None

    danh_sach_bien = _loc_tuong_quan(
        bang,
        danh_sach_bien,
        nguong=0.95,
    )

    if not danh_sach_bien:
        return None

    bang = bang[
        danh_sach_bien
        + [
            bien_phu_thuoc
        ]
    ].dropna()

    if len(bang) < 80:
        return None

    return (
        bang,
        danh_sach_bien,
    )


# ============================================================
# CHẠY HỒI QUY
# ============================================================

def _chay_hoi_quy(
    du_lieu_huan_luyen,
    du_lieu_kiem_tra,
    danh_sach_bien,
    bien_phu_thuoc,
):

    try:
        import statsmodels.api as sm

    except Exception:
        return None

    try:

        bien_huan_luyen = sm.add_constant(
            du_lieu_huan_luyen[
                danh_sach_bien
            ],
            has_constant="add",
        )

        bien_kiem_tra = sm.add_constant(
            du_lieu_kiem_tra[
                danh_sach_bien
            ],
            has_constant="add",
        )

        ket_qua_y = du_lieu_huan_luyen[
            bien_phu_thuoc
        ]

        mo_hinh = sm.OLS(
            ket_qua_y,
            bien_huan_luyen,
        ).fit(
            cov_type="HC3"
        )

        du_bao = mo_hinh.predict(
            bien_kiem_tra
        )

        gia_tri_thuc = (
            du_lieu_kiem_tra[
                bien_phu_thuoc
            ]
            .to_numpy()
        )

        gia_tri_du_bao = (
            np.asarray(
                du_bao
            )
        )

        sai_so_tuyet_doi = np.abs(
            gia_tri_thuc
            - gia_tri_du_bao
        )

        mae = float(
            np.nanmean(
                sai_so_tuyet_doi
            )
        )

        return {
            "mo_hinh": mo_hinh,
            "du_bao": gia_tri_du_bao,
            "mae": mae,
            "r2": float(
                mo_hinh.rsquared
            ),
            "r2_hieu_chinh": float(
                mo_hinh.rsquared_adj
            ),
            "he_so": mo_hinh.params.to_dict(),
            "gia_tri_p": mo_hinh.pvalues.to_dict(),
        }

    except Exception:
        return None


# ============================================================
# CHẠY RỪNG NGẪU NHIÊN
# ============================================================

def _chay_rung_ngau_nhien(
    du_lieu_huan_luyen,
    du_lieu_kiem_tra,
    danh_sach_bien,
    bien_phu_thuoc,
):

    try:

        from sklearn.ensemble import (
            RandomForestRegressor,
        )

        from sklearn.metrics import (
            mean_absolute_error,
            r2_score,
        )

    except Exception:
        return None

    try:

        bien_huan_luyen = (
            du_lieu_huan_luyen[
                danh_sach_bien
            ]
            .astype(float)
        )

        bien_kiem_tra = (
            du_lieu_kiem_tra[
                danh_sach_bien
            ]
            .astype(float)
        )

        ket_qua_y = (
            du_lieu_huan_luyen[
                bien_phu_thuoc
            ]
            .astype(float)
        )

        gia_tri_thuc_kiem_tra = (
            du_lieu_kiem_tra[
                bien_phu_thuoc
            ]
            .astype(float)
        )

        mo_hinh = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        mo_hinh.fit(
            bien_huan_luyen,
            ket_qua_y,
        )

        du_bao_kiem_tra = (
            mo_hinh.predict(
                bien_kiem_tra
            )
        )

        sai_so_tuyet_doi = mean_absolute_error(
            gia_tri_thuc_kiem_tra,
            du_bao_kiem_tra,
        )

        r2 = r2_score(
            gia_tri_thuc_kiem_tra,
            du_bao_kiem_tra,
        )

        tam_quan_trong = pd.Series(
            mo_hinh.feature_importances_,
            index=danh_sach_bien,
        ).sort_values(
            ascending=False
        )

        # ----------------------------------------------------
        # Dự báo phiên kế tiếp
        # ----------------------------------------------------

        hang_moi_nhat = (
            du_lieu_kiem_tra[
                danh_sach_bien
            ]
            .iloc[[-1]]
            .astype(float)
        )

        du_bao_tiep_theo = float(
            mo_hinh.predict(
                hang_moi_nhat
            )[0]
        )

        return {
            "mo_hinh": mo_hinh,
            "du_bao_kiem_tra": du_bao_kiem_tra,
            "mae": float(
                sai_so_tuyet_doi
            ),
            "r2": float(r2),
            "tam_quan_trong": tam_quan_trong,
            "du_bao_tiep_theo": du_bao_tiep_theo,
        }

    except Exception:
        return None


# ============================================================
# CHẠY MÔ HÌNH CHÍNH
# ============================================================

def run_quant(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_DU_BAO_MAC_DINH,
):

    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return None

    if (
        bien_phu_thuoc
        not in du_lieu.columns
    ):
        return None

    bo_du_lieu = chuan_bi_du_lieu_mo_hinh(
        du_lieu,
        bien_phu_thuoc,
    )

    if bo_du_lieu is None:
        return None

    bang, danh_sach_bien = (
        bo_du_lieu
    )

    if len(bang) < 80:
        return None

    # ========================================================
    # CHIA THEO THỜI GIAN
    # ========================================================

    vi_tri_tach = int(
        len(bang)
        * 0.80
    )

    if vi_tri_tach < 50:
        return None

    if vi_tri_tach >= len(bang):
        return None

    du_lieu_huan_luyen = (
        bang.iloc[
            :vi_tri_tach
        ]
        .copy()
    )

    du_lieu_kiem_tra = (
        bang.iloc[
            vi_tri_tach:
        ]
        .copy()
    )

    if (
        du_lieu_huan_luyen.empty
        or du_lieu_kiem_tra.empty
    ):
        return None

    # ========================================================
    # HỒI QUY
    # ========================================================

    ket_qua_hoi_quy = _chay_hoi_quy(
        du_lieu_huan_luyen,
        du_lieu_kiem_tra,
        danh_sach_bien,
        bien_phu_thuoc,
    )

    # ========================================================
    # RỪNG NGẪU NHIÊN
    # ========================================================

    ket_qua_rung = _chay_rung_ngau_nhien(
        du_lieu_huan_luyen,
        du_lieu_kiem_tra,
        danh_sach_bien,
        bien_phu_thuoc,
    )

    if (
        ket_qua_hoi_quy is None
        and ket_qua_rung is None
    ):
        return None

    # ========================================================
    # DỰ BÁO
    # ========================================================

    du_bao_rung = None

    if ket_qua_rung is not None:

        du_bao_rung = _lay_so(
            ket_qua_rung.get(
                "du_bao_tiep_theo"
            ),
            None,
        )

    # ========================================================
    # TỔNG HỢP
    # ========================================================

    return {
        "bien_phu_thuoc": bien_phu_thuoc,
        "danh_sach_bien": danh_sach_bien,
        "danh_sach_bien_tieng_viet": [
            ten_bien_tieng_viet(
                ten_bien
            )
            for ten_bien
            in danh_sach_bien
        ],
        "so_dong": int(
            len(bang)
        ),
        "so_bien": int(
            len(danh_sach_bien)
        ),
        "so_huan_luyen": int(
            len(
                du_lieu_huan_luyen
            )
        ),
        "so_kiem_tra": int(
            len(
                du_lieu_kiem_tra
            )
        ),
        "hoi_quy": (
            ket_qua_hoi_quy
        ),
        "rung_ngau_nhien": (
            ket_qua_rung
        ),
        "du_bao_tiep_theo": (
            du_bao_rung
        ),
    }


# ============================================================
# TƯƠNG THÍCH CODE CŨ
# ============================================================

def chay_mo_hinh(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_DU_BAO_MAC_DINH,
):
    return run_quant(
        du_lieu,
        bien_phu_thuoc,
    )


def phan_tich_dinh_luong(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_DU_BAO_MAC_DINH,
):
    return run_quant(
        du_lieu,
        bien_phu_thuoc,
    )


# ============================================================
# LẤY KẾT QUẢ TÓM TẮT
# ============================================================

def tom_tat_ket_qua(
    ket_qua,
):

    if ket_qua is None:
        return {}

    tom_tat = {
        "so_bien": ket_qua.get(
            "so_bien",
            0,
        ),
        "so_dong": ket_qua.get(
            "so_dong",
            0,
        ),
        "du_bao_tiep_theo": ket_qua.get(
            "du_bao_tiep_theo"
        ),
    }

    hoi_quy = ket_qua.get(
        "hoi_quy"
    )

    if hoi_quy:

        tom_tat[
            "r2_hoi_quy"
        ] = hoi_quy.get(
            "r2"
        )

        tom_tat[
            "mae_hoi_quy"
        ] = hoi_quy.get(
            "mae"
        )

    else:

        tom_tat[
            "r2_hoi_quy"
        ] = None

        tom_tat[
            "mae_hoi_quy"
        ] = None

    rung = ket_qua.get(
        "rung_ngau_nhien"
    )

    if rung:

        tom_tat[
            "r2_rung"
        ] = rung.get(
            "r2"
        )

        tom_tat[
            "mae_rung"
        ] = rung.get(
            "mae"
        )

    else:

        tom_tat[
            "r2_rung"
        ] = None

        tom_tat[
            "mae_rung"
        ] = None

    return tom_tat
