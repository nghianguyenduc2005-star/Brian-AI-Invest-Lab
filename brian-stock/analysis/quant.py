from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CẤU HÌNH
# ============================================================

BIEN_PHU_THUOC_MAC_DINH = "Return"

NGUONG_TUONG_QUAN = 0.95
SO_QUAN_SAT_TOI_THIEU = 80
TY_LE_HUAN_LUYEN = 0.80


# ============================================================
# TÊN BIẾN TIẾNG VIỆT
# ============================================================

TEN_BIEN_TIENg_VIET = {
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

    "Khoang_Cach_SMA20": "Khoảng cách tới SMA20 (%)",
    "Khoang_Cach_SMA50": "Khoảng cách tới SMA50 (%)",
    "Khoang_Cach_SMA200": "Khoảng cách tới SMA200 (%)",
    "Khoang_Cach_EMA20": "Khoảng cách tới EMA20 (%)",
    "Khoang_Cach_EMA50": "Khoảng cách tới EMA50 (%)",

    "Bien_Dong_Gia_2": "Thay đổi giá 2 phiên (%)",
    "Bien_Dong_Gia_3": "Thay đổi giá 3 phiên (%)",
    "Bien_Dong_Gia_5": "Thay đổi giá 5 phiên (%)",
    "Bien_Dong_Gia_10": "Thay đổi giá 10 phiên (%)",
    "Bien_Dong_Gia_20": "Thay đổi giá 20 phiên (%)",
    "Bien_Dong_Gia_60": "Thay đổi giá 60 phiên (%)",
    "Bien_Dong_Gia_120": "Thay đổi giá 120 phiên (%)",
    "Bien_Dong_Gia_252": "Thay đổi giá 252 phiên (%)",

    "Range_Real": "Biên độ thực",
    "Range_Real_Pct": "Biên độ thực (%)",

    "Vi_Tri_Vung_20": "Vị trí trong vùng 20 phiên",
    "Vi_Tri_Vung_50": "Vị trí trong vùng 50 phiên",
    "Vi_Tri_Vung_252": "Vị trí trong vùng 252 phiên",

    "Bien_Do_Ngay": "Biên độ trong ngày (%)",
    "Ty_Le_Khoi_Luong_TB20": "Tỷ lệ khối lượng / TB20",

    "Thu_Trong_Tuan": "Thứ trong tuần",
    "Ngay_Trong_Thang": "Ngày trong tháng",
    "Thang_Trong_Nam": "Tháng trong năm",
}


# ============================================================
# NHÓM BIẾN
# ============================================================

NHOM_BIEN = {
    "Giá": [
        "Open",
        "High",
        "Low",
        "Close",
    ],

    "Khối lượng": [
        "Volume",
        "Volume_SMA5",
        "Volume_SMA20",
        "Volume_SMA50",
        "Volume_Change",
        "Relative_Volume",
    ],

    "Xu hướng": [
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
    ],

    "Động lượng": [
        "Return",
        "ReturnPct",
        "Momentum5",
        "Momentum10",
        "Momentum20",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
    ],

    "Biến động": [
        "Volatility5",
        "Volatility20",
        "Volatility60",
        "ATR14",
        "Range",
        "Range_Percent",
        "Range_Real",
        "Range_Real_Pct",
        "Bien_Do_Ngay",
    ],

    "Bollinger": [
        "Bollinger_Mid",
        "Bollinger_Upper",
        "Bollinger_Lower",
        "Bollinger_Width",
    ],

    "Đỉnh / đáy": [
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
        "Vi_Tri_Vung_20",
        "Vi_Tri_Vung_50",
        "Vi_Tri_Vung_252",
    ],

    "Khoảng trống giá": [
        "Gap_Open_Pct",
        "Gap_Up",
        "Gap_Down",
    ],
}


# ============================================================
# TIỆN ÍCH
# ============================================================

def ten_bien_tieng_viet(
    ten_bien: str,
) -> str:
    return TEN_BIEN_TIENg_VIET.get(
        ten_bien,
        ten_bien,
    )


def _so(
    gia_tri: Any,
    mac_dinh: float | None = None,
):
    try:
        gia_tri = float(gia_tri)

        if pd.isna(gia_tri):
            return mac_dinh

        return gia_tri

    except Exception:
        return mac_dinh


def _la_so_lien_tuc(
    series: pd.Series,
) -> bool:
    try:

        series = pd.to_numeric(
            series,
            errors="coerce",
        )

        series = series.dropna()

        if len(series) < 2:
            return False

        do_lech = float(
            series.std()
        )

        if not np.isfinite(do_lech):
            return False

        if do_lech == 0:
            return False

        return True

    except Exception:
        return False


# ============================================================
# BỔ SUNG BIẾN PHÁI SINH
# ============================================================

def bo_sung_bien_phai_sinh(
    du_lieu: pd.DataFrame,
) -> pd.DataFrame:

    if (
        du_lieu is None
        or not isinstance(
            du_lieu,
            pd.DataFrame,
        )
    ):
        return pd.DataFrame()

    df = du_lieu.copy()

    # --------------------------------------------------------
    # Chuyển dữ liệu số
    # --------------------------------------------------------

    for cot in df.columns:

        try:

            if cot not in {
                "Gap_Up",
                "Gap_Down",
            }:

                df[cot] = pd.to_numeric(
                    df[cot],
                    errors="ignore",
                )

        except Exception:
            pass

    if "Close" not in df.columns:
        return df

    gia_dong_cua = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    # ========================================================
    # GAP
    # ========================================================

    if "Open" in df.columns:

        gia_dong_cua_truoc = (
            gia_dong_cua.shift(1)
        )

        df["Gap_Open_Pct"] = (
            (
                df["Open"]
                / gia_dong_cua_truoc
                - 1
            )
            * 100
        )

        if "High" in df.columns:

            df["Gap_Up"] = (
                df["Open"]
                > df["High"].shift(1)
            ).astype(float)

        else:

            df["Gap_Up"] = 0.0

        if "Low" in df.columns:

            df["Gap_Down"] = (
                df["Open"]
                < df["Low"].shift(1)
            ).astype(float)

        else:

            df["Gap_Down"] = 0.0

    else:

        df["Gap_Open_Pct"] = np.nan
        df["Gap_Up"] = 0.0
        df["Gap_Down"] = 0.0

    # ========================================================
    # KHOẢNG CÁCH GIÁ VỚI MA
    # ========================================================

    for ten_ma in [
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA20",
        "EMA50",
    ]:

        if ten_ma in df.columns:

            ten_cot = (
                f"Khoang_Cach_{ten_ma}"
            )

            df[ten_cot] = (
                (
                    gia_dong_cua
                    / pd.to_numeric(
                        df[ten_ma],
                        errors="coerce",
                    )
                    - 1
                )
                * 100
            )

    # ========================================================
    # THAY ĐỔI GIÁ NHIỀU PHIÊN
    # ========================================================

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

        df[ten_cot] = (
            gia_dong_cua
            / gia_dong_cua.shift(
                so_phien
            )
            - 1
        )

    # ========================================================
    # BIÊN ĐỘ THẬT
    # ========================================================

    if {
        "High",
        "Low",
    }.issubset(
        df.columns
    ):

        gia_cao = pd.to_numeric(
            df["High"],
            errors="coerce",
        )

        gia_thap = pd.to_numeric(
            df["Low"],
            errors="coerce",
        )

        df["Range_Real"] = (
            gia_cao
            - gia_thap
        )

        df["Range_Real_Pct"] = (
            df["Range_Real"]
            / gia_dong_cua
            * 100
        )

        df["Bien_Do_Ngay"] = (
            df["Range_Real_Pct"]
        )

    # ========================================================
    # VỊ TRÍ TRONG VÙNG
    # ========================================================

    for so_phien in [
        20,
        50,
        252,
    ]:

        cot_cao = (
            f"High{so_phien}"
        )

        cot_thap = (
            f"Low{so_phien}"
        )

        ten_cot = (
            f"Vi_Tri_Vung_{so_phien}"
        )

        if (
            cot_cao in df.columns
            and cot_thap in df.columns
        ):

            cao = pd.to_numeric(
                df[cot_cao],
                errors="coerce",
            )

            thap = pd.to_numeric(
                df[cot_thap],
                errors="coerce",
            )

            do_rong = (
                cao - thap
            )

            df[ten_cot] = (
                gia_dong_cua
                - thap
            ) / do_rong

    # ========================================================
    # KHỐI LƯỢNG NÂNG CAO
    # ========================================================

    if "Volume" in df.columns:

        khoi_luong = pd.to_numeric(
            df["Volume"],
            errors="coerce",
        )

        for so_phien in [
            5,
            10,
            20,
            50,
            100,
        ]:

            ten_cot = (
                f"Volume_SMA{so_phien}"
            )

            if ten_cot not in df.columns:

                df[ten_cot] = (
                    khoi_luong
                    .rolling(
                        so_phien,
                        min_periods=so_phien,
                    )
                    .mean()
                )

        if "Volume_SMA20" in df.columns:

            df[
                "Ty_Le_Khoi_Luong_TB20"
            ] = (
                khoi_luong
                / pd.to_numeric(
                    df["Volume_SMA20"],
                    errors="coerce",
                )
            )

    # ========================================================
    # THÔNG TIN THỜI GIAN
    # ========================================================

    try:

        chi_so = pd.DatetimeIndex(
            df.index
        )

        df["Thu_Trong_Tuan"] = (
            chi_so.weekday + 1
        )

        df["Ngay_Trong_Thang"] = (
            chi_so.day
        )

        df["Thang_Trong_Nam"] = (
            chi_so.month
        )

    except Exception:
        pass

    return df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# LẤY TOÀN BỘ BIẾN CÓ THỂ SỬ DỤNG
# ============================================================

def lay_danh_sach_bien(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
) -> list[str]:

    if (
        du_lieu is None
        or du_lieu.empty
    ):
        return []

    df = bo_sung_bien_phai_sinh(
        du_lieu
    )

    danh_sach = []

    for ten_cot in df.columns:

        if ten_cot == bien_phu_thuoc:
            continue

        series = pd.to_numeric(
            df[ten_cot],
            errors="coerce",
        )

        so_hop_le = int(
            series.notna().sum()
        )

        if so_hop_le < 40:
            continue

        if not _la_so_lien_tuc(
            series
        ):
            continue

        danh_sach.append(
            ten_cot
        )

    return danh_sach


# ============================================================
# LỌC TƯƠNG QUAN CAO
# ============================================================

def loc_tuong_quan(
    du_lieu: pd.DataFrame,
    danh_sach_bien: list[str],
    nguong: float = NGUONG_TUONG_QUAN,
) -> list[str]:

    if (
        du_lieu is None
        or du_lieu.empty
        or not danh_sach_bien
    ):
        return []

    df = du_lieu[
        danh_sach_bien
    ].copy()

    for cot in df.columns:

        df[cot] = pd.to_numeric(
            df[cot],
            errors="coerce",
        )

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if df.empty:
        return danh_sach_bien

    if len(df.columns) <= 1:
        return list(
            df.columns
        )

    ma_tran = (
        df.corr()
        .abs()
    )

    loai = set()

    cac_cot = list(
        ma_tran.columns
    )

    for i in range(
        len(cac_cot)
    ):

        bien_a = cac_cot[i]

        if bien_a in loai:
            continue

        for j in range(
            i + 1,
            len(cac_cot),
        ):

            bien_b = cac_cot[j]

            if bien_b in loai:
                continue

            try:

                he_so = float(
                    ma_tran.loc[
                        bien_a,
                        bien_b,
                    ]
                )

            except Exception:
                continue

            if he_so >= nguong:

                loai.add(
                    bien_b
                )

    return [
        bien
        for bien
        in danh_sach_bien
        if bien not in loai
    ]


# ============================================================
# CHUẨN BỊ BỘ DỮ LIỆU
# ============================================================

def chuan_bi_du_lieu(
    du_lieu: pd.DataFrame,
    bien_giai_thich: list[str] | None = None,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
):

    if (
        du_lieu is None
        or not isinstance(
            du_lieu,
            pd.DataFrame,
        )
        or du_lieu.empty
    ):
        return None

    df = bo_sung_bien_phai_sinh(
        du_lieu
    )

    if (
        bien_phu_thuoc
        not in df.columns
    ):
        return None

    # --------------------------------------------------------
    # Nếu người dùng không chỉ định biến,
    # tự lấy toàn bộ biến hợp lệ.
    # --------------------------------------------------------

    if not bien_giai_thich:

        bien_giai_thich = lay_danh_sach_bien(
            df,
            bien_phu_thuoc,
        )

    else:

        bien_giai_thich = [
            x
            for x in bien_giai_thich
            if x in df.columns
            and x != bien_phu_thuoc
        ]

    if not bien_giai_thich:
        return None

    cac_cot = list(
        dict.fromkeys(
            bien_giai_thich
            + [
                bien_phu_thuoc
            ]
        )
    )

    df = df[cac_cot].copy()

    for cot in df.columns:

        df[cot] = pd.to_numeric(
            df[cot],
            errors="coerce",
        )

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    df = df.dropna()

    if len(df) < SO_QUAN_SAT_TOI_THIEU:
        return None

    # --------------------------------------------------------
    # Bỏ biến tương quan quá cao.
    # --------------------------------------------------------

    bien_sau_loc = loc_tuong_quan(
        df,
        bien_giai_thich,
        NGUONG_TUONG_QUAN,
    )

    if not bien_sau_loc:
        return None

    df = df[
        bien_sau_loc
        + [
            bien_phu_thuoc
        ]
    ].dropna()

    if len(df) < SO_QUAN_SAT_TOI_THIEU:
        return None

    return (
        df,
        bien_sau_loc,
    )


# ============================================================
# HỒI QUY OLS
# ============================================================

def chay_ols(
    du_lieu_huan_luyen: pd.DataFrame,
    du_lieu_kiem_tra: pd.DataFrame,
    danh_sach_bien: list[str],
    bien_phu_thuoc: str,
):

    try:

        import statsmodels.api as sm

    except Exception:
        return None

    try:

        x_train = sm.add_constant(
            du_lieu_huan_luyen[
                danh_sach_bien
            ],
            has_constant="add",
        )

        x_test = sm.add_constant(
            du_lieu_kiem_tra[
                danh_sach_bien
            ],
            has_constant="add",
        )

        y_train = (
            du_lieu_huan_luyen[
                bien_phu_thuoc
            ]
            .astype(float)
        )

        y_test = (
            du_lieu_kiem_tra[
                bien_phu_thuoc
            ]
            .astype(float)
        )

        mo_hinh = sm.OLS(
            y_train,
            x_train,
        ).fit(
            cov_type="HC3"
        )

        du_bao = np.asarray(
            mo_hinh.predict(
                x_test
            )
        )

        thuc_te = (
            y_test
            .to_numpy()
        )

        sai_so = (
            thuc_te
            - du_bao
        )

        mae = float(
            np.mean(
                np.abs(
                    sai_so
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    sai_so ** 2
                )
            )
        )

        return {
            "mo_hinh": mo_hinh,
            "du_bao": du_bao,
            "thuc_te": thuc_te,
            "mae": mae,
            "rmse": rmse,
            "r2": float(
                mo_hinh.rsquared
            ),
            "r2_hieu_chinh": float(
                mo_hinh.rsquared_adj
            ),
            "he_so": (
                mo_hinh.params.to_dict()
            ),
            "p_value": (
                mo_hinh.pvalues.to_dict()
            ),
        }

    except Exception:
        return None


# ============================================================
# RANDOM FOREST
# ============================================================

def chay_random_forest(
    du_lieu_huan_luyen: pd.DataFrame,
    du_lieu_kiem_tra: pd.DataFrame,
    danh_sach_bien: list[str],
    bien_phu_thuoc: str,
):

    try:

        from sklearn.ensemble import (
            RandomForestRegressor,
        )

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

    except Exception:
        return None

    try:

        x_train = (
            du_lieu_huan_luyen[
                danh_sach_bien
            ]
            .astype(float)
        )

        x_test = (
            du_lieu_kiem_tra[
                danh_sach_bien
            ]
            .astype(float)
        )

        y_train = (
            du_lieu_huan_luyen[
                bien_phu_thuoc
            ]
            .astype(float)
        )

        y_test = (
            du_lieu_kiem_tra[
                bien_phu_thuoc
            ]
            .astype(float)
        )

        mo_hinh = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        mo_hinh.fit(
            x_train,
            y_train,
        )

        du_bao_kiem_tra = (
            mo_hinh.predict(
                x_test
            )
        )

        mae = float(
            mean_absolute_error(
                y_test,
                du_bao_kiem_tra,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    du_bao_kiem_tra,
                )
            )
        )

        try:

            r2 = float(
                r2_score(
                    y_test,
                    du_bao_kiem_tra,
                )
            )

        except Exception:

            r2 = float(
                "nan"
            )

        tam_quan_trong = pd.Series(
            mo_hinh.feature_importances_,
            index=danh_sach_bien,
        ).sort_values(
            ascending=False
        )

        x_moi_nhat = (
            du_lieu_kiem_tra[
                danh_sach_bien
            ]
            .iloc[[-1]]
            .astype(float)
        )

        du_bao_tiep_theo = float(
            mo_hinh.predict(
                x_moi_nhat
            )[0]
        )

        return {
            "mo_hinh": mo_hinh,
            "du_bao_kiem_tra": (
                du_bao_kiem_tra
            ),
            "du_bao_tiep_theo": (
                du_bao_tiep_theo
            ),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "tam_quan_trong": tam_quan_trong,
        }

    except Exception:
        return None


# ============================================================
# MÔ HÌNH CHÍNH
# ============================================================

def run_quant(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
    bien_giai_thich: list[str] | None = None,
):

    bo_du_lieu = chuan_bi_du_lieu(
        du_lieu,
        bien_giai_thich,
        bien_phu_thuoc,
    )

    if bo_du_lieu is None:
        return None

    df, danh_sach_bien = (
        bo_du_lieu
    )

    so_dong = len(df)

    if so_dong < SO_QUAN_SAT_TOI_THIEU:
        return None

    # ========================================================
    # CHIA TRAIN / TEST THEO THỜI GIAN
    # ========================================================

    vi_tri_tach = int(
        so_dong
        * TY_LE_HUAN_LUYEN
    )

    if vi_tri_tach < 40:
        return None

    if vi_tri_tach >= so_dong:
        return None

    du_lieu_huan_luyen = (
        df.iloc[
            :vi_tri_tach
        ]
        .copy()
    )

    du_lieu_kiem_tra = (
        df.iloc[
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
    # OLS
    # ========================================================

    ket_qua_ols = chay_ols(
        du_lieu_huan_luyen,
        du_lieu_kiem_tra,
        danh_sach_bien,
        bien_phu_thuoc,
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    ket_qua_rf = chay_random_forest(
        du_lieu_huan_luyen,
        du_lieu_kiem_tra,
        danh_sach_bien,
        bien_phu_thuoc,
    )

    if (
        ket_qua_ols is None
        and ket_qua_rf is None
    ):
        return None

    # ========================================================
    # DỰ BÁO CUỐI
    # ========================================================

    du_bao_tiep_theo = None

    if ket_qua_rf is not None:

        du_bao_tiep_theo = _so(
            ket_qua_rf.get(
                "du_bao_tiep_theo"
            ),
            None,
        )

    elif ket_qua_ols is not None:

        try:

            hang_cuoi = (
                du_lieu_kiem_tra[
                    danh_sach_bien
                ]
                .iloc[[-1]]
            )

            du_bao = (
                ket_qua_ols[
                    "mo_hinh"
                ]
                .predict(
                    pd.DataFrame(
                        {
                            ten: hang_cuoi[
                                ten
                            ].astype(float)
                            for ten
                            in danh_sach_bien
                        },
                        index=hang_cuoi.index,
                    ).assign(
                        const=1.0
                    )[
                        [
                            "const"
                        ]
                        + danh_sach_bien
                    ]
                )
            )

            du_bao_tiep_theo = float(
                np.asarray(
                    du_bao
                )[0]
            )

        except Exception:
            du_bao_tiep_theo = None

    # ========================================================
    # KẾT QUẢ
    # ========================================================

    return {
        "bien_phu_thuoc": (
            bien_phu_thuoc
        ),

        "bien_phu_thuoc_tieng_viet": (
            ten_bien_tieng_viet(
                bien_phu_thuoc
            )
        ),

        "danh_sach_bien": (
            danh_sach_bien
        ),

        "danh_sach_bien_tieng_viet": [
            ten_bien_tieng_viet(
                bien
            )
            for bien
            in danh_sach_bien
        ],

        "so_bien": len(
            danh_sach_bien
        ),

        "so_dong": so_dong,

        "so_huan_luyen": len(
            du_lieu_huan_luyen
        ),

        "so_kiem_tra": len(
            du_lieu_kiem_tra
        ),

        "du_lieu": df,

        "du_lieu_huan_luyen": (
            du_lieu_huan_luyen
        ),

        "du_lieu_kiem_tra": (
            du_lieu_kiem_tra
        ),

        "hoi_quy": (
            ket_qua_ols
        ),

        "rung_ngau_nhien": (
            ket_qua_rf
        ),

        "du_bao_tiep_theo": (
            du_bao_tiep_theo
        ),
    }


# ============================================================
# TƯƠNG THÍCH API CŨ
# ============================================================

def build_quant(
    df: pd.DataFrame,
    bien_giai_thich: list[str] | None = None,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
):

    ket_qua = run_quant(
        df,
        bien_phu_thuoc,
        bien_giai_thich,
    )

    if ket_qua is None:
        return None

    ols = None
    rf = None
    metrics = {}
    next_return = (
        ket_qua.get(
            "du_bao_tiep_theo"
        )
    )
    feature_importance = pd.Series(
        dtype=float
    )

    if ket_qua.get("hoi_quy"):

        ols = ket_qua[
            "hoi_quy"
        ].get(
            "mo_hinh"
        )

        metrics[
            "OLS_MAE"
        ] = ket_qua[
            "hoi_quy"
        ].get(
            "mae"
        )

        metrics[
            "OLS_R2"
        ] = ket_qua[
            "hoi_quy"
        ].get(
            "r2"
        )

    if ket_qua.get(
        "rung_ngau_nhien"
    ):

        rf = ket_qua[
            "rung_ngau_nhien"
        ].get(
            "mo_hinh"
        )

        metrics[
            "MAE"
        ] = ket_qua[
            "rung_ngau_nhien"
        ].get(
            "mae"
        )

        metrics[
            "R2"
        ] = ket_qua[
            "rung_ngau_nhien"
        ].get(
            "r2"
        )

        feature_importance = (
            ket_qua[
                "rung_ngau_nhien"
            ].get(
                "tam_quan_trong"
            )
        )

    return (
        ols,
        rf,
        metrics,
        next_return,
        feature_importance,
    )


def phan_tich_dinh_luong(
    du_lieu: pd.DataFrame,
    bien_giai_thich: list[str] | None = None,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
):

    return run_quant(
        du_lieu,
        bien_phu_thuoc,
        bien_giai_thich,
    )


def chay_mo_hinh(
    du_lieu: pd.DataFrame,
    bien_giai_thich: list[str] | None = None,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
):

    return run_quant(
        du_lieu,
        bien_phu_thuoc,
        bien_giai_thich,
    )


# ============================================================
# HỖ TRỢ GIAO DIỆN
# ============================================================

def lay_bien_theo_nhom(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
) -> dict[str, list[str]]:

    danh_sach_hop_le = set(
        lay_danh_sach_bien(
            du_lieu,
            bien_phu_thuoc,
        )
    )

    ket_qua = {}

    for ten_nhom, cac_bien in NHOM_BIEN.items():

        bien_trong_nhom = [
            bien
            for bien
            in cac_bien
            if (
                bien
                in danh_sach_hop_le
            )
        ]

        if bien_trong_nhom:

            ket_qua[
                ten_nhom
            ] = bien_trong_nhom

    # --------------------------------------------------------
    # Các biến phái sinh chưa nằm trong nhóm
    # --------------------------------------------------------

    da_co = set()

    for cac_bien in ket_qua.values():
        da_co.update(
            cac_bien
        )

    bien_khac = [
        bien
        for bien
        in danh_sach_hop_le
        if bien not in da_co
    ]

    if bien_khac:

        ket_qua[
            "Biến phái sinh khác"
        ] = bien_khac

    return ket_qua


def lay_danh_sach_hien_thi(
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
) -> dict[str, str]:

    danh_sach = lay_danh_sach_bien(
        du_lieu,
        bien_phu_thuoc,
    )

    return {
        ten_bien_tieng_viet(
            bien
        ): bien
        for bien
        in danh_sach
    }


def chuyen_ten_hien_thi_sang_ma(
    danh_sach_hien_thi: list[str],
    du_lieu: pd.DataFrame,
    bien_phu_thuoc: str = BIEN_PHU_THUOC_MAC_DINH,
) -> list[str]:

    bang_anh_xa = (
        lay_danh_sach_hien_thi(
            du_lieu,
            bien_phu_thuoc,
        )
    )

    return [
        bang_anh_xa[x]
        for x
        in danh_sach_hien_thi
        if x in bang_anh_xa
    ]


# ============================================================
# THỐNG KÊ MÔ HÌNH
# ============================================================

def tom_tat_ket_qua(
    ket_qua,
):

    if ket_qua is None:
        return {}

    ket_qua_tom_tat = {
        "so_bien": ket_qua.get(
            "so_bien"
        ),

        "so_dong": ket_qua.get(
            "so_dong"
        ),

        "so_huan_luyen": ket_qua.get(
            "so_huan_luyen"
        ),

        "so_kiem_tra": ket_qua.get(
            "so_kiem_tra"
        ),

        "du_bao_tiep_theo": ket_qua.get(
            "du_bao_tiep_theo"
        ),

        "bien_phu_thuoc": ket_qua.get(
            "bien_phu_thuoc_tieng_viet"
        ),
    }

    hoi_quy = ket_qua.get(
        "hoi_quy"
    )

    if hoi_quy:

        ket_qua_tom_tat[
            "ols_r2"
        ] = hoi_quy.get(
            "r2"
        )

        ket_qua_tom_tat[
            "ols_mae"
        ] = hoi_quy.get(
            "mae"
        )

        ket_qua_tom_tat[
            "ols_rmse"
        ] = hoi_quy.get(
            "rmse"
        )

    else:

        ket_qua_tom_tat[
            "ols_r2"
        ] = None

        ket_qua_tom_tat[
            "ols_mae"
        ] = None

        ket_qua_tom_tat[
            "ols_rmse"
        ] = None

    rung = ket_qua.get(
        "rung_ngau_nhien"
    )

    if rung:

        ket_qua_tom_tat[
            "rf_r2"
        ] = rung.get(
            "r2"
        )

        ket_qua_tom_tat[
            "rf_mae"
        ] = rung.get(
            "mae"
        )

        ket_qua_tom_tat[
            "rf_rmse"
        ] = rung.get(
            "rmse"
        )

    else:

        ket_qua_tom_tat[
            "rf_r2"
        ] = None

        ket_qua_tom_tat[
            "rf_mae"
        ] = None

        ket_qua_tom_tat[
            "rf_rmse"
        ] = None

    return ket_qua_tom_tat
