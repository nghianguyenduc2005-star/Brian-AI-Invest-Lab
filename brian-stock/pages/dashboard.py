from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components.ai import (
    dashboard_prompt,
    render_ai_panel,
)

from components.cards import metric_card

from components.charts import (
    price_volume_chart,
)

from data.market import (
    display_symbol,
    load_market_data,
    load_vnindex_data,
    market_snapshot,
    normalize_symbol,
)

from data.news import (
    fetch_market_news,
)


# ============================================================
# CẤU HÌNH
# ============================================================

TTL_STOCK = 300
TTL_VNINDEX = 300
TTL_NEWS = 120
TTL_QUANT = 900


# ============================================================
# TIỆN ÍCH SỐ
# ============================================================

def num(
    value,
    default=None,
):
    try:

        value = float(value)

        if pd.isna(value):
            return default

        if not np.isfinite(value):
            return default

        return value

    except Exception:

        return default


def format_volume(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu cổ phiếu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn cổ phiếu"
        )

    return f"{value:,.0f} cổ phiếu"


def format_value(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:

        return (
            f"{value / 1_000_000_000_000:.2f} nghìn tỷ đồng"
        )

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ đồng"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu đồng"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn đồng"
        )

    return f"{value:,.0f} đồng"


def format_price(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng/cổ phiếu"


def format_percent(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def format_number(
    value,
    decimals=4,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:.{decimals}f}"


# ============================================================
# TÌM CỘT
# ============================================================

def find_column(
    df,
    names,
):
    if (
        df is None
        or df.empty
    ):
        return None

    mapping = {
        str(column)
        .strip()
        .lower(): column
        for column in df.columns
    }

    for name in names:

        key = (
            str(name)
            .strip()
            .lower()
        )

        if key in mapping:
            return mapping[key]

    return None


# ============================================================
# VN-INDEX
# ============================================================

@st.cache_data(
    ttl=TTL_VNINDEX,
    show_spinner=False,
)
def get_vnindex():

    try:

        df = load_vnindex_data()

    except Exception as error:

        return {
            "error": str(error)
        }

    if (
        df is None
        or df.empty
    ):

        return {
            "error": (
                "Không có dữ liệu VN-INDEX."
            )
        }

    df = df.copy()

    price_column = find_column(
        df,
        [
            "Close",
            "close",
            "last",
            "price",
            "index",
            "Đóng cửa",
        ],
    )

    if price_column is None:

        return {
            "error": (
                "Không tìm thấy cột điểm VN-INDEX."
            )
        }

    df[
        price_column
    ] = pd.to_numeric(
        df[
            price_column
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            price_column
        ]
    )

    if df.empty:

        return {
            "error": (
                "VN-INDEX không có điểm hợp lệ."
            )
        }

    current = float(
        df[
            price_column
        ].iloc[-1]
    )

    if len(df) >= 2:

        previous = float(
            df[
                price_column
            ].iloc[-2]
        )

        change = (
            current
            - previous
        )

        if previous != 0:

            change_percent = (
                change
                / previous
                * 100
            )

        else:

            change_percent = 0.0

    else:

        change = None
        change_percent = None

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_column = find_column(
        df,
        [
            "Volume",
            "volume",
            "Vol",
            "total_volume",
            "match_volume",
            "matchvolume",
        ],
    )

    volume = None

    if volume_column is not None:

        df[
            volume_column
        ] = pd.to_numeric(
            df[
                volume_column
            ],
            errors="coerce",
        )

        volume = num(
            df[
                volume_column
            ].iloc[-1]
        )

    # --------------------------------------------------------
    # Value
    # --------------------------------------------------------

    value_column = find_column(
        df,
        [
            "Value",
            "value",
            "ValueTraded",
            "value_traded",
            "trading_value",
            "traded_value",
            "match_value",
            "matchvalue",
            "turnover",
        ],
    )

    traded_value = None

    if value_column is not None:

        df[
            value_column
        ] = pd.to_numeric(
            df[
                value_column
            ],
            errors="coerce",
        )

        traded_value = num(
            df[
                value_column
            ].iloc[-1]
        )

    return {
        "price": current,
        "change": change,
        "change_percent": change_percent,
        "volume": volume,
        "value": traded_value,
        "df": df,
        "error": None,
    }


# ============================================================
# LOAD CỔ PHIẾU
# ============================================================

@st.cache_data(
    ttl=TTL_STOCK,
    show_spinner=False,
)
def _load_stock(
    symbol,
):

    return load_market_data(
        symbol,
        "1y",
    )


# ============================================================
# LOAD NEWS
# ============================================================

@st.cache_data(
    ttl=TTL_NEWS,
    show_spinner=False,
)
def _load_news():

    try:

        return fetch_market_news(
            6
        )

    except Exception:

        return []


# ============================================================
# DATA -> CONTEXT QUANT
# ============================================================

def _lay_toan_bo_bien_so(
    df,
):
    """
    Lấy TOÀN BỘ cột số có trong data.market.

    Không bỏ RSI, MACD, EMA, SMA, Bollinger,
    volatility, volume, ATR, momentum,
    high/low distance...
    """

    if (
        df is None
        or df.empty
    ):
        return []

    cac_bien = []

    for column in df.columns:

        if str(column).lower() in {
            "time",
            "date",
            "datetime",
        }:

            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            cac_bien.append(
                column
            )

    return cac_bien


# ============================================================
# PREPARE QUANT DATA
# ============================================================

def _prepare_quant_data(
    df,
):
    """
    Target = Return của phiên kế tiếp.

    Feature:
    TOÀN BỘ biến số mà data.market đã tạo.
    """

    if (
        df is None
        or df.empty
    ):

        return None, None, []

    work = df.copy()

    if "Return" not in work.columns:

        return None, None, []

    feature_columns = (
        _lay_toan_bo_bien_so(
            work
        )
    )

    # Không cho target hiện tại trở thành feature.
    feature_columns = [
        x
        for x in feature_columns
        if x not in {
            "Return",
            "ReturnPct",
        }
    ]

    if not feature_columns:

        return None, None, []

    work[
        "Target"
    ] = (
        work[
            "Return"
        ].shift(-1)
    )

    # --------------------------------------------------------
    # Chỉ giữ feature + target
    # --------------------------------------------------------

    feature_columns = list(
        dict.fromkeys(
            feature_columns
        )
    )

    work = work[
        feature_columns
        + [
            "Target"
        ]
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for column in feature_columns:

        work[
            column
        ] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work[
        "Target"
    ] = pd.to_numeric(
        work[
            "Target"
        ],
        errors="coerce",
    )

    work = work.dropna(
        axis=0,
        how="all",
    )

    # --------------------------------------------------------
    # Loại feature toàn NaN
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in feature_columns
        if work[
            column
        ].notna().sum() >= 20
    ]

    if not feature_columns:

        return None, None, []

    X = work[
        feature_columns
    ].copy()

    y = work[
        "Target"
    ].copy()

    # --------------------------------------------------------
    # Median fill
    # --------------------------------------------------------

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    valid = (
        y.notna()
        & np.isfinite(y)
    )

    X = X.loc[
        valid
    ]

    y = y.loc[
        valid
    ]

    if len(X) < 80:

        return None, None, []

    return (
        X,
        y,
        feature_columns,
    )


# ============================================================
# QUANT METRICS
# ============================================================

def _regression_metrics(
    y_true,
    y_pred,
):
    try:

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        y_true = np.asarray(
            y_true,
            dtype=float,
        )

        y_pred = np.asarray(
            y_pred,
            dtype=float,
        )

        mae = float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        )

        mse = float(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        rmse = float(
            np.sqrt(mse)
        )

        r2 = float(
            r2_score(
                y_true,
                y_pred,
            )
        )

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }

    except Exception:

        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }


# ============================================================
# CHẠY TẤT CẢ MODEL + TEST
# ============================================================

@st.cache_data(
    ttl=TTL_QUANT,
    show_spinner=False,
)
def run_full_quant(
    df,
):
    """
    Nghiên cứu định lượng đầy đủ.

    Bao gồm:
    - OLS HC3
    - Ridge
    - Lasso
    - ElasticNet
    - Random Forest
    - Extra Trees
    - Gradient Boosting

    Kiểm định:
    - ADF
    - Jarque-Bera
    - Breusch-Pagan
    - White
    - Durbin-Watson
    - Ljung-Box
    - VIF
    """

    result = {
        "ok": False,
        "error": None,
        "feature_count": 0,
        "features": [],
        "train_size": 0,
        "test_size": 0,
        "models": [],
        "feature_importance": None,
        "tests": {},
        "ols": None,
    }

    X, y, feature_columns = (
        _prepare_quant_data(
            df
        )
    )

    if (
        X is None
        or y is None
    ):

        result[
            "error"
        ] = (
            "Không đủ dữ liệu để chạy nghiên cứu định lượng."
        )

        return result

    result[
        "feature_count"
    ] = len(
        feature_columns
    )

    result[
        "features"
    ] = list(
        feature_columns
    )

    # ========================================================
    # TIME SPLIT
    # ========================================================

    split = int(
        len(X) * 0.8
    )

    if (
        split < 50
        or len(X) - split < 20
    ):

        result[
            "error"
        ] = (
            "Tập train/test không đủ kích thước."
        )

        return result

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_train = y.iloc[
        :split
    ]

    y_test = y.iloc[
        split:
    ]

    result[
        "train_size"
    ] = len(
        X_train
    )

    result[
        "test_size"
    ] = len(
        X_test
    )

    # ========================================================
    # OLS
    # ========================================================

    try:

        import statsmodels.api as sm

        X_train_const = sm.add_constant(
            X_train,
            has_constant="add",
        )

        X_test_const = sm.add_constant(
            X_test,
            has_constant="add",
        )

        ols_model = sm.OLS(
            y_train,
            X_train_const,
        ).fit(
            cov_type="HC3"
        )

        ols_pred = ols_model.predict(
            X_test_const
        )

        result[
            "models"
        ].append(
            {
                "Mô hình": "OLS + HC3",
                **_regression_metrics(
                    y_test,
                    ols_pred,
                ),
            }
        )

        result[
            "ols"
        ] = ols_model

    except Exception as error:

        result[
            "models"
        ].append(
            {
                "Mô hình": "OLS + HC3",
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "Lỗi": str(error),
            }
        )

    # ========================================================
    # STANDARDIZATION FOR LINEAR MODELS
    # ========================================================

    try:

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from sklearn.linear_model import (
            Ridge,
            Lasso,
            ElasticNet,
        )

        linear_models = [
            (
                "Ridge",
                Ridge(
                    alpha=1.0
                ),
            ),

            (
                "Lasso",
                Lasso(
                    alpha=0.0001,
                    max_iter=20000,
                    random_state=42,
                ),
            ),

            (
                "Elastic Net",
                ElasticNet(
                    alpha=0.0001,
                    l1_ratio=0.5,
                    max_iter=20000,
                    random_state=42,
                ),
            ),
        ]

        for model_name, estimator in linear_models:

            try:

                model = Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                        (
                            "model",
                            estimator,
                        ),
                    ]
                )

                model.fit(
                    X_train,
                    y_train,
                )

                pred = model.predict(
                    X_test
                )

                result[
                    "models"
                ].append(
                    {
                        "Mô hình": model_name,
                        **_regression_metrics(
                            y_test,
                            pred,
                        ),
                    }
                )

            except Exception as error:

                result[
                    "models"
                ].append(
                    {
                        "Mô hình": model_name,
                        "MAE": np.nan,
                        "MSE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "Lỗi": str(error),
                    }
                )

    except Exception:
        pass

    # ========================================================
    # TREE MODELS
    # ========================================================

    tree_models = []

    try:

        from sklearn.ensemble import (
            RandomForestRegressor,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
        )

        tree_models = [
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),

            (
                "Extra Trees",
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),

            (
                "Gradient Boosting",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=3,
                    min_samples_leaf=4,
                    random_state=42,
                ),
            ),
        ]

    except Exception:

        tree_models = []

    for model_name, model in tree_models:

        try:

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(
                X_test
            )

            result[
                "models"
            ].append(
                {
                    "Mô hình": model_name,
                    **_regression_metrics(
                        y_test,
                        pred,
                    ),
                }
            )

            # --------------------------------------------
            # Feature importance
            # --------------------------------------------

            importance = getattr(
                model,
                "feature_importances_",
                None,
            )

            if importance is not None:

                importance_df = pd.DataFrame(
                    {
                        "Biến": feature_columns,
                        "Mức quan trọng": importance,
                    }
                )

                importance_df = (
                    importance_df
                    .sort_values(
                        "Mức quan trọng",
                        ascending=False,
                    )
                    .reset_index(
                        drop=True
                    )
                )

                # Lấy model cây tốt nhất trong nhóm.
                if result[
                    "feature_importance"
                ] is None:

                    result[
                        "feature_importance"
                    ] = importance_df

        except Exception as error:

            result[
                "models"
            ].append(
                {
                    "Mô hình": model_name,
                    "MAE": np.nan,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Lỗi": str(error),
                }
            )

    # ========================================================
    # STATISTICAL TESTS
    # ========================================================

    try:

        from statsmodels.stats.diagnostic import (
            acorr_ljungbox,
            het_breuschpagan,
            het_white,
        )

        from statsmodels.stats.stattools import (
            durbin_watson,
            jarque_bera,
        )

        from statsmodels.tsa.stattools import (
            adfuller,
        )

        residuals = None

        if result[
            "ols"
        ] is not None:

            residuals = np.asarray(
                result[
                    "ols"
                ].resid,
                dtype=float,
            )

        # ----------------------------------------------------
        # ADF
        # ----------------------------------------------------

        try:

            adf = adfuller(
                y.dropna()
            )

            result[
                "tests"
            ][
                "ADF"
            ] = {
                "statistic": float(
                    adf[0]
                ),
                "p_value": float(
                    adf[1]
                ),
            }

        except Exception:
            pass

        # ----------------------------------------------------
        # Jarque Bera
        # ----------------------------------------------------

        if residuals is not None:

            try:

                jb = jarque_bera(
                    residuals
                )

                result[
                    "tests"
                ][
                    "Jarque-Bera"
                ] = {
                    "statistic": float(
                        jb[0]
                    ),
                    "p_value": float(
                        jb[1]
                    ),
                    "skew": float(
                        jb[2]
                    ),
                    "kurtosis": float(
                        jb[3]
                    ),
                }

            except Exception:
                pass

        # ----------------------------------------------------
        # Durbin Watson
        # ----------------------------------------------------

        if residuals is not None:

            try:

                dw = durbin_watson(
                    residuals
                )

                result[
                    "tests"
                ][
                    "Durbin-Watson"
                ] = {
                    "statistic": float(
                        dw
                    )
                }

            except Exception:
                pass

        # ----------------------------------------------------
        # Breusch Pagan
        # ----------------------------------------------------

        if (
            result[
                "ols"
            ] is not None
        ):

            try:

                X_ols = sm.add_constant(
                    X_train,
                    has_constant="add",
                )

                bp = het_breuschpagan(
                    result[
                        "ols"
                    ].resid,
                    X_ols,
                )

                result[
                    "tests"
                ][
                    "Breusch-Pagan"
                ] = {
                    "LM": float(
                        bp[0]
                    ),
                    "p_value": float(
                        bp[1]
                    ),
                    "F": float(
                        bp[2]
                    ),
                    "F_p_value": float(
                        bp[3]
                    ),
                }

            except Exception:
                pass

        # ----------------------------------------------------
        # White
        # ----------------------------------------------------

        if (
            result[
                "ols"
            ] is not None
        ):

            try:

                # White test có thể rất nặng khi quá nhiều biến.
                # Dùng subset tối đa 12 biến variance-rich.
                white_features = (
                    feature_columns[
                        :min(
                            12,
                            len(
                                feature_columns
                            ),
                        )
                    ]
                )

                X_white = sm.add_constant(
                    X_train[
                        white_features
                    ],
                    has_constant="add",
                )

                white = het_white(
                    result[
                        "ols"
                    ].resid,
                    X_white,
                )

                result[
                    "tests"
                ][
                    "White"
                ] = {
                    "LM": float(
                        white[0]
                    ),
                    "p_value": float(
                        white[1]
                    ),
                    "F": float(
                        white[2]
                    ),
                    "F_p_value": float(
                        white[3]
                    ),
                }

            except Exception:
                pass

        # ----------------------------------------------------
        # Ljung Box
        # ----------------------------------------------------

        if residuals is not None:

            try:

                lag = min(
                    10,
                    max(
                        1,
                        len(
                            residuals
                        )
                        // 10,
                    ),
                )

                lb = acorr_ljungbox(
                    residuals,
                    lags=[
                        lag
                    ],
                    return_df=True,
                )

                result[
                    "tests"
                ][
                    "Ljung-Box"
                ] = {
                    "lag": lag,
                    "statistic": float(
                        lb[
                            "lb_stat"
                        ].iloc[-1]
                    ),
                    "p_value": float(
                        lb[
                            "lb_pvalue"
                        ].iloc[-1]
                    ),
                }

            except Exception:
                pass

    except Exception:
        pass

    # ========================================================
    # VIF
    # ========================================================

    try:

        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor,
        )

        # VIF với toàn bộ biến có thể cực nặng.
        # Chuẩn hóa trước và xử lý inf.
        X_vif = X_train.copy()

        X_vif = X_vif.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        X_vif = X_vif.fillna(
            X_vif.median()
        )

        # Giới hạn số dòng/cột không bỏ biến trong
        # nghiên cứu mô hình chính; VIF chỉ là kiểm định bổ sung.
        if X_vif.shape[1] <= 35:

            vif_rows = []

            values = (
                X_vif.astype(float)
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .fillna(0)
            )

            for index, column in enumerate(
                values.columns
            ):

                try:

                    vif_value = (
                        variance_inflation_factor(
                            values.values,
                            index,
                        )
                    )

                    if not np.isfinite(
                        vif_value
                    ):

                        vif_value = np.nan

                except Exception:

                    vif_value = np.nan

                vif_rows.append(
                    {
                        "Biến": column,
                        "VIF": vif_value,
                    }
                )

            vif_df = pd.DataFrame(
                vif_rows
            )

            result[
                "tests"
            ][
                "VIF"
            ] = vif_df

    except Exception:
        pass

    # ========================================================
    # BEST MODEL
    # ========================================================

    model_df = pd.DataFrame(
        result[
            "models"
        ]
    )

    if not model_df.empty:

        model_df[
            "RMSE"
        ] = pd.to_numeric(
            model_df[
                "RMSE"
            ],
            errors="coerce",
        )

        model_df = (
            model_df
            .sort_values(
                "RMSE",
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        result[
            "models"
        ] = model_df.to_dict(
            "records"
        )

    result[
        "ok"
    ] = True

    return result


# ============================================================
# GIẢI THÍCH MODEL
# ============================================================

MODEL_EXPLANATIONS = {
    "OLS + HC3": (
        "Hồi quy tuyến tính dùng để đo mối quan hệ "
        "có điều kiện giữa toàn bộ biến giải thích và "
        "lợi suất phiên kế tiếp. HC3 giúp sai số chuẩn "
        "bền vững hơn khi có phương sai thay đổi."
    ),

    "Ridge": (
        "Hồi quy tuyến tính có regularization L2. "
        "Hữu ích khi nhiều biến có tương quan mạnh "
        "và OLS dễ gặp đa cộng tuyến."
    ),

    "Lasso": (
        "Hồi quy tuyến tính có regularization L1. "
        "Có khả năng làm một số hệ số về 0, "
        "qua đó chỉ ra các biến ít hữu ích trong mô hình."
    ),

    "Elastic Net": (
        "Kết hợp L1 và L2. Thường hữu ích khi có "
        "nhiều biến tương quan với nhau nhưng vẫn muốn "
        "một mức chọn lọc biến."
    ),

    "Random Forest": (
        "Tập hợp nhiều cây quyết định. Có khả năng bắt "
        "quan hệ phi tuyến và tương tác giữa các chỉ báo "
        "mà hồi quy tuyến tính khó biểu diễn."
    ),

    "Extra Trees": (
        "Biến thể ngẫu nhiên hóa mạnh hơn Random Forest. "
        "Có thể giảm phương sai và cung cấp một góc nhìn "
        "khác về quan hệ phi tuyến."
    ),

    "Gradient Boosting": (
        "Xây dựng cây tuần tự để sửa sai cho mô hình trước. "
        "Có khả năng bắt tín hiệu phi tuyến nhưng cần "
        "kiểm soát overfitting."
    ),
}


TEST_EXPLANATIONS = {
    "ADF": (
        "Kiểm tra tính dừng của chuỗi. "
        "p-value thấp thường ủng hộ giả thuyết chuỗi dừng."
    ),

    "Jarque-Bera": (
        "Kiểm tra phần dư có phù hợp với phân phối chuẩn hay không. "
        "p-value thấp cho thấy phần dư lệch khỏi chuẩn."
    ),

    "Breusch-Pagan": (
        "Kiểm tra phương sai phần dư có thay đổi theo biến giải thích hay không. "
        "p-value thấp gợi ý heteroskedasticity."
    ),

    "White": (
        "Một kiểm định phương sai thay đổi tổng quát hơn Breusch-Pagan. "
        "p-value thấp gợi ý phương sai không đồng nhất."
    ),

    "Durbin-Watson": (
        "Kiểm tra tự tương quan bậc một của phần dư. "
        "Giá trị gần 2 thường là tín hiệu ít tự tương quan."
    ),

    "Ljung-Box": (
        "Kiểm tra tự tương quan tổng thể của phần dư qua nhiều độ trễ. "
        "p-value thấp cho thấy còn cấu trúc chuỗi thời gian chưa được giải thích."
    ),

    "VIF": (
        "Đo mức đa cộng tuyến giữa các biến giải thích. "
        "VIF cao cho thấy các biến chứa thông tin trùng lặp mạnh."
    ),
}


# ============================================================
# RENDER DASHBOARD
# ============================================================

def render_dashboard():

    # ========================================================
    # HERO
    # ========================================================

    st.caption(
        "BRIAN STOCK · INVESTMENT INTELLIGENCE"
    )

    st.title(
        "Góc nhìn dữ liệu cho nhà đầu tư"
    )

    st.write(
        "Dashboard nghiên cứu thị trường, cổ phiếu, "
        "tin tức, AI và nghiên cứu định lượng. "
        "Dữ liệu được lấy từ nguồn thị trường thực."
    )

    # ========================================================
    # VN-INDEX
    # ========================================================

    vn = get_vnindex()

    if vn.get(
        "error"
    ):

        vn_price = "—"
        vn_change = "—"
        vn_percent = "—"
        vn_volume = "—"
        vn_value = "—"

    else:

        vn_price = (
            f"{vn['price']:,.2f} điểm"
            if vn.get(
                "price"
            ) is not None
            else "—"
        )

        vn_change = (
            f"{vn['change']:+,.2f} điểm"
            if vn.get(
                "change"
            ) is not None
            else "—"
        )

        vn_percent = (
            f"{vn['change_percent']:+.2f}%"
            if vn.get(
                "change_percent"
            ) is not None
            else "—"
        )

        vn_volume = (
            format_volume(
                vn.get(
                    "volume"
                )
            )
        )

        vn_value = (
            format_value(
                vn.get(
                    "value"
                )
            )
        )

    # ========================================================
    # QUICK VIEW
    # ========================================================

    st.subheader(
        "📌 Theo dõi nhanh"
    )

    quick_1, quick_2, quick_3, quick_4 = (
        st.columns(4)
    )

    with quick_1:

        metric_card(
            "VN-INDEX",
            vn_price,
            "Điểm chỉ số thị trường",
        )

    with quick_2:

        metric_card(
            "Khối lượng",
            vn_volume,
            "Khối lượng giao dịch toàn thị trường",
        )

    with quick_3:

        metric_card(
            "Giá trị giao dịch",
            vn_value,
            "Tổng giá trị giao dịch",
        )

    with quick_4:

        metric_card(
            "Tin tức",
            "LIVE",
            "Nguồn tin thị trường mới",
        )

    # ========================================================
    # VNINDEX DETAIL
    # ========================================================

    st.subheader(
        "📊 VN-INDEX"
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "Điểm",
            vn_price,
        )

    with a2:

        st.metric(
            "Thay đổi",
            vn_change,
        )

    with a3:

        st.metric(
            "Thay đổi 1D",
            vn_percent,
        )

    with a4:

        st.metric(
            "Khối lượng",
            vn_volume,
        )

    if vn.get(
        "error"
    ):

        st.warning(
            "VN-INDEX chưa tải được: "
            + str(
                vn.get(
                    "error"
                )
            )
        )

    # ========================================================
    # STOCK
    # ========================================================

    st.subheader(
        "📈 Theo dõi cổ phiếu"
    )

    default_symbol = (
        st.session_state.get(
            "dashboard_symbol",
            "HPG",
        )
    )

    symbol_input = st.text_input(
        "Mã cổ phiếu",
        value=default_symbol,
        label_visibility="collapsed",
        placeholder="Ví dụ HPG, FPT, VNM...",
        key="dashboard_stock_input",
    )

    if st.button(
        "🔄 Tải dữ liệu",
        type="primary",
        key="dashboard_load_button",
    ):

        clean_symbol = normalize_symbol(
            symbol_input
        )

        if not clean_symbol:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

        else:

            st.session_state[
                "dashboard_symbol"
            ] = clean_symbol

            st.rerun()

    current_symbol = normalize_symbol(
        st.session_state.get(
            "dashboard_symbol",
            symbol_input,
        )
    )

    # ========================================================
    # LOAD STOCK
    # ========================================================

    try:

        stock = _load_stock(
            current_symbol
        )

    except Exception as error:

        st.error(
            f"Không lấy được dữ liệu "
            f"{display_symbol(current_symbol)}."
        )

        st.code(
            str(error)
        )

        stock = None

    # ========================================================
    # STOCK SNAPSHOT
    # ========================================================

    stock_snapshot = {}

    stock_last_row = None

    if (
        stock is not None
        and not stock.empty
    ):

        stock_snapshot = market_snapshot(
            stock
        )

        stock_last_row = (
            stock.iloc[-1]
        )

        price = num(
            stock_snapshot.get(
                "price"
            )
        )

        change_1d = num(
            stock_snapshot.get(
                "change_1d"
            )
        )

        rsi_value = num(
            stock_snapshot.get(
                "rsi"
            )
        )

        stock_volume = num(
            stock_snapshot.get(
                "volume"
            )
        )

        sma20 = num(
            stock_snapshot.get(
                "sma20"
            )
        )

        sma50 = num(
            stock_snapshot.get(
                "sma50"
            )
        )

        macd = num(
            stock_snapshot.get(
                "macd"
            )
        )

        volatility = num(
            stock_snapshot.get(
                "volatility20"
            )
        )

        atr14 = num(
            stock_last_row.get(
                "ATR14"
            )
        )

        volume_sma20 = num(
            stock_last_row.get(
                "Volume_SMA20"
            )
        )

        relative_volume = None

        if (
            stock_volume is not None
            and volume_sma20 is not None
            and volume_sma20 != 0
        ):

            relative_volume = (
                stock_volume
                / volume_sma20
            )

        # ====================================================
        # TITLE
        # ====================================================

        st.subheader(
            f"📈 {display_symbol(current_symbol)}"
        )

        # ====================================================
        # MAIN METRICS
        # ====================================================

        s1, s2, s3, s4 = (
            st.columns(4)
        )

        with s1:

            st.metric(
                "Giá",
                format_price(
                    price
                ),
            )

        with s2:

            st.metric(
                "Thay đổi 1D",
                format_percent(
                    change_1d
                ),
            )

        with s3:

            st.metric(
                "RSI",
                (
                    f"{rsi_value:.1f}"
                    if rsi_value is not None
                    else "—"
                ),
            )

        with s4:

            st.metric(
                "Khối lượng",
                format_volume(
                    stock_volume
                ),
            )

        # ====================================================
        # INDICATORS
        # ====================================================

        s5, s6, s7, s8 = (
            st.columns(4)
        )

        with s5:

            st.metric(
                "Trung bình 20 phiên",
                format_price(
                    sma20
                ),
            )

        with s6:

            st.metric(
                "Trung bình 50 phiên",
                format_price(
                    sma50
                ),
            )

        with s7:

            st.metric(
                "MACD",
                (
                    f"{macd:.3f}"
                    if macd is not None
                    else "—"
                ),
            )

        with s8:

            st.metric(
                "Biến động 20 phiên",
                (
                    f"{volatility:.2f}%"
                    if volatility is not None
                    else "—"
                ),
            )

        # ====================================================
        # EXTRA
        # ====================================================

        e1, e2, e3, e4 = (
            st.columns(4)
        )

        with e1:

            st.metric(
                "ATR 14",
                format_price(
                    atr14
                ),
            )

        with e2:

            st.metric(
                "Khối lượng TB20",
                format_volume(
                    volume_sma20
                ),
            )

        with e3:

            st.metric(
                "Khối lượng / TB20",
                (
                    f"{relative_volume:.2f} lần"
                    if relative_volume is not None
                    else "—"
                ),
            )

        with e4:

            open_price = num(
                stock_last_row.get(
                    "Open"
                )
            )

            st.metric(
                "Giá mở cửa",
                format_price(
                    open_price
                ),
            )

        # ====================================================
        # CHART
        # ====================================================

        st.subheader(
            "📊 Biểu đồ kỹ thuật"
        )

        try:

            chart = price_volume_chart(
                stock
            )

            if chart is not None:

                st.plotly_chart(
                    chart,
                    width="stretch",
                    config={
                        "displaylogo": False,
                    },
                )

        except Exception as error:

            st.warning(
                "Không thể hiển thị biểu đồ."
            )

            st.code(
                str(error)
            )

    # ========================================================
    # NEWS
    # ========================================================

    st.subheader(
        "📰 Tin mới"
    )

    news = _load_news()

    if not news:

        st.info(
            "Hiện chưa lấy được tin tức mới."
        )

    else:

        for item in news:

            title = str(
                item.get(
                    "title",
                    "Không có tiêu đề",
                )
            ).strip()

            source = str(
                item.get(
                    "source",
                    "Nguồn không xác định",
                )
            ).strip()

            published = str(
                item.get(
                    "published",
                    "",
                )
            ).strip()

            link = str(
                item.get(
                    "link",
                    "",
                )
            ).strip()

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{title}**"
                )

                if source and published:

                    st.caption(
                        f"{source} · {published}"
                    )

                elif source:

                    st.caption(
                        source
                    )

                elif published:

                    st.caption(
                        published
                    )

                if link:

                    st.markdown(
                        f"[Đọc bài ↗]({link})"
                    )

    # ========================================================
    # NGHIÊN CỨU ĐỊNH LƯỢNG
    # ========================================================

    st.subheader(
        "🧮 Nghiên cứu định lượng"
    )

    st.caption(
        "Khi chạy, hệ thống sử dụng toàn bộ biến số "
        "có trong dữ liệu kỹ thuật của mã đang xem, "
        "sau đó chạy đồng thời nhiều mô hình và kiểm định."
    )

    if (
        stock is None
        or stock.empty
    ):

        st.info(
            "Cần tải dữ liệu cổ phiếu trước khi chạy nghiên cứu định lượng."
        )

    else:

        run_quant = st.button(
            "🧮 Chạy nghiên cứu định lượng",
            type="primary",
            key="dashboard_run_full_quant",
        )

        if run_quant:

            with st.spinner(
                "Đang chạy toàn bộ mô hình và kiểm định..."
            ):

                quant_result = run_full_quant(
                    stock
                )

            st.session_state[
                "dashboard_quant_result"
            ] = quant_result

            st.session_state[
                "dashboard_quant_symbol"
            ] = current_symbol

        quant_result = (
            st.session_state.get(
                "dashboard_quant_result"
            )
        )

        quant_symbol = (
            st.session_state.get(
                "dashboard_quant_symbol"
            )
        )

        # Nếu đổi mã -> không dùng kết quả mã cũ.
        if quant_symbol != current_symbol:

            quant_result = None

        if quant_result is None:

            st.info(
                "Bấm «Chạy nghiên cứu định lượng» "
                "để chạy toàn bộ mô hình và kiểm định."
            )

        elif not quant_result.get(
            "ok",
            False,
        ):

            st.error(
                quant_result.get(
                    "error",
                    "Nghiên cứu định lượng thất bại."
                )
            )

        else:

            # =================================================
            # SUMMARY
            # =================================================

            st.markdown(
                "### 📌 Tổng quan nghiên cứu"
            )

            q1, q2, q3, q4 = (
                st.columns(4)
            )

            with q1:

                st.metric(
                    "Số biến sử dụng",
                    str(
                        quant_result.get(
                            "feature_count",
                            0,
                        )
                    ),
                )

            with q2:

                st.metric(
                    "Tập train",
                    str(
                        quant_result.get(
                            "train_size",
                            0,
                        )
                    ),
                )

            with q3:

                st.metric(
                    "Tập test",
                    str(
                        quant_result.get(
                            "test_size",
                            0,
                        )
                    ),
                )

            with q4:

                st.metric(
                    "Số mô hình",
                    str(
                        len(
                            quant_result.get(
                                "models",
                                [],
                            )
                        )
                    ),
                )

            # =================================================
            # TOÀN BỘ BIẾN
            # =================================================

            st.markdown(
                "### 🔢 Toàn bộ biến được sử dụng"
            )

            features = quant_result.get(
                "features",
                [],
            )

            if features:

                feature_df = pd.DataFrame(
                    {
                        "Biến": features
                    }
                )

                st.dataframe(
                    feature_df,
                    width="stretch",
                    height=320,
                    hide_index=True,
                )

            # =================================================
            # MODEL COMPARISON
            # =================================================

            st.markdown(
                "### 🤖 So sánh mô hình"
            )

            model_df = pd.DataFrame(
                quant_result.get(
                    "models",
                    [],
                )
            )

            if not model_df.empty:

                display_model_df = (
                    model_df.copy()
                )

                for column in [
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R2",
                ]:

                    if column in display_model_df.columns:

                        display_model_df[
                            column
                        ] = pd.to_numeric(
                            display_model_df[
                                column
                            ],
                            errors="coerce",
                        ).round(6)

                st.dataframe(
                    display_model_df,
                    width="stretch",
                    hide_index=True,
                )

                # ---------------------------------------------
                # Best model
                # ---------------------------------------------

                valid_models = (
                    model_df[
                        pd.to_numeric(
                            model_df["RMSE"],
                            errors="coerce",
                        ).notna()
                    ]
                    if "RMSE" in model_df.columns
                    else pd.DataFrame()
                )

                if not valid_models.empty:

                    best_model = (
                        valid_models
                        .sort_values(
                            "RMSE"
                        )
                        .iloc[0]
                    )

                    st.success(
                        "Mô hình có RMSE thấp nhất trên tập test: "
                        f"{best_model['Mô hình']} "
                        f"(RMSE = {best_model['RMSE']:.6f})"
                    )

            # =================================================
            # MODEL EXPLANATION
            # =================================================

            st.markdown(
                "### 📖 Ý nghĩa từng mô hình"
            )

            for model_name in [
                "OLS + HC3",
                "Ridge",
                "Lasso",
                "Elastic Net",
                "Random Forest",
                "Extra Trees",
                "Gradient Boosting",
            ]:

                with st.expander(
                    model_name
                ):

                    st.write(
                        MODEL_EXPLANATIONS.get(
                            model_name,
                            "Mô hình định lượng."
                        )
                    )

            # =================================================
            # FEATURE IMPORTANCE
            # =================================================

            importance_df = (
                quant_result.get(
                    "feature_importance"
                )
            )

            if (
                importance_df is not None
                and not importance_df.empty
            ):

                st.markdown(
                    "### 🌳 Feature importance"
                )

                importance_show = (
                    importance_df.head(
                        30
                    ).copy()
                )

                importance_show[
                    "Mức quan trọng"
                ] = (
                    pd.to_numeric(
                        importance_show[
                            "Mức quan trọng"
                        ],
                        errors="coerce",
                    )
                    .round(6)
                )

                st.dataframe(
                    importance_show,
                    width="stretch",
                    hide_index=True,
                )

            # =================================================
            # STATISTICAL TESTS
            # =================================================

            st.markdown(
                "### 🧪 Kiểm định thống kê"
            )

            tests = quant_result.get(
                "tests",
                {},
            )

            # ---------------------------------------------
            # ADF
            # ---------------------------------------------

            if "ADF" in tests:

                adf = tests[
                    "ADF"
                ]

                with st.expander(
                    "ADF — Augmented Dickey-Fuller"
                ):

                    st.write(
                        TEST_EXPLANATIONS[
                            "ADF"
                        ]
                    )

                    st.metric(
                        "p-value",
                        format_number(
                            adf.get(
                                "p_value"
                            ),
                            6,
                        ),
                    )

                    if (
                        adf.get(
                            "p_value"
                        ) is not None
                        and adf.get(
                            "p_value"
                        ) < 0.05
                    ):

                        st.success(
                            "p-value < 0.05: có bằng chứng chuỗi dừng."
                        )

                    else:

                        st.info(
                            "p-value >= 0.05: chưa đủ bằng chứng bác bỏ giả thuyết có nghiệm đơn vị."
                        )

            # ---------------------------------------------
            # Jarque Bera
            # ---------------------------------------------

            if "Jarque-Bera" in tests:

                jb = tests[
                    "Jarque-Bera"
                ]

                with st.expander(
                    "Jarque–Bera"
                ):

                    st.write(
                        TEST_EXPLANATIONS[
                            "Jarque-Bera"
                        ]
                    )

                    st.metric(
                        "p-value",
                        format_number(
                            jb.get(
                                "p_value"
                            ),
                            6,
                        ),
                    )

            # ---------------------------------------------
            # Breusch Pagan
            # ---------------------------------------------

            if "Breusch-Pagan" in tests:

                bp = tests[
                    "Breusch-Pagan"
                ]

                with st.expander(
                    "Breusch–Pagan"
                ):

                    st.write(
                        TEST_EXPLANATIONS[
                            "Breusch-Pagan"
                        ]
                    )

                    st.metric(
                        "p-value",
                        format_number(
                            bp.get(
                                "p_value"
                            ),
                            6,
                        ),
                    )

                    if (
                        bp.get(
                            "p_value"
                        ) is not None
                        and bp.get(
                            "p_value"
                        ) < 0.05
                    ):

                        st.warning(
                            "Có dấu hiệu phương sai thay đổi."
                        )

                    else:

                        st.success(
                            "Chưa thấy bằng chứng rõ về phương sai thay đổi."
                        )

            # ---------------------------------------------
            # White
            # ---------------------------------------------

            if "White" in tests:

                white = tests[
                    "White"
                ]

                with st.expander(
                    "White Test"
                ):

                    st.write(
                        TEST_EXPLANATIONS[
                            "White"
                        ]
                    )

                    st.metric(
                        "p-value",
                        format_number(
                            white.get(
                                "p_value"
                            ),
                            6,
                        ),
                    )

            # ---------------------------------------------
            # Durbin Watson
            # ---------------------------------------------

            if "Durbin-Watson" in tests:

                dw = tests[
                    "Durbin-Watson"
                ]

                with st.expander(
                    "Durbin–Watson"
                ):

                    st.write(
                        TEST_EXPLANATIONS[
                            "Durbin-Watson"
                        ]
                    )

                    dw_value = num(
                        dw.get(
                            "statistic"
                        )
                    )

                    st.metric(
                        "Statistic",
                        (
                            f"{dw_value:.3f}"
                            if dw_value is not None
                            else "—"
                        ),
                    )

            # ---------------------------------------------
            # Ljung Box
            # ---------------------------------------------

            if "Ljung-Box" in tests:

                lb = tests[
                    "Ljung-Box"
                ]

                with st.expander(
                    "Ljung–Box"
                ):

                    st.write(
                        TEST_EXPLANATIONS[
                            "Ljung-Box"
                        ]
                    )

                    st.metric(
                        "p-value",
                        format_number(
                            lb.get(
                                "p_value"
                            ),
                            6,
                        ),
                    )

            # ---------------------------------------------
            # VIF
            # ---------------------------------------------

            if "VIF" in tests:

                vif_df = tests[
                    "VIF"
                ]

                with st.expander(
                    "VIF — Đa cộng tuyến"
                ):

                    st.write(
                        TEST_EXPLANATIONS[
                            "VIF"
                        ]
                    )

                    display_vif = (
                        vif_df.copy()
                    )

                    display_vif[
                        "VIF"
                    ] = pd.to_numeric(
                        display_vif[
                            "VIF"
                        ],
                        errors="coerce",
                    ).round(3)

                    display_vif = (
                        display_vif
                        .sort_values(
                            "VIF",
                            ascending=False,
                            na_position="last",
                        )
                    )

                    st.dataframe(
                        display_vif,
                        width="stretch",
                        height=360,
                        hide_index=True,
                    )

                    high_vif = display_vif[
                        display_vif[
                            "VIF"
                        ] > 10
                    ]

                    if not high_vif.empty:

                        st.warning(
                            "Có biến có VIF > 10, "
                            "cho thấy đa cộng tuyến mạnh."
                        )

                    else:

                        st.success(
                            "Không phát hiện VIF > 10 trong bảng kiểm định."
                        )

            # =================================================
            # KẾT LUẬN TỰ ĐỘNG
            # =================================================

            st.markdown(
                "### 🧠 Diễn giải nghiên cứu"
            )

            interpretation_lines = []

            if not model_df.empty:

                valid_models = model_df[
                    pd.to_numeric(
                        model_df[
                            "RMSE"
                        ],
                        errors="coerce",
                    ).notna()
                ]

                if not valid_models.empty:

                    best = (
                        valid_models
                        .sort_values(
                            "RMSE"
                        )
                        .iloc[0]
                    )

                    interpretation_lines.append(
                        f"Mô hình có sai số RMSE thấp nhất "
                        f"trên tập test là {best['Mô hình']}."
                    )

            if "ADF" in tests:

                p = num(
                    tests[
                        "ADF"
                    ].get(
                        "p_value"
                    )
                )

                if p is not None:

                    if p < 0.05:

                        interpretation_lines.append(
                            "ADF cho thấy chuỗi mục tiêu có bằng chứng dừng."
                        )

                    else:

                        interpretation_lines.append(
                            "ADF chưa cho đủ bằng chứng để kết luận chuỗi mục tiêu dừng."
                        )

            if "Breusch-Pagan" in tests:

                p = num(
                    tests[
                        "Breusch-Pagan"
                    ].get(
                        "p_value"
                    )
                )

                if p is not None:

                    if p < 0.05:

                        interpretation_lines.append(
                            "Breusch–Pagan cho thấy cần lưu ý phương sai thay đổi."
                        )

                    else:

                        interpretation_lines.append(
                            "Breusch–Pagan chưa cho thấy bằng chứng rõ về phương sai thay đổi."
                        )

            if "Durbin-Watson" in tests:

                dw_value = num(
                    tests[
                        "Durbin-Watson"
                    ].get(
                        "statistic"
                    )
                )

                if dw_value is not None:

                    if 1.5 <= dw_value <= 2.5:

                        interpretation_lines.append(
                            "Durbin–Watson nằm tương đối gần 2, "
                            "cho thấy không có dấu hiệu tự tương quan bậc một quá mạnh."
                        )

                    else:

                        interpretation_lines.append(
                            "Durbin–Watson lệch đáng kể khỏi 2, "
                            "nên kiểm tra thêm cấu trúc tự tương quan của phần dư."
                        )

            if "Ljung-Box" in tests:

                p = num(
                    tests[
                        "Ljung-Box"
                    ].get(
                        "p_value"
                    )
                )

                if p is not None:

                    if p < 0.05:

                        interpretation_lines.append(
                            "Ljung–Box cho thấy phần dư còn có cấu trúc tự tương quan đáng chú ý."
                        )

                    else:

                        interpretation_lines.append(
                            "Ljung–Box chưa phát hiện bằng chứng mạnh về tự tương quan còn sót lại."
                        )

            if "VIF" in tests:

                vif_df = tests[
                    "VIF"
                ]

                if (
                    isinstance(
                        vif_df,
                        pd.DataFrame,
                    )
                    and not vif_df.empty
                ):

                    max_vif = num(
                        vif_df[
                            "VIF"
                        ].max()
                    )

                    if max_vif is not None:

                        if max_vif > 10:

                            interpretation_lines.append(
                                "Có đa cộng tuyến mạnh ở ít nhất một biến "
                                "dựa trên VIF."
                            )

                        elif max_vif > 5:

                            interpretation_lines.append(
                                "Một số biến có đa cộng tuyến ở mức cần lưu ý."
                            )

                        else:

                            interpretation_lines.append(
                                "Đa cộng tuyến chưa ở mức quá nghiêm trọng theo VIF."
                            )

            if not interpretation_lines:

                interpretation_lines.append(
                    "Chưa đủ kiểm định để đưa ra diễn giải tự động."
                )

            for line in interpretation_lines:

                st.write(
                    "• "
                    + line
                )

    # ========================================================
    # AI DASHBOARD
    # ========================================================

    ai_vn_index = {
        "diem": vn.get(
            "price"
        ),
        "thay_doi": vn.get(
            "change"
        ),
        "phan_tram": vn.get(
            "change_percent"
        ),
        "khoi_luong": vn.get(
            "volume"
        ),
        "gia_tri": vn.get(
            "value"
        ),
    }

    ai_stock_snapshot = (
        stock_snapshot
        if isinstance(
            stock_snapshot,
            dict,
        )
        else {}
    )

    ai_news = (
        news
        if isinstance(
            news,
            list,
        )
        else []
    )

    ai_prompt = dashboard_prompt(
        vn_index=ai_vn_index,
        stock_symbol=display_symbol(
            current_symbol
        ),
        stock_snapshot=ai_stock_snapshot,
        news=ai_news,
    )

    render_ai_panel(
        title="🤖 AI Market Brief",
        description=(
            "AI tổng hợp VN-INDEX, cổ phiếu đang theo dõi "
            "và tin tức thị trường."
        ),
        prompt=ai_prompt,
        button_label="🤖 Phân tích dashboard bằng AI",
        key="dashboard_ai_analysis",
    )
