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
# TIỆN ÍCH
# ============================================================

def num(
    value: Any,
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


def format_volume(value: Any):
    value = num(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ cổ phiếu"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu cổ phiếu"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn cổ phiếu"

    return f"{value:,.0f} cổ phiếu"


def format_value(value: Any):
    value = num(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:
        return (
            f"{value / 1_000_000_000_000:.2f} nghìn tỷ đồng"
        )

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ đồng"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu đồng"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn đồng"

    return f"{value:,.0f} đồng"


def format_price(value: Any):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng/cổ phiếu"


def format_percent(value: Any):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def format_decimal(
    value: Any,
    digits: int = 4,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:.{digits}f}"


# ============================================================
# TÌM CỘT
# ============================================================

def find_column(
    df: pd.DataFrame,
    names: list[str],
):
    if df is None or df.empty:
        return None

    mapping = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in names:
        key = str(name).strip().lower()

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
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": pd.DataFrame(),
            "error": str(error),
        }

    if df is None or df.empty:
        return {
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": pd.DataFrame(),
            "error": "Không có dữ liệu VN-INDEX.",
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
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": df,
            "error": "Không tìm thấy cột điểm VN-INDEX.",
        }

    df[price_column] = pd.to_numeric(
        df[price_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[price_column]
    )

    if df.empty:
        return {
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": df,
            "error": "VN-INDEX không có điểm hợp lệ.",
        }

    current = float(
        df[price_column].iloc[-1]
    )

    if len(df) >= 2:
        previous = float(
            df[price_column].iloc[-2]
        )

        change = current - previous

        if previous != 0:
            change_percent = (
                change / previous * 100
            )
        else:
            change_percent = 0.0

    else:
        change = None
        change_percent = None

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
        df[volume_column] = pd.to_numeric(
            df[volume_column],
            errors="coerce",
        )

        volume = num(
            df[volume_column].iloc[-1]
        )

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
        df[value_column] = pd.to_numeric(
            df[value_column],
            errors="coerce",
        )

        traded_value = num(
            df[value_column].iloc[-1]
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
def load_dashboard_stock(
    symbol: str,
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
def load_dashboard_news():

    try:
        result = fetch_market_news(6)

        if isinstance(result, list):
            return result

        return []

    except Exception:
        return []


# ============================================================
# LẤY TOÀN BỘ BIẾN NUMERIC
# ============================================================

def get_all_numeric_features(
    df: pd.DataFrame,
):
    if df is None or df.empty:
        return []

    features = []

    for column in df.columns:

        name = str(
            column
        ).strip()

        if name.lower() in {
            "time",
            "date",
            "datetime",
            "timestamp",
        }:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            features.append(column)

    return list(
        dict.fromkeys(
            features
        )
    )


# ============================================================
# CHUẨN BỊ DATASET QUANT
# ============================================================

def prepare_quant_dataset(
    df: pd.DataFrame,
):
    if (
        df is None
        or df.empty
        or "Return" not in df.columns
    ):
        return None, None, []

    work = df.copy()

    all_features = (
        get_all_numeric_features(
            work
        )
    )

    # Chống leakage.
    features = [
        column
        for column in all_features
        if column not in {
            "Return",
            "ReturnPct",
            "Target",
        }
    ]

    if not features:
        return None, None, []

    work["Target"] = (
        pd.to_numeric(
            work["Return"],
            errors="coerce",
        ).shift(-1)
    )

    for column in features:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work["Target"] = pd.to_numeric(
        work["Target"],
        errors="coerce",
    )

    work = work.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # Chỉ loại biến hoàn toàn thiếu hoặc gần như thiếu.
    features = [
        column
        for column in features
        if work[column].notna().sum() >= 20
    ]

    if not features:
        return None, None, []

    X = work[
        features
    ].copy()

    y = work[
        "Target"
    ].copy()

    # Fill missing median.
    for column in features:

        median = X[
            column
        ].median()

        if pd.isna(median):
            median = 0.0

        X[column] = X[
            column
        ].fillna(
            median
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
        X.astype(float),
        y.astype(float),
        features,
    )


# ============================================================
# METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):
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

    try:
        r2 = float(
            r2_score(
                y_true,
                y_pred,
            )
        )
    except Exception:
        r2 = np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# VIF TOÀN BỘ BIẾN
# ============================================================

def calculate_vif_all(
    X: pd.DataFrame,
):
    """
    Tính VIF từ inverse correlation matrix.

    Dùng pseudoinverse để không làm page chết khi
    các chỉ báo có tương quan tuyến tính rất mạnh.
    """

    try:

        work = X.copy()

        work = work.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        for column in work.columns:

            work[column] = pd.to_numeric(
                work[column],
                errors="coerce",
            )

            median = work[
                column
            ].median()

            if pd.isna(median):
                median = 0.0

            work[column] = work[
                column
            ].fillna(
                median
            )

        # VIF không có ý nghĩa nếu variance = 0.
        work = work.loc[
            :,
            work.nunique(
                dropna=True
            ) > 1,
        ]

        if work.empty:
            return pd.DataFrame()

        corr = work.corr()

        corr = corr.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).fillna(0.0)

        inverse_corr = np.linalg.pinv(
            corr.values
        )

        vif_values = np.diag(
            inverse_corr
        )

        result = pd.DataFrame(
            {
                "Biến": corr.columns,
                "VIF": vif_values,
            }
        )

        result["VIF"] = result[
            "VIF"
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        return (
            result
            .sort_values(
                "VIF",
                ascending=False,
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# FULL QUANT
# ============================================================

@st.cache_data(
    ttl=TTL_QUANT,
    show_spinner=False,
)
def run_full_quant(
    df: pd.DataFrame,
):
    """
    Toàn bộ biến numeric được dùng cho các model chính.

    Models:
        OLS + HC3
        Ridge
        Lasso
        Elastic Net
        Random Forest
        Extra Trees
        Gradient Boosting

    Tests:
        ADF
        Jarque-Bera
        Breusch-Pagan
        White
        Durbin-Watson
        Ljung-Box
        VIF
    """

    result = {
        "ok": False,
        "error": None,
        "features": [],
        "feature_count": 0,
        "train_size": 0,
        "test_size": 0,
        "models": [],
        "tests": {},
        "feature_importance": pd.DataFrame(),
        "ols_summary": None,
        "residuals": None,
        "rank_info": None,
    }

    X, y, features = (
        prepare_quant_dataset(
            df
        )
    )

    if (
        X is None
        or y is None
        or not features
    ):

        result[
            "error"
        ] = (
            "Không đủ dữ liệu để chạy nghiên cứu định lượng."
        )

        return result

    result[
        "features"
    ] = features

    result[
        "feature_count"
    ] = len(features)

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
            "Tập train/test không đủ dữ liệu."
        )

        return result

    X_train = X.iloc[
        :split
    ].copy()

    X_test = X.iloc[
        split:
    ].copy()

    y_train = y.iloc[
        :split
    ].copy()

    y_test = y.iloc[
        split:
    ].copy()

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
    # RANK / CONDITION NUMBER
    # ========================================================

    try:

        matrix = np.asarray(
            X_train,
            dtype=float,
        )

        matrix = np.nan_to_num(
            matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        rank = int(
            np.linalg.matrix_rank(
                matrix
            )
        )

        columns = int(
            matrix.shape[1]
        )

        condition_number = float(
            np.linalg.cond(
                matrix
            )
        )

        result[
            "rank_info"
        ] = {
            "rank": rank,
            "columns": columns,
            "condition_number": condition_number,
            "rank_deficient": rank < columns,
        }

    except Exception:
        pass

    # ========================================================
    # OLS + HC3
    # ========================================================

    ols_model = None

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

        pred = ols_model.predict(
            X_test_const
        )

        result[
            "models"
        ].append(
            {
                "Mô hình": "OLS + HC3",
                **regression_metrics(
                    y_test,
                    pred,
                ),
            }
        )

        result[
            "ols_summary"
        ] = {
            "rsquared": num(
                getattr(
                    ols_model,
                    "rsquared",
                    None,
                )
            ),
            "adj_rsquared": num(
                getattr(
                    ols_model,
                    "rsquared_adj",
                    None,
                )
            ),
            "aic": num(
                getattr(
                    ols_model,
                    "aic",
                    None,
                )
            ),
            "bic": num(
                getattr(
                    ols_model,
                    "bic",
                    None,
                )
            ),
        }

        result[
            "residuals"
        ] = np.asarray(
            ols_model.resid,
            dtype=float,
        )

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
    # RIDGE / LASSO / ELASTIC NET
    # ========================================================

    try:

        from sklearn.linear_model import (
            ElasticNet,
            Lasso,
            Ridge,
        )

        from sklearn.pipeline import (
            Pipeline,
        )

        from sklearn.preprocessing import (
            StandardScaler,
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
                    max_iter=50000,
                ),
            ),
            (
                "Elastic Net",
                ElasticNet(
                    alpha=0.0001,
                    l1_ratio=0.5,
                    max_iter=50000,
                ),
            ),
        ]

        for name, estimator in linear_models:

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
                        "Mô hình": name,
                        **regression_metrics(
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
                        "Mô hình": name,
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

    try:

        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
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

        importance_list = []

        for name, model in tree_models:

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
                        "Mô hình": name,
                        **regression_metrics(
                            y_test,
                            pred,
                        ),
                    }
                )

                feature_importance = getattr(
                    model,
                    "feature_importances_",
                    None,
                )

                if feature_importance is not None:

                    temp = pd.DataFrame(
                        {
                            "Biến": features,
                            "Mức quan trọng": (
                                feature_importance
                            ),
                            "Mô hình": name,
                        }
                    )

                    importance_list.append(
                        temp
                    )

            except Exception as error:

                result[
                    "models"
                ].append(
                    {
                        "Mô hình": name,
                        "MAE": np.nan,
                        "MSE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "Lỗi": str(error),
                    }
                )

        if importance_list:

            importance_all = pd.concat(
                importance_list,
                ignore_index=True,
            )

            importance_avg = (
                importance_all
                .groupby(
                    "Biến",
                    as_index=False,
                )[
                    "Mức quan trọng"
                ]
                .mean()
                .sort_values(
                    "Mức quan trọng",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            result[
                "feature_importance"
            ] = importance_avg

    except Exception:
        pass

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
                "lags": int(
                    adf[2]
                ),
                "observations": int(
                    adf[3]
                ),
            }

        except Exception:
            pass

        residuals = result.get(
            "residuals"
        )

        # ----------------------------------------------------
        # Jarque-Bera
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

                result[
                    "tests"
                ][
                    "Durbin-Watson"
                ] = {
                    "statistic": float(
                        durbin_watson(
                            residuals
                        )
                    )
                }

            except Exception:
                pass

        # ----------------------------------------------------
        # BREUSCH-PAGAN
        #
        # Dùng toàn bộ feature của model OLS.
        # ----------------------------------------------------

        if ols_model is not None:

            try:

                X_bp = sm.add_constant(
                    X_train,
                    has_constant="add",
                )

                bp = het_breuschpagan(
                    ols_model.resid,
                    X_bp,
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
        # WHITE
        #
        # White test đầy đủ với hàng chục biến sẽ sinh
        # rất nhiều interaction. Để tránh treo app,
        # dùng 8 biến variance cao nhất cho TEST WHITE.
        #
        # Toàn bộ biến vẫn được giữ trong model chính.
        # ----------------------------------------------------

        if ols_model is not None:

            try:

                variances = (
                    X_train
                    .var()
                    .sort_values(
                        ascending=False
                    )
                )

                white_features = (
                    variances
                    .head(
                        min(
                            8,
                            len(
                                variances
                            ),
                        )
                    )
                    .index
                    .tolist()
                )

                if white_features:

                    X_white = sm.add_constant(
                        X_train[
                            white_features
                        ],
                        has_constant="add",
                    )

                    white = het_white(
                        ols_model.resid,
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
                        "feature_count_used": len(
                            white_features
                        ),
                        "features_used": white_features,
                    }

            except Exception:
                pass

        # ----------------------------------------------------
        # LJUNG BOX
        # ----------------------------------------------------

        if residuals is not None:

            try:

                lag = min(
                    10,
                    max(
                        1,
                        len(
                            residuals
                        ) // 10,
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

        vif_df = calculate_vif_all(
            X_train
        )

        if (
            vif_df is not None
            and not vif_df.empty
        ):

            result[
                "tests"
            ][
                "VIF"
            ] = vif_df

    except Exception:
        pass

    # ========================================================
    # SORT MODEL
    # ========================================================

    model_df = pd.DataFrame(
        result[
            "models"
        ]
    )

    if not model_df.empty:

        for column in [
            "MAE",
            "MSE",
            "RMSE",
            "R2",
        ]:

            if column in model_df.columns:

                model_df[
                    column
                ] = pd.to_numeric(
                    model_df[
                        column
                    ],
                    errors="coerce",
                )

        if "RMSE" in model_df.columns:

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
        "Hồi quy tuyến tính dùng để đo mối quan hệ có điều kiện "
        "giữa các biến giải thích và lợi suất phiên kế tiếp. "
        "HC3 cung cấp sai số chuẩn robust hơn khi phương sai thay đổi. "
        "Nếu ma trận thiết kế rank-deficient, không nên diễn giải "
        "hệ số riêng lẻ như các tham số độc nhất."
    ),

    "Ridge": (
        "Hồi quy tuyến tính với penalty L2. Giúp ổn định mô hình "
        "khi rất nhiều biến có tương quan mạnh."
    ),

    "Lasso": (
        "Hồi quy với penalty L1. Có thể đưa một số hệ số về gần 0, "
        "hỗ trợ xác định nhóm biến ít đóng góp."
    ),

    "Elastic Net": (
        "Kết hợp L1 và L2. Phù hợp khi có nhiều biến tương quan "
        "và vẫn muốn một mức regularization/chọn lọc biến."
    ),

    "Random Forest": (
        "Ensemble nhiều cây quyết định. Bắt quan hệ phi tuyến "
        "và tương tác giữa các chỉ báo."
    ),

    "Extra Trees": (
        "Ensemble cây với mức randomization mạnh hơn. "
        "Cho một góc nhìn khác về quan hệ phi tuyến."
    ),

    "Gradient Boosting": (
        "Xây cây tuần tự để sửa sai cho mô hình trước, "
        "phù hợp với tín hiệu phi tuyến nhưng cần lưu ý overfitting."
    ),
}


TEST_EXPLANATIONS = {
    "ADF": (
        "Kiểm tra tính dừng của chuỗi. p-value < 0.05 thường "
        "ủng hộ việc bác bỏ giả thuyết có nghiệm đơn vị."
    ),

    "Jarque-Bera": (
        "Kiểm tra phần dư có gần phân phối chuẩn hay không."
    ),

    "Breusch-Pagan": (
        "Kiểm tra heteroskedasticity. p-value < 0.05 là dấu hiệu "
        "phương sai phần dư thay đổi."
    ),

    "White": (
        "Kiểm tra phương sai thay đổi tổng quát. Với rất nhiều "
        "biến, kiểm định dùng tập biến variance cao nhất để tránh "
        "bùng nổ interaction."
    ),

    "Durbin-Watson": (
        "Kiểm tra tự tương quan bậc một của phần dư. "
        "Giá trị gần 2 thường ít đáng lo hơn."
    ),

    "Ljung-Box": (
        "Kiểm tra tự tương quan phần dư trên nhiều độ trễ."
    ),

    "VIF": (
        "Đo đa cộng tuyến giữa các biến giải thích. "
        "VIF càng cao, biến càng có thông tin trùng lặp "
        "với các biến khác."
    ),
}


# ============================================================
# BẢNG KIỂM ĐỊNH
# ============================================================

def make_test_summary(
    tests: dict,
):

    rows = []

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    if "ADF" in tests:

        item = tests["ADF"]

        statistic = num(
            item.get("statistic")
        )

        p_value = num(
            item.get("p_value")
        )

        if (
            p_value is not None
            and p_value < 0.05
        ):

            conclusion = (
                "✅ Bác bỏ nghiệm đơn vị"
            )

        else:

            conclusion = (
                "⚠️ Chưa đủ bằng chứng dừng"
            )

        rows.append(
            {
                "Kiểm định": "ADF",
                "Statistic": statistic,
                "p-value": p_value,
                "Kết luận": conclusion,
            }
        )

    # --------------------------------------------------------
    # Jarque-Bera
    # --------------------------------------------------------

    if "Jarque-Bera" in tests:

        item = tests[
            "Jarque-Bera"
        ]

        statistic = num(
            item.get("statistic")
        )

        p_value = num(
            item.get("p_value")
        )

        if (
            p_value is not None
            and p_value < 0.05
        ):

            conclusion = (
                "⚠️ Phần dư không gần chuẩn"
            )

        else:

            conclusion = (
                "✅ Chưa thấy lệch chuẩn mạnh"
            )

        rows.append(
            {
                "Kiểm định": "Jarque-Bera",
                "Statistic": statistic,
                "p-value": p_value,
                "Kết luận": conclusion,
            }
        )

    # --------------------------------------------------------
    # Breusch-Pagan
    # --------------------------------------------------------

    if "Breusch-Pagan" in tests:

        item = tests[
            "Breusch-Pagan"
        ]

        statistic = num(
            item.get("LM")
        )

        p_value = num(
            item.get("p_value")
        )

        if (
            p_value is not None
            and p_value < 0.05
        ):

            conclusion = (
                "⚠️ Có dấu hiệu phương sai thay đổi"
            )

        else:

            conclusion = (
                "✅ Chưa phát hiện rõ"
            )

        rows.append(
            {
                "Kiểm định": "Breusch-Pagan",
                "Statistic": statistic,
                "p-value": p_value,
                "Kết luận": conclusion,
            }
        )

    # --------------------------------------------------------
    # White
    # --------------------------------------------------------

    if "White" in tests:

        item = tests[
            "White"
        ]

        statistic = num(
            item.get("LM")
        )

        p_value = num(
            item.get("p_value")
        )

        if (
            p_value is not None
            and p_value < 0.05
        ):

            conclusion = (
                "⚠️ Có dấu hiệu phương sai thay đổi"
            )

        else:

            conclusion = (
                "✅ Chưa phát hiện rõ"
            )

        rows.append(
            {
                "Kiểm định": "White",
                "Statistic": statistic,
                "p-value": p_value,
                "Kết luận": conclusion,
            }
        )

    # --------------------------------------------------------
    # Durbin-Watson
    # --------------------------------------------------------

    if "Durbin-Watson" in tests:

        item = tests[
            "Durbin-Watson"
        ]

        statistic = num(
            item.get("statistic")
        )

        if statistic is None:

            conclusion = "⚪ Không xác định"

        elif 1.5 <= statistic <= 2.5:

            conclusion = "✅ Tương đối gần 2"

        elif statistic < 1.5:

            conclusion = (
                "⚠️ Dấu hiệu tự tương quan dương"
            )

        else:

            conclusion = (
                "⚠️ Dấu hiệu tự tương quan âm"
            )

        rows.append(
            {
                "Kiểm định": "Durbin-Watson",
                "Statistic": statistic,
                "p-value": np.nan,
                "Kết luận": conclusion,
            }
        )

    # --------------------------------------------------------
    # Ljung-Box
    # --------------------------------------------------------

    if "Ljung-Box" in tests:

        item = tests[
            "Ljung-Box"
        ]

        statistic = num(
            item.get("statistic")
        )

        p_value = num(
            item.get("p_value")
        )

        if (
            p_value is not None
            and p_value < 0.05
        ):

            conclusion = (
                "⚠️ Còn tự tương quan"
            )

        else:

            conclusion = (
                "✅ Chưa phát hiện rõ"
            )

        rows.append(
            {
                "Kiểm định": "Ljung-Box",
                "Statistic": statistic,
                "p-value": p_value,
                "Kết luận": conclusion,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# KẾT LUẬN TỰ ĐỘNG
# ============================================================

def build_quant_conclusion(
    quant_result: dict,
):

    conclusions = []

    # ========================================================
    # MODEL
    # ========================================================

    model_df = pd.DataFrame(
        quant_result.get(
            "models",
            [],
        )
    )

    if (
        not model_df.empty
        and "RMSE" in model_df.columns
    ):

        model_df[
            "RMSE"
        ] = pd.to_numeric(
            model_df[
                "RMSE"
            ],
            errors="coerce",
        )

        valid = model_df[
            model_df[
                "RMSE"
            ].notna()
        ]

        if not valid.empty:

            best = (
                valid
                .sort_values(
                    "RMSE"
                )
                .iloc[0]
            )

            conclusions.append(
                f"Mô hình có RMSE thấp nhất trên tập test là "
                f"**{best['Mô hình']}** "
                f"với RMSE = {best['RMSE']:.6f}."
            )

            best_r2 = num(
                best.get(
                    "R2"
                )
            )

            if best_r2 is not None:

                conclusions.append(
                    f"R² của mô hình này = {best_r2:.4f}."
                )

    # ========================================================
    # RANK
    # ========================================================

    rank_info = quant_result.get(
        "rank_info"
    )

    if isinstance(
        rank_info,
        dict,
    ):

        rank = rank_info.get(
            "rank"
        )

        columns = rank_info.get(
            "columns"
        )

        condition_number = num(
            rank_info.get(
                "condition_number"
            )
        )

        rank_deficient = bool(
            rank_info.get(
                "rank_deficient",
                False,
            )
        )

        if rank_deficient:

            conclusions.append(
                f"Ma trận biến có rank {rank}/{columns}. "
                "OLS bị đa cộng tuyến/rank deficiency; "
                "không nên coi các hệ số OLS riêng lẻ là duy nhất."
            )

        elif (
            condition_number is not None
            and condition_number > 1e8
        ):

            conclusions.append(
                "Ma trận biến có condition number rất lớn, "
                "cho thấy vấn đề conditioning/đa cộng tuyến cần lưu ý."
            )

    # ========================================================
    # ADF
    # ========================================================

    tests = quant_result.get(
        "tests",
        {},
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

                conclusions.append(
                    "ADF: có bằng chứng bác bỏ nghiệm đơn vị "
                    "ở mức 5%."
                )

            else:

                conclusions.append(
                    "ADF: chưa đủ bằng chứng để kết luận "
                    "chuỗi mục tiêu dừng."
                )

    # ========================================================
    # HETEROSKEDASTICITY
    # ========================================================

    bp_p = None
    white_p = None

    if "Breusch-Pagan" in tests:

        bp_p = num(
            tests[
                "Breusch-Pagan"
            ].get(
                "p_value"
            )
        )

    if "White" in tests:

        white_p = num(
            tests[
                "White"
            ].get(
                "p_value"
            )
        )

    if (
        (
            bp_p is not None
            and bp_p < 0.05
        )
        or
        (
            white_p is not None
            and white_p < 0.05
        )
    ):

        conclusions.append(
            "Có dấu hiệu phương sai thay đổi; "
            "kết quả OLS nên được đọc cùng sai số chuẩn robust."
        )

    else:

        conclusions.append(
            "Các kiểm định phương sai chưa cho thấy "
            "bằng chứng mạnh về heteroskedasticity."
        )

    # ========================================================
    # AUTOCORRELATION
    # ========================================================

    dw_value = None
    lb_p = None

    if "Durbin-Watson" in tests:

        dw_value = num(
            tests[
                "Durbin-Watson"
            ].get(
                "statistic"
            )
        )

    if "Ljung-Box" in tests:

        lb_p = num(
            tests[
                "Ljung-Box"
            ].get(
                "p_value"
            )
        )

    if (
        (
            dw_value is not None
            and (
                dw_value < 1.5
                or dw_value > 2.5
            )
        )
        or
        (
            lb_p is not None
            and lb_p < 0.05
        )
    ):

        conclusions.append(
            "Phần dư có dấu hiệu tự tương quan; "
            "mô hình có thể chưa khai thác hết cấu trúc thời gian."
        )

    else:

        conclusions.append(
            "Chưa thấy bằng chứng mạnh về tự tương quan còn lại."
        )

    # ========================================================
    # VIF
    # ========================================================

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

            finite = vif_df[
                np.isfinite(
                    vif_df[
                        "VIF"
                    ]
                )
            ]

            if not finite.empty:

                max_vif = num(
                    finite[
                        "VIF"
                    ].max()
                )

                if (
                    max_vif is not None
                    and max_vif > 10
                ):

                    conclusions.append(
                        f"VIF cao nhất = {max_vif:.2f}; "
                        "đa cộng tuyến mạnh."
                    )

                elif (
                    max_vif is not None
                    and max_vif > 5
                ):

                    conclusions.append(
                        f"VIF cao nhất = {max_vif:.2f}; "
                        "đa cộng tuyến cần lưu ý."
                    )

                else:

                    conclusions.append(
                        "VIF chưa cho thấy đa cộng tuyến nghiêm trọng."
                    )

    return conclusions


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
        "tin tức, AI và định lượng."
    )

    # ========================================================
    # VN-INDEX
    # ========================================================

    vn = get_vnindex()

    vn_price = (
        f"{vn['price']:,.2f} điểm"
        if vn.get("price") is not None
        else "—"
    )

    vn_change = (
        f"{vn['change']:+,.2f} điểm"
        if vn.get("change") is not None
        else "—"
    )

    vn_percent = (
        f"{vn['change_percent']:+.2f}%"
        if vn.get("change_percent") is not None
        else "—"
    )

    vn_volume = format_volume(
        vn.get("volume")
    )

    vn_value = format_value(
        vn.get("value")
    )

    # ========================================================
    # QUICK VIEW
    # ========================================================

    st.subheader(
        "📌 Theo dõi nhanh"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        metric_card(
            "VN-INDEX",
            vn_price,
            "Điểm chỉ số thị trường",
        )

    with q2:

        metric_card(
            "Khối lượng",
            vn_volume,
            "Khối lượng giao dịch toàn thị trường",
        )

    with q3:

        metric_card(
            "Giá trị giao dịch",
            vn_value,
            "Tổng giá trị giao dịch",
        )

    with q4:

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
    # CỔ PHIẾU
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

    load_button = st.button(
        "🔄 Tải dữ liệu",
        type="primary",
        key="dashboard_load_button",
    )

    if load_button:

        clean_symbol = normalize_symbol(
            symbol_input
        )

        if clean_symbol:

            st.session_state[
                "dashboard_symbol"
            ] = clean_symbol

            # Xóa Quant của mã cũ.
            st.session_state.pop(
                "dashboard_quant_result",
                None,
            )

            st.session_state.pop(
                "dashboard_quant_symbol",
                None,
            )

            st.rerun()

        else:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

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

        stock = load_dashboard_stock(
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

    stock_snapshot = {}

    # ========================================================
    # STOCK DATA
    # ========================================================

    if (
        stock is not None
        and not stock.empty
    ):

        stock_snapshot = market_snapshot(
            stock
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

        volume = num(
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

        volatility20 = num(
            stock_snapshot.get(
                "volatility20"
            )
        )

        last_row = stock.iloc[-1]

        atr14 = num(
            last_row.get(
                "ATR14"
            )
        )

        volume_sma20 = num(
            last_row.get(
                "Volume_SMA20"
            )
        )

        relative_volume = None

        if (
            volume is not None
            and volume_sma20 is not None
            and volume_sma20 != 0
        ):

            relative_volume = (
                volume
                / volume_sma20
            )

        st.subheader(
            f"📈 {display_symbol(current_symbol)}"
        )

        # ----------------------------------------------------
        # MAIN
        # ----------------------------------------------------

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Giá",
                format_price(
                    price
                ),
            )

        with c2:

            st.metric(
                "Thay đổi 1D",
                format_percent(
                    change_1d
                ),
            )

        with c3:

            st.metric(
                "RSI",
                (
                    f"{rsi_value:.1f}"
                    if rsi_value is not None
                    else "—"
                ),
            )

        with c4:

            st.metric(
                "Khối lượng",
                format_volume(
                    volume
                ),
            )

        # ----------------------------------------------------
        # INDICATORS
        # ----------------------------------------------------

        c5, c6, c7, c8 = (
            st.columns(4)
        )

        with c5:

            st.metric(
                "Trung bình 20 phiên",
                format_price(
                    sma20
                ),
            )

        with c6:

            st.metric(
                "Trung bình 50 phiên",
                format_price(
                    sma50
                ),
            )

        with c7:

            st.metric(
                "MACD",
                (
                    f"{macd:.3f}"
                    if macd is not None
                    else "—"
                ),
            )

        with c8:

            st.metric(
                "Biến động 20 phiên",
                (
                    f"{volatility20:.2f}%"
                    if volatility20 is not None
                    else "—"
                ),
            )

        # ----------------------------------------------------
        # EXTRA
        # ----------------------------------------------------

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
                last_row.get(
                    "Open"
                )
            )

            st.metric(
                "Giá mở cửa",
                format_price(
                    open_price
                ),
            )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

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
                        "displaylogo": False
                    },
                )

        except Exception as error:

            st.warning(
                "Không thể hiển thị biểu đồ."
            )

            st.caption(
                str(error)
            )

    # ========================================================
    # NEWS
    # ========================================================

    st.subheader(
        "📰 Tin mới"
    )

    news = load_dashboard_news()

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
    # QUANT
    # ========================================================

    st.subheader(
        "🧮 Nghiên cứu định lượng"
    )

    st.caption(
        "Các mô hình chính sử dụng toàn bộ biến numeric "
        "hợp lệ từ dữ liệu kỹ thuật của mã đang xem. "
        "Nghiên cứu chỉ chạy khi bấm nút."
    )

    quant_result = None

    if (
        stock is None
        or stock.empty
    ):

        st.info(
            "Cần tải dữ liệu cổ phiếu trước."
        )

    else:

        run_quant = st.button(
            "🧮 Chạy toàn bộ mô hình + kiểm định",
            type="primary",
            key="dashboard_quant_button",
        )

        if run_quant:

            with st.spinner(
                "Đang chạy toàn bộ nghiên cứu định lượng..."
            ):

                quant_result = (
                    run_full_quant(
                        stock
                    )
                )

            st.session_state[
                "dashboard_quant_result"
            ] = quant_result

            st.session_state[
                "dashboard_quant_symbol"
            ] = current_symbol

        else:

            quant_result = (
                st.session_state.get(
                    "dashboard_quant_result"
                )
            )

        saved_symbol = (
            st.session_state.get(
                "dashboard_quant_symbol"
            )
        )

        if saved_symbol != current_symbol:

            quant_result = None

        if quant_result is None:

            st.info(
                "Bấm nút để chạy mô hình và kiểm định."
            )

        elif not quant_result.get(
            "ok",
            False,
        ):

            st.error(
                quant_result.get(
                    "error",
                    "Nghiên cứu thất bại.",
                )
            )

        else:

            # =================================================
            # QUANT OVERVIEW
            # =================================================

            st.markdown(
                "### 📌 Tổng quan nghiên cứu"
            )

            q1, q2, q3, q4 = (
                st.columns(4)
            )

            with q1:

                st.metric(
                    "Số biến",
                    f"{quant_result.get('feature_count', 0):,}",
                )

            with q2:

                st.metric(
                    "Train",
                    f"{quant_result.get('train_size', 0):,}",
                )

            with q3:

                st.metric(
                    "Test",
                    f"{quant_result.get('test_size', 0):,}",
                )

            with q4:

                st.metric(
                    "Số model",
                    f"{len(quant_result.get('models', [])):,}",
                )

            # =================================================
            # RANK
            # =================================================

            rank_info = quant_result.get(
                "rank_info"
            )

            if isinstance(
                rank_info,
                dict,
            ):

                rank = rank_info.get(
                    "rank"
                )

                columns = rank_info.get(
                    "columns"
                )

                condition_number = num(
                    rank_info.get(
                        "condition_number"
                    )
                )

                if rank is not None and columns is not None:

                    r1, r2, r3 = (
                        st.columns(3)
                    )

                    with r1:

                        st.metric(
                            "Rank",
                            f"{rank}/{columns}",
                        )

                    with r2:

                        st.metric(
                            "Condition number",
                            (
                                format_decimal(
                                    condition_number,
                                    2,
                                )
                            ),
                        )

                    with r3:

                        if rank < columns:

                            st.warning(
                                "Rank-deficient / đa cộng tuyến"
                            )

                        else:

                            st.success(
                                "Full rank"
                            )

            # =================================================
            # ALL FEATURES
            # =================================================

            st.markdown(
                "### 🔢 Toàn bộ biến sử dụng"
            )

            features = (
                quant_result.get(
                    "features",
                    [],
                )
            )

            if features:

                feature_df = pd.DataFrame(
                    {
                        "STT": range(
                            1,
                            len(features)
                            + 1,
                        ),
                        "Biến": features,
                    }
                )

                st.dataframe(
                    feature_df,
                    width="stretch",
                    height=320,
                    hide_index=True,
                )

            # =================================================
            # MODELS
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

                for column in [
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R2",
                ]:

                    if column in model_df.columns:

                        model_df[
                            column
                        ] = pd.to_numeric(
                            model_df[
                                column
                            ],
                            errors="coerce",
                        )

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
                        ] = display_model_df[
                            column
                        ].round(6)

                st.dataframe(
                    display_model_df,
                    width="stretch",
                    hide_index=True,
                )

                valid_models = model_df[
                    model_df[
                        "RMSE"
                    ].notna()
                ]

                if not valid_models.empty:

                    best_model = (
                        valid_models
                        .sort_values(
                            "RMSE"
                        )
                        .iloc[0]
                    )

                    st.success(
                        "Model tốt nhất theo RMSE tập test: "
                        f"**{best_model['Mô hình']}** · "
                        f"RMSE = {best_model['RMSE']:.6f}"
                    )

            # =================================================
            # MODEL EXPLANATIONS
            # =================================================

            st.markdown(
                "### 📖 Ý nghĩa các mô hình"
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

                st.markdown(
                    f"**{model_name}:** "
                    + MODEL_EXPLANATIONS[
                        model_name
                    ]
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
                isinstance(
                    importance_df,
                    pd.DataFrame,
                )
                and not importance_df.empty
            ):

                st.markdown(
                    "### 🌳 Feature importance"
                )

                show_importance = (
                    importance_df
                    .head(30)
                    .copy()
                )

                show_importance[
                    "Mức quan trọng"
                ] = pd.to_numeric(
                    show_importance[
                        "Mức quan trọng"
                    ],
                    errors="coerce",
                ).round(6)

                st.dataframe(
                    show_importance,
                    width="stretch",
                    hide_index=True,
                )

            # =================================================
            # TESTS
            # =================================================

            st.markdown(
                "### 🧪 Kết quả kiểm định"
            )

            tests = quant_result.get(
                "tests",
                {},
            )

            test_summary = (
                make_test_summary(
                    tests
                )
            )

            if not test_summary.empty:

                st.dataframe(
                    test_summary,
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.warning(
                    "Không có kiểm định trả kết quả."
                )

            # =================================================
            # TEST MEANINGS
            # =================================================

            st.markdown(
                "### 📖 Ý nghĩa kiểm định"
            )

            for test_name in [
                "ADF",
                "Jarque-Bera",
                "Breusch-Pagan",
                "White",
                "Durbin-Watson",
                "Ljung-Box",
                "VIF",
            ]:

                st.markdown(
                    f"**{test_name}:** "
                    + TEST_EXPLANATIONS[
                        test_name
                    ]
                )

            # =================================================
            # ADF DETAIL
            # =================================================

            if "ADF" in tests:

                item = tests[
                    "ADF"
                ]

                p = num(
                    item.get(
                        "p_value"
                    )
                )

                statistic = num(
                    item.get(
                        "statistic"
                    )
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "ADF Statistic",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

                    st.metric(
                        "ADF p-value",
                        format_decimal(
                            p,
                            6,
                        ),
                    )

                if (
                    p is not None
                    and p < 0.05
                ):

                    st.success(
                        "ADF: bác bỏ giả thuyết có nghiệm đơn vị."
                    )

                else:

                    st.warning(
                        "ADF: chưa đủ bằng chứng bác bỏ giả thuyết có nghiệm đơn vị."
                    )

            # =================================================
            # JARQUE-BERA
            # =================================================

            if "Jarque-Bera" in tests:

                item = tests[
                    "Jarque-Bera"
                ]

                p = num(
                    item.get(
                        "p_value"
                    )
                )

                statistic = num(
                    item.get(
                        "statistic"
                    )
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "JB Statistic",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

                    st.metric(
                        "JB p-value",
                        format_decimal(
                            p,
                            6,
                        ),
                    )

                if (
                    p is not None
                    and p < 0.05
                ):

                    st.warning(
                        "Jarque–Bera: phần dư lệch khỏi giả định chuẩn."
                    )

                else:

                    st.success(
                        "Jarque–Bera: chưa có bằng chứng mạnh phần dư lệch chuẩn."
                    )

            # =================================================
            # BREUSCH-PAGAN
            # =================================================

            if "Breusch-Pagan" in tests:

                item = tests[
                    "Breusch-Pagan"
                ]

                p = num(
                    item.get(
                        "p_value"
                    )
                )

                statistic = num(
                    item.get(
                        "LM"
                    )
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "BP LM",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

                    st.metric(
                        "BP p-value",
                        format_decimal(
                            p,
                            6,
                        ),
                    )

                if (
                    p is not None
                    and p < 0.05
                ):

                    st.warning(
                        "Breusch–Pagan: có dấu hiệu phương sai thay đổi."
                    )

                else:

                    st.success(
                        "Breusch–Pagan: chưa phát hiện phương sai thay đổi rõ."
                    )

            # =================================================
            # WHITE
            # =================================================

            if "White" in tests:

                item = tests[
                    "White"
                ]

                p = num(
                    item.get(
                        "p_value"
                    )
                )

                statistic = num(
                    item.get(
                        "LM"
                    )
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "White LM",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

                    st.metric(
                        "White p-value",
                        format_decimal(
                            p,
                            6,
                        ),
                    )

                if (
                    p is not None
                    and p < 0.05
                ):

                    st.warning(
                        "White: có dấu hiệu phương sai thay đổi."
                    )

                else:

                    st.success(
                        "White: chưa phát hiện phương sai thay đổi rõ."
                    )

                used_features = item.get(
                    "features_used",
                    [],
                )

                if used_features:

                    st.caption(
                        "White test dùng "
                        f"{len(used_features)} biến variance cao "
                        "để tránh tạo interaction quá lớn: "
                        + ", ".join(
                            map(
                                str,
                                used_features,
                            )
                        )
                    )

            # =================================================
            # DURBIN-WATSON
            # =================================================

            if "Durbin-Watson" in tests:

                statistic = num(
                    tests[
                        "Durbin-Watson"
                    ].get(
                        "statistic"
                    )
                )

                st.metric(
                    "Durbin-Watson",
                    format_decimal(
                        statistic,
                        4,
                    ),
                )

                if statistic is None:

                    st.info(
                        "Không xác định."
                    )

                elif 1.5 <= statistic <= 2.5:

                    st.success(
                        "Statistic tương đối gần 2."
                    )

                elif statistic < 1.5:

                    st.warning(
                        "Có dấu hiệu tự tương quan dương."
                    )

                else:

                    st.warning(
                        "Có dấu hiệu tự tương quan âm."
                    )

            # =================================================
            # LJUNG-BOX
            # =================================================

            if "Ljung-Box" in tests:

                item = tests[
                    "Ljung-Box"
                ]

                lag = item.get(
                    "lag"
                )

                statistic = num(
                    item.get(
                        "statistic"
                    )
                )

                p = num(
                    item.get(
                        "p_value"
                    )
                )

                l1, l2, l3 = (
                    st.columns(3)
                )

                with l1:

                    st.metric(
                        "Lag",
                        str(
                            lag
                        ),
                    )

                with l2:

                    st.metric(
                        "Statistic",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with l3:

                    st.metric(
                        "p-value",
                        format_decimal(
                            p,
                            6,
                        ),
                    )

                if (
                    p is not None
                    and p < 0.05
                ):

                    st.warning(
                        "Ljung–Box: phần dư còn tự tương quan."
                    )

                else:

                    st.success(
                        "Ljung–Box: chưa phát hiện tự tương quan còn lại rõ."
                    )

            # =================================================
            # VIF
            # =================================================

            if "VIF" in tests:

                vif_df = tests[
                    "VIF"
                ]

                st.markdown(
                    "#### 🔗 VIF — toàn bộ biến"
                )

                if (
                    isinstance(
                        vif_df,
                        pd.DataFrame,
                    )
                    and not vif_df.empty
                ):

                    vif_show = vif_df.copy()

                    vif_show[
                        "VIF"
                    ] = pd.to_numeric(
                        vif_show[
                            "VIF"
                        ],
                        errors="coerce",
                    )

                    display_vif = (
                        vif_show.copy()
                    )

                    display_vif[
                        "VIF"
                    ] = display_vif[
                        "VIF"
                    ].round(3)

                    st.dataframe(
                        display_vif,
                        width="stretch",
                        height=420,
                        hide_index=True,
                    )

                    finite_vif = vif_show[
                        np.isfinite(
                            vif_show[
                                "VIF"
                            ]
                        )
                    ]

                    if not finite_vif.empty:

                        max_vif = num(
                            finite_vif[
                                "VIF"
                            ].max()
                        )

                        if (
                            max_vif is not None
                            and max_vif > 10
                        ):

                            st.error(
                                f"VIF cao nhất = {max_vif:.2f}. "
                                "Đa cộng tuyến mạnh."
                            )

                        elif (
                            max_vif is not None
                            and max_vif > 5
                        ):

                            st.warning(
                                f"VIF cao nhất = {max_vif:.2f}. "
                                "Có đa cộng tuyến cần lưu ý."
                            )

                        else:

                            st.success(
                                "VIF chưa cho thấy đa cộng tuyến nghiêm trọng."
                            )

            # =================================================
            # FINAL CONCLUSION
            # =================================================

            st.markdown(
                "### 🧠 Kết luận nghiên cứu"
            )

            conclusions = (
                build_quant_conclusion(
                    quant_result
                )
            )

            for conclusion in conclusions:

                st.markdown(
                    "• "
                    + conclusion
                )

            if not conclusions:

                st.info(
                    "Chưa đủ bằng chứng để tạo kết luận tự động."
                )

            st.warning(
                "Kết luận định lượng chỉ phản ánh bằng chứng "
                "trên mẫu dữ liệu hiện tại, không bảo đảm biến động giá tương lai."
            )

    # ========================================================
    # AI MARKET BRIEF
    # ========================================================

    st.subheader(
        "🤖 AI Market Brief"
    )

    st.caption(
        "AI chỉ được gọi khi bấm nút; lỗi 503 từ model "
        "không làm Dashboard sập."
    )

    ai_button = st.button(
        "🤖 Phân tích dashboard bằng AI",
        type="primary",
        key="dashboard_ai_button",
    )

    if ai_button:

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

        ai_snapshot = (
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

        try:

            ai_prompt = dashboard_prompt(
                vn_index=ai_vn_index,
                stock_symbol=display_symbol(
                    current_symbol
                ),
                stock_snapshot=ai_snapshot,
                news=ai_news,
            )

            render_ai_panel(
                title="🤖 AI Market Brief",
                description=(
                    "AI tổng hợp VN-INDEX, cổ phiếu đang theo dõi "
                    "và tin tức thị trường."
                ),
                prompt=ai_prompt,
                button_label="Phân tích",
                key="dashboard_ai_analysis",
            )

        except Exception as error:

            error_text = str(
                error
            )

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text.lower()
            ):

                st.warning(
                    "AI provider hiện đang quá tải/tạm thời không khả dụng. "
                    "Dữ liệu Dashboard và Quant vẫn hoạt động bình thường."
                )

            else:

                st.error(
                    "Không thể khởi tạo AI."
                )

                st.caption(
                    error_text
                )


# ============================================================
# TƯƠNG THÍCH TÊN CŨ
# ============================================================

def render_dashboard_page():
    render_dashboard()
