from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components.ai import (
    dashboard_prompt,
    render_ai_panel,
)

from components.cards import (
    metric_card,
)

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


def format_volume(
    value: Any,
):
    value = num(
        value,
        None,
    )

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

    return (
        f"{value:,.0f} cổ phiếu"
    )


def format_value(
    value: Any,
):
    value = num(
        value,
        None,
    )

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

    return (
        f"{value:,.0f} đồng"
    )


def format_price(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    return (
        f"{value:,.0f} đồng/cổ phiếu"
    )


def format_percent(
    value: Any,
):
    value = num(
        value,
        None,
    )

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def format_decimal(
    value: Any,
    digits: int = 4,
):
    value = num(
        value,
        None,
    )

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
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": pd.DataFrame(),
            "error": str(error),
        }

    if (
        df is None
        or df.empty
    ):

        return {
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": pd.DataFrame(),
            "error": (
                "Không có dữ liệu VN-INDEX."
            ),
        }

    df = df.copy()

    # --------------------------------------------------------
    # Điểm
    # --------------------------------------------------------

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
            "error": (
                "Không tìm thấy cột điểm VN-INDEX."
            ),
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
            "price": None,
            "change": None,
            "change_percent": None,
            "volume": None,
            "value": None,
            "df": df,
            "error": (
                "VN-INDEX không có điểm hợp lệ."
            ),
        }

    current = float(
        df[
            price_column
        ].iloc[-1]
    )

    # --------------------------------------------------------
    # Thay đổi
    # --------------------------------------------------------

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
    # Khối lượng
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
    # Giá trị giao dịch
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
# CỔ PHIẾU
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
# TIN TỨC
# ============================================================

@st.cache_data(
    ttl=TTL_NEWS,
    show_spinner=False,
)
def load_dashboard_news():

    try:

        return fetch_market_news(
            6
        )

    except Exception:

        return []


# ============================================================
# LẤY TOÀN BỘ BIẾN ĐỊNH LƯỢNG
# ============================================================

def get_all_numeric_features(
    df: pd.DataFrame,
):
    if (
        df is None
        or df.empty
    ):
        return []

    features = []

    for column in df.columns:

        column_name = str(
            column
        ).strip()

        # Không lấy index dạng ngày nếu nó xuất hiện dưới dạng cột.
        if column_name.lower() in {
            "time",
            "date",
            "datetime",
        }:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            features.append(
                column
            )

    # Loại trùng nhưng giữ thứ tự.
    return list(
        dict.fromkeys(
            features
        )
    )


# ============================================================
# PREPARE QUANT
# ============================================================

def prepare_quant_dataset(
    df: pd.DataFrame,
):
    """
    Target:
        Return của phiên kế tiếp.

    Feature:
        TOÀN BỘ biến numeric có trong data.market,
        trừ chính Return và ReturnPct để tránh leakage.
    """

    if (
        df is None
        or df.empty
    ):

        return None, None, []

    work = df.copy()

    if "Return" not in work.columns:

        return (
            None,
            None,
            [],
        )

    all_features = (
        get_all_numeric_features(
            work
        )
    )

    # --------------------------------------------------------
    # Chống target leakage.
    #
    # Return hiện tại là nguồn tạo Target.
    # ReturnPct là bản % của Return.
    # --------------------------------------------------------

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

        return (
            None,
            None,
            [],
        )

    work["Target"] = (
        pd.to_numeric(
            work["Return"],
            errors="coerce",
        ).shift(-1)
    )

    # --------------------------------------------------------
    # Ép toàn bộ feature về numeric
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Bỏ feature hoàn toàn rỗng.
    # Không bỏ biến có một vài NaN.
    # --------------------------------------------------------

    features = [
        column
        for column in features
        if work[
            column
        ].notna().sum() >= 20
    ]

    if not features:

        return (
            None,
            None,
            [],
        )

    X = work[
        features
    ].copy()

    y = work[
        "Target"
    ].copy()

    # --------------------------------------------------------
    # Fill missing bằng median của feature.
    # --------------------------------------------------------

    for column in features:

        median = X[
            column
        ].median()

        if pd.isna(
            median
        ):

            median = 0.0

        X[
            column
        ] = X[
            column
        ].fillna(
            median
        )

    valid_target = (
        y.notna()
        & np.isfinite(y)
    )

    X = X.loc[
        valid_target
    ]

    y = y.loc[
        valid_target
    ]

    if len(X) < 80:

        return (
            None,
            None,
            [],
        )

    return (
        X,
        y,
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
        np.sqrt(
            mse
        )
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
    Tính VIF cho toàn bộ biến bằng ma trận tương quan.

    Cách này nhanh hơn gọi variance_inflation_factor
    hàng chục lần khi số feature lớn.
    """

    try:

        X_work = X.copy()

        X_work = X_work.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        for column in X_work.columns:

            X_work[column] = pd.to_numeric(
                X_work[column],
                errors="coerce",
            )

            median = X_work[
                column
            ].median()

            if pd.isna(
                median
            ):
                median = 0.0

            X_work[
                column
            ] = X_work[
                column
            ].fillna(
                median
            )

        # Loại biến không có variance.
        X_work = X_work.loc[
            :,
            X_work.nunique(
                dropna=True
            ) > 1,
        ]

        if X_work.empty:

            return pd.DataFrame()

        corr = (
            X_work.corr()
        )

        corr = corr.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).fillna(
            0.0
        )

        # Pseudoinverse chống ma trận singular.
        inverse = np.linalg.pinv(
            corr.values
        )

        vif_values = np.diag(
            inverse
        )

        vif_values = np.where(
            np.isfinite(
                vif_values
            ),
            vif_values,
            np.inf,
        )

        result = pd.DataFrame(
            {
                "Biến": corr.columns,
                "VIF": vif_values,
            }
        )

        return (
            result
            .sort_values(
                "VIF",
                ascending=False,
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
    Chạy toàn bộ nghiên cứu định lượng trong một lần.

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
        "target": None,
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

    result[
        "target"
    ] = "Return phiên kế tiếp"

    # ========================================================
    # TIME SPLIT
    # ========================================================

    split = int(
        len(X)
        * 0.8
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

        metrics = regression_metrics(
            y_test,
            pred,
        )

        result[
            "models"
        ].append(
            {
                "Mô hình": "OLS + HC3",
                **metrics,
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
                "Lỗi": str(
                    error
                ),
            }
        )

    # ========================================================
    # LINEAR MODELS
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

                metrics = regression_metrics(
                    y_test,
                    pred,
                )

                result[
                    "models"
                ].append(
                    {
                        "Mô hình": name,
                        **metrics,
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
                        "Lỗi": str(
                            error
                        ),
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

        tree_importances = []

        for name, model in tree_models:

            try:

                model.fit(
                    X_train,
                    y_train,
                )

                pred = model.predict(
                    X_test
                )

                metrics = regression_metrics(
                    y_test,
                    pred,
                )

                result[
                    "models"
                ].append(
                    {
                        "Mô hình": name,
                        **metrics,
                    }
                )

                importance = getattr(
                    model,
                    "feature_importances_",
                    None,
                )

                if importance is not None:

                    temp = pd.DataFrame(
                        {
                            "Biến": features,
                            "Mức quan trọng": (
                                importance
                            ),
                        }
                    )

                    temp[
                        "Mô hình"
                    ] = name

                    tree_importances.append(
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
                        "Lỗi": str(
                            error
                        ),
                    }
                )

        # ----------------------------------------------------
        # Gộp importance các model cây.
        # ----------------------------------------------------

        if tree_importances:

            all_importance = pd.concat(
                tree_importances,
                ignore_index=True,
            )

            average_importance = (
                all_importance
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
            ] = average_importance

    except Exception:
        pass

    # ========================================================
    # TESTS
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

            adf_result = adfuller(
                y.dropna()
            )

            result[
                "tests"
            ][
                "ADF"
            ] = {
                "statistic": float(
                    adf_result[0]
                ),
                "p_value": float(
                    adf_result[1]
                ),
                "lags": int(
                    adf_result[2]
                ),
                "observations": int(
                    adf_result[3]
                ),
            }

        except Exception:
            pass

        residuals = result.get(
            "residuals"
        )

        if residuals is not None:

            # ------------------------------------------------
            # Jarque-Bera
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Durbin-Watson
            # ------------------------------------------------

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
                    ),
                }

            except Exception:
                pass

        # ----------------------------------------------------
        # Breusch-Pagan
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
        # White
        #
        # White test với toàn bộ feature có thể tạo ra lượng
        # interaction cực lớn.
        #
        # Vì vậy giữ toàn bộ feature cho model chính, nhưng
        # White dùng một bộ biến có variance cao để kiểm định
        # heteroskedasticity mà không làm treo app.
        # ----------------------------------------------------

        if ols_model is not None:

            try:

                feature_variance = (
                    X_train
                    .var()
                    .sort_values(
                        ascending=False
                    )
                )

                white_features = (
                    feature_variance
                    .head(
                        min(
                            8,
                            len(
                                feature_variance
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
        # Ljung-Box
        # ----------------------------------------------------

        if residuals is not None:

            try:

                max_lag = min(
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
                        max_lag
                    ],
                    return_df=True,
                )

                result[
                    "tests"
                ][
                    "Ljung-Box"
                ] = {
                    "lag": max_lag,
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
# DIỄN GIẢI MODEL
# ============================================================

MODEL_EXPLANATIONS = {
    "OLS + HC3": (
        "Hồi quy tuyến tính. Dùng để đo quan hệ có điều kiện "
        "giữa các biến giải thích và lợi suất phiên kế tiếp. "
        "HC3 làm sai số chuẩn bền vững hơn khi phương sai thay đổi."
    ),

    "Ridge": (
        "Hồi quy tuyến tính với regularization L2. "
        "Giúp ổn định mô hình khi nhiều biến có tương quan cao."
    ),

    "Lasso": (
        "Hồi quy tuyến tính với regularization L1. "
        "Một số hệ số có thể bị kéo về 0, hỗ trợ chọn biến."
    ),

    "Elastic Net": (
        "Kết hợp L1 và L2. Hữu ích khi có nhiều biến tương quan "
        "và vẫn muốn một phần chọn lọc biến."
    ),

    "Random Forest": (
        "Nhiều cây quyết định kết hợp với nhau. "
        "Có thể mô hình hóa quan hệ phi tuyến và tương tác giữa biến."
    ),

    "Extra Trees": (
        "Một ensemble cây có mức randomization cao hơn Random Forest. "
        "Cho thêm một góc nhìn về quan hệ phi tuyến."
    ),

    "Gradient Boosting": (
        "Các cây được xây tuần tự để giảm sai số của cây trước. "
        "Có khả năng bắt tín hiệu phi tuyến nhưng cần chú ý overfitting."
    ),
}


# ============================================================
# DIỄN GIẢI TEST
# ============================================================

TEST_EXPLANATIONS = {
    "ADF": (
        "Kiểm tra tính dừng của chuỗi. "
        "p-value thấp thường ủng hộ việc bác bỏ giả thuyết "
        "có nghiệm đơn vị."
    ),

    "Jarque-Bera": (
        "Kiểm tra phần dư có gần phân phối chuẩn hay không. "
        "p-value thấp cho thấy phần dư lệch khỏi normality."
    ),

    "Breusch-Pagan": (
        "Kiểm tra heteroskedasticity. "
        "p-value thấp cho thấy phương sai phần dư thay đổi."
    ),

    "White": (
        "Kiểm định phương sai thay đổi tổng quát hơn "
        "Breusch-Pagan."
    ),

    "Durbin-Watson": (
        "Kiểm tra tự tương quan bậc một của phần dư. "
        "Giá trị gần 2 thường ít đáng lo hơn."
    ),

    "Ljung-Box": (
        "Kiểm tra tự tương quan của phần dư trên nhiều độ trễ."
    ),

    "VIF": (
        "Đo đa cộng tuyến giữa các biến giải thích. "
        "VIF cao nghĩa là các biến có thông tin trùng lặp mạnh."
    ),
}


# ============================================================
# RENDER TEST RESULT
# ============================================================

def render_test_summary(
    tests: dict,
):

    rows = []

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    if "ADF" in tests:

        item = tests[
            "ADF"
        ]

        statistic = num(
            item.get(
                "statistic"
            )
        )

        p_value = num(
            item.get(
                "p_value"
            )
        )

        if (
            p_value is not None
            and p_value < 0.05
        ):

            conclusion = (
                "✅ Có bằng chứng chuỗi dừng"
            )

        else:

            conclusion = (
                "⚠️ Chưa đủ bằng chứng chuỗi dừng"
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
            item.get(
                "statistic"
            )
        )

        p_value = num(
            item.get(
                "p_value"
            )
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
                "✅ Chưa thấy bằng chứng lệch chuẩn mạnh"
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
            item.get(
                "LM"
            )
        )

        p_value = num(
            item.get(
                "p_value"
            )
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
                "✅ Chưa phát hiện phương sai thay đổi rõ"
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
            item.get(
                "LM"
            )
        )

        p_value = num(
            item.get(
                "p_value"
            )
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
                "✅ Chưa phát hiện phương sai thay đổi rõ"
            )

        rows.append(
            {
                "Kiểm định": (
                    "White"
                ),
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
            item.get(
                "statistic"
            )
        )

        if statistic is None:

            conclusion = (
                "⚪ Không xác định"
            )

        elif 1.5 <= statistic <= 2.5:

            conclusion = (
                "✅ Tương đối gần 2"
            )

        elif statistic < 1.5:

            conclusion = (
                "⚠️ Có dấu hiệu tự tương quan dương"
            )

        else:

            conclusion = (
                "⚠️ Có dấu hiệu tự tương quan âm"
            )

        rows.append(
            {
                "Kiểm định": (
                    "Durbin-Watson"
                ),
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
            item.get(
                "statistic"
            )
        )

        p_value = num(
            item.get(
                "p_value"
            )
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
                "✅ Chưa phát hiện tự tương quan còn lại rõ"
            )

        rows.append(
            {
                "Kiểm định": (
                    "Ljung-Box"
                ),
                "Statistic": statistic,
                "p-value": p_value,
                "Kết luận": conclusion,
            }
        )

    if not rows:

        return pd.DataFrame()

    summary = pd.DataFrame(
        rows
    )

    summary[
        "Statistic"
    ] = pd.to_numeric(
        summary[
            "Statistic"
        ],
        errors="coerce",
    ).round(5)

    summary[
        "p-value"
    ] = pd.to_numeric(
        summary[
            "p-value"
        ],
        errors="coerce",
    ).round(6)

    return summary


# ============================================================
# KẾT LUẬN NGHIÊN CỨU
# ============================================================

def build_quant_conclusions(
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
                "Mô hình có RMSE thấp nhất trên tập test là "
                f"**{best['Mô hình']}** "
                f"với RMSE = {best['RMSE']:.6f}."
            )

    # ========================================================
    # R2 OLS
    # ========================================================

    ols_summary = quant_result.get(
        "ols_summary"
    )

    if isinstance(
        ols_summary,
        dict,
    ):

        r2 = num(
            ols_summary.get(
                "rsquared"
            )
        )

        adj_r2 = num(
            ols_summary.get(
                "adj_rsquared"
            )
        )

        if r2 is not None:

            conclusions.append(
                f"OLS có R² = {r2:.4f}"
                + (
                    f" và adjusted R² = {adj_r2:.4f}"
                    if adj_r2 is not None
                    else ""
                )
                + "."
            )

    # ========================================================
    # TESTS
    # ========================================================

    tests = quant_result.get(
        "tests",
        {},
    )

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

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
                    "ADF: có bằng chứng chuỗi mục tiêu dừng "
                    "ở mức ý nghĩa 5%."
                )

            else:

                conclusions.append(
                    "ADF: chưa đủ bằng chứng để kết luận "
                    "chuỗi mục tiêu dừng."
                )

    # --------------------------------------------------------
    # Heteroskedasticity
    # --------------------------------------------------------

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
            "Các kiểm định phương sai cho thấy cần lưu ý "
            "heteroskedasticity; vì vậy OLS nên được đọc cùng "
            "sai số chuẩn robust."
        )

    else:

        conclusions.append(
            "Breusch–Pagan và White chưa cho thấy bằng chứng mạnh "
            "về phương sai thay đổi."
        )

    # --------------------------------------------------------
    # Autocorrelation
    # --------------------------------------------------------

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
            "mô hình có thể chưa khai thác hết cấu trúc theo thời gian."
        )

    else:

        conclusions.append(
            "Chưa thấy bằng chứng mạnh về tự tương quan còn lại "
            "của phần dư."
        )

    # --------------------------------------------------------
    # VIF
    # --------------------------------------------------------

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
                ].replace(
                    np.inf,
                    np.nan,
                ).max()
            )

            if max_vif is not None:

                if max_vif > 10:

                    conclusions.append(
                        f"VIF cao nhất khoảng {max_vif:.2f}; "
                        "có dấu hiệu đa cộng tuyến mạnh."
                    )

                elif max_vif > 5:

                    conclusions.append(
                        f"VIF cao nhất khoảng {max_vif:.2f}; "
                        "có đa cộng tuyến cần lưu ý."
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
        "tin tức, AI và định lượng. "
        "Dữ liệu được lấy từ nguồn thị trường thực."
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

    q1, q2, q3, q4 = st.columns(
        4
    )

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
    # VN-INDEX DETAIL
    # ========================================================

    st.subheader(
        "📊 VN-INDEX"
    )

    v1, v2, v3, v4 = st.columns(
        4
    )

    with v1:

        st.metric(
            "Điểm",
            vn_price,
        )

    with v2:

        st.metric(
            "Thay đổi",
            vn_change,
        )

    with v3:

        st.metric(
            "Thay đổi 1D",
            vn_percent,
        )

    with v4:

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

    load_stock_button = st.button(
        "🔄 Tải dữ liệu",
        type="primary",
        key="dashboard_load_button",
    )

    if load_stock_button:

        clean_symbol = normalize_symbol(
            symbol_input
        )

        if clean_symbol:

            st.session_state[
                "dashboard_symbol"
            ] = clean_symbol

            # Xóa kết quả Quant cũ khi đổi mã.
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
    # STOCK DETAILS
    # ========================================================

    if (
        stock is not None
        and not stock.empty
    ):

        stock_snapshot = (
            market_snapshot(
                stock
            )
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

        macd_value = num(
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

        s1, s2, s3, s4 = st.columns(
            4
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

        s5, s6, s7, s8 = st.columns(
            4
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
                    f"{macd_value:.3f}"
                    if macd_value is not None
                    else "—"
                ),
            )

        with s8:

            st.metric(
                "Biến động 20 phiên",
                (
                    f"{volatility20:.2f}%"
                    if volatility20 is not None
                    else "—"
                ),
            )

        # ====================================================
        # EXTRA
        # ====================================================

        e1, e2, e3, e4 = st.columns(
            4
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
    # NGHIÊN CỨU ĐỊNH LƯỢNG
    # ========================================================

    st.subheader(
        "🧮 Nghiên cứu định lượng"
    )

    st.caption(
        "Tất cả biến numeric từ dữ liệu kỹ thuật được đưa vào "
        "pipeline. Mô hình và kiểm định chỉ chạy khi bấm nút."
    )

    if (
        stock is None
        or stock.empty
    ):

        st.info(
            "Cần có dữ liệu cổ phiếu để chạy Quant."
        )

    else:

        run_quant_button = st.button(
            "🧮 Chạy toàn bộ mô hình + kiểm định",
            type="primary",
            key="dashboard_run_quant",
        )

        if run_quant_button:

            with st.spinner(
                "Đang chạy toàn bộ nghiên cứu định lượng..."
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

        if quant_symbol != current_symbol:

            quant_result = None

        if quant_result is None:

            st.info(
                "Bấm nút trên để chạy toàn bộ mô hình "
                "và kiểm định."
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
            # OVERVIEW
            # =================================================

            st.markdown(
                "### 📌 Tổng quan nghiên cứu"
            )

            o1, o2, o3, o4 = st.columns(
                4
            )

            with o1:

                st.metric(
                    "Số biến",
                    f"{quant_result.get('feature_count', 0):,}",
                )

            with o2:

                st.metric(
                    "Train",
                    f"{quant_result.get('train_size', 0):,}",
                )

            with o3:

                st.metric(
                    "Test",
                    f"{quant_result.get('test_size', 0):,}",
                )

            with o4:

                st.metric(
                    "Số mô hình",
                    f"{len(quant_result.get('models', [])):,}",
                )

            # =================================================
            # ALL FEATURES
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
                    height=300,
                    hide_index=True,
                )

            # =================================================
            # MODEL COMPARISON
            # =================================================

            st.markdown(
                "### 🤖 So sánh tất cả mô hình"
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

                display_models = (
                    model_df.copy()
                )

                for column in [
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R2",
                ]:

                    if column in display_models.columns:

                        display_models[
                            column
                        ] = display_models[
                            column
                        ].round(6)

                st.dataframe(
                    display_models,
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
            # MODEL EXPLANATION
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
                    + MODEL_EXPLANATIONS.get(
                        model_name,
                        "",
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
                isinstance(
                    importance_df,
                    pd.DataFrame,
                )
                and not importance_df.empty
            ):

                st.markdown(
                    "### 🌳 Feature importance"
                )

                importance_show = (
                    importance_df
                    .head(30)
                    .copy()
                )

                importance_show[
                    "Mức quan trọng"
                ] = pd.to_numeric(
                    importance_show[
                        "Mức quan trọng"
                    ],
                    errors="coerce",
                ).round(6)

                st.dataframe(
                    importance_show,
                    width="stretch",
                    hide_index=True,
                )

            # =================================================
            # TEST RESULTS
            # =================================================

            st.markdown(
                "### 🧪 Kết quả kiểm định"
            )

            tests = quant_result.get(
                "tests",
                {},
            )

            test_summary = (
                render_test_summary(
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
                    "Chưa có kết quả kiểm định."
                )

            # =================================================
            # TEST EXPLANATION
            # =================================================

            st.markdown(
                "### 📖 Ý nghĩa từng kiểm định"
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
            # CHI TIẾT TEST
            # =================================================

            # -------------------------------------------------
            # ADF
            # -------------------------------------------------

            if "ADF" in tests:

                adf = tests[
                    "ADF"
                ]

                p = num(
                    adf.get(
                        "p_value"
                    )
                )

                statistic = num(
                    adf.get(
                        "statistic"
                    )
                )

                st.markdown(
                    "#### ADF"
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "Statistic",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

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

                    st.success(
                        "ADF: bác bỏ giả thuyết có nghiệm đơn vị "
                        "ở mức 5%."
                    )

                else:

                    st.warning(
                        "ADF: chưa đủ bằng chứng bác bỏ giả thuyết "
                        "có nghiệm đơn vị."
                    )

            # -------------------------------------------------
            # Jarque-Bera
            # -------------------------------------------------

            if "Jarque-Bera" in tests:

                jb = tests[
                    "Jarque-Bera"
                ]

                p = num(
                    jb.get(
                        "p_value"
                    )
                )

                statistic = num(
                    jb.get(
                        "statistic"
                    )
                )

                st.markdown(
                    "#### Jarque–Bera"
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "Statistic",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

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
                        "Phần dư lệch khỏi giả định phân phối chuẩn."
                    )

                else:

                    st.success(
                        "Chưa có bằng chứng mạnh cho thấy phần dư lệch chuẩn."
                    )

            # -------------------------------------------------
            # Breusch-Pagan
            # -------------------------------------------------

            if "Breusch-Pagan" in tests:

                bp = tests[
                    "Breusch-Pagan"
                ]

                p = num(
                    bp.get(
                        "p_value"
                    )
                )

                statistic = num(
                    bp.get(
                        "LM"
                    )
                )

                st.markdown(
                    "#### Breusch–Pagan"
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "LM",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

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
                        "Có dấu hiệu heteroskedasticity."
                    )

                else:

                    st.success(
                        "Chưa phát hiện heteroskedasticity rõ."
                    )

            # -------------------------------------------------
            # White
            # -------------------------------------------------

            if "White" in tests:

                white = tests[
                    "White"
                ]

                p = num(
                    white.get(
                        "p_value"
                    )
                )

                statistic = num(
                    white.get(
                        "LM"
                    )
                )

                st.markdown(
                    "#### White Test"
                )

                a1, a2 = st.columns(2)

                with a1:

                    st.metric(
                        "LM",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a2:

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
                        "White test cho thấy dấu hiệu phương sai thay đổi."
                    )

                else:

                    st.success(
                        "White test chưa cho thấy bằng chứng mạnh "
                        "về phương sai thay đổi."
                    )

                used_count = white.get(
                    "feature_count_used"
                )

                used_features = white.get(
                    "features_used"
                )

                if used_count:

                    st.caption(
                        "White test dùng "
                        f"{used_count} biến có phương sai cao "
                        "để tránh bùng nổ số interaction; "
                        "toàn bộ biến vẫn được dùng trong các mô hình chính."
                    )

                    if used_features:

                        st.caption(
                            "Biến White test: "
                            + ", ".join(
                                map(
                                    str,
                                    used_features,
                                )
                            )
                        )

            # -------------------------------------------------
            # Durbin-Watson
            # -------------------------------------------------

            if "Durbin-Watson" in tests:

                dw = tests[
                    "Durbin-Watson"
                ]

                statistic = num(
                    dw.get(
                        "statistic"
                    )
                )

                st.markdown(
                    "#### Durbin–Watson"
                )

                st.metric(
                    "Statistic",
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

            # -------------------------------------------------
            # Ljung-Box
            # -------------------------------------------------

            if "Ljung-Box" in tests:

                lb = tests[
                    "Ljung-Box"
                ]

                statistic = num(
                    lb.get(
                        "statistic"
                    )
                )

                p = num(
                    lb.get(
                        "p_value"
                    )
                )

                lag = lb.get(
                    "lag"
                )

                st.markdown(
                    "#### Ljung–Box"
                )

                a1, a2, a3 = st.columns(3)

                with a1:

                    st.metric(
                        "Lag",
                        str(
                            lag
                            if lag is not None
                            else "—"
                        ),
                    )

                with a2:

                    st.metric(
                        "Statistic",
                        format_decimal(
                            statistic,
                            5,
                        ),
                    )

                with a3:

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
                        "Phần dư còn dấu hiệu tự tương quan."
                    )

                else:

                    st.success(
                        "Chưa phát hiện tự tương quan còn lại rõ."
                    )

            # -------------------------------------------------
            # VIF
            # -------------------------------------------------

            if "VIF" in tests:

                vif_df = tests[
                    "VIF"
                ]

                st.markdown(
                    "#### VIF — Đa cộng tuyến"
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

                    vif_display = (
                        vif_show.copy()
                    )

                    vif_display[
                        "VIF"
                    ] = vif_display[
                        "VIF"
                    ].replace(
                        np.inf,
                        np.nan,
                    ).round(3)

                    st.dataframe(
                        vif_display,
                        width="stretch",
                        height=400,
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
                build_quant_conclusions(
                    quant_result
                )
            )

            if conclusions:

                for conclusion in conclusions:

                    st.markdown(
                        "• "
                        + conclusion
                    )

            else:

                st.info(
                    "Chưa đủ kết quả để tạo kết luận."
                )

            st.warning(
                "Kết luận này mô tả bằng chứng thống kê trên mẫu dữ liệu hiện tại; "
                "không phải bảo đảm cho diễn biến giá trong tương lai."
            )

    # ========================================================
    # AI MARKET BRIEF
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
