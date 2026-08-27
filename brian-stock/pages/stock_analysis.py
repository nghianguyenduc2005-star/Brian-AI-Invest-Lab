from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components.ai import (
    render_ai_panel,
    stock_analysis_prompt,
)

from components.charts import (
    price_volume_chart,
)

from data.market import (
    display_symbol,
    load_market_data,
    market_snapshot,
    normalize_symbol,
)


# ============================================================
# CẤU HÌNH
# ============================================================

DISPLAY_PERIOD = "1y"

RESEARCH_PERIODS = [
    "1mo",
    "3mo",
    "1y",
    "3y",
    "5y",
    "10y",
]

RESEARCH_PERIOD_LABELS = {
    "1mo": "1 tháng",
    "3mo": "3 tháng",
    "1y": "1 năm",
    "3y": "3 năm",
    "5y": "5 năm",
    "10y": "10 năm",
}

RESEARCH_HORIZONS = [
    "1D",
    "5D",
    "20D",
]

RESEARCH_CACHE_TTL = 900


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


def format_price(
    value: Any,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng"


def format_percent(
    value: Any,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def format_volume(
    value: Any,
):
    value = num(value)

    if value is None:
        return "—"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn"

    return f"{value:,.0f}"


def format_decimal(
    value: Any,
    digits=4,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:.{digits}f}"


# ============================================================
# TRẠNG THÁI KỸ THUẬT
# ============================================================

def rsi_status(
    value: Any,
):
    value = num(value)

    if value is None:
        return "Không xác định"

    if value >= 70:
        return "Quá mua"

    if value <= 30:
        return "Quá bán"

    return "Trung tính"


def price_vs_ma_status(
    price: Any,
    sma20: Any,
    sma50: Any,
):
    price = num(price)
    sma20 = num(sma20)
    sma50 = num(sma50)

    if price is None:
        return "Không xác định"

    if (
        sma20 is not None
        and sma50 is not None
    ):
        if price > sma20 > sma50:
            return "Xu hướng tăng"

        if price < sma20 < sma50:
            return "Xu hướng giảm"

        return "Đang giằng co"

    if sma20 is not None:

        if price > sma20:
            return "Trên MA20"

        if price < sma20:
            return "Dưới MA20"

    return "Không xác định"


def macd_status(
    value: Any,
):
    value = num(value)

    if value is None:
        return "Không xác định"

    if value > 0:
        return "MACD dương"

    if value < 0:
        return "MACD âm"

    return "MACD trung tính"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_display_data(
    symbol: str,
):
    return load_market_data(
        symbol,
        DISPLAY_PERIOD,
    )


@st.cache_data(
    ttl=RESEARCH_CACHE_TTL,
    show_spinner=False,
)
def load_research_data(
    symbol: str,
    period: str,
):
    return load_market_data(
        symbol,
        period,
    )


# ============================================================
# NUMERIC FEATURES
# ============================================================

def get_numeric_features(
    df: pd.DataFrame,
):
    if (
        df is None
        or df.empty
    ):
        return []

    features = []

    for column in df.columns:

        name = str(
            column
        ).strip().lower()

        if name in {
            "time",
            "date",
            "datetime",
            "timestamp",
        }:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            features.append(
                column
            )

    return list(
        dict.fromkeys(
            features
        )
    )


# ============================================================
# BUILD FACTOR DATASET
# ============================================================

def build_factor_dataset(
    df: pd.DataFrame,
):
    """
    X_t:
        toàn bộ biến numeric hợp lệ tại thời điểm t

    Y:
        lợi suất tương lai 1D, 5D, 20D

    Không dùng:
        Target_*
        ReturnPct vì là bản sao của Return
    """

    if (
        df is None
        or df.empty
        or "Close" not in df.columns
    ):
        return None

    work = df.copy()

    numeric_features = (
        get_numeric_features(
            work
        )
    )

    if "Return" not in work.columns:

        close = pd.to_numeric(
            work["Close"],
            errors="coerce",
        )

        work[
            "Return"
        ] = close.pct_change()

        if "Return" not in numeric_features:
            numeric_features.append(
                "Return"
            )

    features = [
        column
        for column in numeric_features
        if column not in {
            "ReturnPct",
            "Target_1D",
            "Target_5D",
            "Target_20D",
        }
    ]

    # Return hiện tại được phép dùng vì nó là thông tin
    # đã biết tại thời điểm t.
    if (
        "Return" in work.columns
        and "Return" not in features
    ):
        features.append(
            "Return"
        )

    for column in features:

        work[
            column
        ] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    close = pd.to_numeric(
        work["Close"],
        errors="coerce",
    )

    work[
        "Target_1D"
    ] = (
        close.shift(-1)
        / close
        - 1
    )

    work[
        "Target_5D"
    ] = (
        close.shift(-5)
        / close
        - 1
    )

    work[
        "Target_20D"
    ] = (
        close.shift(-20)
        / close
        - 1
    )

    work = work.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # Giữ feature nếu có ít nhất 50 giá trị thực.
    features = [
        column
        for column in features
        if work[
            column
        ].notna().sum() >= 50
    ]

    if not features:
        return None

    X = work[
        features
    ].copy()

    # Điền missing theo median.
    for column in features:

        median = X[
            column
        ].median()

        if pd.isna(median):
            median = 0.0

        X[
            column
        ] = X[
            column
        ].fillna(
            median
        )

    datasets = {}

    for horizon in RESEARCH_HORIZONS:

        target = work[
            f"Target_{horizon}"
        ]

        valid = (
            target.notna()
            & np.isfinite(
                target
            )
        )

        X_h = X.loc[
            valid
        ].copy()

        y_h = target.loc[
            valid
        ].copy()

        if len(X_h) >= 80:

            datasets[
                horizon
            ] = {
                "X": X_h.astype(float),
                "y": y_h.astype(float),
            }

    if not datasets:
        return None

    return {
        "features": features,
        "datasets": datasets,
    }


# ============================================================
# REGRESSION METRICS
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
# CORRELATION
# ============================================================

def calculate_correlations(
    X: pd.DataFrame,
    y: pd.Series,
):
    rows = []

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    for column in X.columns:

        x = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        mask = (
            x.notna()
            & y.notna()
            & np.isfinite(x)
            & np.isfinite(y)
        )

        x_valid = x.loc[
            mask
        ]

        y_valid = y.loc[
            mask
        ]

        if len(x_valid) < 50:
            continue

        try:

            pearson = float(
                x_valid.corr(
                    y_valid,
                    method="pearson",
                )
            )

        except Exception:

            pearson = np.nan

        try:

            spearman = float(
                x_valid.corr(
                    y_valid,
                    method="spearman",
                )
            )

        except Exception:

            spearman = np.nan

        rows.append(
            {
                "Biến": column,
                "Pearson": pearson,
                "Spearman": spearman,
                "|Spearman|": (
                    abs(spearman)
                    if np.isfinite(
                        spearman
                    )
                    else np.nan
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "|Spearman|",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# STANDARDIZED OLS
# ============================================================

def standardized_ols(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    try:

        import statsmodels.api as sm
        from sklearn.preprocessing import (
            StandardScaler,
        )

        scaler = StandardScaler()

        X_scaled_array = scaler.fit_transform(
            X_train
        )

        X_scaled = pd.DataFrame(
            X_scaled_array,
            columns=X_train.columns,
            index=X_train.index,
        )

        X_with_const = sm.add_constant(
            X_scaled,
            has_constant="add",
        )

        model = sm.OLS(
            y_train,
            X_with_const,
        ).fit(
            cov_type="HC3"
        )

        rows = []

        for column in X_train.columns:

            beta = num(
                model.params.get(
                    column
                ),
                None,
            )

            p_value = num(
                model.pvalues.get(
                    column
                ),
                None,
            )

            rows.append(
                {
                    "Biến": column,
                    "Beta chuẩn hóa": beta,
                    "p-value": p_value,
                    "|Beta|": (
                        abs(beta)
                        if beta is not None
                        else np.nan
                    ),
                }
            )

        table = (
            pd.DataFrame(
                rows
            )
            .sort_values(
                "|Beta|",
                ascending=False,
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        return {
            "model": model,
            "table": table,
            "r2": num(
                model.rsquared,
                None,
            ),
            "adj_r2": num(
                model.rsquared_adj,
                None,
            ),
        }

    except Exception:
        return None


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def permutation_importance_table(
    X_train,
    X_test,
    y_train,
    y_test,
):
    try:

        from sklearn.ensemble import (
            ExtraTreesRegressor,
        )

        from sklearn.inspection import (
            permutation_importance,
        )

        model = ExtraTreesRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        metrics = regression_metrics(
            y_test,
            predictions,
        )

        permutation = (
            permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=8,
                random_state=42,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
        )

        table = pd.DataFrame(
            {
                "Biến": X_train.columns,
                "Permutation": (
                    permutation.importances_mean
                ),
                "Permutation_STD": (
                    permutation.importances_std
                ),
                "TreeImportance": (
                    model.feature_importances_
                ),
            }
        )

        table[
            "Permutation"
        ] = table[
            "Permutation"
        ].clip(
            lower=0
        )

        table = (
            table
            .sort_values(
                "Permutation",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        return {
            "model": model,
            "metrics": metrics,
            "table": table,
        }

    except Exception:
        return None


# ============================================================
# VIF
# ============================================================

def calculate_vif(
    X: pd.DataFrame,
):
    try:

        work = X.copy()

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

            work[
                column
            ] = work[
                column
            ].fillna(
                median
            )

        # Biến không có variation không hữu ích cho VIF.
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

        inverse = np.linalg.pinv(
            corr.values
        )

        vif = np.diag(
            inverse
        )

        result = pd.DataFrame(
            {
                "Biến": corr.columns,
                "VIF": vif,
            }
        )

        result[
            "VIF"
        ] = result[
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
# FACTOR RANKING
# ============================================================

def build_factor_ranking(
    correlation: pd.DataFrame,
    ols_table: pd.DataFrame | None,
    permutation_table: pd.DataFrame | None,
):
    if (
        correlation is None
        or correlation.empty
    ):
        return pd.DataFrame()

    result = correlation[
        [
            "Biến",
            "Pearson",
            "Spearman",
            "|Spearman|",
        ]
    ].copy()

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    if (
        ols_table is not None
        and not ols_table.empty
    ):

        result = result.merge(
            ols_table[
                [
                    "Biến",
                    "Beta chuẩn hóa",
                    "p-value",
                    "|Beta|",
                ]
            ],
            on="Biến",
            how="left",
        )

    else:

        result[
            "Beta chuẩn hóa"
        ] = np.nan

        result[
            "p-value"
        ] = np.nan

        result[
            "|Beta|"
        ] = np.nan

    # --------------------------------------------------------
    # Permutation
    # --------------------------------------------------------

    if (
        permutation_table is not None
        and not permutation_table.empty
    ):

        result = result.merge(
            permutation_table[
                [
                    "Biến",
                    "Permutation",
                    "Permutation_STD",
                    "TreeImportance",
                ]
            ],
            on="Biến",
            how="left",
        )

    else:

        result[
            "Permutation"
        ] = np.nan

        result[
            "Permutation_STD"
        ] = np.nan

        result[
            "TreeImportance"
        ] = np.nan

    # --------------------------------------------------------
    # Percentile scores
    # --------------------------------------------------------

    result[
        "Score_corr"
    ] = result[
        "|Spearman|"
    ].rank(
        ascending=False,
        pct=True,
    )

    result[
        "Score_ols"
    ] = result[
        "|Beta|"
    ].rank(
        ascending=False,
        pct=True,
    )

    result[
        "Score_perm"
    ] = result[
        "Permutation"
    ].rank(
        ascending=False,
        pct=True,
    )

    score_columns = [
        "Score_corr",
        "Score_ols",
        "Score_perm",
    ]

    result[
        "RankScore"
    ] = result[
        score_columns
    ].mean(
        axis=1,
        skipna=True,
    )

    result[
        "Mức độ ảnh hưởng tương đối"
    ] = (
        1
        - result[
            "RankScore"
        ]
    ) * 100

    # --------------------------------------------------------
    # Chiều quan hệ
    # --------------------------------------------------------

    def get_direction(row):

        beta = num(
            row.get(
                "Beta chuẩn hóa"
            ),
            None,
        )

        spearman = num(
            row.get(
                "Spearman"
            ),
            None,
        )

        value = (
            beta
            if beta is not None
            else spearman
        )

        if value is None:
            return "Không xác định"

        if value > 0:
            return "Cùng chiều"

        if value < 0:
            return "Ngược chiều"

        return "Trung tính"

    result[
        "Chiều quan hệ"
    ] = result.apply(
        get_direction,
        axis=1,
    )

    # --------------------------------------------------------
    # OLS significance
    # --------------------------------------------------------

    def get_significance(row):

        p = num(
            row.get(
                "p-value"
            ),
            None,
        )

        if p is None:
            return "Không có p-value"

        if p < 0.01:
            return "Rất mạnh"

        if p < 0.05:
            return "Có ý nghĩa 5%"

        if p < 0.10:
            return "Có ý nghĩa 10%"

        return "Chưa có ý nghĩa"

    result[
        "Ý nghĩa OLS"
    ] = result.apply(
        get_significance,
        axis=1,
    )

    return (
        result
        .sort_values(
            "Mức độ ảnh hưởng tương đối",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# FULL RESEARCH ENGINE
# ============================================================

@st.cache_data(
    ttl=RESEARCH_CACHE_TTL,
    show_spinner=False,
)
def run_full_factor_research(
    df: pd.DataFrame,
):
    """
    Nghiên cứu hoàn chỉnh chạy LOCAL.

    Không gọi API.
    """

    result = {
        "ok": False,
        "error": None,
        "features": [],
        "horizons": {},
    }

    prepared = build_factor_dataset(
        df
    )

    if prepared is None:

        result[
            "error"
        ] = (
            "Không đủ dữ liệu để xây dựng dataset nghiên cứu."
        )

        return result

    result[
        "features"
    ] = prepared[
        "features"
    ]

    for horizon in RESEARCH_HORIZONS:

        if horizon not in prepared[
            "datasets"
        ]:
            continue

        X = prepared[
            "datasets"
        ][
            horizon
        ][
            "X"
        ].copy()

        y = prepared[
            "datasets"
        ][
            horizon
        ][
            "y"
        ].copy()

        # ====================================================
        # TIME SPLIT
        # ====================================================

        split = int(
            len(X) * 0.8
        )

        if (
            split < 50
            or len(X) - split < 20
        ):
            continue

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

        # ====================================================
        # CORRELATION
        # ====================================================

        correlations = (
            calculate_correlations(
                X,
                y,
            )
        )

        # ====================================================
        # STANDARDIZED OLS
        # ====================================================

        ols = standardized_ols(
            X_train,
            y_train,
        )

        ols_table = None

        if ols is not None:
            ols_table = ols[
                "table"
            ]

        # ====================================================
        # PERMUTATION
        # ====================================================

        permutation = (
            permutation_importance_table(
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

        permutation_table = None

        if permutation is not None:
            permutation_table = (
                permutation[
                    "table"
                ]
            )

        # ====================================================
        # MODELS
        # ====================================================

        models = []

        # ----------------------------------------------------
        # Ridge / Lasso / Elastic Net
        # ----------------------------------------------------

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

                    prediction = (
                        model.predict(
                            X_test
                        )
                    )

                    models.append(
                        {
                            "Mô hình": model_name,
                            **regression_metrics(
                                y_test,
                                prediction,
                            ),
                        }
                    )

                except Exception:
                    pass

        except Exception:
            pass

        # ----------------------------------------------------
        # Tree models
        # ----------------------------------------------------

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
                        n_estimators=250,
                        max_depth=8,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                (
                    "Extra Trees",
                    ExtraTreesRegressor(
                        n_estimators=250,
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

            for model_name, model in tree_models:

                try:

                    model.fit(
                        X_train,
                        y_train,
                    )

                    prediction = (
                        model.predict(
                            X_test
                        )
                    )

                    models.append(
                        {
                            "Mô hình": model_name,
                            **regression_metrics(
                                y_test,
                                prediction,
                            ),
                        }
                    )

                except Exception:
                    pass

        except Exception:
            pass

        # ====================================================
        # STATISTICAL TESTS
        # ====================================================

        tests = {}

        # ----------------------------------------------------
        # ADF
        # ----------------------------------------------------

        try:

            from statsmodels.tsa.stattools import (
                adfuller,
            )

            adf = adfuller(
                y.dropna()
            )

            tests[
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

        # ----------------------------------------------------
        # OLS residual tests
        # ----------------------------------------------------

        residuals = None

        if ols is not None:

            try:

                residuals = np.asarray(
                    ols[
                        "model"
                    ].resid,
                    dtype=float,
                )

            except Exception:
                residuals = None

        # ----------------------------------------------------
        # Jarque-Bera + DW
        # ----------------------------------------------------

        if residuals is not None:

            try:

                from statsmodels.stats.stattools import (
                    durbin_watson,
                    jarque_bera,
                )

                jb = jarque_bera(
                    residuals
                )

                tests[
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

                tests[
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
        # Breusch-Pagan + White
        # ----------------------------------------------------

        if ols is not None:

            try:

                import statsmodels.api as sm

                from statsmodels.stats.diagnostic import (
                    het_breuschpagan,
                    het_white,
                )

                X_const = sm.add_constant(
                    X_train,
                    has_constant="add",
                )

                bp = het_breuschpagan(
                    ols[
                        "model"
                    ].resid,
                    X_const,
                )

                tests[
                    "Breusch-Pagan"
                ] = {
                    "statistic": float(
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

                # White test:
                # dùng tối đa 8 biến có variance lớn nhất
                # để tránh số interaction bùng nổ.
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
                        ols[
                            "model"
                        ].resid,
                        X_white,
                    )

                    tests[
                        "White"
                    ] = {
                        "statistic": float(
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
                        "features_used": white_features,
                    }

            except Exception:
                pass

        # ----------------------------------------------------
        # Ljung-Box
        # ----------------------------------------------------

        if residuals is not None:

            try:

                from statsmodels.stats.diagnostic import (
                    acorr_ljungbox,
                )

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

                tests[
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

        # ----------------------------------------------------
        # VIF
        # ----------------------------------------------------

        try:

            vif = calculate_vif(
                X_train
            )

            if not vif.empty:

                tests[
                    "VIF"
                ] = vif

        except Exception:
            pass

        # ====================================================
        # FACTOR RANKING
        # ====================================================

        ranking = build_factor_ranking(
            correlations,
            ols_table,
            permutation_table,
        )

        # ====================================================
        # MODEL TABLE
        # ====================================================

        model_df = pd.DataFrame(
            models
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

            model_df = (
                model_df
                .sort_values(
                    "RMSE",
                    ascending=True,
                    na_position="last",
                )
                .reset_index(
                    drop=True
                )
            )

        # ====================================================
        # SAVE
        # ====================================================

        result[
            "horizons"
        ][
            horizon
        ] = {
            "observations": len(X),
            "train": len(X_train),
            "test": len(X_test),
            "correlation": correlations,
            "ols": ols,
            "ols_table": ols_table,
            "permutation": permutation,
            "permutation_table": permutation_table,
            "ranking": ranking,
            "models": model_df,
            "tests": tests,
        }

    if not result[
        "horizons"
    ]:

        result[
            "error"
        ] = (
            "Không có horizon nào đủ dữ liệu."
        )

        return result

    result[
        "ok"
    ] = True

    return result


# ============================================================
# GIẢI THÍCH YẾU TỐ
# ============================================================

def explain_factor(
    row: pd.Series,
):
    name = str(
        row.get(
            "Biến",
            "",
        )
    )

    direction = str(
        row.get(
            "Chiều quan hệ",
            "Không xác định",
        )
    )

    score = num(
        row.get(
            "Mức độ ảnh hưởng tương đối"
        ),
        None,
    )

    spearman = num(
        row.get(
            "Spearman"
        ),
        None,
    )

    beta = num(
        row.get(
            "Beta chuẩn hóa"
        ),
        None,
    )

    p_value = num(
        row.get(
            "p-value"
        ),
        None,
    )

    permutation = num(
        row.get(
            "Permutation"
        ),
        None,
    )

    text = (
        f"**{name}**"
    )

    if score is not None:

        text += (
            f" — điểm ảnh hưởng tương đối "
            f"{score:.1f}/100"
        )

    text += (
        f", quan hệ {direction.lower()}"
    )

    if spearman is not None:

        text += (
            f", Spearman {spearman:+.4f}"
        )

    if beta is not None:

        text += (
            f", beta chuẩn hóa {beta:+.4f}"
        )

    if p_value is not None:

        text += (
            f", p-value {p_value:.5f}"
        )

    if permutation is not None:

        text += (
            f", permutation {permutation:.6f}"
        )

    return (
        text
        + "."
    )


# ============================================================
# KẾT LUẬN HORIZON
# ============================================================

def build_horizon_conclusion(
    horizon_result: dict,
    horizon_label: str,
):
    lines = []

    ranking = horizon_result.get(
        "ranking"
    )

    # --------------------------------------------------------
    # TOP FACTOR
    # --------------------------------------------------------

    if (
        isinstance(
            ranking,
            pd.DataFrame,
        )
        and not ranking.empty
    ):

        top = ranking.iloc[
            0
        ]

        factor = str(
            top.get(
                "Biến",
                "Không xác định",
            )
        )

        direction = str(
            top.get(
                "Chiều quan hệ",
                "Không xác định",
            )
        )

        score = num(
            top.get(
                "Mức độ ảnh hưởng tương đối"
            )
        )

        spearman = num(
            top.get(
                "Spearman"
            )
        )

        lines.append(
            f"Ở horizon **{horizon_label}**, "
            f"yếu tố nổi bật nhất là **{factor}**, "
            f"có quan hệ {direction.lower()}"
            + (
                f" và điểm ảnh hưởng tương đối "
                f"{score:.1f}/100"
                if score is not None
                else ""
            )
            + (
                f" (Spearman = {spearman:+.4f})."
                if spearman is not None
                else "."
            )
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    models = horizon_result.get(
        "models"
    )

    if (
        isinstance(
            models,
            pd.DataFrame,
        )
        and not models.empty
    ):

        valid = models[
            models[
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

            best_r2 = num(
                best.get(
                    "R2"
                )
            )

            model_text = (
                f"Model có RMSE thấp nhất là "
                f"**{best['Mô hình']}** "
                f"với RMSE = {best['RMSE']:.6f}"
            )

            if best_r2 is not None:

                model_text += (
                    f", R² = {best_r2:.4f}"
                )

                if best_r2 < 0:

                    model_text += (
                        ". Tuy nhiên R² âm cho thấy "
                        "khả năng dự báo ngoài mẫu yếu."
                    )

                elif best_r2 < 0.10:

                    model_text += (
                        ". Khả năng giải thích còn thấp."
                    )

                elif best_r2 < 0.25:

                    model_text += (
                        ". Có tín hiệu nhưng chưa mạnh."
                    )

                else:

                    model_text += (
                        ". Tín hiệu dự báo đáng chú ý hơn."
                    )

            else:

                model_text += "."

            lines.append(
                model_text
            )

    # --------------------------------------------------------
    # TEST SUMMARY
    # --------------------------------------------------------

    tests = horizon_result.get(
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

            lines.append(
                (
                    "ADF: bác bỏ giả thuyết nghiệm đơn vị "
                    "ở mức 5%."
                    if p < 0.05
                    else
                    "ADF: chưa đủ bằng chứng bác bỏ giả thuyết nghiệm đơn vị."
                )
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

            lines.append(
                (
                    "Breusch–Pagan: có dấu hiệu "
                    "phương sai thay đổi."
                    if p < 0.05
                    else
                    "Breusch–Pagan: chưa thấy dấu hiệu "
                    "phương sai thay đổi mạnh."
                )
            )

    if "Durbin-Watson" in tests:

        dw = num(
            tests[
                "Durbin-Watson"
            ].get(
                "statistic"
            )
        )

        if dw is not None:

            lines.append(
                (
                    f"Durbin–Watson = {dw:.4f}, "
                    "tương đối gần 2."
                    if 1.5 <= dw <= 2.5
                    else
                    f"Durbin–Watson = {dw:.4f}, "
                    "cần lưu ý khả năng tự tương quan."
                )
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

            lines.append(
                (
                    "Ljung–Box: chưa phát hiện "
                    "tự tương quan còn lại rõ."
                    if p >= 0.05
                    else
                    "Ljung–Box: phần dư còn dấu hiệu "
                    "tự tương quan."
                )
            )

    return lines


# ============================================================
# RENDER RESEARCH
# ============================================================

def render_research_panel(
    symbol: str,
):
    st.header(
        "🧪 Nghiên cứu yếu tố ảnh hưởng"
    )

    st.write(
        "Mục tiêu là xác định biến nào có thông tin mạnh nhất "
        "đối với lợi suất tương lai của cổ phiếu, thay vì chỉ đưa ra "
        "RMSE/R² mà không có diễn giải."
    )

    st.info(
        "Biến phụ thuộc là lợi suất tương lai của cổ phiếu. "
        "Các biến giải thích gồm toàn bộ biến numeric hợp lệ từ dữ liệu kỹ thuật."
    )

    period = st.selectbox(
        "Khoảng dữ liệu nghiên cứu",
        RESEARCH_PERIODS,
        index=2,
        format_func=lambda x: (
            RESEARCH_PERIOD_LABELS.get(
                x,
                x,
            )
        ),
        key="stock_research_period",
    )

    run_button = st.button(
        "🧪 Chạy nghiên cứu toàn diện",
        type="primary",
        key="stock_run_full_research",
    )

    research = st.session_state.get(
        "stock_research_result"
    )

    saved_symbol = st.session_state.get(
        "stock_research_symbol"
    )

    saved_period = st.session_state.get(
        "stock_research_period_saved"
    )

    if run_button:

        try:

            with st.spinner(
                "Đang tải dữ liệu lịch sử nghiên cứu..."
            ):

                research_data = (
                    load_research_data(
                        symbol,
                        period,
                    )
                )

            if (
                research_data is None
                or research_data.empty
            ):

                st.error(
                    "Không có dữ liệu đủ dài cho nghiên cứu."
                )

                return

            # ------------------------------------------------
            # SAMPLE INFO
            # ------------------------------------------------

            raw_observations = len(
                research_data
            )

            start_date = (
                research_data.index.min()
                if hasattr(
                    research_data,
                    "index",
                )
                else None
            )

            end_date = (
                research_data.index.max()
                if hasattr(
                    research_data,
                    "index",
                )
                else None
            )

            st.markdown(
                "### 📚 Mẫu dữ liệu"
            )

            d1, d2, d3, d4 = st.columns(
                4
            )

            with d1:

                st.metric(
                    "Khoảng",
                    RESEARCH_PERIOD_LABELS.get(
                        period,
                        period,
                    ),
                )

            with d2:

                st.metric(
                    "Quan sát thô",
                    f"{raw_observations:,}",
                )

            with d3:

                st.metric(
                    "Từ",
                    (
                        str(
                            start_date.date()
                        )
                        if start_date is not None
                        else "—"
                    ),
                )

            with d4:

                st.metric(
                    "Đến",
                    (
                        str(
                            end_date.date()
                        )
                        if end_date is not None
                        else "—"
                    ),
                )

            # ------------------------------------------------
            # WARNING SAMPLE
            # ------------------------------------------------

            if period == "1mo":

                st.warning(
                    "1 tháng có ít quan sát; kết quả chỉ nên xem là tín hiệu ngắn hạn."
                )

            elif period == "3mo":

                st.info(
                    "3 tháng phù hợp để xem tín hiệu ngắn hạn, "
                    "nhưng sức mạnh thống kê còn hạn chế."
                )

            elif period == "1y":

                st.info(
                    "1 năm phù hợp cho phân tích ngắn đến trung hạn."
                )

            else:

                st.success(
                    "Khoảng dữ liệu dài giúp tăng số quan sát và ổn định hơn."
                )

            # ------------------------------------------------
            # RUN
            # ------------------------------------------------

            with st.spinner(
                "Đang chạy toàn bộ mô hình và kiểm định..."
            ):

                research = (
                    run_full_factor_research(
                        research_data
                    )
                )

            st.session_state[
                "stock_research_result"
            ] = research

            st.session_state[
                "stock_research_symbol"
            ] = symbol

            st.session_state[
                "stock_research_period_saved"
            ] = period

        except Exception as error:

            st.error(
                "Không thể chạy nghiên cứu."
            )

            st.code(
                str(error)
            )

            return

    if (
        research is None
        or saved_symbol != symbol
        or saved_period is None
        or saved_period != period
    ):

        st.caption(
            "Chọn khoảng dữ liệu và bấm "
            "'Chạy nghiên cứu toàn diện'."
        )

        return

    if not research.get(
        "ok",
        False,
    ):

        st.error(
            research.get(
                "error",
                "Nghiên cứu thất bại.",
            )
        )

        return

    # ========================================================
    # RESEARCH OVERVIEW
    # ========================================================

    st.markdown(
        "### 📌 Tổng quan"
    )

    feature_count = len(
        research.get(
            "features",
            [],
        )
    )

    o1, o2, o3 = st.columns(
        3
    )

    with o1:

        st.metric(
            "Số biến",
            f"{feature_count:,}",
        )

    with o2:

        st.metric(
            "Số horizon",
            f"{len(research.get('horizons', {})):,}",
        )

    with o3:

        st.metric(
            "Dữ liệu",
            RESEARCH_PERIOD_LABELS.get(
                saved_period,
                saved_period,
            ),
        )

    with st.expander(
        "🔢 Xem toàn bộ biến được nghiên cứu",
        expanded=False,
    ):

        features = research.get(
            "features",
            [],
        )

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
            hide_index=True,
        )

    # ========================================================
    # HORIZON RESULTS
    # ========================================================

    for horizon in RESEARCH_HORIZONS:

        if horizon not in research[
            "horizons"
        ]:
            continue

        horizon_result = research[
            "horizons"
        ][
            horizon
        ]

        label = {
            "1D": "1 ngày",
            "5D": "5 ngày",
            "20D": "20 ngày",
        }[
            horizon
        ]

        st.divider()

        st.markdown(
            f"## 📈 Kết quả cho lợi suất {label}"
        )

        # ----------------------------------------------------
        # SAMPLE
        # ----------------------------------------------------

        s1, s2, s3 = st.columns(
            3
        )

        with s1:

            st.metric(
                "Quan sát",
                f"{horizon_result.get('observations', 0):,}",
            )

        with s2:

            st.metric(
                "Train",
                f"{horizon_result.get('train', 0):,}",
            )

        with s3:

            st.metric(
                "Test",
                f"{horizon_result.get('test', 0):,}",
            )

        # ----------------------------------------------------
        # TOP FACTORS
        # ----------------------------------------------------

        st.markdown(
            "### 🏆 Yếu tố nổi bật nhất"
        )

        ranking = horizon_result.get(
            "ranking"
        )

        if (
            isinstance(
                ranking,
                pd.DataFrame,
            )
            and not ranking.empty
        ):

            top = ranking.head(
                10
            ).copy()

            display = top[
                [
                    "Biến",
                    "Mức độ ảnh hưởng tương đối",
                    "Chiều quan hệ",
                    "Spearman",
                    "Beta chuẩn hóa",
                    "p-value",
                    "Permutation",
                    "Ý nghĩa OLS",
                ]
            ].copy()

            for column in [
                "Mức độ ảnh hưởng tương đối",
                "Spearman",
                "Beta chuẩn hóa",
                "p-value",
                "Permutation",
            ]:

                display[
                    column
                ] = pd.to_numeric(
                    display[
                        column
                    ],
                    errors="coerce",
                )

            display[
                "Mức độ ảnh hưởng tương đối"
            ] = display[
                "Mức độ ảnh hưởng tương đối"
            ].round(1)

            display[
                "Spearman"
            ] = display[
                "Spearman"
            ].round(4)

            display[
                "Beta chuẩn hóa"
            ] = display[
                "Beta chuẩn hóa"
            ].round(4)

            display[
                "p-value"
            ] = display[
                "p-value"
            ].round(5)

            display[
                "Permutation"
            ] = display[
                "Permutation"
            ].round(6)

            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
            )

            st.markdown(
                "#### 🧠 Diễn giải"
            )

            for _, row in top.head(
                5
            ).iterrows():

                st.markdown(
                    "• "
                    + explain_factor(
                        row
                    )
                )

        else:

            st.info(
                "Không có bảng xếp hạng yếu tố."
            )

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        st.markdown(
            "### 🤖 So sánh mô hình"
        )

        models = horizon_result.get(
            "models"
        )

        if (
            isinstance(
                models,
                pd.DataFrame,
            )
            and not models.empty
        ):

            model_display = models.copy()

            for column in [
                "MAE",
                "MSE",
                "RMSE",
                "R2",
            ]:

                if column in model_display.columns:

                    model_display[
                        column
                    ] = pd.to_numeric(
                        model_display[
                            column
                        ],
                        errors="coerce",
                    ).round(6)

            st.dataframe(
                model_display,
                width="stretch",
                hide_index=True,
            )

            valid = models[
                models[
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

                r2 = num(
                    best.get(
                        "R2"
                    )
                )

                st.success(
                    f"Model tốt nhất tương đối: "
                    f"**{best['Mô hình']}** · "
                    f"RMSE = {best['RMSE']:.6f}"
                    + (
                        f" · R² = {r2:.4f}"
                        if r2 is not None
                        else ""
                    )
                )

        # ----------------------------------------------------
        # OLS
        # ----------------------------------------------------

        ols_table = horizon_result.get(
            "ols_table"
        )

        if (
            isinstance(
                ols_table,
                pd.DataFrame,
            )
            and not ols_table.empty
        ):

            with st.expander(
                "📐 Chi tiết OLS chuẩn hóa",
                expanded=False,
            ):

                ols_show = (
                    ols_table.head(
                        30
                    ).copy()
                )

                ols_show[
                    "Beta chuẩn hóa"
                ] = ols_show[
                    "Beta chuẩn hóa"
                ].round(5)

                ols_show[
                    "p-value"
                ] = ols_show[
                    "p-value"
                ].round(6)

                st.dataframe(
                    ols_show[
                        [
                            "Biến",
                            "Beta chuẩn hóa",
                            "p-value",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

        # ----------------------------------------------------
        # PERMUTATION
        # ----------------------------------------------------

        permutation_table = (
            horizon_result.get(
                "permutation_table"
            )
        )

        if (
            isinstance(
                permutation_table,
                pd.DataFrame,
            )
            and not permutation_table.empty
        ):

            with st.expander(
                "🌳 Permutation importance",
                expanded=False,
            ):

                perm_show = (
                    permutation_table.head(
                        30
                    ).copy()
                )

                perm_show[
                    "Permutation"
                ] = perm_show[
                    "Permutation"
                ].round(6)

                perm_show[
                    "Permutation_STD"
                ] = perm_show[
                    "Permutation_STD"
                ].round(6)

                st.dataframe(
                    perm_show[
                        [
                            "Biến",
                            "Permutation",
                            "Permutation_STD",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

        # ----------------------------------------------------
        # TESTS
        # ----------------------------------------------------

        tests = horizon_result.get(
            "tests",
            {},
        )

        st.markdown(
            "### 🧪 Kiểm định"
        )

        # ADF
        if "ADF" in tests:

            item = tests[
                "ADF"
            ]

            p = num(
                item.get(
                    "p_value"
                )
            )

            stat = num(
                item.get(
                    "statistic"
                )
            )

            a, b = st.columns(2)

            with a:

                st.metric(
                    "ADF Statistic",
                    format_decimal(
                        stat,
                        5,
                    ),
                )

            with b:

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
                    "ADF: có bằng chứng bác bỏ nghiệm đơn vị."
                )

            else:

                st.warning(
                    "ADF: chưa đủ bằng chứng bác bỏ nghiệm đơn vị."
                )

        # Jarque-Bera
        if "Jarque-Bera" in tests:

            item = tests[
                "Jarque-Bera"
            ]

            p = num(
                item.get(
                    "p_value"
                )
            )

            stat = num(
                item.get(
                    "statistic"
                )
            )

            a, b = st.columns(2)

            with a:

                st.metric(
                    "JB Statistic",
                    format_decimal(
                        stat,
                        5,
                    ),
                )

            with b:

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
                    "Jarque–Bera: phần dư không gần phân phối chuẩn."
                )

            else:

                st.success(
                    "Jarque–Bera: chưa phát hiện lệch chuẩn mạnh."
                )

        # Breusch-Pagan
        if "Breusch-Pagan" in tests:

            item = tests[
                "Breusch-Pagan"
            ]

            p = num(
                item.get(
                    "p_value"
                )
            )

            stat = num(
                item.get(
                    "statistic"
                )
            )

            a, b = st.columns(2)

            with a:

                st.metric(
                    "BP Statistic",
                    format_decimal(
                        stat,
                        5,
                    ),
                )

            with b:

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

        # White
        if "White" in tests:

            item = tests[
                "White"
            ]

            p = num(
                item.get(
                    "p_value"
                )
            )

            stat = num(
                item.get(
                    "statistic"
                )
            )

            a, b = st.columns(2)

            with a:

                st.metric(
                    "White Statistic",
                    format_decimal(
                        stat,
                        5,
                    ),
                )

            with b:

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

            used = item.get(
                "features_used",
                [],
            )

            if used:

                st.caption(
                    "White test dùng "
                    f"{len(used)} biến variance cao nhất."
                )

        # DW
        if "Durbin-Watson" in tests:

            stat = num(
                tests[
                    "Durbin-Watson"
                ].get(
                    "statistic"
                )
            )

            st.metric(
                "Durbin-Watson",
                format_decimal(
                    stat,
                    4,
                ),
            )

            if stat is None:

                st.info(
                    "Không xác định."
                )

            elif 1.5 <= stat <= 2.5:

                st.success(
                    "Statistic tương đối gần 2."
                )

            elif stat < 1.5:

                st.warning(
                    "Có dấu hiệu tự tương quan dương."
                )

            else:

                st.warning(
                    "Có dấu hiệu tự tương quan âm."
                )

        # Ljung-Box
        if "Ljung-Box" in tests:

            item = tests[
                "Ljung-Box"
            ]

            lag = item.get(
                "lag"
            )

            stat = num(
                item.get(
                    "statistic"
                )
            )

            p = num(
                item.get(
                    "p_value"
                )
            )

            a, b, c = st.columns(3)

            with a:

                st.metric(
                    "Lag",
                    str(
                        lag
                    ),
                )

            with b:

                st.metric(
                    "Statistic",
                    format_decimal(
                        stat,
                        5,
                    ),
                )

            with c:

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

        # VIF
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

                with st.expander(
                    "🔗 VIF — toàn bộ biến",
                    expanded=False,
                ):

                    vif_display = (
                        vif_df.copy()
                    )

                    vif_display[
                        "VIF"
                    ] = pd.to_numeric(
                        vif_display[
                            "VIF"
                        ],
                        errors="coerce",
                    ).round(3)

                    st.dataframe(
                        vif_display,
                        width="stretch",
                        height=420,
                        hide_index=True,
                    )

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

                            st.error(
                                f"VIF cao nhất = {max_vif:.2f}: "
                                "đa cộng tuyến mạnh."
                            )

                        elif (
                            max_vif is not None
                            and max_vif > 5
                        ):

                            st.warning(
                                f"VIF cao nhất = {max_vif:.2f}: "
                                "đa cộng tuyến cần lưu ý."
                            )

                        else:

                            st.success(
                                "VIF chưa cho thấy đa cộng tuyến nghiêm trọng."
                            )

        # ----------------------------------------------------
        # CONCLUSION
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Kết luận nghiên cứu"
        )

        conclusions = (
            build_horizon_conclusion(
                horizon_result,
                label,
            )
        )

        if conclusions:

            for line in conclusions:

                st.markdown(
                    "• "
                    + line
                )

        else:

            st.info(
                "Chưa đủ bằng chứng để tạo kết luận tự động."
            )

    # ========================================================
    # CROSS HORIZON
    # ========================================================

    st.divider()

    st.header(
        "🏆 Yếu tố nhất quán qua các horizon"
    )

    cross_rows = []

    for horizon, horizon_result in (
        research[
            "horizons"
        ].items()
    ):

        ranking = horizon_result.get(
            "ranking"
        )

        if (
            isinstance(
                ranking,
                pd.DataFrame,
            )
            and not ranking.empty
        ):

            for rank, (_, row) in enumerate(
                ranking.head(
                    20
                ).iterrows(),
                start=1,
            ):

                cross_rows.append(
                    {
                        "Biến": row[
                            "Biến"
                        ],
                        "Horizon": horizon,
                        "Rank": rank,
                        "Score": row[
                            "Mức độ ảnh hưởng tương đối"
                        ],
                        "Chiều": row[
                            "Chiều quan hệ"
                        ],
                    }
                )

    if cross_rows:

        cross_df = pd.DataFrame(
            cross_rows
        )

        stable = (
            cross_df
            .groupby(
                "Biến",
                as_index=False,
            )
            .agg(
                So_horizon=(
                    "Horizon",
                    "nunique",
                ),
                Rank_trung_binh=(
                    "Rank",
                    "mean",
                ),
                Score_trung_binh=(
                    "Score",
                    "mean",
                ),
            )
        )

        stable = (
            stable
            .sort_values(
                [
                    "So_horizon",
                    "Score_trung_binh",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        stable.columns = [
            "Biến",
            "Số horizon xuất hiện",
            "Rank trung bình",
            "Điểm ảnh hưởng TB",
        ]

        stable[
            "Rank trung bình"
        ] = stable[
            "Rank trung bình"
        ].round(2)

        stable[
            "Điểm ảnh hưởng TB"
        ] = stable[
            "Điểm ảnh hưởng TB"
        ].round(1)

        st.dataframe(
            stable.head(
                20
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "Chưa có dữ liệu so sánh cross-horizon."
        )

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    st.header(
        "🎯 Câu trả lời nghiên cứu"
    )

    final_rows = []

    for horizon, horizon_result in (
        research[
            "horizons"
        ].items()
    ):

        ranking = horizon_result.get(
            "ranking"
        )

        if (
            isinstance(
                ranking,
                pd.DataFrame,
            )
            and not ranking.empty
        ):

            top = ranking.iloc[
                0
            ]

            final_rows.append(
                {
                    "Horizon": horizon,
                    "Yếu tố mạnh nhất": top[
                        "Biến"
                    ],
                    "Chiều quan hệ": top[
                        "Chiều quan hệ"
                    ],
                    "Điểm ảnh hưởng": round(
                        num(
                            top[
                                "Mức độ ảnh hưởng tương đối"
                            ],
                            0,
                        ),
                        1,
                    ),
                    "Spearman": round(
                        num(
                            top[
                                "Spearman"
                            ],
                            0,
                        ),
                        4,
                    ),
                    "Beta": round(
                        num(
                            top[
                                "Beta chuẩn hóa"
                            ],
                            0,
                        ),
                        4,
                    ),
                }
            )

    if final_rows:

        final_df = pd.DataFrame(
            final_rows
        )

        st.dataframe(
            final_df,
            width="stretch",
            hide_index=True,
        )

        counts = (
            final_df[
                "Yếu tố mạnh nhất"
            ]
            .value_counts()
        )

        if not counts.empty:

            dominant_factor = (
                counts.index[0]
            )

            appearances = int(
                counts.iloc[0]
            )

            st.success(
                f"**{dominant_factor}** là yếu tố nổi bật nhất "
                f"ở {appearances}/{len(final_df)} horizon."
            )

    st.warning(
        "Điểm ảnh hưởng ở đây là mức độ liên hệ/thông tin dự báo "
        "trong dữ liệu và mô hình. Nó không chứng minh rằng yếu tố đó "
        "gây ra biến động giá theo nghĩa nhân quả."
    )


# ============================================================
# MAIN PAGE
# ============================================================

def render_stock_analysis():

    # ========================================================
    # HEADER
    # ========================================================

    st.caption(
        "BRIAN STOCK · STOCK RESEARCH"
    )

    st.title(
        "Phân tích cổ phiếu"
    )

    st.write(
        "Phân tích kỹ thuật và nghiên cứu định lượng "
        "để xác định yếu tố nào có thông tin mạnh nhất "
        "đối với lợi suất tương lai."
    )

    # ========================================================
    # SYMBOL INPUT
    # ========================================================

    current_symbol = (
        st.session_state.get(
            "stock_analysis_symbol",
            "HPG",
        )
    )

    symbol_input = st.text_input(
        "Mã cổ phiếu",
        value=current_symbol,
        placeholder="Ví dụ HPG, FPT, VNM...",
        key="stock_analysis_symbol_input",
    )

    if st.button(
        "🔄 Tải dữ liệu",
        type="primary",
        key="stock_analysis_load_button",
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
                "stock_analysis_symbol"
            ] = clean_symbol

            # Xóa nghiên cứu cũ.
            st.session_state.pop(
                "stock_research_result",
                None,
            )

            st.session_state.pop(
                "stock_research_symbol",
                None,
            )

            st.session_state.pop(
                "stock_research_period_saved",
                None,
            )

            st.rerun()

    symbol = normalize_symbol(
        st.session_state.get(
            "stock_analysis_symbol",
            symbol_input,
        )
    )

    # ========================================================
    # LOAD DISPLAY DATA
    # ========================================================

    try:

        data = load_display_data(
            symbol
        )

    except Exception as error:

        st.error(
            f"Không thể tải dữ liệu {display_symbol(symbol)}."
        )

        st.code(
            str(error)
        )

        return

    if (
        data is None
        or data.empty
    ):

        st.warning(
            "Không có dữ liệu cổ phiếu."
        )

        return

    # ========================================================
    # SNAPSHOT
    # ========================================================

    snapshot = market_snapshot(
        data
    )

    price = num(
        snapshot.get(
            "price"
        )
    )

    change_1d = num(
        snapshot.get(
            "change_1d"
        )
    )

    rsi_value = num(
        snapshot.get(
            "rsi"
        )
    )

    volume = num(
        snapshot.get(
            "volume"
        )
    )

    sma20 = num(
        snapshot.get(
            "sma20"
        )
    )

    sma50 = num(
        snapshot.get(
            "sma50"
        )
    )

    macd = num(
        snapshot.get(
            "macd"
        )
    )

    volatility = num(
        snapshot.get(
            "volatility20"
        )
    )

    last_row = data.iloc[
        -1
    ]

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

    # ========================================================
    # TITLE
    # ========================================================

    st.subheader(
        f"📈 {display_symbol(symbol)}"
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
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

    # ========================================================
    # SECONDARY
    # ========================================================

    c5, c6, c7, c8 = st.columns(
        4
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
                f"{volatility:.2f}%"
                if volatility is not None
                else "—"
            ),
        )

    # ========================================================
    # TECHNICAL STATUS
    # ========================================================

    st.subheader(
        "🧭 Trạng thái kỹ thuật"
    )

    t1, t2, t3, t4 = st.columns(
        4
    )

    with t1:

        st.metric(
            "Xu hướng",
            price_vs_ma_status(
                price,
                sma20,
                sma50,
            ),
        )

    with t2:

        st.metric(
            "RSI",
            rsi_status(
                rsi_value
            ),
        )

    with t3:

        st.metric(
            "MACD",
            macd_status(
                macd
            ),
        )

    with t4:

        st.metric(
            "Thanh khoản",
            (
                f"{relative_volume:.2f}x TB20"
                if relative_volume is not None
                else "—"
            ),
        )

    # ========================================================
    # EXTRA
    # ========================================================

    st.subheader(
        "📋 Chỉ báo bổ sung"
    )

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

        opening_price = num(
            last_row.get(
                "Open"
            )
        )

        st.metric(
            "Giá mở cửa",
            format_price(
                opening_price
            ),
        )

    with e4:

        high = num(
            last_row.get(
                "High"
            )
        )

        low = num(
            last_row.get(
                "Low"
            )
        )

        close = num(
            last_row.get(
                "Close"
            )
        )

        if (
            high is not None
            and low is not None
            and close is not None
            and high != low
        ):

            position = (
                (
                    close
                    - low
                )
                / (
                    high
                    - low
                )
                * 100
            )

            st.metric(
                "Vị trí trong biên ngày",
                f"{position:.1f}%",
            )

        else:

            st.metric(
                "Vị trí trong biên ngày",
                "—",
            )

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "📊 Biểu đồ kỹ thuật"
    )

    try:

        chart = price_volume_chart(
            data
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

        st.caption(
            str(error)
        )

    # ========================================================
    # QUANT RESEARCH
    # ========================================================

    st.divider()

    render_research_panel(
        symbol
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_analysis():
    render_stock_analysis()
