from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import price_volume_chart

from data.market import (
    display_symbol,
    load_market_data,
    load_multifactor_research_history,
    market_snapshot,
    normalize_symbol,
)


# ============================================================
# CẤU HÌNH
# ============================================================

CACHE_TTL_RESEARCH = 900

HORIZONS = {
    "1D": 1,
    "5D": 5,
    "20D": 20,
}

HORIZON_LABELS = {
    "1D": "1 phiên",
    "5D": "5 phiên",
    "20D": "20 phiên",
}

PERIOD_PRESETS = {
    "1 tháng": 31,
    "3 tháng": 93,
    "6 tháng": 186,
    "1 năm": 365,
    "3 năm": 1095,
    "5 năm": 1825,
    "10 năm": 3652,
}

TRAIN_RATIO = 0.80

MIN_OBSERVATIONS = 60

MAX_FEATURES_MODEL = 80

MAX_RANKING_ROWS = 40


# ============================================================
# TIỆN ÍCH SỐ
# ============================================================

def num(
    value: Any,
    default=None,
):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def format_price(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng"


def format_percent(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def format_number(
    value,
    digits=4,
):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:.{digits}f}"


def format_volume(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    value = abs(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu"

    if value >= 1_000:
        return f"{value / 1_000:.2f} nghìn"

    return f"{value:,.0f}"


def format_money(
    value,
):
    value = num(value)

    if value is None:
        return "—"

    value = abs(value)

    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f} nghìn tỷ"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu"

    return f"{value:,.0f}"


# ============================================================
# TRẠNG THÁI KỸ THUẬT
# ============================================================

def rsi_status(
    value,
):
    value = num(value)

    if value is None:
        return "Không xác định"

    if value >= 70:
        return "Quá mua"

    if value <= 30:
        return "Quá bán"

    return "Trung tính"


def ma_status(
    price,
    sma20,
    sma50,
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
    value,
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
# DATE HELPERS
# ============================================================

def normalize_datetime_index(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    work = df.copy()

    work.index = pd.to_datetime(
        work.index,
        errors="coerce",
    )

    work = work[
        ~work.index.isna()
    ].copy()

    work = work.sort_index()

    work = work[
        ~work.index.duplicated(
            keep="last"
        )
    ].copy()

    return work


def slice_date_range(
    df,
    start_date,
    end_date,
):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    work = normalize_datetime_index(
        df
    )

    start_ts = pd.Timestamp(
        start_date
    ).normalize()

    end_ts = (
        pd.Timestamp(
            end_date
        ).normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )

    return work.loc[
        (work.index >= start_ts)
        & (work.index <= end_ts)
    ].copy()


def get_actual_date_range(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return None, None

    work = normalize_datetime_index(
        df
    )

    if work.empty:
        return None, None

    return (
        work.index.min().date(),
        work.index.max().date(),
    )


# ============================================================
# FEATURE GROUPS
# ============================================================

def classify_feature(
    feature,
):
    name = str(
        feature
    )

    lower = name.lower()

    if lower.startswith(
        "foreign_"
    ):
        return "Khối ngoại"

    if lower.startswith(
        "proprietary_"
    ):
        return "Tự doanh"

    if lower.startswith(
        "sector_"
    ):
        return "Nhóm ngành"

    if lower.startswith(
        "market_"
    ):
        return "Thị trường chung"

    if lower.startswith(
        "flow_"
    ):
        return "Dòng tiền"

    if any(
        token in lower
        for token in [
            "trading_value",
            "relative_volume",
            "volume_",
            "volume",
            "dollar_volume",
            "obv",
        ]
    ):
        return "Dòng tiền"

    return "Kỹ thuật"


def feature_group_name(
    feature,
):
    group = classify_feature(
        feature
    )

    labels = {
        "Kỹ thuật": "Kỹ thuật",
        "Dòng tiền": "Dòng tiền",
        "Khối ngoại": "Khối ngoại",
        "Tự doanh": "Tự doanh",
        "Thị trường chung": "Thị trường chung",
        "Nhóm ngành": "Nhóm ngành",
    }

    return labels.get(
        group,
        group,
    )


def group_features(
    features,
):
    groups = {
        "Kỹ thuật": [],
        "Dòng tiền": [],
        "Khối ngoại": [],
        "Tự doanh": [],
        "Thị trường chung": [],
        "Nhóm ngành": [],
    }

    for feature in features:

        group = classify_feature(
            feature
        )

        groups.setdefault(
            group,
            []
        ).append(
            feature
        )

    return groups


# ============================================================
# EXCLUDE TARGET / LEAKAGE
# ============================================================

def get_model_features(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return []

    excluded_exact = {
        "Target_1D",
        "Target_5D",
        "Target_20D",
        "Target_1D_Pct",
        "Target_5D_Pct",
        "Target_20D_Pct",
    }

    excluded_contains = [
        "future",
        "target_",
    ]

    features = []

    for column in df.columns:

        name = str(
            column
        )

        if name in excluded_exact:
            continue

        if any(
            token in name.lower()
            for token in excluded_contains
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        features.append(
            column
        )

    return list(
        dict.fromkeys(
            features
        )
    )


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_research_data(
    sample,
):
    if (
        sample is None
        or sample.empty
    ):
        return None

    if "Close" not in sample.columns:
        return None

    df = normalize_datetime_index(
        sample
    )

    if df.empty:
        return None

    # Numeric conversion.
    for column in df.columns:

        try:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        except Exception:
            pass

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    # ========================================================
    # ENSURE TARGETS
    # ========================================================

    for horizon, periods in HORIZONS.items():

        target = (
            close.shift(
                -periods
            )
            / close
            - 1
        )

        df[
            f"Target_{horizon}"
        ] = target

        df[
            f"Target_{horizon}_Pct"
        ] = (
            target
            * 100
        )

    # ========================================================
    # FEATURES
    # ========================================================

    features = get_model_features(
        df
    )

    # Current price itself is allowed for price-model context,
    # but the research factors prioritize returns / normalized
    # technical variables.
    return {
        "data": df,
        "features": features,
    }


# ============================================================
# CLEAN X
# ============================================================

def prepare_xy(
    df,
    features,
    target_column,
):
    if (
        df is None
        or df.empty
        or target_column not in df.columns
    ):
        return None

    available = [
        column
        for column in features
        if column in df.columns
    ]

    if not available:
        return None

    X = df[
        available
    ].copy()

    y = pd.to_numeric(
        df[
            target_column
        ],
        errors="coerce",
    )

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    y = y.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # Median fill.
    for column in X.columns:

        value = X[
            column
        ].median()

        if pd.isna(
            value
        ):
            value = 0.0

        X[
            column
        ] = X[
            column
        ].fillna(
            value
        )

    valid = y.notna()

    X = X.loc[
        valid
    ]

    y = y.loc[
        valid
    ]

    # Remove zero variance from actual model matrix.
    usable = []

    for column in X.columns:

        try:

            variance = float(
                X[
                    column
                ].var(
                    ddof=0
                )
            )

            if (
                np.isfinite(
                    variance
                )
                and variance > 0
            ):
                usable.append(
                    column
                )

        except Exception:
            pass

    if not usable:
        return None

    X = X[
        usable
    ]

    return X.astype(
        float
    ), y.astype(
        float
    )


# ============================================================
# CORRELATIONS
# ============================================================

def correlation_table(
    X,
    y,
):
    rows = []

    for column in X.columns:

        x = X[
            column
        ]

        valid = (
            x.notna()
            & y.notna()
        )

        if valid.sum() < 20:
            continue

        xx = x.loc[
            valid
        ]

        yy = y.loc[
            valid
        ]

        try:
            pearson = float(
                xx.corr(
                    yy,
                    method="pearson",
                )
            )
        except Exception:
            pearson = np.nan

        try:
            spearman = float(
                xx.corr(
                    yy,
                    method="spearman",
                )
            )
        except Exception:
            spearman = np.nan

        rows.append(
            {
                "Biến": column,
                "Nhóm": feature_group_name(
                    column
                ),
                "Pearson": pearson,
                "Spearman": spearman,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(
        rows
    )

    result[
        "|Spearman|"
    ] = result[
        "Spearman"
    ].abs()

    return (
        result
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
    X_train,
    y_train,
):
    try:

        import statsmodels.api as sm

        from sklearn.preprocessing import (
            StandardScaler,
        )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X_train
        )

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=X_train.columns,
            index=X_train.index,
        )

        model = sm.OLS(
            y_train,
            sm.add_constant(
                X_scaled,
                has_constant="add",
            ),
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": min(
                    5,
                    max(
                        1,
                        len(X_train) // 20,
                    ),
                )
            },
        )

        rows = []

        for column in X_train.columns:

            beta = num(
                model.params.get(
                    column
                )
            )

            pvalue = num(
                model.pvalues.get(
                    column
                )
            )

            stderr = num(
                model.bse.get(
                    column
                )
            )

            rows.append(
                {
                    "Biến": column,
                    "Nhóm": feature_group_name(
                        column
                    ),
                    "Beta": beta,
                    "p-value": pvalue,
                    "Std Error": stderr,
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
                model.rsquared
            ),
            "adj_r2": num(
                model.rsquared_adj
            ),
        }

    except Exception as error:

        return {
            "model": None,
            "table": pd.DataFrame(),
            "r2": np.nan,
            "adj_r2": np.nan,
            "error": str(error),
        }


# ============================================================
# VIF
# ============================================================

def calculate_vif(
    X,
):
    try:

        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor,
        )

        clean = X.copy()

        clean = clean.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        clean = clean.fillna(
            clean.median()
        )

        columns = []

        for column in clean.columns:

            variance = clean[
                column
            ].var()

            if (
                pd.notna(
                    variance
                )
                and variance > 0
            ):
                columns.append(
                    column
                )

        clean = clean[
            columns
        ]

        if clean.empty:
            return pd.DataFrame()

        values = clean.values.astype(
            float
        )

        rows = []

        for index, column in enumerate(
            clean.columns
        ):

            try:

                vif = float(
                    variance_inflation_factor(
                        values,
                        index,
                    )
                )

            except Exception:

                vif = np.inf

            rows.append(
                {
                    "Biến": column,
                    "Nhóm": feature_group_name(
                        column
                    ),
                    "VIF": vif,
                }
            )

        return (
            pd.DataFrame(
                rows
            )
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
# MODELS
# ============================================================

def run_model_suite(
    X_train,
    X_test,
    y_train,
    y_test,
):
    try:

        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
            HistGradientBoostingRegressor,
        )

        from sklearn.linear_model import (
            Ridge,
            ElasticNet,
        )

        from sklearn.pipeline import (
            Pipeline,
        )

        from sklearn.preprocessing import (
            StandardScaler,
        )

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

    except Exception:
        return (
            pd.DataFrame(),
            {},
            {},
        )

    models = [
        (
            "Ridge",
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        Ridge(
                            alpha=1.0
                        ),
                    ),
                ]
            ),
        ),
        (
            "Elastic Net",
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        ElasticNet(
                            alpha=0.0001,
                            l1_ratio=0.5,
                            max_iter=50000,
                        ),
                    ),
                ]
            ),
        ),
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
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42,
            ),
        ),
        (
            "Hist Gradient Boosting",
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
    ]

    rows = []
    fitted = {}
    predictions = {}

    actual = np.asarray(
        y_test,
        dtype=float,
    )

    baseline = np.repeat(
        np.mean(
            np.asarray(
                y_train,
                dtype=float,
            )
        ),
        len(actual),
    )

    baseline_mae = float(
        mean_absolute_error(
            actual,
            baseline,
        )
    )

    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                baseline,
            )
        )
    )

    for name, model in models:

        try:

            model.fit(
                X_train,
                y_train,
            )

            pred = np.asarray(
                model.predict(
                    X_test
                ),
                dtype=float,
            )

            mae = float(
                mean_absolute_error(
                    actual,
                    pred,
                )
            )

            mse = float(
                mean_squared_error(
                    actual,
                    pred,
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
                        actual,
                        pred,
                    )
                )
            except Exception:
                r2 = np.nan

            rows.append(
                {
                    "Mô hình": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2,
                    "So với baseline":
                        "Tốt hơn"
                        if rmse < baseline_rmse
                        else "Kém hơn / tương đương",
                }
            )

            fitted[
                name
            ] = model

            predictions[
                name
            ] = pred

        except Exception:
            continue

    if not rows:
        return (
            pd.DataFrame(),
            {},
            {},
        )

    models_df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "RMSE",
                "MAE",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        models_df,
        fitted,
        predictions,
    )


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def permutation_table(
    model,
    X_test,
    y_test,
):
    try:

        from sklearn.inspection import (
            permutation_importance,
        )

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )

        table = pd.DataFrame(
            {
                "Biến": X_test.columns,
                "Permutation":
                    result.importances_mean,
                "Permutation_STD":
                    result.importances_std,
            }
        )

        table[
            "AbsPermutation"
        ] = table[
            "Permutation"
        ].abs()

        table[
            "Nhóm"
        ] = table[
            "Biến"
        ].apply(
            feature_group_name
        )

        return (
            table
            .sort_values(
                "AbsPermutation",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# TREE IMPORTANCE
# ============================================================

def tree_importance_table(
    model,
    features,
):
    try:

        raw = model

        if hasattr(
            model,
            "named_steps",
        ):

            raw = model.named_steps.get(
                "model",
                model,
            )

        if not hasattr(
            raw,
            "feature_importances_",
        ):
            return pd.DataFrame()

        values = np.asarray(
            raw.feature_importances_,
            dtype=float,
        )

        table = pd.DataFrame(
            {
                "Biến": features,
                "TreeImportance": values,
            }
        )

        table[
            "Nhóm"
        ] = table[
            "Biến"
        ].apply(
            feature_group_name
        )

        return (
            table
            .sort_values(
                "TreeImportance",
                ascending=False,
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
    correlations,
    ols_table,
    permutation,
    tree_importance,
):
    if (
        correlations is None
        or correlations.empty
    ):
        return pd.DataFrame()

    result = correlations.copy()

    # OLS
    if (
        isinstance(
            ols_table,
            pd.DataFrame,
        )
        and not ols_table.empty
    ):

        result = result.merge(
            ols_table[
                [
                    "Biến",
                    "Beta",
                    "p-value",
                    "|Beta|",
                ]
            ],
            on="Biến",
            how="left",
        )

    else:

        result[
            "Beta"
        ] = np.nan

        result[
            "p-value"
        ] = np.nan

        result[
            "|Beta|"
        ] = np.nan

    # Permutation
    if (
        isinstance(
            permutation,
            pd.DataFrame,
        )
        and not permutation.empty
    ):

        result = result.merge(
            permutation[
                [
                    "Biến",
                    "Permutation",
                    "AbsPermutation",
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
            "AbsPermutation"
        ] = np.nan

    # Tree
    if (
        isinstance(
            tree_importance,
            pd.DataFrame,
        )
        and not tree_importance.empty
    ):

        result = result.merge(
            tree_importance[
                [
                    "Biến",
                    "TreeImportance",
                ]
            ],
            on="Biến",
            how="left",
        )

    else:

        result[
            "TreeImportance"
        ] = np.nan

    # ========================================================
    # NORMALIZED SCORES
    # ========================================================

    def percentile(
        series,
    ):
        series = pd.to_numeric(
            series,
            errors="coerce",
        )

        if series.notna().sum() <= 1:
            return pd.Series(
                0.0,
                index=series.index,
            )

        return (
            series.rank(
                pct=True,
                na_option="keep",
            )
            * 100
        )

    result[
        "CorrScore"
    ] = percentile(
        result[
            "|Spearman|"
        ]
    )

    result[
        "BetaScore"
    ] = percentile(
        result[
            "|Beta|"
        ]
    )

    result[
        "PermutationScore"
    ] = percentile(
        result[
            "AbsPermutation"
        ]
    )

    result[
        "TreeScore"
    ] = percentile(
        result[
            "TreeImportance"
        ]
    )

    # ========================================================
    # COMPOSITE SCORE
    #
    # Correlation + OLS + model importance.
    # ========================================================

    result[
        "Score"
    ] = (
        result[
            "CorrScore"
        ].fillna(0)
        * 0.25
        + result[
            "BetaScore"
        ].fillna(0)
        * 0.25
        + result[
            "PermutationScore"
        ].fillna(0)
        * 0.25
        + result[
            "TreeScore"
        ].fillna(0)
        * 0.25
    )

    # ========================================================
    # SIGN / DIRECTION
    # ========================================================

    def get_direction(
        row,
    ):
        beta = num(
            row.get(
                "Beta"
            ),
            None,
        )

        spear = num(
            row.get(
                "Spearman"
            ),
            None,
        )

        if beta is not None:
            value = beta
        elif spear is not None:
            value = spear
        else:
            return "Không xác định"

        if value > 0:
            return "Cùng chiều"

        if value < 0:
            return "Ngược chiều"

        return "Trung tính"

    result[
        "Quan hệ"
    ] = result.apply(
        get_direction,
        axis=1,
    )

    # ========================================================
    # STATISTICAL EVIDENCE
    # ========================================================

    def get_significance(
        value,
    ):
        p = num(
            value,
            None,
        )

        if p is None:
            return "Không có p-value"

        if p < 0.01:
            return "Rất mạnh"

        if p < 0.05:
            return "Có ý nghĩa 5%"

        if p < 0.10:
            return "Có tín hiệu 10%"

        return "Chưa rõ"

    result[
        "Ý nghĩa"
    ] = result[
        "p-value"
    ].apply(
        get_significance
    )

    return (
        result
        .sort_values(
            "Score",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# GROUP IMPORTANCE
# ============================================================

def build_group_summary(
    ranking,
):
    if (
        ranking is None
        or ranking.empty
    ):
        return pd.DataFrame()

    work = ranking.copy()

    work[
        "Score"
    ] = pd.to_numeric(
        work[
            "Score"
        ],
        errors="coerce",
    )

    grouped = (
        work
        .groupby(
            "Nhóm",
            as_index=False,
        )
        .agg(
            Score_TB=(
                "Score",
                "mean",
            ),
            Score_Max=(
                "Score",
                "max",
            ),
            So_bien=(
                "Biến",
                "count",
            ),
        )
    )

    grouped[
        "Score_TB"
    ] = grouped[
        "Score_TB"
    ].round(
        1
    )

    grouped[
        "Score_Max"
    ] = grouped[
        "Score_Max"
    ].round(
        1
    )

    return (
        grouped
        .sort_values(
            [
                "Score_TB",
                "Score_Max",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# STATISTICAL TESTS
# ============================================================

def run_tests(
    X_train,
    y_train,
    ols_result,
):
    tests = {}

    # ========================================================
    # ADF
    # ========================================================

    try:

        from statsmodels.tsa.stattools import (
            adfuller,
        )

        result = adfuller(
            np.asarray(
                y_train,
                dtype=float,
            ),
            autolag="AIC",
        )

        tests[
            "ADF"
        ] = {
            "statistic": float(
                result[0]
            ),
            "p_value": float(
                result[1]
            ),
        }

    except Exception:
        pass

    # ========================================================
    # RESIDUALS
    # ========================================================

    residuals = None

    try:

        if (
            isinstance(
                ols_result,
                dict,
            )
            and ols_result.get(
                "model"
            ) is not None
        ):

            residuals = np.asarray(
                ols_result[
                    "model"
                ].resid,
                dtype=float,
            )

    except Exception:
        residuals = None

    if residuals is None:
        return tests

    # ========================================================
    # JB
    # ========================================================

    try:

        from statsmodels.stats.stattools import (
            jarque_bera,
        )

        value = jarque_bera(
            residuals
        )

        tests[
            "Jarque-Bera"
        ] = {
            "statistic": float(
                value[0]
            ),
            "p_value": float(
                value[1]
            ),
        }

    except Exception:
        pass

    # ========================================================
    # DURBIN-WATSON
    # ========================================================

    try:

        from statsmodels.stats.stattools import (
            durbin_watson,
        )

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

    # ========================================================
    # LJUNG BOX
    # ========================================================

    try:

        from statsmodels.stats.diagnostic import (
            acorr_ljungbox,
        )

        lag = min(
            10,
            max(
                1,
                len(residuals) // 10,
            ),
        )

        result = acorr_ljungbox(
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
                result[
                    "lb_stat"
                ].iloc[-1]
            ),
            "p_value": float(
                result[
                    "lb_pvalue"
                ].iloc[-1]
            ),
        }

    except Exception:
        pass

    # ========================================================
    # BREUSCH PAGAN
    # ========================================================

    try:

        import statsmodels.api as sm

        from statsmodels.stats.diagnostic import (
            het_breuschpagan,
        )

        X_const = sm.add_constant(
            X_train,
            has_constant="add",
        )

        bp = het_breuschpagan(
            residuals,
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
        }

    except Exception:
        pass

    return tests


# ============================================================
# CONCLUSION
# ============================================================

def build_conclusion(
    item,
    current_price,
):
    conclusions = []

    ranking = item.get(
        "ranking"
    )

    # ========================================================
    # TOP FACTOR
    # ========================================================

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

        group = str(
            top.get(
                "Nhóm",
                "Không xác định",
            )
        )

        direction = str(
            top.get(
                "Quan hệ",
                "Không xác định",
            )
        )

        score = num(
            top.get(
                "Score"
            )
        )

        pvalue = num(
            top.get(
                "p-value"
            )
        )

        text = (
            f"**Yếu tố nổi bật nhất:** "
            f"{factor} ({group})"
        )

        if score is not None:
            text += (
                f", Score {score:.1f}/100"
            )

        text += (
            f", quan hệ {direction.lower()}"
        )

        if pvalue is not None:
            text += (
                f", p-value {pvalue:.5f}"
            )

        text += "."

        conclusions.append(
            text
        )

        if (
            pvalue is not None
            and pvalue < 0.05
        ):

            conclusions.append(
                f"**{factor}** có bằng chứng "
                "thống kê ở mức 5% trong mô hình OLS."
            )

        elif (
            pvalue is not None
            and pvalue < 0.10
        ):

            conclusions.append(
                f"**{factor}** có tín hiệu thống kê "
                "ở mức 10%, nhưng chưa đạt ngưỡng 5%."
            )

        else:

            conclusions.append(
                f"**{factor}** đứng đầu về tín hiệu tổng hợp "
                "nhưng chưa có bằng chứng thống kê đủ mạnh "
                "để xem là quan hệ ổn định."
            )

    # ========================================================
    # MODEL
    # ========================================================

    models = item.get(
        "models"
    )

    if (
        isinstance(
            models,
            pd.DataFrame,
        )
        and not models.empty
    ):

        best = models.iloc[
            0
        ]

        model_name = str(
            best["Mô hình"]
        )

        rmse = num(
            best["RMSE"]
        )

        r2 = num(
            best["R²"]
        )

        text = (
            f"Mô hình tốt nhất trên tập test là "
            f"**{model_name}**"
        )

        if rmse is not None:
            text += (
                f" với RMSE {rmse:.6f}"
            )

        if r2 is not None:
            text += (
                f", R² {r2:.4f}"
            )

        text += "."

        conclusions.append(
            text
        )

        if (
            r2 is not None
            and r2 < 0
        ):

            conclusions.append(
                "R² âm trên tập test: mô hình hiện tại "
                "chưa vượt benchmark trung bình. Không nên "
                "dùng dự báo điểm như một mục tiêu giá chắc chắn."
            )

        elif (
            r2 is not None
            and r2 < 0.10
        ):

            conclusions.append(
                "Khả năng giải thích biến động lợi suất tương lai "
                "còn thấp; tín hiệu nên được xem như tham khảo."
            )

        elif (
            r2 is not None
            and r2 < 0.25
        ):

            conclusions.append(
                "Mô hình có tín hiệu dự báo nhưng sức giải thích "
                "vẫn ở mức vừa phải."
            )

        elif r2 is not None:

            conclusions.append(
                "Mô hình có khả năng giải thích đáng kể "
                "trong tập kiểm tra của mẫu này."
            )

    # ========================================================
    # TESTS
    # ========================================================

    tests = item.get(
        "tests",
        {},
    )

    bp = tests.get(
        "Breusch-Pagan"
    )

    if isinstance(
        bp,
        dict,
    ):

        p = num(
            bp.get(
                "p_value"
            )
        )

        if (
            p is not None
            and p < 0.05
        ):

            conclusions.append(
                "Breusch-Pagan: có bằng chứng "
                "phương sai thay đổi ở mức 5%."
            )

    dw = tests.get(
        "Durbin-Watson"
    )

    if isinstance(
        dw,
        dict,
    ):

        value = num(
            dw.get(
                "statistic"
            )
        )

        if value is not None:

            if value < 1.5:

                conclusions.append(
                    f"Durbin-Watson = {value:.2f}: "
                    "có dấu hiệu tự tương quan dương."
                )

            elif value > 2.5:

                conclusions.append(
                    f"Durbin-Watson = {value:.2f}: "
                    "có dấu hiệu tự tương quan âm."
                )

            else:

                conclusions.append(
                    f"Durbin-Watson = {value:.2f}: "
                    "không cho thấy tự tương quan mạnh theo "
                    "quy tắc thực hành."
                )

    # ========================================================
    # CURRENT PRICE
    # ========================================================

    forecast = item.get(
        "forecast"
    )

    if (
        current_price is not None
        and isinstance(
            forecast,
            dict,
        )
    ):

        horizon_parts = []

        for horizon in [
            "1D",
            "5D",
            "20D",
        ]:

            value = forecast.get(
                horizon
            )

            if not isinstance(
                value,
                dict,
            ):
                continue

            predicted_return = num(
                value.get(
                    "predicted_return"
                )
            )

            predicted_price = num(
                value.get(
                    "predicted_price"
                )
            )

            if (
                predicted_return is not None
                and predicted_price is not None
            ):

                horizon_parts.append(
                    f"{horizon}: "
                    f"{predicted_return:+.2f}% "
                    f"→ {predicted_price:,.0f}đ"
                )

        if horizon_parts:

            conclusions.append(
                "**Dự báo mô hình:** "
                + " · ".join(
                    horizon_parts
                )
            )

    return conclusions


# ============================================================
# EXECUTE ONE HORIZON
# ============================================================

def execute_horizon(
    df,
    features,
    horizon,
):
    target_column = (
        f"Target_{horizon}"
    )

    xy = prepare_xy(
        df,
        features,
        target_column,
    )

    if xy is None:
        return None

    X, y = xy

    if len(X) < MIN_OBSERVATIONS:
        return None

    split = int(
        len(X)
        * TRAIN_RATIO
    )

    if (
        split < 30
        or len(X) - split < 10
    ):
        return None

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

    # ========================================================
    # CORRELATION
    # ========================================================

    correlations = correlation_table(
        X,
        y,
    )

    # ========================================================
    # OLS
    # ========================================================

    ols = standardized_ols(
        X_train,
        y_train,
    )

    ols_table = (
        ols.get(
            "table"
        )
        if isinstance(
            ols,
            dict,
        )
        else pd.DataFrame()
    )

    # ========================================================
    # MODELS
    # ========================================================

    models, fitted, predictions = (
        run_model_suite(
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model = None
    best_model_name = None

    if (
        isinstance(
            models,
            pd.DataFrame,
        )
        and not models.empty
    ):

        best_model_name = str(
            models.iloc[
                0
            ][
                "Mô hình"
            ]
        )

        best_model = fitted.get(
            best_model_name
        )

    # ========================================================
    # PERMUTATION
    # ========================================================

    permutation = pd.DataFrame()

    if best_model is not None:

        permutation = permutation_table(
            best_model,
            X_test,
            y_test,
        )

    # ========================================================
    # TREE
    # ========================================================

    tree_importance = pd.DataFrame()

    if best_model is not None:

        tree_importance = tree_importance_table(
            best_model,
            X_train.columns,
        )

    # ========================================================
    # FACTOR RANKING
    # ========================================================

    ranking = build_factor_ranking(
        correlations,
        ols_table,
        permutation,
        tree_importance,
    )

    # ========================================================
    # GROUP SUMMARY
    # ========================================================

    group_summary = build_group_summary(
        ranking
    )

    # ========================================================
    # VIF
    # ========================================================

    vif = calculate_vif(
        X_train[
            [
                column
                for column
                in ranking[
                    "Biến"
                ].tolist()
                if column in X_train.columns
            ]
        ]
        if (
            isinstance(
                ranking,
                pd.DataFrame,
            )
            and not ranking.empty
        )
        else X_train
    )

    # ========================================================
    # TESTS
    # ========================================================

    tests = run_tests(
        X_train,
        y_train,
        ols,
    )

    # ========================================================
    # FORECAST
    # ========================================================

    forecast_return = None
    forecast_price = None
    current_price = None

    last_close = num(
        df[
            "Close"
        ].iloc[
            -1
        ]
    )

    if last_close is not None:

        current_price = last_close

    if best_model is not None:

        try:

            last_features = (
                df[
                    list(
                        X_train.columns
                    )
                ]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .copy()
            )

            for column in last_features.columns:

                median = last_features[
                    column
                ].median()

                if pd.isna(
                    median
                ):
                    median = 0.0

                last_features[
                    column
                ] = last_features[
                    column
                ].fillna(
                    median
                )

            latest = last_features.iloc[
                [
                    -1
                ]
            ]

            forecast_return = float(
                best_model.predict(
                    latest
                )[0]
            )

            if current_price is not None:

                forecast_price = (
                    current_price
                    * (
                        1
                        + forecast_return
                    )
                )

        except Exception:
            forecast_return = None
            forecast_price = None

    # ========================================================
    # MODEL QUALITY
    # ========================================================

    model_quality = "Chưa đánh giá"

    if (
        isinstance(
            models,
            pd.DataFrame,
        )
        and not models.empty
    ):

        best_r2 = num(
            models.iloc[
                0
            ].get(
                "R²"
            )
        )

        best_rmse = num(
            models.iloc[
                0
            ].get(
                "RMSE"
            )
        )

        if best_r2 is not None:

            if best_r2 < 0:
                model_quality = "Yếu"
            elif best_r2 < 0.10:
                model_quality = "Thấp"
            elif best_r2 < 0.25:
                model_quality = "Trung bình"
            else:
                model_quality = "Khá"

        elif best_rmse is not None:

            model_quality = "Có kết quả"

    return {
        "horizon": horizon,
        "observations": len(X),
        "train": len(X_train),
        "test": len(X_test),
        "features_used": list(
            X_train.columns
        ),
        "correlations": correlations,
        "ols": ols,
        "ols_table": ols_table,
        "models": models,
        "fitted": fitted,
        "predictions": predictions,
        "best_model": best_model_name,
        "permutation": permutation,
        "tree_importance": tree_importance,
        "ranking": ranking,
        "group_summary": group_summary,
        "vif": vif,
        "tests": tests,
        "forecast": {
            "current_price": current_price,
            "predicted_return": forecast_return,
            "predicted_price": forecast_price,
        },
        "model_quality": model_quality,
    }


# ============================================================
# EXECUTE FULL RESEARCH
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL_RESEARCH,
    show_spinner=False,
)
def execute_full_research(
    sample,
):
    prepared = prepare_research_data(
        sample
    )

    if prepared is None:
        return {
            "ok": False,
            "error": "Không thể chuẩn bị dữ liệu nghiên cứu.",
        }

    df = prepared[
        "data"
    ]

    original_features = prepared[
        "features"
    ]

    # ========================================================
    # KHÔNG GIỚI HẠN VỀ DATASET
    #
    # Nhưng model quá nhiều biến khi sample ngắn sẽ
    # gây bất ổn. Ta giữ toàn bộ dataset để báo cáo,
    # rồi giới hạn model matrix nếu cần.
    # ========================================================

    features = list(
        original_features
    )

    # Ưu tiên biến có độ biến thiên.
    feature_variances = []

    for column in features:

        try:

            value = float(
                df[
                    column
                ].var(
                    ddof=0
                )
            )

            if np.isfinite(
                value
            ):
                feature_variances.append(
                    (
                        column,
                        value,
                    )
                )

        except Exception:
            pass

    feature_variances.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    model_features = [
        column
        for column, _ in feature_variances
    ]

    # Giữ đủ nhóm.
    groups = group_features(
        model_features
    )

    selected_model_features = []

    for group_name in [
        "Kỹ thuật",
        "Dòng tiền",
        "Khối ngoại",
        "Tự doanh",
        "Thị trường chung",
        "Nhóm ngành",
    ]:

        selected_model_features.extend(
            groups.get(
                group_name,
                [],
            )
        )

    # Không để dataset quá cao chiều so với sample.
    # Nhưng không bỏ khỏi "toàn bộ biến".
    if len(
        selected_model_features
    ) > MAX_FEATURES_MODEL:

        # Lấy theo variance trong từng nhóm.
        final_features = []

        for group_name in [
            "Kỹ thuật",
            "Dòng tiền",
            "Khối ngoại",
            "Tự doanh",
            "Thị trường chung",
            "Nhóm ngành",
        ]:

            group_list = groups.get(
                group_name,
                [],
            )

            final_features.extend(
                group_list[
                    :max(
                        1,
                        MAX_FEATURES_MODEL
                        // 6,
                    )
                ]
            )

        # Fill remaining slots by variance.
        remaining = [
            column
            for column in model_features
            if column not in final_features
        ]

        final_features.extend(
            remaining[
                :(
                    MAX_FEATURES_MODEL
                    - len(final_features)
                )
            ]
        )

        model_features = final_features

    result = {
        "ok": True,
        "data": df,
        "all_features": original_features,
        "model_features": model_features,
        "feature_groups": group_features(
            original_features
        ),
        "horizons": {},
    }

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = execute_horizon(
            df,
            model_features,
            horizon,
        )

        if item is not None:

            result[
                "horizons"
            ][
                horizon
            ] = item

    if not result[
        "horizons"
    ]:

        result[
            "ok"
        ] = False

        result[
            "error"
        ] = (
            "Không có horizon nào đủ dữ liệu "
            "để chạy nghiên cứu."
        )

    return result


# ============================================================
# RENDER SAMPLE CONTROLS
# ============================================================

def render_research_controls(
    symbol,
):
    st.subheader(
        "🧪 Nghiên cứu định lượng"
    )

    st.write(
        "Chọn khoảng dữ liệu. Hệ thống sẽ nghiên cứu "
        "lợi suất tương lai 1D, 5D và 20D trên cùng mẫu, "
        "đồng thời xét kỹ thuật, dòng tiền, khối ngoại, "
        "tự doanh, thị trường chung và nhóm ngành."
    )

    mode = st.radio(
        "Cách chọn mẫu",
        [
            "Preset",
            "Khoảng ngày tùy chọn",
        ],
        horizontal=True,
        key="research_sample_mode",
    )

    today = (
        pd.Timestamp.today()
        .date()
    )

    if mode == "Preset":

        preset_names = list(
            PERIOD_PRESETS.keys()
        )

        preset = st.selectbox(
            "Khoảng thời gian",
            preset_names,
            index=3,
            key="research_period_preset",
        )

        days = PERIOD_PRESETS[
            preset
        ]

        end_date = today

        start_date = (
            pd.Timestamp(
                end_date
            )
            - pd.Timedelta(
                days=days
            )
        ).date()

    else:

        default_start = (
            pd.Timestamp(
                today
            )
            - pd.Timedelta(
                days=365
            )
        ).date()

        c1, c2 = st.columns(2)

        with c1:

            start_date = st.date_input(
                "Từ ngày",
                value=default_start,
                max_value=today,
                key="research_start_date",
            )

        with c2:

            end_date = st.date_input(
                "Đến ngày",
                value=today,
                max_value=today,
                key="research_end_date",
            )

    if start_date > end_date:

        st.error(
            "Ngày bắt đầu phải nhỏ hơn ngày kết thúc."
        )

        return None

    st.caption(
        f"Mẫu yêu cầu: "
        f"{start_date.strftime('%d/%m/%Y')}"
        f" → "
        f"{end_date.strftime('%d/%m/%Y')}"
    )

    return (
        start_date,
        end_date,
        mode,
        (
            preset
            if mode == "Preset"
            else "Tùy chọn"
        ),
    )


# ============================================================
# RENDER FORECAST
# ============================================================

def render_forecast_block(
    item,
):
    forecast = item.get(
        "forecast",
        {},
    )

    current = num(
        forecast.get(
            "current_price"
        )
    )

    predicted = num(
        forecast.get(
            "predicted_return"
        )
    )

    predicted_price = num(
        forecast.get(
            "predicted_price"
        )
    )

    st.subheader(
        f"🎯 Dự báo {item['horizon']}"
    )

    if (
        current is None
        or predicted is None
        or predicted_price is None
    ):

        st.info(
            "Chưa tạo được dự báo giá từ model."
        )

        return

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Giá hiện tại",
            format_price(
                current
            ),
        )

    with c2:

        st.metric(
            "Lợi suất dự báo",
            format_percent(
                predicted
                * 100
            ),
        )

    with c3:

        st.metric(
            "Giá dự báo",
            format_price(
                predicted_price
            ),
        )

    with c4:

        quality = item.get(
            "model_quality",
            "Chưa đánh giá",
        )

        st.metric(
            "Chất lượng model",
            quality,
        )

    if predicted > 0:

        st.success(
            f"Mô hình đang nghiêng về **tăng** "
            f"{predicted * 100:+.2f}% "
            f"cho horizon {item['horizon']}."
        )

    elif predicted < 0:

        st.warning(
            f"Mô hình đang nghiêng về **giảm** "
            f"{predicted * 100:+.2f}% "
            f"cho horizon {item['horizon']}."
        )

    else:

        st.info(
            f"Mô hình đang nghiêng về **đi ngang** "
            f"cho horizon {item['horizon']}."
        )


# ============================================================
# RENDER FACTOR GROUP
# ============================================================

def render_group_summary(
    item,
):
    group_summary = item.get(
        "group_summary"
    )

    if (
        not isinstance(
            group_summary,
            pd.DataFrame,
        )
        or group_summary.empty
    ):
        return

    st.subheader(
        "🧩 Nhóm yếu tố nào đang nổi bật?"
    )

    show = group_summary.rename(
        columns={
            "Nhóm": "Nhóm yếu tố",
            "Score_TB": "Score TB",
            "Score_Max": "Score cao nhất",
            "So_bien": "Số biến",
        }
    ).copy()

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
    )

    strongest = group_summary.iloc[
        0
    ]

    st.info(
        f"Nhóm nổi bật nhất trong horizon "
        f"**{item['horizon']}** hiện là "
        f"**{strongest['Nhóm']}**, "
        f"Score TB {strongest['Score_TB']:.1f}/100."
    )


# ============================================================
# RENDER FACTOR RANKING
# ============================================================

def render_factor_ranking(
    item,
):
    ranking = item.get(
        "ranking"
    )

    if (
        not isinstance(
            ranking,
            pd.DataFrame,
        )
        or ranking.empty
    ):
        st.info(
            "Chưa có bảng xếp hạng yếu tố."
        )
        return

    st.subheader(
        "🏆 Yếu tố ảnh hưởng / dự báo nổi bật"
    )

    top = ranking.head(
        MAX_RANKING_ROWS
    ).copy()

    columns = [
        "Biến",
        "Nhóm",
        "Score",
        "Quan hệ",
        "Pearson",
        "Spearman",
        "Beta",
        "p-value",
        "Permutation",
        "TreeImportance",
        "Ý nghĩa",
    ]

    columns = [
        column
        for column in columns
        if column in top.columns
    ]

    for column in [
        "Score",
        "Pearson",
        "Spearman",
        "Beta",
        "p-value",
        "Permutation",
        "TreeImportance",
    ]:

        if column in top.columns:

            top[
                column
            ] = pd.to_numeric(
                top[
                    column
                ],
                errors="coerce",
            )

    if "Score" in top.columns:
        top[
            "Score"
        ] = top[
            "Score"
        ].round(
            1
        )

    if "Pearson" in top.columns:
        top[
            "Pearson"
        ] = top[
            "Pearson"
        ].round(
            4
        )

    if "Spearman" in top.columns:
        top[
            "Spearman"
        ] = top[
            "Spearman"
        ].round(
            4
        )

    if "Beta" in top.columns:
        top[
            "Beta"
        ] = top[
            "Beta"
        ].round(
            4
        )

    if "p-value" in top.columns:
        top[
            "p-value"
        ] = top[
            "p-value"
        ].round(
            5
        )

    if "Permutation" in top.columns:
        top[
            "Permutation"
        ] = top[
            "Permutation"
        ].round(
            6
        )

    if "TreeImportance" in top.columns:
        top[
            "TreeImportance"
        ] = top[
            "TreeImportance"
        ].round(
            6
        )

    st.dataframe(
        top[
            columns
        ],
        width="stretch",
        hide_index=True,
    )

    # ========================================================
    # TOP THREE EXPLANATIONS
    # ========================================================

    st.markdown(
        "#### 🧠 Đọc kết quả"
    )

    for _, row in ranking.head(
        5
    ).iterrows():

        feature = str(
            row.get(
                "Biến"
            )
        )

        group = str(
            row.get(
                "Nhóm"
            )
        )

        direction = str(
            row.get(
                "Quan hệ",
                "Không xác định",
            )
        )

        spear = num(
            row.get(
                "Spearman"
            )
        )

        beta = num(
            row.get(
                "Beta"
            )
        )

        pvalue = num(
            row.get(
                "p-value"
            )
        )

        score = num(
            row.get(
                "Score"
            )
        )

        text = (
            f"**{feature}** · {group}"
        )

        if score is not None:
            text += (
                f" · Score {score:.1f}/100"
            )

        text += (
            f" · {direction.lower()}"
        )

        if spear is not None:
            text += (
                f" · Spearman {spear:+.4f}"
            )

        if beta is not None:
            text += (
                f" · Beta {beta:+.4f}"
            )

        if pvalue is not None:
            text += (
                f" · p-value {pvalue:.5f}"
            )

        st.markdown(
            "• " + text
        )


# ============================================================
# RENDER MODEL
# ============================================================

def render_model_results(
    item,
):
    st.subheader(
        "🤖 So sánh mô hình"
    )

    models = item.get(
        "models"
    )

    if (
        not isinstance(
            models,
            pd.DataFrame,
        )
        or models.empty
    ):

        st.warning(
            "Không có kết quả model."
        )

        return

    display = models.copy()

    for column in [
        "MAE",
        "RMSE",
        "R²",
    ]:

        if column in display.columns:

            display[
                column
            ] = pd.to_numeric(
                display[
                    column
                ],
                errors="coerce",
            ).round(
                6
            )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )

    best = models.iloc[
        0
    ]

    best_name = str(
        best[
            "Mô hình"
        ]
    )

    r2 = num(
        best.get(
            "R²"
        )
    )

    rmse = num(
        best.get(
            "RMSE"
        )
    )

    text = (
        f"**Model tốt nhất:** {best_name}"
    )

    if rmse is not None:
        text += (
            f" · RMSE {rmse:.6f}"
        )

    if r2 is not None:
        text += (
            f" · R² {r2:.4f}"
        )

    if r2 is not None:

        if r2 < 0:

            st.error(
                text
                + " — model chưa vượt benchmark."
            )

        elif r2 < 0.10:

            st.warning(
                text
                + " — khả năng giải thích thấp."
            )

        elif r2 < 0.25:

            st.info(
                text
                + " — tín hiệu có nhưng chưa mạnh."
            )

        else:

            st.success(
                text
                + " — khả năng giải thích đáng chú ý."
            )

    else:

        st.info(
            text
        )


# ============================================================
# RENDER OLS
# ============================================================

def render_ols(
    item,
):
    st.subheader(
        "📐 Hồi quy OLS chuẩn hóa"
    )

    ols = item.get(
        "ols"
    )

    table = item.get(
        "ols_table"
    )

    if (
        not isinstance(
            ols,
            dict,
        )
        or ols.get(
            "model"
        ) is None
    ):

        st.info(
            "OLS không chạy được trên mẫu này."
        )

        return

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "R²",
            format_number(
                ols.get(
                    "r2"
                ),
                4,
            ),
        )

    with c2:

        st.metric(
            "Adjusted R²",
            format_number(
                ols.get(
                    "adj_r2"
                ),
                4,
            ),
        )

    if (
        isinstance(
            table,
            pd.DataFrame,
        )
        and not table.empty
    ):

        show = table.head(
            MAX_RANKING_ROWS
        ).copy()

        for column in [
            "Beta",
            "p-value",
            "Std Error",
        ]:

            if column in show.columns:

                show[
                    column
                ] = pd.to_numeric(
                    show[
                        column
                    ],
                    errors="coerce",
                ).round(
                    6
                )

        st.dataframe(
            show[
                [
                    "Biến",
                    "Nhóm",
                    "Beta",
                    "p-value",
                    "Std Error",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Beta chuẩn hóa dùng để so sánh độ lớn tương đối "
        "giữa các biến; không đọc Beta như phần trăm tăng giá."
    )


# ============================================================
# RENDER TESTS
# ============================================================

def render_tests(
    item,
):
    st.subheader(
        "🧪 Kiểm định mô hình"
    )

    tests = item.get(
        "tests",
        {},
    )

    if not tests:

        st.info(
            "Chưa có kiểm định."
        )

        return

    # ADF
    adf = tests.get(
        "ADF"
    )

    if isinstance(
        adf,
        dict,
    ):

        with st.expander(
            "ADF — tính dừng của target",
            expanded=False,
        ):

            p = num(
                adf.get(
                    "p_value"
                )
            )

            st.write(
                f"Statistic: "
                f"{format_number(adf.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: "
                f"{format_number(p, 6)}"
            )

            if (
                p is not None
                and p < 0.05
            ):

                st.success(
                    "Bác bỏ nghiệm đơn vị ở mức 5%."
                )

            else:

                st.info(
                    "Chưa đủ bằng chứng bác bỏ nghiệm đơn vị."
                )

    # Jarque-Bera
    jb = tests.get(
        "Jarque-Bera"
    )

    if isinstance(
        jb,
        dict,
    ):

        with st.expander(
            "Jarque-Bera — phân phối phần dư",
            expanded=False,
        ):

            p = num(
                jb.get(
                    "p_value"
                )
            )

            st.write(
                f"Statistic: "
                f"{format_number(jb.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: "
                f"{format_number(p, 6)}"
            )

            if (
                p is not None
                and p < 0.05
            ):

                st.warning(
                    "Phần dư lệch khỏi giả định chuẩn."
                )

            else:

                st.info(
                    "Chưa có bằng chứng mạnh chống lại giả định chuẩn."
                )

    # BP
    bp = tests.get(
        "Breusch-Pagan"
    )

    if isinstance(
        bp,
        dict,
    ):

        with st.expander(
            "Breusch-Pagan — phương sai thay đổi",
            expanded=False,
        ):

            p = num(
                bp.get(
                    "p_value"
                )
            )

            st.write(
                f"Statistic: "
                f"{format_number(bp.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: "
                f"{format_number(p, 6)}"
            )

            if (
                p is not None
                and p < 0.05
            ):

                st.warning(
                    "Có bằng chứng phương sai thay đổi."
                )

            else:

                st.success(
                    "Chưa có bằng chứng rõ về phương sai thay đổi."
                )

    # DW
    dw = tests.get(
        "Durbin-Watson"
    )

    if isinstance(
        dw,
        dict,
    ):

        with st.expander(
            "Durbin-Watson — tự tương quan",
            expanded=False,
        ):

            value = num(
                dw.get(
                    "statistic"
                )
            )

            st.metric(
                "Statistic",
                format_number(
                    value,
                    4,
                ),
            )

            if value is not None:

                if value < 1.5:

                    st.warning(
                        "Có dấu hiệu tự tương quan dương."
                    )

                elif value > 2.5:

                    st.warning(
                        "Có dấu hiệu tự tương quan âm."
                    )

                else:

                    st.success(
                        "Chưa thấy tự tương quan mạnh."
                    )

    # Ljung Box
    lb = tests.get(
        "Ljung-Box"
    )

    if isinstance(
        lb,
        dict,
    ):

        with st.expander(
            "Ljung-Box — tự tương quan phần dư",
            expanded=False,
        ):

            p = num(
                lb.get(
                    "p_value"
                )
            )

            st.write(
                f"Lag: {lb.get('lag', '—')}"
            )

            st.write(
                f"Statistic: "
                f"{format_number(lb.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: "
                f"{format_number(p, 6)}"
            )

            if (
                p is not None
                and p < 0.05
            ):

                st.warning(
                    "Còn bằng chứng tự tương quan trong phần dư."
                )

            else:

                st.success(
                    "Chưa thấy tự tương quan phần dư rõ."
                )


# ============================================================
# RENDER VIF
# ============================================================

def render_vif(
    item,
):
    st.subheader(
        "🔗 Đa cộng tuyến — VIF"
    )

    vif = item.get(
        "vif"
    )

    if (
        not isinstance(
            vif,
            pd.DataFrame,
        )
        or vif.empty
    ):

        st.info(
            "Chưa có VIF."
        )

        return

    show = vif.head(
        30
    ).copy()

    show[
        "VIF"
    ] = pd.to_numeric(
        show[
            "VIF"
        ],
        errors="coerce",
    ).round(
        3
    )

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
    )

    clean_vif = vif[
        "VIF"
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    max_vif = num(
        clean_vif.max()
    )

    if (
        max_vif is not None
        and max_vif >= 10
    ):

        st.warning(
            "Có đa cộng tuyến mạnh. "
            "Không nên diễn giải hệ số OLS riêng lẻ như quan hệ nhân quả."
        )

    elif (
        max_vif is not None
        and max_vif >= 5
    ):

        st.info(
            "Có dấu hiệu đa cộng tuyến đáng chú ý."
        )

    else:

        st.success(
            "Không thấy đa cộng tuyến quá mạnh theo VIF."
        )


# ============================================================
# RENDER CROSS HORIZON
# ============================================================

def render_cross_horizon(
    result,
):
    st.divider()

    st.header(
        "🏆 Yếu tố nhất quán giữa các horizon"
    )

    rows = []

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = result[
            "horizons"
        ].get(
            horizon
        )

        if item is None:
            continue

        ranking = item.get(
            "ranking"
        )

        if (
            not isinstance(
                ranking,
                pd.DataFrame,
            )
            or ranking.empty
        ):
            continue

        for rank, (_, row) in enumerate(
            ranking.head(
                20
            ).iterrows(),
            start=1,
        ):

            rows.append(
                {
                    "Horizon": horizon,
                    "Biến": row[
                        "Biến"
                    ],
                    "Nhóm": row.get(
                        "Nhóm"
                    ),
                    "Rank": rank,
                    "Score": num(
                        row.get(
                            "Score"
                        )
                    ),
                    "Quan hệ": row.get(
                        "Quan hệ"
                    ),
                }
            )

    if not rows:

        st.info(
            "Chưa có dữ liệu để so sánh horizon."
        )

        return

    df = pd.DataFrame(
        rows
    )

    stable = (
        df
        .groupby(
            [
                "Biến",
                "Nhóm",
            ],
            as_index=False,
        )
        .agg(
            So_horizon=(
                "Horizon",
                "nunique",
            ),
            Rank_TB=(
                "Rank",
                "mean",
            ),
            Score_TB=(
                "Score",
                "mean",
            ),
        )
    )

    stable[
        "Rank_TB"
    ] = stable[
        "Rank_TB"
    ].round(
        2
    )

    stable[
        "Score_TB"
    ] = stable[
        "Score_TB"
    ].round(
        1
    )

    stable = (
        stable
        .sort_values(
            [
                "So_horizon",
                "Score_TB",
                "Rank_TB",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    stable.columns = [
        "Biến",
        "Nhóm",
        "Số horizon",
        "Rank TB",
        "Score TB",
    ]

    st.dataframe(
        stable.head(
            30
        ),
        width="stretch",
        hide_index=True,
    )

    if not stable.empty:

        best = stable.iloc[
            0
        ]

        st.success(
            f"Yếu tố nhất quán nhất: "
            f"**{best['Biến']}** "
            f"({int(best['Số horizon'])} horizon, "
            f"Score TB {best['Score TB']:.1f}/100)."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def render_final_summary(
    result,
    symbol,
):
    st.divider()

    st.header(
        "🎯 Kết luận cho nhà đầu tư"
    )

    rows = []

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = result[
            "horizons"
        ].get(
            horizon
        )

        if item is None:
            continue

        ranking = item.get(
            "ranking"
        )

        forecast = item.get(
            "forecast",
            {},
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

            rows.append(
                {
                    "Horizon": horizon,
                    "Yếu tố nổi bật": top.get(
                        "Biến"
                    ),
                    "Nhóm": top.get(
                        "Nhóm"
                    ),
                    "Quan hệ": top.get(
                        "Quan hệ"
                    ),
                    "Score": num(
                        top.get(
                            "Score"
                        )
                    ),
                    "Dự báo %": (
                        num(
                            forecast.get(
                                "predicted_return"
                            )
                        )
                        * 100
                        if num(
                            forecast.get(
                                "predicted_return"
                            )
                        ) is not None
                        else np.nan
                    ),
                    "Giá dự báo": num(
                        forecast.get(
                            "predicted_price"
                        )
                    ),
                }
            )

    if not rows:

        st.info(
            "Chưa đủ kết quả để tạo tổng kết."
        )

        return

    summary = pd.DataFrame(
        rows
    )

    summary[
        "Score"
    ] = summary[
        "Score"
    ].round(
        1
    )

    summary[
        "Dự báo %"
    ] = summary[
        "Dự báo %"
    ].round(
        2
    )

    summary[
        "Giá dự báo"
    ] = summary[
        "Giá dự báo"
    ].round(
        0
    )

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # MOST IMPORTANT
    # --------------------------------------------------------

    group_scores = []

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = result[
            "horizons"
        ].get(
            horizon
        )

        if item is None:
            continue

        groups = item.get(
            "group_summary"
        )

        if (
            isinstance(
                groups,
                pd.DataFrame,
            )
            and not groups.empty
        ):

            for _, row in groups.iterrows():

                group_scores.append(
                    {
                        "Horizon": horizon,
                        "Nhóm": row[
                            "Nhóm"
                        ],
                        "Score": num(
                            row[
                                "Score_TB"
                            ]
                        ),
                    }
                )

    if group_scores:

        group_df = pd.DataFrame(
            group_scores
        )

        group_avg = (
            group_df
            .groupby(
                "Nhóm",
                as_index=False,
            )[
                "Score"
            ]
            .mean()
            .sort_values(
                "Score",
                ascending=False,
            )
        )

        if not group_avg.empty:

            strongest_group = (
                group_avg.iloc[
                    0
                ]
            )

            st.success(
                f"Trên toàn bộ nghiên cứu, nhóm "
                f"**{strongest_group['Nhóm']}** "
                f"có tín hiệu tổng hợp nổi bật nhất "
                f"với Score TB "
                f"{strongest_group['Score']:.1f}/100."
            )

    st.caption(
        "Dự báo giá là kết quả của model trên mẫu đã chọn, "
        "không phải giá mục tiêu chắc chắn. 'Yếu tố nổi bật' "
        "là yếu tố có tín hiệu thống kê/dự báo mạnh trong mẫu, "
        "không phải bằng chứng nhân quả."
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
        "Phân tích kỹ thuật, dòng tiền, khối ngoại, tự doanh, "
        "thị trường chung, nhóm ngành và dự báo lợi suất."
    )

    # ========================================================
    # SYMBOL
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

        clean = normalize_symbol(
            symbol_input
        )

        if not clean:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

            return

        st.session_state[
            "stock_analysis_symbol"
        ] = clean

        # Xóa kết quả cũ.
        for key in [
            "stock_research_result",
            "stock_research_symbol",
            "stock_research_dates",
        ]:

            st.session_state.pop(
                key,
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
    # DISPLAY DATA
    # ========================================================

    try:

        data = load_market_data(
            symbol,
            "1y",
        )

    except Exception as error:

        st.error(
            f"Không thể tải dữ liệu "
            f"{display_symbol(symbol)}."
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
            "Không có dữ liệu."
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

    change = num(
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

    last = data.iloc[
        -1
    ]

    atr14 = num(
        last.get(
            "ATR14"
        )
    )

    volume_sma20 = num(
        last.get(
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
    # STOCK TITLE
    # ========================================================

    st.subheader(
        f"📈 {display_symbol(symbol)}"
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Giá",
            format_price(
                price
            ),
        )

    with b:

        st.metric(
            "Thay đổi 1D",
            format_percent(
                change
            ),
        )

    with c:

        st.metric(
            "RSI",
            format_number(
                rsi_value,
                1,
            ),
        )

    with d:

        st.metric(
            "Khối lượng",
            format_volume(
                volume
            ),
        )

    # ========================================================
    # SECONDARY
    # ========================================================

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "MA20",
            format_price(
                sma20
            ),
        )

    with b:

        st.metric(
            "MA50",
            format_price(
                sma50
            ),
        )

    with c:

        st.metric(
            "MACD",
            format_number(
                macd,
                3,
            ),
        )

    with d:

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

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Xu hướng",
            ma_status(
                price,
                sma20,
                sma50,
            ),
        )

    with b:

        st.metric(
            "RSI",
            rsi_status(
                rsi_value
            ),
        )

    with c:

        st.metric(
            "MACD",
            macd_status(
                macd
            ),
        )

    with d:

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

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "ATR 14",
            format_price(
                atr14
            ),
        )

    with b:

        st.metric(
            "Khối lượng TB20",
            format_volume(
                volume_sma20
            ),
        )

    with c:

        st.metric(
            "Giá mở cửa",
            format_price(
                num(
                    last.get(
                        "Open"
                    )
                )
            ),
        )

    with d:

        high = num(
            last.get(
                "High"
            )
        )

        low = num(
            last.get(
                "Low"
            )
        )

        close = num(
            last.get(
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
    # RESEARCH CONTROL
    # ========================================================

    controls = render_research_controls(
        symbol
    )

    if controls is None:
        return

    start_date, end_date, mode, period_label = controls

    # ========================================================
    # LOAD MULTIFACTOR RESEARCH DATA
    # ========================================================

    st.subheader(
        "📚 Dữ liệu nghiên cứu"
    )

    load_placeholder = st.empty()

    with load_placeholder.container():

        st.info(
            "Đang tải dữ liệu multifactor..."
        )

    try:

        research_data = (
            load_multifactor_research_history(
                symbol,
                start_date,
                end_date,
            )
        )

    except Exception as error:

        load_placeholder.empty()

        st.error(
            "Không tải được dữ liệu multifactor."
        )

        st.code(
            str(error)
        )

        return

    load_placeholder.empty()

    sample = slice_date_range(
        research_data,
        start_date,
        end_date,
    )

    if (
        sample is None
        or sample.empty
    ):

        st.error(
            "Không có phiên giao dịch trong khoảng đã chọn."
        )

        return

    actual_start, actual_end = (
        get_actual_date_range(
            sample
        )
    )

    # ========================================================
    # SAMPLE INFO
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Khoảng",
            period_label,
        )

    with c2:

        st.metric(
            "Quan sát",
            f"{len(sample):,}",
        )

    with c3:

        st.metric(
            "Ngày đầu",
            (
                actual_start.strftime(
                    "%d/%m/%Y"
                )
                if actual_start
                else "—"
            ),
        )

    with c4:

        st.metric(
            "Ngày cuối",
            (
                actual_end.strftime(
                    "%d/%m/%Y"
                )
                if actual_end
                else "—"
            ),
        )

    if len(sample) < 60:

        st.warning(
            "Mẫu dưới 60 quan sát. Có thể chạy nhưng "
            "độ ổn định của các kết luận định lượng thấp."
        )

    elif len(sample) < 120:

        st.info(
            "Mẫu đủ để chạy phân tích cơ bản; mẫu dài hơn "
            "thường cho kết quả ổn định hơn."
        )

    else:

        st.success(
            f"Mẫu có {len(sample):,} quan sát thực tế."
        )

    # ========================================================
    # DATASET GROUPS
    # ========================================================

    prepared = prepare_research_data(
        sample
    )

    if prepared is None:

        st.error(
            "Không thể chuẩn bị dữ liệu nghiên cứu."
        )

        return

    all_features = prepared[
        "features"
    ]

    groups = group_features(
        all_features
    )

    st.subheader(
        "🧬 Phạm vi biến nghiên cứu"
    )

    group_columns = st.columns(3)

    group_items = [
        (
            "Kỹ thuật",
            "technical",
        ),
        (
            "Dòng tiền",
            "flow",
        ),
        (
            "Khối ngoại",
            "foreign",
        ),
        (
            "Tự doanh",
            "proprietary",
        ),
        (
            "Thị trường chung",
            "market",
        ),
        (
            "Nhóm ngành",
            "sector",
        ),
    ]

    # 6 cards.
    for index, (
        label,
        key,
    ) in enumerate(
        group_items
    ):

        with group_columns[
            index % 3
        ]:

            count = len(
                groups.get(
                    label,
                    [],
                )
            )

            st.metric(
                label,
                f"{count} biến",
            )

    with st.expander(
        "Xem toàn bộ biến theo nhóm",
        expanded=False,
    ):

        for label in [
            "Kỹ thuật",
            "Dòng tiền",
            "Khối ngoại",
            "Tự doanh",
            "Thị trường chung",
            "Nhóm ngành",
        ]:

            st.markdown(
                f"**{label}** "
                f"({len(groups.get(label, []))} biến)"
            )

            values = groups.get(
                label,
                [],
            )

            if values:

                st.write(
                    ", ".join(
                        str(x)
                        for x in values
                    )
                )

            else:

                st.caption(
                    "Chưa có biến từ nguồn dữ liệu."
                )

    # ========================================================
    # RUN
    # ========================================================

    run = st.button(
        "🚀 Chạy toàn bộ nghiên cứu",
        type="primary",
        width="stretch",
        key="stock_quant_run",
    )

    if run:

        with st.spinner(
            "Đang chạy multifactor research..."
        ):

            result = execute_full_research(
                sample
            )

        st.session_state[
            "stock_research_result"
        ] = result

        st.session_state[
            "stock_research_symbol"
        ] = symbol

        st.session_state[
            "stock_research_dates"
        ] = (
            start_date,
            end_date,
        )

    # ========================================================
    # RESULT
    # ========================================================

    result = st.session_state.get(
        "stock_research_result"
    )

    saved_symbol = st.session_state.get(
        "stock_research_symbol"
    )

    if (
        result is None
        or saved_symbol != symbol
    ):

        st.caption(
            "Chọn mẫu rồi bấm "
            "“Chạy toàn bộ nghiên cứu”."
        )

        return

    if not result.get(
        "ok",
        False,
    ):

        st.error(
            result.get(
                "error",
                "Nghiên cứu thất bại.",
            )
        )

        return

    # ========================================================
    # OVERVIEW
    # ========================================================

    st.divider()

    st.header(
        "📊 Kết quả nghiên cứu"
    )

    model_feature_count = len(
        result.get(
            "model_features",
            [],
        )
    )

    st.caption(
        f"{display_symbol(symbol)} · "
        f"{len(all_features):,} biến trong dataset · "
        f"{model_feature_count:,} biến sử dụng trong model matrix"
    )

    # ========================================================
    # HORIZONS
    # ========================================================

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = result[
            "horizons"
        ].get(
            horizon
        )

        if item is None:
            continue

        st.divider()

        st.header(
            f"📈 Horizon {horizon} — "
            f"{HORIZON_LABELS[horizon]}"
        )

        # Sample.
        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Quan sát",
                f"{item['observations']:,}",
            )

        with c2:

            st.metric(
                "Train",
                f"{item['train']:,}",
            )

        with c3:

            st.metric(
                "Test",
                f"{item['test']:,}",
            )

        with c4:

            st.metric(
                "Biến model",
                f"{len(item['features_used']):,}",
            )

        # ====================================================
        # FORECAST
        # ====================================================

        render_forecast_block(
            item
        )

        # ====================================================
        # GROUP
        # ====================================================

        render_group_summary(
            item
        )

        # ====================================================
        # FACTOR
        # ====================================================

        render_factor_ranking(
            item
        )

        # ====================================================
        # MODEL
        # ====================================================

        render_model_results(
            item
        )

        # ====================================================
        # OLS
        # ====================================================

        render_ols(
            item
        )

        # ====================================================
        # VIF
        # ====================================================

        render_vif(
            item
        )

        # ====================================================
        # TESTS
        # ====================================================

        render_tests(
            item
        )

        # ====================================================
        # CONCLUSION
        # ====================================================

        st.subheader(
            "🧠 Kết luận horizon"
        )

        conclusions = build_conclusion(
            item,
            price,
        )

        if conclusions:

            for conclusion in conclusions:

                st.markdown(
                    "• "
                    + conclusion
                )

        else:

            st.info(
                "Chưa đủ thông tin để tạo kết luận."
            )

        st.warning(
            "Các kết quả mô hình là quan hệ thống kê / "
            "khả năng dự báo trong mẫu, không phải bằng chứng nhân quả."
        )

    # ========================================================
    # CROSS HORIZON
    # ========================================================

    render_cross_horizon(
        result
    )

    # ========================================================
    # FINAL
    # ========================================================

    render_final_summary(
        result,
        symbol,
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_analysis():
    render_stock_analysis()
