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
# CONFIG
# ============================================================

DISPLAY_CACHE_TTL = 300
RESEARCH_CACHE_TTL = 900
AI_CACHE_TTL = 1800

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

MIN_OBSERVATIONS = 60
TRAIN_RATIO = 0.80

# OLS không dùng toàn bộ matrix vì các biến như:
# Close, EMA, SMA, Bollinger... có thể phụ thuộc tuyến tính mạnh.
MAX_OLS_FEATURES = 15
OLS_CORR_LIMIT = 0.90

MAX_DISPLAY_FACTORS = 40

GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


# ============================================================
# BASIC
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


def safe_text(
    value: Any,
    default="",
):
    if value is None:
        return default

    try:
        text = str(value).strip()
    except Exception:
        return default

    return text if text else default


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


# ============================================================
# TECHNICAL STATUS
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

    if sma20 is not None and sma50 is not None:

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
# DATE
# ============================================================

def normalize_index(
    df,
):
    if df is None or df.empty:
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
    work = normalize_index(df)

    if work.empty:
        return work

    start_ts = pd.Timestamp(
        start_date
    ).normalize()

    end_ts = (
        pd.Timestamp(
            end_date
        )
        .normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )

    return work.loc[
        (work.index >= start_ts)
        & (work.index <= end_ts)
    ].copy()


def actual_range(
    df,
):
    work = normalize_index(df)

    if work.empty:
        return None, None

    return (
        work.index.min().date(),
        work.index.max().date(),
    )


# ============================================================
# FACTOR GROUP
# ============================================================

def feature_group(
    feature,
):
    name = str(feature)
    low = name.lower()

    if low.startswith("foreign_"):
        return "Khối ngoại"

    if low.startswith("proprietary_"):
        return "Tự doanh"

    if low.startswith("sector_"):
        return "Nhóm ngành"

    if low.startswith("market_"):
        return "Thị trường chung"

    if low.startswith("flow_"):
        return "Dòng tiền"

    if any(
        token in low
        for token in [
            "volume",
            "trading_value",
            "dollar_volume",
            "obv",
            "flow",
        ]
    ):
        return "Dòng tiền"

    return "Kỹ thuật"


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
        group = feature_group(feature)

        groups.setdefault(
            group,
            [],
        ).append(
            feature
        )

    return groups


# ============================================================
# TARGET
# ============================================================

def ensure_targets(
    df,
):
    work = normalize_index(df)

    if (
        work.empty
        or "Close" not in work.columns
    ):
        return work

    close = pd.to_numeric(
        work["Close"],
        errors="coerce",
    )

    for horizon, periods in HORIZONS.items():

        target = (
            close.shift(-periods)
            / close
            - 1
        )

        work[
            f"Target_{horizon}"
        ] = target

        work[
            f"Target_{horizon}_Pct"
        ] = target * 100

    return work


# ============================================================
# ALL NUMERIC FEATURES
# ============================================================

def get_all_features(
    df,
):
    if df is None or df.empty:
        return []

    excluded = {
        "Target_1D",
        "Target_5D",
        "Target_20D",
        "Target_1D_Pct",
        "Target_5D_Pct",
        "Target_20D_Pct",
    }

    features = []

    for column in df.columns:

        name = str(column)

        if name in excluded:
            continue

        if name.lower().startswith("target_"):
            continue

        if name in {
            "Time",
            "Date",
        }:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        features.append(column)

    return list(
        dict.fromkeys(features)
    )


# ============================================================
# PREPARE XY
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
        feature
        for feature in features
        if feature in df.columns
    ]

    if not available:
        return None

    X = df[
        available
    ].copy()

    y = pd.to_numeric(
        df[target_column],
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

    # --------------------------------------------------------
    # Median fill
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        median = X[column].median()

        if pd.isna(median):
            median = 0.0

        X[column] = X[
            column
        ].fillna(
            median
        )

    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    if X.empty:
        return None

    # --------------------------------------------------------
    # Drop only zero variance from model matrix.
    #
    # Dataset itself vẫn giữ nguyên tất cả biến.
    # --------------------------------------------------------

    usable = []

    for column in X.columns:

        try:
            variance = float(
                X[column].var(
                    ddof=0
                )
            )

            if (
                np.isfinite(variance)
                and variance > 0
            ):
                usable.append(column)

        except Exception:
            continue

    if not usable:
        return None

    return (
        X[usable].astype(float),
        y.astype(float),
    )


# ============================================================
# CORRELATION
# ============================================================

def correlation_table(
    X,
    y,
):
    if X is None or X.empty:
        return pd.DataFrame()

    rows = []

    for column in X.columns:

        x = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        yy = pd.to_numeric(
            y,
            errors="coerce",
        )

        valid = (
            x.notna()
            & yy.notna()
            & np.isfinite(x)
            & np.isfinite(yy)
        )

        x2 = x.loc[valid]
        y2 = yy.loc[valid]

        if len(x2) < 20:
            continue

        try:
            pearson = float(
                x2.corr(
                    y2,
                    method="pearson",
                )
            )
        except Exception:
            pearson = np.nan

        try:
            spearman = float(
                x2.corr(
                    y2,
                    method="spearman",
                )
            )
        except Exception:
            spearman = np.nan

        rows.append(
            {
                "Biến": column,
                "Nhóm": feature_group(
                    column
                ),
                "Pearson": pearson,
                "Spearman": spearman,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result[
        "AbsSpearman"
    ] = result[
        "Spearman"
    ].abs()

    return (
        result
        .sort_values(
            "AbsSpearman",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# OLS SAFE FEATURE SELECTION
# ============================================================

def select_ols_features(
    X,
    y,
):
    if X is None or X.empty:
        return []

    correlation = correlation_table(
        X,
        y,
    )

    ranked = (
        correlation["Biến"].tolist()
        if not correlation.empty
        else list(X.columns)
    )

    selected = []

    for feature in ranked:

        if len(selected) >= MAX_OLS_FEATURES:
            break

        if not selected:
            selected.append(feature)
            continue

        independent_enough = True

        for existing in selected:

            try:
                corr = abs(
                    float(
                        X[feature].corr(
                            X[existing]
                        )
                    )
                )

            except Exception:
                corr = 0.0

            if corr >= OLS_CORR_LIMIT:

                independent_enough = False
                break

        if independent_enough:
            selected.append(feature)

    if not selected:

        selected = list(
            X.columns[:MAX_OLS_FEATURES]
        )

    return selected


# ============================================================
# OLS
# ============================================================

def run_safe_ols(
    X_train,
    y_train,
):
    try:

        import statsmodels.api as sm

        from sklearn.preprocessing import (
            StandardScaler,
        )

        features = select_ols_features(
            X_train,
            y_train,
        )

        if not features:
            return {
                "model": None,
                "table": pd.DataFrame(),
                "features": [],
            }

        X = X_train[
            features
        ].copy()

        # ----------------------------------------------------
        # Remove exact constant columns
        # ----------------------------------------------------

        variable_features = []

        for feature in X.columns:

            try:

                if (
                    X[feature].nunique(
                        dropna=False
                    )
                    > 1
                ):
                    variable_features.append(
                        feature
                    )

            except Exception:
                pass

        X = X[
            variable_features
        ]

        features = list(
            X.columns
        )

        if not features:
            return {
                "model": None,
                "table": pd.DataFrame(),
                "features": [],
            }

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=features,
            index=X.index,
        )

        X_const = sm.add_constant(
            X_scaled,
            has_constant="add",
        )

        model = sm.OLS(
            y_train,
            X_const,
        ).fit(
            cov_type="HC3"
        )

        rows = []

        for feature in features:

            beta = num(
                model.params.get(
                    feature
                )
            )

            pvalue = num(
                model.pvalues.get(
                    feature
                )
            )

            stderr = num(
                model.bse.get(
                    feature
                )
            )

            rows.append(
                {
                    "Biến": feature,
                    "Nhóm": feature_group(
                        feature
                    ),
                    "Beta": beta,
                    "p-value": pvalue,
                    "Std Error": stderr,
                    "AbsBeta": (
                        abs(beta)
                        if beta is not None
                        else np.nan
                    ),
                }
            )

        table = (
            pd.DataFrame(rows)
            .sort_values(
                "AbsBeta",
                ascending=False,
                na_position="last",
            )
            .reset_index(drop=True)
        )

        return {
            "model": model,
            "table": table,
            "features": features,
            "r2": num(
                model.rsquared
            ),
            "adj_r2": num(
                model.rsquared_adj
            ),
            "scaler": scaler,
        }

    except Exception as error:

        return {
            "model": None,
            "table": pd.DataFrame(),
            "features": [],
            "r2": np.nan,
            "adj_r2": np.nan,
            "error": str(error),
        }


# ============================================================
# VIF
# ============================================================

def run_vif(
    X,
):
    try:

        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor,
        )

        if X is None or X.empty:
            return pd.DataFrame()

        clean = X.copy()

        clean = clean.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        for column in clean.columns:

            median = clean[column].median()

            if pd.isna(median):
                median = 0.0

            clean[column] = clean[
                column
            ].fillna(
                median
            )

        # Drop constant columns only for VIF.
        keep = [
            column
            for column in clean.columns
            if clean[column].nunique(
                dropna=False
            ) > 1
        ]

        clean = clean[keep]

        if clean.empty:
            return pd.DataFrame()

        matrix = clean.astype(
            float
        ).values

        rows = []

        for index, column in enumerate(
            clean.columns
        ):

            try:

                value = float(
                    variance_inflation_factor(
                        matrix,
                        index,
                    )
                )

            except Exception:

                value = np.inf

            rows.append(
                {
                    "Biến": column,
                    "Nhóm": feature_group(
                        column
                    ),
                    "VIF": value,
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(
                "VIF",
                ascending=False,
                na_position="last",
            )
            .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# ML MODELS
# ============================================================

def run_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    try:

        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )

        from sklearn.linear_model import (
            ElasticNet,
            Ridge,
        )

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        from sklearn.pipeline import (
            Pipeline,
        )

        from sklearn.preprocessing import (
            StandardScaler,
        )

    except Exception:
        return pd.DataFrame(), {}

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
                min_samples_leaf=3,
                random_state=42,
            ),
        ),
        (
            "Hist Gradient Boosting",
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
    ]

    rows = []
    fitted = {}

    actual = np.asarray(
        y_test,
        dtype=float,
    )

    train_values = np.asarray(
        y_train,
        dtype=float,
    )

    train_mean = float(
        np.mean(
            train_values
        )
    )

    baseline_prediction = np.repeat(
        train_mean,
        len(actual),
    )

    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                baseline_prediction,
            )
        )
    )

    for name, model in models:

        try:

            model.fit(
                X_train,
                y_train,
            )

            prediction = np.asarray(
                model.predict(
                    X_test
                ),
                dtype=float,
            )

            mae = float(
                mean_absolute_error(
                    actual,
                    prediction,
                )
            )

            mse = float(
                mean_squared_error(
                    actual,
                    prediction,
                )
            )

            rmse = float(
                np.sqrt(mse)
            )

            try:

                r2 = float(
                    r2_score(
                        actual,
                        prediction,
                    )
                )

            except Exception:

                r2 = np.nan

            rows.append(
                {
                    "Mô hình": name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R²": r2,
                    "Baseline_RMSE": baseline_rmse,
                    "So_baseline": (
                        "Tốt hơn"
                        if rmse < baseline_rmse
                        else "Kém hơn"
                    ),
                }
            )

            fitted[name] = model

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(), {}

    result = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "RMSE",
                "MAE",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return result, fitted


# ============================================================
# PERMUTATION
# ============================================================

def run_permutation(
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
            n_repeats=5,
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
            feature_group
        )

        return (
            table
            .sort_values(
                "AbsPermutation",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# TREE IMPORTANCE
# ============================================================

def run_tree_importance(
    model,
    features,
):
    try:

        raw_model = model

        if hasattr(
            model,
            "named_steps",
        ):

            raw_model = (
                model
                .named_steps
                .get(
                    "model",
                    model,
                )
            )

        if not hasattr(
            raw_model,
            "feature_importances_",
        ):
            return pd.DataFrame()

        values = np.asarray(
            raw_model.feature_importances_,
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
            feature_group
        )

        return (
            table
            .sort_values(
                "TreeImportance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# FACTOR RANKING
# ============================================================

def _percentile_score(
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


def build_factor_ranking(
    correlation,
    ols,
    permutation,
    tree,
):
    if (
        correlation is None
        or correlation.empty
    ):
        return pd.DataFrame()

    result = correlation.copy()

    if (
        isinstance(
            ols,
            pd.DataFrame,
        )
        and not ols.empty
    ):

        result = result.merge(
            ols[
                [
                    "Biến",
                    "Beta",
                    "p-value",
                    "AbsBeta",
                ]
            ],
            on="Biến",
            how="left",
        )

    else:

        result["Beta"] = np.nan
        result["p-value"] = np.nan
        result["AbsBeta"] = np.nan

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

        result["Permutation"] = np.nan
        result["AbsPermutation"] = np.nan

    if (
        isinstance(
            tree,
            pd.DataFrame,
        )
        and not tree.empty
    ):

        result = result.merge(
            tree[
                [
                    "Biến",
                    "TreeImportance",
                ]
            ],
            on="Biến",
            how="left",
        )

    else:

        result["TreeImportance"] = np.nan

    result[
        "CorrScore"
    ] = _percentile_score(
        result["AbsSpearman"]
    )

    result[
        "BetaScore"
    ] = _percentile_score(
        result["AbsBeta"]
    )

    result[
        "PermutationScore"
    ] = _percentile_score(
        result["AbsPermutation"]
    )

    result[
        "TreeScore"
    ] = _percentile_score(
        result["TreeImportance"]
    )

    result[
        "Score"
    ] = (
        result[
            "CorrScore"
        ].fillna(0) * 0.20
        + result[
            "BetaScore"
        ].fillna(0) * 0.20
        + result[
            "PermutationScore"
        ].fillna(0) * 0.30
        + result[
            "TreeScore"
        ].fillna(0) * 0.30
    )

    def get_direction(
        row,
    ):
        beta = num(
            row.get("Beta")
        )

        spear = num(
            row.get("Spearman")
        )

        value = (
            beta
            if beta is not None
            else spear
        )

        if value is None:
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

    def significance(
        value,
    ):
        p = num(value)

        if p is None:
            return "Không có p-value"

        if p < 0.01:
            return "Rất mạnh (<1%)"

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
        significance
    )

    return (
        result
        .sort_values(
            [
                "Score",
                "AbsSpearman",
            ],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# GROUP SUMMARY
# ============================================================

def group_summary(
    ranking,
):
    if (
        ranking is None
        or ranking.empty
    ):
        return pd.DataFrame()

    result = (
        ranking
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

    return (
        result
        .sort_values(
            [
                "Score_TB",
                "Score_Max",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
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

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    try:

        from statsmodels.tsa.stattools import (
            adfuller,
        )

        value = adfuller(
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
                value[0]
            ),
            "p_value": float(
                value[1]
            ),
            "nobs": int(
                value[3]
            ),
        }

    except Exception:
        pass

    # --------------------------------------------------------
    # OLS residual
    # --------------------------------------------------------

    residuals = None

    try:

        model = ols_result.get(
            "model"
        )

        if model is not None:

            residuals = np.asarray(
                model.resid,
                dtype=float,
            )

    except Exception:
        pass

    if residuals is None:
        return tests

    # --------------------------------------------------------
    # Jarque-Bera
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Durbin-Watson
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ljung-Box
    # --------------------------------------------------------

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

        lb = acorr_ljungbox(
            residuals,
            lags=[lag],
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

    # --------------------------------------------------------
    # Breusch-Pagan
    # --------------------------------------------------------

    try:

        import statsmodels.api as sm

        from statsmodels.stats.diagnostic import (
            het_breuschpagan,
        )

        ols_features = ols_result.get(
            "features",
            [],
        )

        selected = [
            column
            for column in ols_features
            if column in X_train.columns
        ]

        if selected:

            X_bp = sm.add_constant(
                X_train[selected],
                has_constant="add",
            )

            bp = het_breuschpagan(
                residuals,
                X_bp,
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

    # --------------------------------------------------------
    # White
    #
    # Chỉ chạy nếu matrix OLS nhỏ,
    # tránh bùng số biến chéo.
    # --------------------------------------------------------

    try:

        import statsmodels.api as sm

        from statsmodels.stats.diagnostic import (
            het_white,
        )

        ols_features = ols_result.get(
            "features",
            [],
        )

        selected = [
            column
            for column in ols_features
            if column in X_train.columns
        ]

        if 1 <= len(selected) <= 10:

            X_white = sm.add_constant(
                X_train[selected],
                has_constant="add",
            )

            white = het_white(
                residuals,
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
                "features": selected,
            }

    except Exception:
        pass

    return tests


# ============================================================
# ONE HORIZON
# ============================================================

def execute_one_horizon(
    df,
    all_features,
    horizon,
):
    xy = prepare_xy(
        df,
        all_features,
        f"Target_{horizon}",
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

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlations = correlation_table(
        X,
        y,
    )

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    ols = run_safe_ols(
        X_train,
        y_train,
    )

    ols_table = ols.get(
        "table",
        pd.DataFrame(),
    )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    models, fitted = run_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    best_name = None
    best_model = None

    if (
        isinstance(
            models,
            pd.DataFrame,
        )
        and not models.empty
    ):

        best_name = str(
            models.iloc[
                0
            ]["Mô hình"]
        )

        best_model = fitted.get(
            best_name
        )

    # --------------------------------------------------------
    # Permutation
    # --------------------------------------------------------

    permutation = pd.DataFrame()

    if best_model is not None:

        permutation = run_permutation(
            best_model,
            X_test,
            y_test,
        )

    # --------------------------------------------------------
    # Tree importance
    # --------------------------------------------------------

    tree = pd.DataFrame()

    if best_model is not None:

        tree = run_tree_importance(
            best_model,
            X_train.columns,
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranking = build_factor_ranking(
        correlations,
        ols_table,
        permutation,
        tree,
    )

    # --------------------------------------------------------
    # Group
    # --------------------------------------------------------

    groups = group_summary(
        ranking
    )

    # --------------------------------------------------------
    # VIF
    # --------------------------------------------------------

    ols_features = ols.get(
        "features",
        [],
    )

    vif = pd.DataFrame()

    if ols_features:

        vif = run_vif(
            X_train[
                ols_features
            ]
        )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    tests = run_tests(
        X_train,
        y_train,
        ols,
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    current_price = num(
        df[
            "Close"
        ].iloc[-1]
    )

    predicted_return = None
    predicted_price = None

    if best_model is not None:

        try:

            latest = (
                df[
                    list(
                        X.columns
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

            for column in latest.columns:

                median = latest[
                    column
                ].median()

                if pd.isna(median):
                    median = 0.0

                latest[column] = latest[
                    column
                ].fillna(
                    median
                )

            predicted_return = float(
                best_model.predict(
                    latest.iloc[
                        [-1]
                    ]
                )[0]
            )

            if current_price is not None:

                predicted_price = (
                    current_price
                    * (
                        1
                        + predicted_return
                    )
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Model quality
    # --------------------------------------------------------

    quality = "Chưa đánh giá"

    if (
        isinstance(
            models,
            pd.DataFrame,
        )
        and not models.empty
    ):

        r2 = num(
            models.iloc[
                0
            ].get(
                "R²"
            )
        )

        if r2 is not None:

            if r2 < 0:
                quality = "Yếu"

            elif r2 < 0.10:
                quality = "Thấp"

            elif r2 < 0.25:
                quality = "Trung bình"

            else:
                quality = "Khá"


    return {
        "horizon": horizon,
        "observations": len(X),
        "train": len(X_train),
        "test": len(X_test),
        "features_used": list(
            X.columns
        ),
        "ols_features": ols_features,
        "correlations": correlations,
        "ols": ols,
        "ols_table": ols_table,
        "models": models,
        "fitted": fitted,
        "best_model": best_name,
        "permutation": permutation,
        "tree": tree,
        "ranking": ranking,
        "groups": groups,
        "vif": vif,
        "tests": tests,
        "forecast": {
            "current_price": current_price,
            "predicted_return": predicted_return,
            "predicted_price": predicted_price,
        },
        "quality": quality,
    }


# ============================================================
# FULL RESEARCH
# ============================================================

@st.cache_data(
    ttl=RESEARCH_CACHE_TTL,
    show_spinner=False,
)
def execute_full_research(
    sample,
):
    prepared = ensure_targets(
        sample
    )

    if prepared.empty:

        return {
            "ok": False,
            "error": "Không có dữ liệu nghiên cứu.",
        }

    all_features = get_all_features(
        prepared
    )

    if not all_features:

        return {
            "ok": False,
            "error": "Không tìm thấy biến định lượng.",
        }

    horizon_results = {}

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = execute_one_horizon(
            prepared,
            all_features,
            horizon,
        )

        if item is not None:

            horizon_results[
                horizon
            ] = item

    if not horizon_results:

        return {
            "ok": False,
            "error": "Không có horizon nào đủ dữ liệu.",
        }

    return {
        "ok": True,
        "data": prepared,
        "all_features": all_features,
        "groups": group_features(
            all_features
        ),
        "horizons": horizon_results,
    }


# ============================================================
# DISPLAY DATA
# ============================================================

@st.cache_data(
    ttl=DISPLAY_CACHE_TTL,
    show_spinner=False,
)
def load_display_data(
    symbol,
):
    return load_market_data(
        symbol,
        "1y",
    )


# ============================================================
# FORECAST UI
# ============================================================

def render_forecast(
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

    predicted_return = num(
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
        "🎯 Dự báo"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Giá hiện tại",
            format_price(
                current
            ),
        )

    with b:

        st.metric(
            "Lợi suất dự báo",
            (
                format_percent(
                    predicted_return * 100
                )
                if predicted_return is not None
                else "—"
            ),
        )

    with c:

        st.metric(
            "Giá dự báo",
            format_price(
                predicted_price
            ),
        )

    with d:

        st.metric(
            "Model",
            safe_text(
                item.get(
                    "best_model",
                    "—",
                ),
                "—",
            ),
        )

    if predicted_return is not None:

        if predicted_return > 0:

            st.success(
                f"Mô hình đang nghiêng về **TĂNG** "
                f"{predicted_return * 100:+.2f}% "
                f"cho horizon {item['horizon']}."
            )

        elif predicted_return < 0:

            st.warning(
                f"Mô hình đang nghiêng về **GIẢM** "
                f"{predicted_return * 100:+.2f}% "
                f"cho horizon {item['horizon']}."
            )

        else:

            st.info(
                "Mô hình đang nghiêng về đi ngang."
            )


# ============================================================
# GROUP UI
# ============================================================

def render_groups(
    item,
):
    groups = item.get(
        "groups"
    )

    if (
        not isinstance(
            groups,
            pd.DataFrame,
        )
        or groups.empty
    ):
        return

    st.subheader(
        "🧩 Nhóm yếu tố"
    )

    show = groups.rename(
        columns={
            "Nhóm": "Nhóm yếu tố",
            "Score_TB": "Score TB",
            "Score_Max": "Score cao nhất",
            "So_bien": "Số biến",
        }
    )

    show[
        "Score TB"
    ] = pd.to_numeric(
        show[
            "Score TB"
        ],
        errors="coerce",
    ).round(1)

    show[
        "Score cao nhất"
    ] = pd.to_numeric(
        show[
            "Score cao nhất"
        ],
        errors="coerce",
    ).round(1)

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
    )

    strongest = groups.iloc[
        0
    ]

    st.info(
        f"Nhóm nổi bật nhất là **{strongest['Nhóm']}**, "
        f"Score trung bình {strongest['Score_TB']:.1f}/100."
    )


# ============================================================
# FACTOR UI
# ============================================================

def render_factors(
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
            "Chưa có xếp hạng yếu tố."
        )

        return

    st.subheader(
        "🏆 Yếu tố liên hệ / dự báo"
    )

    show = ranking.head(
        MAX_DISPLAY_FACTORS
    ).copy()

    display_columns = [
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

    display_columns = [
        column
        for column in display_columns
        if column in show.columns
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

        if column in show.columns:

            show[column] = pd.to_numeric(
                show[column],
                errors="coerce",
            )

    st.dataframe(
        show[
            display_columns
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "#### 🎯 5 yếu tố nổi bật nhất"
    )

    for _, row in ranking.head(5).iterrows():

        feature = safe_text(
            row.get("Biến"),
            "Không xác định",
        )

        group = safe_text(
            row.get("Nhóm"),
            "Không xác định",
        )

        score = num(
            row.get("Score")
        )

        direction = safe_text(
            row.get(
                "Quan hệ"
            ),
            "Không xác định",
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

        text = (
            f"**{feature}** ({group})"
        )

        if score is not None:

            text += (
                f" · Score {score:.1f}/100"
            )

        text += (
            f" · {direction}"
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
# MODEL UI
# ============================================================

def render_models(
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
            "Không có kết quả mô hình."
        )

        return

    show = models.copy()

    for column in [
        "MAE",
        "MSE",
        "RMSE",
        "R²",
        "Baseline_RMSE",
    ]:

        if column in show.columns:

            show[column] = pd.to_numeric(
                show[column],
                errors="coerce",
            ).round(6)

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
    )

    best = models.iloc[
        0
    ]

    rmse = num(
        best.get(
            "RMSE"
        )
    )

    r2 = num(
        best.get(
            "R²"
        )
    )

    baseline = num(
        best.get(
            "Baseline_RMSE"
        )
    )

    status = safe_text(
        best.get(
            "So_baseline"
        ),
        "Không rõ",
    )

    st.info(
        f"Model tốt nhất theo RMSE: "
        f"**{best['Mô hình']}** · "
        f"RMSE {format_number(rmse, 6)} · "
        f"R² {format_number(r2, 4)} · "
        f"so benchmark: **{status}**"
    )

    if (
        baseline is not None
        and rmse is not None
    ):

        if rmse < baseline:

            improvement = (
                (
                    baseline - rmse
                )
                / baseline
                * 100
                if baseline != 0
                else 0
            )

            st.success(
                f"Model tốt hơn benchmark khoảng "
                f"{improvement:.2f}% theo RMSE."
            )

        else:

            st.warning(
                "Model không vượt benchmark trung bình trên tập test."
            )


# ============================================================
# OLS UI
# ============================================================

def render_ols(
    item,
):
    st.subheader(
        "📐 OLS"
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

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "R²",
            format_number(
                ols.get(
                    "r2"
                ),
                4,
            ),
        )

    with b:

        st.metric(
            "Adjusted R²",
            format_number(
                ols.get(
                    "adj_r2"
                ),
                4,
            ),
        )

    with c:

        st.metric(
            "Biến OLS",
            len(
                ols.get(
                    "features",
                    [],
                )
            ),
        )

    if (
        isinstance(
            table,
            pd.DataFrame,
        )
        and not table.empty
    ):

        show = table.copy()

        for column in [
            "Beta",
            "p-value",
            "Std Error",
        ]:

            if column in show.columns:

                show[column] = pd.to_numeric(
                    show[column],
                    errors="coerce",
                ).round(6)

        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "OLS dùng tập biến đã giảm đa cộng tuyến để hệ số "
        "ổn định hơn; các biến còn lại vẫn được giữ trong "
        "dataset và các mô hình máy học."
    )


# ============================================================
# VIF UI
# ============================================================

def render_vif(
    item,
):
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
        return

    st.subheader(
        "🔗 Đa cộng tuyến — VIF"
    )

    show = vif.copy()

    show["VIF"] = pd.to_numeric(
        show["VIF"],
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).round(3)

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
    )

    max_vif = num(
        show["VIF"].max()
    )

    if max_vif is None:

        return

    if max_vif >= 10:

        st.error(
            f"VIF cao nhất {max_vif:.2f}: "
            "đa cộng tuyến mạnh."
        )

    elif max_vif >= 5:

        st.warning(
            f"VIF cao nhất {max_vif:.2f}: "
            "có đa cộng tuyến đáng chú ý."
        )

    else:

        st.success(
            f"VIF cao nhất {max_vif:.2f}: "
            "chưa thấy đa cộng tuyến quá mạnh trong tập OLS."
        )


# ============================================================
# TEST UI
# ============================================================

def _render_pvalue_status(
    pvalue,
    good_text,
    bad_text,
):
    pvalue = num(
        pvalue
    )

    if pvalue is None:

        st.info(
            "Không đủ dữ liệu để kết luận."
        )

    elif pvalue < 0.05:

        st.warning(
            bad_text
        )

    else:

        st.success(
            good_text
        )


def render_tests(
    item,
):
    tests = item.get(
        "tests",
        {},
    )

    if not tests:

        st.info(
            "Chưa có kiểm định khả dụng."
        )

        return

    st.subheader(
        "🧪 Kiểm định thống kê"
    )

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    adf = tests.get(
        "ADF"
    )

    if isinstance(
        adf,
        dict,
    ):

        p = num(
            adf.get(
                "p_value"
            )
        )

        with st.expander(
            "ADF — tính dừng",
            expanded=False,
        ):

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
                    "Bác bỏ giả thuyết nghiệm đơn vị ở mức 5%."
                )

            else:

                st.info(
                    "Chưa đủ bằng chứng bác bỏ nghiệm đơn vị ở mức 5%."
                )

    # --------------------------------------------------------
    # Jarque-Bera
    # --------------------------------------------------------

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

            st.write(
                f"Statistic: "
                f"{format_number(jb.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: "
                f"{format_number(jb.get('p_value'), 6)}"
            )

            p = num(
                jb.get(
                    "p_value"
                )
            )

            if (
                p is not None
                and p < 0.05
            ):

                st.warning(
                    "Phần dư có dấu hiệu lệch khỏi phân phối chuẩn."
                )

            else:

                st.success(
                    "Chưa thấy bằng chứng mạnh chống lại giả định chuẩn."
                )

    # --------------------------------------------------------
    # Breusch-Pagan
    # --------------------------------------------------------

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

            st.write(
                f"Statistic: "
                f"{format_number(bp.get('statistic'), 5)}"
            )

            p = num(
                bp.get(
                    "p_value"
                )
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
                    "Chưa thấy bằng chứng rõ về phương sai thay đổi."
                )

    # --------------------------------------------------------
    # White
    # --------------------------------------------------------

    white = tests.get(
        "White"
    )

    if isinstance(
        white,
        dict,
    ):

        with st.expander(
            "White Test — phương sai thay đổi",
            expanded=False,
        ):

            st.write(
                f"Statistic: "
                f"{format_number(white.get('statistic'), 5)}"
            )

            p = num(
                white.get(
                    "p_value"
                )
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
                    "White Test cho thấy dấu hiệu phương sai thay đổi."
                )

            else:

                st.success(
                    "White Test chưa cho thấy phương sai thay đổi rõ."
                )

    # --------------------------------------------------------
    # Durbin-Watson
    # --------------------------------------------------------

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

                if 1.5 <= value <= 2.5:

                    st.success(
                        "Statistic tương đối gần 2; "
                        "chưa thấy tự tương quan mạnh."
                    )

                elif value < 1.5:

                    st.warning(
                        "Có dấu hiệu tự tương quan dương."
                    )

                else:

                    st.warning(
                        "Có dấu hiệu tự tương quan âm."
                    )

    # --------------------------------------------------------
    # Ljung Box
    # --------------------------------------------------------

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

            st.write(
                f"Lag: {lb.get('lag', '—')}"
            )

            st.write(
                f"Statistic: "
                f"{format_number(lb.get('statistic'), 5)}"
            )

            p = num(
                lb.get(
                    "p_value"
                )
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
                    "Có bằng chứng tự tương quan còn lại trong phần dư."
                )

            else:

                st.success(
                    "Chưa thấy tự tương quan phần dư rõ."
                )


# ============================================================
# HORIZON CONCLUSION
# ============================================================

def render_conclusion(
    item,
):
    ranking = item.get(
        "ranking"
    )

    st.subheader(
        "🧠 Kết luận của horizon"
    )

    if (
        not isinstance(
            ranking,
            pd.DataFrame,
        )
        or ranking.empty
    ):

        st.info(
            "Chưa đủ kết quả để kết luận."
        )

        return

    top = ranking.iloc[
        0
    ]

    factor = safe_text(
        top.get(
            "Biến"
        ),
        "Không xác định",
    )

    group = safe_text(
        top.get(
            "Nhóm"
        ),
        "Không xác định",
    )

    score = num(
        top.get(
            "Score"
        )
    )

    relation = safe_text(
        top.get(
            "Quan hệ"
        ),
        "Không xác định",
    )

    pvalue = num(
        top.get(
            "p-value"
        )
    )

    spear = num(
        top.get(
            "Spearman"
        )
    )

    text = (
        f"Yếu tố nổi bật nhất là "
        f"**{factor}** ({group})"
    )

    if score is not None:

        text += (
            f", Score {score:.1f}/100"
        )

    text += (
        f", quan hệ {relation.lower()}"
    )

    if spear is not None:

        text += (
            f", Spearman {spear:+.4f}"
        )

    if pvalue is not None:

        text += (
            f", p-value {pvalue:.5f}"
        )

    text += "."

    st.markdown(
        "• " + text
    )

    if (
        pvalue is not None
        and pvalue < 0.05
    ):

        st.markdown(
            f"• **{factor}** có bằng chứng "
            "thống kê trong mô hình OLS ở mức 5%."
        )

    else:

        st.markdown(
            f"• **{factor}** có tín hiệu nổi bật "
            "nhưng chưa đủ bằng chứng để gọi là quan hệ nhân quả."
        )

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

        rmse = num(
            best.get("RMSE")
        )

        r2 = num(
            best.get("R²")
        )

        model_name = safe_text(
            best.get(
                "Mô hình"
            ),
            "Không xác định",
        )

        baseline = num(
            best.get(
                "Baseline_RMSE"
            )
        )

        comparison = safe_text(
            best.get(
                "So_baseline"
            ),
            "Không rõ",
        )

        line = (
            f"• Model tốt nhất: **{model_name}**"
        )

        if rmse is not None:

            line += (
                f", RMSE {rmse:.6f}"
            )

        if r2 is not None:

            line += (
                f", R² {r2:.4f}"
            )

        line += (
            f", {comparison.lower()} benchmark."
        )

        st.markdown(
            line
        )

        if (
            baseline is not None
            and rmse is not None
            and rmse >= baseline
        ):

            st.warning(
                "Model chưa vượt benchmark trên tập test."
            )

        elif (
            baseline is not None
            and rmse is not None
        ):

            st.success(
                "Model đã vượt benchmark trên tập test."
            )

    forecast = item.get(
        "forecast",
        {},
    )

    predicted_return = num(
        forecast.get(
            "predicted_return"
        )
    )

    predicted_price = num(
        forecast.get(
            "predicted_price"
        )
    )

    if (
        predicted_return is not None
        and predicted_price is not None
    ):

        st.markdown(
            f"• Dự báo theo model: "
            f"**{predicted_return * 100:+.2f}%** "
            f"→ khoảng **{predicted_price:,.0f} đồng**."
        )


# ============================================================
# CROSS HORIZON
# ============================================================

def render_cross_horizon(
    result,
):
    st.divider()

    st.header(
        "🏆 Yếu tố nhất quán giữa 1D / 5D / 20D"
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
            ranking.head(20).iterrows(),
            start=1,
        ):

            rows.append(
                {
                    "Horizon": horizon,
                    "Biến": row.get(
                        "Biến"
                    ),
                    "Nhóm": row.get(
                        "Nhóm"
                    ),
                    "Rank": rank,
                    "Score": num(
                        row.get(
                            "Score"
                        )
                    ),
                }
            )

    if not rows:

        st.info(
            "Chưa có đủ dữ liệu để so sánh các horizon."
        )

        return

    work = pd.DataFrame(
        rows
    )

    stable = (
        work
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
        .reset_index(drop=True)
    )

    stable[
        "Rank_TB"
    ] = stable[
        "Rank_TB"
    ].round(2)

    stable[
        "Score_TB"
    ] = stable[
        "Score_TB"
    ].round(1)

    stable.columns = [
        "Biến",
        "Nhóm",
        "Số horizon",
        "Rank TB",
        "Score TB",
    ]

    st.dataframe(
        stable.head(30),
        width="stretch",
        hide_index=True,
    )

    best = stable.iloc[
        0
    ]

    st.success(
        f"Yếu tố ổn định nhất: **{best['Biến']}** "
        f"({int(best['Số horizon'])} horizon)."
    )


# ============================================================
# AI
# ============================================================

def get_gemini_api_key():
    try:
        key = st.secrets.get(
            "GEMINI_API_KEY"
        )
    except Exception:
        key = None

    if key is None:
        return None

    key = str(key).strip()

    return key if key else None


def dataframe_to_ai_text(
    df,
    columns=None,
    limit=25,
):
    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):
        return "Không có dữ liệu."

    work = df.copy()

    if columns:

        selected = [
            column
            for column in columns
            if column in work.columns
        ]

        if selected:
            work = work[selected]

    work = work.head(
        limit
    ).copy()

    for column in work.columns:

        if pd.api.types.is_numeric_dtype(
            work[column]
        ):

            work[
                column
            ] = pd.to_numeric(
                work[
                    column
                ],
                errors="coerce",
            ).round(6)

    return work.to_string(
        index=False
    )


def build_research_ai_context(
    result,
    symbol,
    start_date,
    end_date,
):
    blocks = []

    blocks.append(
        "=== RESEARCH SCOPE ==="
    )

    blocks.append(
        f"Mã cổ phiếu: {symbol}"
    )

    blocks.append(
        f"Khoảng nghiên cứu: "
        f"{start_date} → {end_date}"
    )

    all_features = result.get(
        "all_features",
        [],
    )

    blocks.append(
        f"Tổng số biến: {len(all_features)}"
    )

    groups = result.get(
        "groups",
        {},
    )

    for group_name, values in groups.items():

        blocks.append(
            f"{group_name}: "
            f"{len(values)} biến"
        )

    horizons = result.get(
        "horizons",
        {},
    )

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        item = horizons.get(
            horizon
        )

        if not isinstance(
            item,
            dict,
        ):
            continue

        blocks.append(
            "\n"
            + "=" * 70
        )

        blocks.append(
            f"HORIZON {horizon}"
        )

        blocks.append(
            "=" * 70
        )

        blocks.append(
            f"Quan sát: "
            f"{item.get('observations')}"
        )

        blocks.append(
            f"Train: "
            f"{item.get('train')}"
        )

        blocks.append(
            f"Test: "
            f"{item.get('test')}"
        )

        blocks.append(
            f"Model tốt nhất: "
            f"{item.get('best_model')}"
        )

        blocks.append(
            f"Quality: "
            f"{item.get('quality')}"
        )

        # ----------------------------------------------------
        # TOP FACTORS
        # ----------------------------------------------------

        blocks.append(
            "\nTOP FACTORS"
        )

        blocks.append(
            dataframe_to_ai_text(
                item.get(
                    "ranking"
                ),
                [
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
                ],
                20,
            )
        )

        # ----------------------------------------------------
        # GROUPS
        # ----------------------------------------------------

        blocks.append(
            "\nFACTOR GROUPS"
        )

        blocks.append(
            dataframe_to_ai_text(
                item.get(
                    "groups"
                ),
                [
                    "Nhóm",
                    "Score_TB",
                    "Score_Max",
                    "So_bien",
                ],
                20,
            )
        )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        blocks.append(
            "\nMODEL PERFORMANCE"
        )

        blocks.append(
            dataframe_to_ai_text(
                item.get(
                    "models"
                ),
                [
                    "Mô hình",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R²",
                    "Baseline_RMSE",
                    "So_baseline",
                ],
                10,
            )
        )

        # ----------------------------------------------------
        # OLS
        # ----------------------------------------------------

        ols = item.get(
            "ols"
        )

        if isinstance(
            ols,
            dict,
        ):

            blocks.append(
                "\nOLS SUMMARY"
            )

            blocks.append(
                f"R²: {ols.get('r2')}"
            )

            blocks.append(
                f"Adjusted R²: {ols.get('adj_r2')}"
            )

            blocks.append(
                "OLS features: "
                + ", ".join(
                    map(
                        str,
                        ols.get(
                            "features",
                            [],
                        ),
                    )
                )
            )

        blocks.append(
            "\nOLS COEFFICIENTS"
        )

        blocks.append(
            dataframe_to_ai_text(
                item.get(
                    "ols_table"
                ),
                [
                    "Biến",
                    "Nhóm",
                    "Beta",
                    "p-value",
                    "Std Error",
                ],
                20,
            )
        )

        # ----------------------------------------------------
        # VIF
        # ----------------------------------------------------

        blocks.append(
            "\nVIF"
        )

        blocks.append(
            dataframe_to_ai_text(
                item.get(
                    "vif"
                ),
                [
                    "Biến",
                    "Nhóm",
                    "VIF",
                ],
                20,
            )
        )

        # ----------------------------------------------------
        # TESTS
        # ----------------------------------------------------

        tests = item.get(
            "tests",
            {},
        )

        blocks.append(
            "\nSTATISTICAL TESTS"
        )

        for test_name, value in tests.items():

            blocks.append(
                f"{test_name}: {value}"
            )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        forecast = item.get(
            "forecast",
            {},
        )

        blocks.append(
            "\nFORECAST"
        )

        blocks.append(
            f"Current price: "
            f"{forecast.get('current_price')}"
        )

        blocks.append(
            f"Predicted return: "
            f"{forecast.get('predicted_return')}"
        )

        blocks.append(
            f"Predicted price: "
            f"{forecast.get('predicted_price')}"
        )

    return "\n".join(
        blocks
    )


def build_research_ai_prompt(
    context,
    question,
):
    question = safe_text(
        question
    )

    if question:

        question_block = f"""
CÂU HỎI CỤ THỂ CỦA NGƯỜI DÙNG:
{question}
""".strip()

    else:

        question_block = """
Không có câu hỏi cụ thể.
Hãy tự đọc toàn bộ nghiên cứu và chỉ ra kết luận quan trọng nhất.
""".strip()

    return f"""
Bạn là Brian AI, một chuyên viên đọc nghiên cứu định lượng
cổ phiếu Việt Nam.

Bạn KHÔNG chạy lại mô hình.
Bạn CHỈ đọc các kết quả nghiên cứu đã được cung cấp.

{question_block}

============================================================
MỤC TIÊU
============================================================

Người đọc phải hiểu ngay:

- Giá/lợi suất đang liên hệ mạnh nhất với yếu tố nào?
- Yếu tố đó thuộc nhóm nào?
- Cùng chiều hay ngược chiều?
- Bằng chứng thống kê có mạnh không?
- Kết quả có ổn định giữa 1D / 5D / 20D không?
- Model dự báo có thực sự tốt hơn benchmark không?
- Có vấn đề đa cộng tuyến, phương sai thay đổi,
  tự tương quan hoặc phân phối phần dư không?
- Nói đơn giản nghiên cứu này đang cho thấy điều gì?

============================================================
QUY TẮC
============================================================

1. Chỉ dùng số liệu được cung cấp.
2. Không tự bịa thêm dữ liệu.
3. Không bịa biến.
4. Không biến correlation thành quan hệ nhân quả.
5. Không nói "X làm giá tăng" nếu dữ liệu chỉ cho correlation.
6. p-value < 0.05 mới gọi là có ý nghĩa thống kê ở mức 5%.
7. Nếu R² âm thì nói thẳng model không vượt benchmark.
8. Không nhồi hàng chục biến vào câu trả lời.
9. Chỉ chọn 3–5 yếu tố thực sự đáng chú ý.
10. Ưu tiên yếu tố được nhiều phương pháp cùng chỉ ra.
11. Ưu tiên yếu tố ổn định qua nhiều horizon.
12. Nếu không đủ bằng chứng phải nói rõ.
13. Viết tiếng Việt tự nhiên, như một chuyên viên đang
    giải thích cho nhà đầu tư.
14. Không viết kiểu giáo trình.
15. Không khuyến nghị mua/bán.
16. Không cần liệt kê kiểm định nếu chúng không ảnh hưởng
    đến kết luận chính.

============================================================
FORMAT
============================================================

# 🧠 Brian AI — Đọc nghiên cứu

## Kết luận chính

3–4 câu.

Nói thẳng yếu tố nổi bật nhất và nghiên cứu có đáng tin
cho việc dự báo hay không.

## 🎯 Giá đang liên hệ với gì?

Chọn tối đa 5 yếu tố.

Mỗi yếu tố:

**Tên biến — Nhóm**
- Quan hệ:
- Spearman:
- Beta:
- p-value:
- Ý nghĩa:

## 📈 1D / 5D / 20D

Chỉ ra điểm khác biệt quan trọng giữa các horizon.

Nêu yếu tố nào lặp lại nhiều nhất.

## 🤖 Model

Nói:
- Model tốt nhất
- RMSE
- R²
- So với benchmark
- Có nên xem đây là model dự báo mạnh hay chỉ là tín hiệu tham khảo

## ⚠️ Điều cần lưu ý

Chỉ nêu các vấn đề thật sự xuất hiện.

## Nói đơn giản

3–5 câu.

Giải thích nghiên cứu này như đang nói chuyện với
một người đầu tư không chuyên thống kê.

============================================================
DỮ LIỆU NGHIÊN CỨU
============================================================

{context}
""".strip()


def _call_gemini(
    api_key,
    model,
    prompt,
):
    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    response = (
        client
        .models
        .generate_content(
            model=model,
            contents=prompt,
        )
    )

    text = safe_text(
        getattr(
            response,
            "text",
            "",
        )
    )

    if not text:

        raise RuntimeError(
            "Gemini trả về nội dung rỗng."
        )

    return text


def _is_temporary_ai_error(
    error,
):
    text = str(
        error
    ).lower()

    tokens = [
        "503",
        "unavailable",
        "high demand",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "resource exhausted",
        "deadline exceeded",
        "internal server error",
    ]

    return any(
        token in text
        for token in tokens
    )


def _is_fatal_ai_error(
    error,
):
    text = str(
        error
    ).lower()

    tokens = [
        "401",
        "403",
        "invalid api key",
        "api key not valid",
        "permission denied",
        "unauthenticated",
        "billing",
    ]

    return any(
        token in text
        for token in tokens
    )


def generate_research_ai(
    context,
    question,
):
    api_key = get_gemini_api_key()

    if not api_key:

        return {
            "ok": False,
            "text": "",
            "model": None,
            "error": (
                "Thiếu GEMINI_API_KEY trong Streamlit Secrets."
            ),
        }

    try:

        import google.genai  # noqa: F401

    except Exception as error:

        return {
            "ok": False,
            "text": "",
            "model": None,
            "error": (
                "Chưa cài google-genai: "
                f"{error}"
            ),
        }

    prompt = build_research_ai_prompt(
        context,
        question,
    )

    errors = []

    for model in GEMINI_MODELS:

        try:

            output = _call_gemini(
                api_key,
                model,
                prompt,
            )

            return {
                "ok": True,
                "text": output,
                "model": model,
                "error": "",
            }

        except Exception as error:

            error_text = str(
                error
            )

            errors.append(
                f"{model}: {error_text}"
            )

            if _is_fatal_ai_error(
                error
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            if _is_temporary_ai_error(
                error
            ):

                continue

            return {
                "ok": False,
                "text": "",
                "model": model,
                "error": error_text,
            }

    return {
        "ok": False,
        "text": "",
        "model": None,
        "error": (
            "Không có model Gemini nào khả dụng.\n\n"
            + "\n".join(
                errors
            )
        ),
    }


@st.cache_data(
    ttl=AI_CACHE_TTL,
    show_spinner=False,
)
def research_ai_cached(
    context,
    question,
):
    return generate_research_ai(
        context,
        question,
    )


# ============================================================
# RESEARCH AI PANEL
# ============================================================

def render_research_ai(
    result,
    symbol,
    start_date,
    end_date,
):
    st.divider()

    st.header(
        "🧠 Brian AI — Đọc nghiên cứu"
    )

    st.caption(
        "AI đọc kết quả định lượng đã chạy và giải thích "
        "yếu tố nào đang liên hệ mạnh với lợi suất cổ phiếu."
    )

    if (
        not isinstance(
            result,
            dict,
        )
        or not result.get(
            "ok",
            False,
        )
    ):

        st.info(
            "Chạy nghiên cứu định lượng trước."
        )

        return

    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    question = st.text_area(
        "Hỏi Brian AI",
        placeholder=(
            "Ví dụ:\n"
            "Biến nào ảnh hưởng giá mạnh nhất?\n"
            "Khối ngoại có tác động đáng kể không?\n"
            "Dòng tiền hay kỹ thuật quan trọng hơn?\n"
            "Thị trường chung có ảnh hưởng mạnh không?\n"
            "1D, 5D hay 20D đáng tin hơn?\n"
            "Tại sao R² lại âm?\n"
            "Nói ngắn gọn nghiên cứu này đang cho thấy gì?"
        ),
        height=130,
        key="research_ai_question",
    ).strip()

    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    st.markdown(
        "#### ⚡ Hỏi nhanh"
    )

    q1, q2, q3, q4 = st.columns(4)

    quick_question = None

    with q1:

        if st.button(
            "🎯 Yếu tố mạnh nhất",
            key="research_ai_q1",
            width="stretch",
        ):

            quick_question = (
                "Yếu tố nào đang liên hệ mạnh nhất với lợi suất "
                "cổ phiếu? Hãy nói rõ nhóm yếu tố, chiều tác động "
                "và bằng chứng thống kê."
            )

    with q2:

        if st.button(
            "💰 Dòng tiền",
            key="research_ai_q2",
            width="stretch",
        ):

            quick_question = (
                "Dòng tiền, khối ngoại và tự doanh có vai trò "
                "gì trong nghiên cứu? Hãy chỉ ra yếu tố đáng chú ý nhất."
            )

    with q3:

        if st.button(
            "📈 1D / 5D / 20D",
            key="research_ai_q3",
            width="stretch",
        ):

            quick_question = (
                "So sánh 1D, 5D và 20D. Horizon nào cho tín hiệu "
                "ổn định nhất và yếu tố nào xuất hiện lặp lại?"
            )

    with q4:

        if st.button(
            "🔬 Độ tin cậy",
            key="research_ai_q4",
            width="stretch",
        ):

            quick_question = (
                "Đọc toàn bộ model và kiểm định rồi đánh giá "
                "nghiên cứu đáng tin ở mức nào. Nói thẳng điểm mạnh "
                "và điểm yếu."
            )

    if quick_question:
        question = quick_question

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    if st.button(
        "🤖 AI đọc toàn bộ nghiên cứu",
        type="primary",
        width="stretch",
        key="research_ai_run",
    ):

        context = build_research_ai_context(
            result,
            symbol,
            start_date,
            end_date,
        )

        with st.spinner(
            "Brian AI đang đọc số liệu nghiên cứu..."
        ):

            ai_result = research_ai_cached(
                context,
                question,
            )

        st.session_state[
            "research_ai_result"
        ] = ai_result

        st.session_state[
            "research_ai_scope"
        ] = (
            symbol,
            str(start_date),
            str(end_date),
        )

        st.session_state[
            "research_ai_question_used"
        ] = question

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    ai_result = st.session_state.get(
        "research_ai_result"
    )

    saved_scope = st.session_state.get(
        "research_ai_scope"
    )

    current_scope = (
        symbol,
        str(start_date),
        str(end_date),
    )

    if (
        ai_result is None
        or saved_scope != current_scope
    ):

        st.info(
            "Nhập câu hỏi hoặc để trống rồi bấm "
            "“AI đọc toàn bộ nghiên cứu”."
        )

        return

    if not ai_result.get(
        "ok",
        False,
    ):

        st.error(
            "Brian AI chưa chạy được."
        )

        error = safe_text(
            ai_result.get(
                "error"
            )
        )

        if error:
            st.code(error)

        return

    model = safe_text(
        ai_result.get(
            "model"
        )
    )

    if model:

        st.caption(
            f"Model: {model}"
        )

    used_question = safe_text(
        st.session_state.get(
            "research_ai_question_used",
            ""
        )
    )

    if used_question:

        with st.container(
            border=True
        ):

            st.caption(
                "Câu hỏi đã gửi:"
            )

            st.write(
                used_question
            )

    output = safe_text(
        ai_result.get(
            "text"
        )
    )

    if output:

        with st.container(
            border=True
        ):

            st.markdown(
                output
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
        "📈 Phân tích cổ phiếu"
    )

    st.write(
        "Phân tích kỹ thuật, nghiên cứu định lượng, "
        "dòng tiền, khối ngoại, tự doanh, thị trường chung "
        "và nhóm ngành."
    )

    # ========================================================
    # SYMBOL
    # ========================================================

    current_symbol = st.session_state.get(
        "stock_analysis_symbol",
        "HPG",
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

            return

        st.session_state[
            "stock_analysis_symbol"
        ] = clean_symbol

        # Clear old research
        for key in [
            "stock_research_result",
            "stock_research_symbol",
            "stock_research_dates",
            "research_ai_result",
            "research_ai_scope",
            "research_ai_question_used",
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

        display_data = load_display_data(
            symbol
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
        display_data is None
        or display_data.empty
    ):

        st.warning(
            "Không có dữ liệu."
        )

        return

    # ========================================================
    # SNAPSHOT
    # ========================================================

    snapshot = market_snapshot(
        display_data
    )

    price = num(
        snapshot.get("price")
    )

    change = num(
        snapshot.get("change_1d")
    )

    rsi_value = num(
        snapshot.get("rsi")
    )

    volume = num(
        snapshot.get("volume")
    )

    sma20 = num(
        snapshot.get("sma20")
    )

    sma50 = num(
        snapshot.get("sma50")
    )

    macd = num(
        snapshot.get("macd")
    )

    volatility = num(
        snapshot.get("volatility20")
    )

    last = display_data.iloc[
        -1
    ]

    atr14 = num(
        last.get("ATR14")
    )

    volume_sma20 = num(
        last.get("Volume_SMA20")
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
    # STOCK HEADER
    # ========================================================

    st.subheader(
        f"📈 {display_symbol(symbol)}"
    )

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
    # INDICATORS
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
    # STATUS
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
            "ATR14",
            format_price(
                atr14
            ),
        )

    with b:

        st.metric(
            "Volume TB20",
            format_volume(
                volume_sma20
            ),
        )

    with c:

        st.metric(
            "Giá mở cửa",
            format_price(
                num(
                    last.get("Open")
                )
            ),
        )

    with d:

        high = num(
            last.get("High")
        )

        low = num(
            last.get("Low")
        )

        close = num(
            last.get("Close")
        )

        if (
            high is not None
            and low is not None
            and close is not None
            and high != low
        ):

            position = (
                (
                    close - low
                )
                / (
                    high - low
                )
                * 100
            )

            st.metric(
                "Vị trí biên ngày",
                f"{position:.1f}%",
            )

        else:

            st.metric(
                "Vị trí biên ngày",
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
            display_data
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
            f"Không thể hiển thị biểu đồ: {error}"
        )

    # ========================================================
    # QUANTITATIVE RESEARCH
    # ========================================================

    st.divider()

    st.header(
        "🧪 Nghiên cứu định lượng"
    )

    st.write(
        "Chọn đúng mẫu dữ liệu trước khi chạy. "
        "Hệ thống chỉ tải dữ liệu multifactor khi bấm "
        "“Chạy toàn bộ nghiên cứu”."
    )

    mode = st.radio(
        "Kiểu chọn mẫu",
        [
            "Preset",
            "Khoảng ngày tùy chọn",
        ],
        horizontal=True,
        key="research_mode",
    )

    today = (
        pd.Timestamp.today()
        .date()
    )

    if mode == "Preset":

        preset = st.selectbox(
            "Khoảng thời gian",
            list(
                PERIOD_PRESETS.keys()
            ),
            index=3,
            key="research_preset",
        )

        start_date = (
            pd.Timestamp(
                today
            )
            - pd.Timedelta(
                days=PERIOD_PRESETS[
                    preset
                ]
            )
        ).date()

        end_date = today

    else:

        c1, c2 = st.columns(2)

        with c1:

            start_date = st.date_input(
                "Từ ngày",
                value=(
                    pd.Timestamp(
                        today
                    )
                    - pd.Timedelta(
                        days=365
                    )
                ).date(),
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

        preset = "Tùy chọn"

    if start_date > end_date:

        st.error(
            "Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc."
        )

        return

    st.info(
        f"Mẫu yêu cầu: "
        f"{start_date.strftime('%d/%m/%Y')}"
        f" → "
        f"{end_date.strftime('%d/%m/%Y')}"
    )

    # ========================================================
    # RUN
    # ========================================================

    run_research = st.button(
        "🚀 Chạy toàn bộ nghiên cứu",
        type="primary",
        width="stretch",
        key="run_stock_research",
    )

    if run_research:

        st.session_state.pop(
            "stock_research_result",
            None,
        )

        # AI cũ không được dùng cho sample mới.
        st.session_state.pop(
            "research_ai_result",
            None,
        )

        st.session_state.pop(
            "research_ai_scope",
            None,
        )

        st.session_state.pop(
            "research_ai_question_used",
            None,
        )

        with st.status(
            "Đang nghiên cứu...",
            expanded=True,
        ) as status:

            st.write(
                "1/3 Đang tải dữ liệu kỹ thuật, dòng tiền, khối ngoại, "
                "tự doanh, thị trường và nhóm ngành..."
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

                status.update(
                    label="Không lấy được dữ liệu nghiên cứu",
                    state="error",
                )

                st.error(
                    "Không thể tải dữ liệu nghiên cứu."
                )

                st.code(
                    str(error)
                )

                return

            st.write(
                "2/3 Đã tải dữ liệu. Đang chuẩn hóa toàn bộ biến..."
            )

            sample = slice_date_range(
                research_data,
                start_date,
                end_date,
            )

            if sample.empty:

                status.update(
                    label="Không có phiên giao dịch",
                    state="error",
                )

                st.error(
                    "Không có phiên giao dịch trong khoảng đã chọn."
                )

                return

            st.write(
                f"Mẫu thực tế: {len(sample):,} quan sát."
            )

            st.write(
                "3/3 Đang chạy 1D, 5D, 20D + model + kiểm định..."
            )

            result = execute_full_research(
                sample
            )

            status.update(
                label="Đã hoàn tất nghiên cứu",
                state="complete",
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
    # LOAD SAVED RESULT
    # ========================================================

    result = st.session_state.get(
        "stock_research_result"
    )

    saved_symbol = st.session_state.get(
        "stock_research_symbol"
    )

    saved_dates = st.session_state.get(
        "stock_research_dates"
    )

    current_dates = (
        start_date,
        end_date,
    )

    if (
        result is None
        or saved_symbol != symbol
        or saved_dates != current_dates
    ):

        if result is not None:

            st.caption(
                "Mẫu hiện tại đã thay đổi. "
                "Bấm “Chạy toàn bộ nghiên cứu” để chạy mẫu mới."
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
    # SAMPLE SUMMARY
    # ========================================================

    research_data = result.get(
        "data"
    )

    actual_start, actual_end = actual_range(
        research_data
    )

    st.divider()

    st.header(
        "📚 Mẫu nghiên cứu"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Khoảng chọn",
            preset,
        )

    with b:

        st.metric(
            "Quan sát",
            f"{len(research_data):,}",
        )

    with c:

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

    with d:

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

    if len(research_data) < 60:

        st.error(
            "Mẫu dưới 60 quan sát. "
            "Kết luận định lượng cần cực kỳ thận trọng."
        )

    elif len(research_data) < 120:

        st.warning(
            "Mẫu đủ cho phân tích cơ bản nhưng chưa thật dài."
        )

    else:

        st.success(
            f"Mẫu có {len(research_data):,} quan sát thực tế."
        )

    # ========================================================
    # ALL VARIABLES
    # ========================================================

    all_features = result.get(
        "all_features",
        [],
    )

    groups = result.get(
        "groups",
        {},
    )

    st.header(
        "🧬 Toàn bộ biến nghiên cứu"
    )

    a, b, c, d, e, f = st.columns(6)

    with a:

        st.metric(
            "Tổng biến",
            len(all_features),
        )

    with b:

        st.metric(
            "Kỹ thuật",
            len(
                groups.get(
                    "Kỹ thuật",
                    [],
                )
            ),
        )

    with c:

        st.metric(
            "Dòng tiền",
            len(
                groups.get(
                    "Dòng tiền",
                    [],
                )
            ),
        )

    with d:

        st.metric(
            "Khối ngoại",
            len(
                groups.get(
                    "Khối ngoại",
                    [],
                )
            ),
        )

    with e:

        st.metric(
            "Tự doanh",
            len(
                groups.get(
                    "Tự doanh",
                    [],
                )
            ),
        )

    with f:

        st.metric(
            "Market + Sector",
            len(
                groups.get(
                    "Thị trường chung",
                    [],
                )
            )
            + len(
                groups.get(
                    "Nhóm ngành",
                    [],
                )
            ),
        )

    with st.expander(
        "Xem toàn bộ danh sách biến",
        expanded=False,
    ):

        for group_name, values in groups.items():

            st.markdown(
                f"**{group_name} ({len(values)})**"
            )

            if values:

                st.write(
                    ", ".join(
                        map(
                            str,
                            values,
                        )
                    )
                )

            else:

                st.caption(
                    "Không có dữ liệu nhóm này."
                )

    # ========================================================
    # EACH HORIZON
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
            f"📈 {horizon} — "
            f"{HORIZON_LABELS[horizon]}"
        )

        a, b, c, d = st.columns(4)

        with a:

            st.metric(
                "Quan sát",
                f"{item['observations']:,}",
            )

        with b:

            st.metric(
                "Train",
                f"{item['train']:,}",
            )

        with c:

            st.metric(
                "Test",
                f"{item['test']:,}",
            )

        with d:

            st.metric(
                "Số biến",
                f"{len(item['features_used']):,}",
            )

        # ----------------------------------------------------
        # Best factor BEFORE details
        # ----------------------------------------------------

        ranking = item.get(
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

            factor = safe_text(
                top.get(
                    "Biến"
                ),
                "Không xác định",
            )

            group = safe_text(
                top.get(
                    "Nhóm"
                ),
                "Không xác định",
            )

            score = num(
                top.get(
                    "Score"
                )
            )

            direction = safe_text(
                top.get(
                    "Quan hệ"
                ),
                "Không xác định",
            )

            if score is not None:

                st.success(
                    f"🎯 Yếu tố nổi bật nhất: **{factor}** "
                    f"({group}) · Score {score:.1f}/100 · {direction}"
                )

            else:

                st.info(
                    f"🎯 Yếu tố nổi bật nhất: "
                    f"**{factor}** ({group})"
                )

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

        render_forecast(
            item
        )

        # ----------------------------------------------------
        # Group
        # ----------------------------------------------------

        render_groups(
            item
        )

        # ----------------------------------------------------
        # Factors
        # ----------------------------------------------------

        render_factors(
            item
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        render_models(
            item
        )

        # ----------------------------------------------------
        # OLS
        # ----------------------------------------------------

        render_ols(
            item
        )

        # ----------------------------------------------------
        # VIF
        # ----------------------------------------------------

        render_vif(
            item
        )

        # ----------------------------------------------------
        # Tests
        # ----------------------------------------------------

        render_tests(
            item
        )

        # ----------------------------------------------------
        # Conclusion
        # ----------------------------------------------------

        render_conclusion(
            item
        )

    # ========================================================
    # CROSS HORIZON
    # ========================================================

    render_cross_horizon(
        result
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.divider()

    st.header(
        "🎯 Kết luận cuối cùng"
    )

    final_rows = []

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
            not isinstance(
                ranking,
                pd.DataFrame,
            )
            or ranking.empty
        ):
            continue

        top = ranking.iloc[
            0
        ]

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

        final_rows.append(
            {
                "Horizon": horizon,
                "Yếu tố": top.get(
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
                "Spearman": num(
                    top.get(
                        "Spearman"
                    )
                ),
                "p-value": num(
                    top.get(
                        "p-value"
                    )
                ),
                "Dự báo %": (
                    predicted * 100
                    if predicted is not None
                    else np.nan
                ),
                "Giá dự báo": predicted_price,
            }
        )

    if final_rows:

        summary = pd.DataFrame(
            final_rows
        )

        summary[
            "Score"
        ] = summary[
            "Score"
        ].round(1)

        summary[
            "Spearman"
        ] = summary[
            "Spearman"
        ].round(4)

        summary[
            "p-value"
        ] = summary[
            "p-value"
        ].round(5)

        summary[
            "Dự báo %"
        ] = summary[
            "Dự báo %"
        ].round(2)

        summary[
            "Giá dự báo"
        ] = summary[
            "Giá dự báo"
        ].round(0)

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

        factor_counts = (
            summary[
                "Yếu tố"
            ]
            .value_counts()
        )

        if not factor_counts.empty:

            common_factor = str(
                factor_counts.index[0]
            )

            common_count = int(
                factor_counts.iloc[0]
            )

            st.success(
                f"Yếu tố xuất hiện nhiều nhất giữa các horizon: "
                f"**{common_factor}** "
                f"({common_count}/{len(summary)} horizon)."
            )

    # ========================================================
    # AI RESEARCH READER
    # ========================================================

    render_research_ai(
        result=result,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    # ========================================================
    # FINAL WARNING
    # ========================================================

    st.warning(
        "Kết quả phản ánh quan hệ thống kê và khả năng dự báo "
        "trên mẫu đã chọn; không phải bằng chứng nhân quả và "
        "không phải cam kết giá tương lai."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_analysis():
    return render_stock_analysis()
