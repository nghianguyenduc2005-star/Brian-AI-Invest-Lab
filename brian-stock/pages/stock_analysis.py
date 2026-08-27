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

CACHE_TTL_DISPLAY = 300
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

MIN_OBSERVATIONS = 60

TRAIN_RATIO = 0.80

# OLS không nên nhét toàn bộ các biến cực kỳ tương quan
# (Close/SMA/EMA/MACD...) vào cùng một ma trận.
MAX_OLS_FEATURES = 15

# Ngưỡng tương quan để bỏ bớt biến trùng thông tin cho OLS.
OLS_CORR_LIMIT = 0.90

MAX_DISPLAY_FEATURES = 100


# ============================================================
# BASIC HELPERS
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

def normalize_index(
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

    work = normalize_index(
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


def actual_range(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return None, None

    work = normalize_index(
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

def feature_group(
    feature,
):
    name = str(
        feature
    )

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

        groups.setdefault(
            feature_group(feature),
            [],
        ).append(
            feature
        )

    return groups


# ============================================================
# RESEARCH TARGETS
# ============================================================

def ensure_targets(
    df,
):
    work = normalize_index(
        df
    )

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
# ALL FEATURES
# ============================================================

def get_all_features(
    df,
):
    if (
        df is None
        or df.empty
    ):
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

        name = str(
            column
        )

        if name in excluded:
            continue

        if name.lower().startswith(
            "target_"
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
# PREPARE X / Y
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

    # Median fill toàn bộ biến.
    for column in X.columns:

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

    valid = y.notna()

    X = X.loc[
        valid
    ]

    y = y.loc[
        valid
    ]

    if X.empty:
        return None

    # Không bỏ biến khỏi DATASET,
    # chỉ bỏ variance=0 khỏi matrix model.
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
                np.isfinite(variance)
                and variance > 0
            ):
                usable.append(
                    column
                )

        except Exception:
            continue

    if not usable:
        return None

    return (
        X[
            usable
        ].astype(float),
        y.astype(float),
    )


# ============================================================
# CORRELATION
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

        try:

            pearson = float(
                x.corr(
                    y,
                    method="pearson",
                )
            )

        except Exception:

            pearson = np.nan

        try:

            spearman = float(
                x.corr(
                    y,
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

    result = pd.DataFrame(
        rows
    )

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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# BUILD OLS-SAFE FEATURES
#
# Toàn bộ biến vẫn được giữ trong dataset.
# Chỉ ma trận OLS mới giảm các biến cực kỳ trùng nhau.
# ============================================================

def select_ols_features(
    X,
    y,
):
    if (
        X is None
        or X.empty
    ):
        return []

    corr = correlation_table(
        X,
        y,
    )

    if corr.empty:
        return list(
            X.columns[
                :MAX_OLS_FEATURES
            ]
        )

    ranked = corr[
        "Biến"
    ].tolist()

    selected = []

    for feature in ranked:

        if len(selected) >= MAX_OLS_FEATURES:
            break

        if not selected:

            selected.append(
                feature
            )
            continue

        okay = True

        for existing in selected:

            try:

                pair = abs(
                    float(
                        X[
                            feature
                        ].corr(
                            X[
                                existing
                            ]
                        )
                    )
                )

            except Exception:

                pair = 0.0

            if pair >= OLS_CORR_LIMIT:

                okay = False
                break

        if okay:

            selected.append(
                feature
            )

    if not selected:

        selected = list(
            X.columns[
                :MAX_OLS_FEATURES
            ]
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

        selected = select_ols_features(
            X_train,
            y_train,
        )

        if not selected:
            return {
                "model": None,
                "table": pd.DataFrame(),
                "features": [],
            }

        X = X_train[
            selected
        ].copy()

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=selected,
            index=X.index,
        )

        X_const = sm.add_constant(
            X_scaled,
            has_constant="add",
        )

        # HC3: robust SE, tránh một phần vấn đề
        # phương sai thay đổi.
        model = sm.OLS(
            y_train,
            X_const,
        ).fit(
            cov_type="HC3"
        )

        rows = []

        for feature in selected:

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
            pd.DataFrame(
                rows
            )
            .sort_values(
                "AbsBeta",
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
            "features": selected,
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
            "features": [],
            "r2": np.nan,
            "adj_r2": np.nan,
            "error": str(error),
        }


# ============================================================
# VIF — ONLY OLS SAFE FEATURES
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

        clean = clean.fillna(
            clean.median()
        )

        rows = []

        values = clean.values.astype(
            float
        )

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
                    "Nhóm": feature_group(
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
#
# All usable variables are included here.
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

    train_mean = float(
        np.mean(
            np.asarray(
                y_train,
                dtype=float,
            )
        )
    )

    baseline = np.repeat(
        train_mean,
        len(actual),
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

            fitted[
                name
            ] = model

        except Exception:
            continue

    if not rows:

        return (
            pd.DataFrame(),
            {},
        )

    return (
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
        ),
        fitted,
    )


# ============================================================
# PERMUTATION IMPORTANCE
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
            .reset_index(
                drop=True
            )
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

        importance = np.asarray(
            raw_model.feature_importances_,
            dtype=float,
        )

        table = pd.DataFrame(
            {
                "Biến": features,
                "TreeImportance": importance,
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

    # OLS
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

        result[
            "Beta"
        ] = np.nan

        result[
            "p-value"
        ] = np.nan

        result[
            "AbsBeta"
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

        result[
            "TreeImportance"
        ] = np.nan

    # --------------------------------------------------------
    # Percentile score
    # --------------------------------------------------------

    def pct(
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
    ] = pct(
        result[
            "AbsSpearman"
        ]
    )

    result[
        "BetaScore"
    ] = pct(
        result[
            "AbsBeta"
        ]
    )

    result[
        "PermutationScore"
    ] = pct(
        result[
            "AbsPermutation"
        ]
    )

    result[
        "TreeScore"
    ] = pct(
        result[
            "TreeImportance"
        ]
    )

    # --------------------------------------------------------
    # Composite
    # --------------------------------------------------------

    result[
        "Score"
    ] = (
        result[
            "CorrScore"
        ].fillna(0)
        * 0.20
        + result[
            "BetaScore"
        ].fillna(0)
        * 0.20
        + result[
            "PermutationScore"
        ].fillna(0)
        * 0.30
        + result[
            "TreeScore"
        ].fillna(0)
        * 0.30
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    def direction(
        row,
    ):
        beta = num(
            row.get(
                "Beta"
            )
        )

        spear = num(
            row.get(
                "Spearman"
            )
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
        direction,
        axis=1,
    )

    # --------------------------------------------------------
    # Significance
    # --------------------------------------------------------

    def significance(
        value,
    ):
        p = num(
            value
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
        significance
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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TESTS
# ============================================================

def run_tests(
    X_train,
    y_train,
    ols_result,
):
    tests = {}

    # ADF
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
        }

    except Exception:
        pass

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

    # JB
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

    # DW
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
            ),
        }

    except Exception:
        pass

    # Ljung Box
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
                lb["lb_stat"].iloc[-1]
            ),
            "p_value": float(
                lb["lb_pvalue"].iloc[-1]
            ),
        }

    except Exception:
        pass

    # Breusch Pagan
    try:

        import statsmodels.api as sm

        from statsmodels.stats.diagnostic import (
            het_breuschpagan,
        )

        ols_features = (
            ols_result.get(
                "features",
                [],
            )
        )

        X_bp = X_train[
            [
                column
                for column
                in ols_features
                if column in X_train.columns
            ]
        ]

        X_bp = sm.add_constant(
            X_bp,
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

    return tests


# ============================================================
# EXECUTE ONE HORIZON
# ============================================================

def execute_one_horizon(
    df,
    features,
    horizon,
):
    xy = prepare_xy(
        df,
        features,
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

    # Correlation ALL FEATURES
    correlations = correlation_table(
        X,
        y,
    )

    # OLS SAFE SUBSET
    ols = run_safe_ols(
        X_train,
        y_train,
    )

    ols_table = ols.get(
        "table",
        pd.DataFrame(),
    )

    # ALL FEATURES MODELS
    models, fitted = run_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Best model
    best_name = None
    best_model = None

    if (
        not models.empty
    ):

        best_name = str(
            models.iloc[
                0
            ][
                "Mô hình"
            ]
        )

        best_model = fitted.get(
            best_name
        )

    # Permutation
    permutation = pd.DataFrame()

    if best_model is not None:

        permutation = run_permutation(
            best_model,
            X_test,
            y_test,
        )

    # Tree
    tree = pd.DataFrame()

    if best_model is not None:

        tree = run_tree_importance(
            best_model,
            X_train.columns,
        )

    # Ranking
    ranking = build_factor_ranking(
        correlations,
        ols_table,
        permutation,
        tree,
    )

    # Group
    groups = group_summary(
        ranking
    )

    # VIF only safe OLS variables
    vif = pd.DataFrame()

    ols_features = ols.get(
        "features",
        [],
    )

    if ols_features:

        vif = run_vif(
            X_train[
                ols_features
            ]
        )

    # Tests
    tests = run_tests(
        X_train,
        y_train,
        ols,
    )

    # ========================================================
    # FORECAST
    # ========================================================

    current_price = num(
        df[
            "Close"
        ].iloc[
            -1
        ]
    )

    predicted_return = None
    predicted_price = None

    if best_model is not None:

        try:

            latest = (
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

            for column in latest.columns:

                median = latest[
                    column
                ].median()

                if pd.isna(
                    median
                ):
                    median = 0.0

                latest[
                    column
                ] = latest[
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

    # Model quality
    quality = "Chưa đánh giá"

    if not models.empty:

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
    ttl=CACHE_TTL_RESEARCH,
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

    # EVERYTHING numeric = dataset features
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
            "error": (
                "Không có horizon nào đủ dữ liệu."
            ),
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
    ttl=CACHE_TTL_DISPLAY,
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
# RENDER FORECAST
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
            str(
                item.get(
                    "best_model",
                    "—",
                )
            ),
        )

    if predicted_return is not None:

        if predicted_return > 0:

            st.success(
                f"Mô hình nghiêng về **TĂNG** "
                f"{predicted_return * 100:+.2f}% "
                f"trong {item['horizon']}."
            )

        elif predicted_return < 0:

            st.warning(
                f"Mô hình nghiêng về **GIẢM** "
                f"{predicted_return * 100:+.2f}% "
                f"trong {item['horizon']}."
            )

        else:

            st.info(
                "Mô hình nghiêng về đi ngang."
            )


# ============================================================
# RENDER GROUPS
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

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
    )

    strongest = groups.iloc[
        0
    ]

    st.info(
        f"Nhóm nổi bật nhất: "
        f"**{strongest['Nhóm']}** "
        f"với Score TB "
        f"{strongest['Score_TB']:.1f}/100."
    )


# ============================================================
# RENDER FACTORS
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
        "🏆 Yếu tố ảnh hưởng / dự báo"
    )

    show = ranking.head(
        MAX_DISPLAY_FEATURES
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
        c
        for c in columns
        if c in show.columns
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

            show[
                column
            ] = pd.to_numeric(
                show[
                    column
                ],
                errors="coerce",
            )

    st.dataframe(
        show[
            columns
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "#### 🧠 5 yếu tố nổi bật nhất"
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

        score = num(
            row.get(
                "Score"
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

        pvalue = num(
            row.get(
                "p-value"
            )
        )

        text = (
            f"**{feature}** "
            f"({group})"
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

        if pvalue is not None:

            text += (
                f" · p-value {pvalue:.5f}"
            )

        st.markdown(
            "• " + text
        )


# ============================================================
# RENDER MODELS
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
        "RMSE",
        "R²",
        "Baseline_RMSE",
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
        show,
        width="stretch",
        hide_index=True,
    )

    best = models.iloc[
        0
    ]

    st.info(
        f"Model tốt nhất theo RMSE: "
        f"**{best['Mô hình']}** · "
        f"RMSE {best['RMSE']:.6f} · "
        f"R² {best['R²']:.4f}"
    )


# ============================================================
# RENDER OLS
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
            "OLS không chạy được."
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

        st.dataframe(
            table.head(
                30
            ),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "OLS dùng tập biến đã giảm đa cộng tuyến; "
        "toàn bộ biến vẫn được giữ trong dataset và các model phi tuyến."
    )


# ============================================================
# RENDER VIF
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
        "🔗 VIF"
    )

    st.dataframe(
        vif.head(30),
        width="stretch",
        hide_index=True,
    )

    finite = vif[
        "VIF"
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    max_vif = num(
        finite.max()
    )

    if (
        max_vif is not None
        and max_vif >= 10
    ):

        st.warning(
            f"VIF cao nhất {max_vif:.2f}: "
            "đa cộng tuyến mạnh."
        )

    elif (
        max_vif is not None
        and max_vif >= 5
    ):

        st.info(
            f"VIF cao nhất {max_vif:.2f}: "
            "có đa cộng tuyến đáng chú ý."
        )

    else:

        st.success(
            "VIF của tập OLS an toàn."
        )


# ============================================================
# RENDER TESTS
# ============================================================

def render_tests(
    item,
):
    tests = item.get(
        "tests",
        {},
    )

    if not tests:
        return

    st.subheader(
        "🧪 Kiểm định"
    )

    # ADF
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
            "ADF",
            expanded=False,
        ):

            st.write(
                f"Statistic: {format_number(adf.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: {format_number(p, 6)}"
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

    # JB
    jb = tests.get(
        "Jarque-Bera"
    )

    if isinstance(
        jb,
        dict,
    ):

        p = num(
            jb.get(
                "p_value"
            )
        )

        with st.expander(
            "Jarque-Bera",
            expanded=False,
        ):

            st.write(
                f"Statistic: {format_number(jb.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: {format_number(p, 6)}"
            )

            if (
                p is not None
                and p < 0.05
            ):

                st.warning(
                    "Phần dư không phù hợp tốt với chuẩn."
                )

            else:

                st.success(
                    "Chưa có bằng chứng mạnh chống lại chuẩn."
                )

    # BP
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

        with st.expander(
            "Breusch-Pagan",
            expanded=False,
        ):

            st.write(
                f"Statistic: {format_number(bp.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: {format_number(p, 6)}"
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
                    "Chưa thấy phương sai thay đổi rõ."
                )

    # DW
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

        with st.expander(
            "Durbin-Watson",
            expanded=False,
        ):

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
                        "Không thấy tự tương quan mạnh."
                    )

                elif value < 1.5:

                    st.warning(
                        "Có dấu hiệu tự tương quan dương."
                    )

                else:

                    st.warning(
                        "Có dấu hiệu tự tương quan âm."
                    )

    # Ljung Box
    lb = tests.get(
        "Ljung-Box"
    )

    if isinstance(
        lb,
        dict,
    ):

        p = num(
            lb.get(
                "p_value"
            )
        )

        with st.expander(
            "Ljung-Box",
            expanded=False,
        ):

            st.write(
                f"Lag: {lb.get('lag', '—')}"
            )

            st.write(
                f"Statistic: {format_number(lb.get('statistic'), 5)}"
            )

            st.write(
                f"p-value: {format_number(p, 6)}"
            )

            if (
                p is not None
                and p < 0.05
            ):

                st.warning(
                    "Còn tự tương quan trong phần dư."
                )

            else:

                st.success(
                    "Chưa thấy tự tương quan phần dư rõ."
                )


# ============================================================
# RENDER HORIZON CONCLUSION
# ============================================================

def render_conclusion(
    item,
):
    ranking = item.get(
        "ranking"
    )

    st.subheader(
        "🧠 Kết luận"
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

    factor = str(
        top.get(
            "Biến"
        )
    )

    group = str(
        top.get(
            "Nhóm"
        )
    )

    score = num(
        top.get(
            "Score"
        )
    )

    direction = str(
        top.get(
            "Quan hệ",
            "Không xác định",
        )
    )

    p = num(
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
        f", quan hệ {direction.lower()}."
    )

    st.markdown(
        "• " + text
    )

    if (
        p is not None
        and p < 0.05
    ):

        st.markdown(
            f"• OLS cho thấy **{factor}** "
            f"có ý nghĩa thống kê ở mức 5%."
        )

    else:

        st.markdown(
            f"• **{factor}** nổi bật về tín hiệu "
            "dự báo nhưng chưa đủ bằng chứng để xem là quan hệ nhân quả."
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

        model_name = str(
            best.get(
                "Mô hình"
            )
        )

        st.markdown(
            f"• Model tốt nhất: **{model_name}** "
            f"(RMSE {rmse:.6f}, R² {r2:.4f})."
            if (
                rmse is not None
                and r2 is not None
            )
            else
            f"• Model tốt nhất: **{model_name}**."
        )

        if r2 is not None:

            if r2 < 0:

                st.warning(
                    "Model chưa vượt benchmark trên tập test."
                )

            elif r2 < 0.10:

                st.info(
                    "Khả năng giải thích/dự báo còn thấp."
                )

            elif r2 < 0.25:

                st.info(
                    "Có tín hiệu nhưng sức giải thích vừa phải."
                )

            else:

                st.success(
                    "Model có sức giải thích đáng chú ý trên mẫu test."
                )

    forecast = item.get(
        "forecast",
        {},
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

    if (
        predicted is not None
        and predicted_price is not None
    ):

        st.markdown(
            f"• Dự báo {item['horizon']}: "
            f"**{predicted * 100:+.2f}%** "
            f"→ giá khoảng **{predicted_price:,.0f} đồng**."
        )


# ============================================================
# CROSS HORIZON
# ============================================================

def render_cross_horizon(
    result,
):
    st.divider()

    st.header(
        "🏆 Yếu tố nhất quán 1D / 5D / 20D"
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

    if not stable.empty:

        best = stable.iloc[
            0
        ]

        st.success(
            f"Yếu tố nhất quán nhất: "
            f"**{best['Biến']}** "
            f"({int(best['Số horizon'])} horizon)."
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
        "thị trường chung, nhóm ngành và nghiên cứu định lượng."
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

        symbol_clean = normalize_symbol(
            symbol_input
        )

        if not symbol_clean:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

            return

        st.session_state[
            "stock_analysis_symbol"
        ] = symbol_clean

        # Xóa research cũ.
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
    #
    # CHỈ 1Y DISPLAY.
    # KHÔNG LOAD MULTIFACTOR Ở ĐÂY.
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

    snapshot = market_snapshot(
        display_data
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

    last = display_data.iloc[
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
    # STOCK
    # ========================================================

    st.subheader(
        f"📈 {display_symbol(symbol)}"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Giá",
            format_price(price),
        )

    with b:

        st.metric(
            "Thay đổi 1D",
            format_percent(change),
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

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "MA20",
            format_price(sma20),
        )

    with b:

        st.metric(
            "MA50",
            format_price(sma50),
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
    # RESEARCH
    #
    # QUAN TRỌNG:
    # KHÔNG LOAD MULTIFACTOR CHO ĐẾN KHI BẤM BUTTON.
    # ========================================================

    st.divider()

    st.header(
        "🧪 Nghiên cứu định lượng"
    )

    st.write(
        "Chọn mẫu trước. Hệ thống chỉ tải dữ liệu "
        "multifactor khi bấm chạy nghiên cứu."
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
            "Ngày bắt đầu phải nhỏ hơn ngày kết thúc."
        )

        return

    st.info(
        f"Mẫu: "
        f"{start_date.strftime('%d/%m/%Y')}"
        f" → "
        f"{end_date.strftime('%d/%m/%Y')}"
    )

    # ========================================================
    # RUN RESEARCH
    # ========================================================

    run_research = st.button(
        "🚀 Chạy toàn bộ nghiên cứu",
        type="primary",
        width="stretch",
        key="run_stock_research",
    )

    if run_research:

        # Xóa result cũ nếu đổi khoảng.
        st.session_state.pop(
            "stock_research_result",
            None,
        )

        with st.status(
            "Đang nghiên cứu...",
            expanded=True,
        ) as status:

            st.write(
                "1/4 Đang lấy dữ liệu multifactor..."
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
                    label="Nghiên cứu thất bại",
                    state="error",
                )

                st.error(
                    "Không lấy được dữ liệu nghiên cứu."
                )

                st.code(
                    str(error)
                )

                return

            st.write(
                "2/4 Đã lấy dữ liệu. Đang chuẩn hóa mẫu..."
            )

            sample = slice_date_range(
                research_data,
                start_date,
                end_date,
            )

            if sample.empty:

                status.update(
                    label="Không có dữ liệu",
                    state="error",
                )

                st.error(
                    "Không có phiên giao dịch trong khoảng chọn."
                )

                return

            st.write(
                f"3/4 Mẫu thực tế: {len(sample):,} quan sát."
            )

            st.write(
                "4/4 Đang chạy model và kiểm định..."
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
    # RESULT
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

    # Không hiển thị result sai symbol / sai khoảng.
    if (
        result is None
        or saved_symbol != symbol
        or saved_dates != (
            start_date,
            end_date,
        )
    ):

        if result is not None:

            st.caption(
                "Đang chọn mẫu mới. Bấm "
                "“Chạy toàn bộ nghiên cứu” để chạy lại."
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

    actual_start, actual_end = (
        actual_range(
            research_data
        )
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

        st.warning(
            "Mẫu dưới 60 quan sát: kết luận định lượng cần thận trọng."
        )

    elif len(research_data) < 120:

        st.info(
            "Mẫu đủ cho nghiên cứu cơ bản."
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

    st.header(
        "🧬 Toàn bộ biến"
    )

    groups = result.get(
        "groups",
        {},
    )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Tổng biến",
            len(all_features),
        )

    with b:

        st.metric(
            "Kỹ thuật + dòng tiền",
            len(
                groups.get(
                    "Kỹ thuật",
                    [],
                )
            )
            + len(
                groups.get(
                    "Dòng tiền",
                    [],
                )
            ),
        )

    with c:

        st.metric(
            "Market + Sector + Flow",
            sum(
                len(
                    groups.get(
                        name,
                        [],
                    )
                )
                for name in [
                    "Khối ngoại",
                    "Tự doanh",
                    "Thị trường chung",
                    "Nhóm ngành",
                ]
            ),
        )

    with st.expander(
        "Xem toàn bộ biến",
        expanded=False,
    ):

        for group_name, values in groups.items():

            st.markdown(
                f"**{group_name} ({len(values)})**"
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
                    "Không có dữ liệu nhóm này."
                )

    # ========================================================
    # HORIZON RESULTS
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
                "Biến model",
                f"{len(item['features_used']):,}",
            )

        render_forecast(
            item
        )

        render_groups(
            item
        )

        render_factors(
            item
        )

        render_models(
            item
        )

        render_ols(
            item
        )

        render_vif(
            item
        )

        render_tests(
            item
        )

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
    # FINAL
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

        pred = num(
            forecast.get(
                "predicted_return"
            )
        )

        price_forecast = num(
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
                "Dự báo %": (
                    pred * 100
                    if pred is not None
                    else np.nan
                ),
                "Giá dự báo": price_forecast,
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

        # ----------------------------------------------------
        # Most consistent factor
        # ----------------------------------------------------

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
                f"Yếu tố xuất hiện nhiều nhất trong "
                f"các horizon: **{common_factor}** "
                f"({common_count}/"
                f"{len(summary)} horizon)."
            )

    st.warning(
        "Đây là kết quả thống kê và dự báo trên mẫu đã chọn, "
        "không phải bằng chứng nhân quả hay cam kết giá tương lai."
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_analysis():
    render_stock_analysis()
