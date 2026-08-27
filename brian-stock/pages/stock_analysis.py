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
CACHE_TTL_AI = 1800

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

# OLS không nhét toàn bộ biến rất tương quan vào cùng ma trận.
MAX_OLS_FEATURES = 15
OLS_CORR_LIMIT = 0.90

MAX_DISPLAY_FEATURES = 100

# Gemini hiện tại.
GEMINI_MODEL = "gemini-3.7-flash"


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


def text(
    value: Any,
    default: str = "",
):
    if value is None:
        return default

    try:
        value = str(value).strip()
    except Exception:
        return default

    return value or default


def format_price(value):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:,.0f} đồng"


def format_percent(value):
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


def format_volume(value):
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

def rsi_status(value):
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


def macd_status(value):
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

def normalize_index(df):
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

    work = normalize_index(df)

    if work.empty:
        return work

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


def actual_range(df):
    if (
        df is None
        or df.empty
    ):
        return None, None

    work = normalize_index(df)

    if work.empty:
        return None, None

    return (
        work.index.min().date(),
        work.index.max().date(),
    )


# ============================================================
# FEATURE GROUP
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
        ]
    ):
        return "Dòng tiền"

    return "Kỹ thuật"


def group_features(features):
    groups = {
        "Kỹ thuật": [],
        "Dòng tiền": [],
        "Khối ngoại": [],
        "Tự doanh": [],
        "Thị trường chung": [],
        "Nhóm ngành": [],
    }

    for feature in features:

        group = feature_group(
            feature
        )

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

def ensure_targets(df):
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
# ALL FEATURES
# ============================================================

def get_all_features(df):
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

        name = str(column)

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

    # Median fill.
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

    # Chỉ loại variance=0 khỏi model matrix.
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

        x2 = x.loc[
            valid
        ]

        y2 = yy.loc[
            valid
        ]

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
# OLS FEATURE SELECTION
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

        if len(
            selected
        ) >= MAX_OLS_FEATURES:
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
                "r2": np.nan,
                "adj_r2": np.nan,
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
            "error": str(
                error
            ),
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

        clean = clean.fillna(
            clean.median()
        )

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

    train_values = np.asarray(
        y_train,
        dtype=float,
    )

    train_mean = float(
        np.mean(
            train_values
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
                np.sqrt(
                    mse
                )
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

def _percentile(
    series,
):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if values.notna().sum() <= 1:

        return pd.Series(
            0.0,
            index=values.index,
        )

    return (
        values.rank(
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

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PERMUTATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TREE
    # --------------------------------------------------------

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
    # Scores
    # --------------------------------------------------------

    result[
        "CorrScore"
    ] = _percentile(
        result[
            "AbsSpearman"
        ]
    )

    result[
        "BetaScore"
    ] = _percentile(
        result[
            "AbsBeta"
        ]
    )

    result[
        "PermutationScore"
    ] = _percentile(
        result[
            "AbsPermutation"
        ]
    )

    result[
        "TreeScore"
    ] = _percentile(
        result[
            "TreeImportance"
        ]
    )

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
        row
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

        if beta is not None:

            value = beta

        else:

            value = spear

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
        value
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

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    try:

        from statsmodels.tsa.stattools import (
            adfuller,
        )

        values = np.asarray(
            y_train,
            dtype=float,
        )

        if len(values) >= 20:

            result = adfuller(
                values,
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

    # --------------------------------------------------------
    # RESIDUALS
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
    # JARQUE BERA
    # --------------------------------------------------------

    try:

        from statsmodels.stats.stattools import (
            jarque_bera,
        )

        result = jarque_bera(
            residuals
        )

        tests[
            "Jarque-Bera"
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

    # --------------------------------------------------------
    # DURBIN WATSON
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
            ),
        }

    except Exception:
        pass

    # --------------------------------------------------------
    # LJUNG BOX
    # --------------------------------------------------------

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

        result = acorr_ljungbox(
            residuals,
            lags=[lag],
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

    # --------------------------------------------------------
    # BREUSCH PAGAN
    # --------------------------------------------------------

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

        selected = [
            column
            for column
            in ols_features
            if column in X_train.columns
        ]

        if selected:

            X_bp = X_train[
                selected
            ]

            X_bp = sm.add_constant(
                X_bp,
                has_constant="add",
            )

            result = het_breuschpagan(
                residuals,
                X_bp,
            )

            tests[
                "Breusch-Pagan"
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

    # --------------------------------------------------------
    # WHITE
    #
    # Chỉ dùng OLS features để tránh bùng ma trận.
    # --------------------------------------------------------

    try:

        import statsmodels.api as sm

        from statsmodels.stats.diagnostic import (
            het_white,
        )

        ols_features = (
            ols_result.get(
                "features",
                [],
            )
        )

        selected = [
            column
            for column
            in ols_features
            if column in X_train.columns
        ]

        selected = selected[
            :10
        ]

        if selected:

            X_white = sm.add_constant(
                X_train[
                    selected
                ],
                has_constant="add",
            )

            result = het_white(
                residuals,
                X_white,
            )

            tests[
                "White"
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

    return tests


# ============================================================
# ONE HORIZON
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

    # --------------------------------------------------------
    # Correlation: ALL FEATURES
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
    # ALL FEATURES MODELS
    # --------------------------------------------------------

    models, fitted = run_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

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
            ][
                "Mô hình"
            ]
        )

        best_model = fitted.get(
            best_name
        )

    # --------------------------------------------------------
    # PERMUTATION
    # --------------------------------------------------------

    permutation = pd.DataFrame()

    if best_model is not None:

        permutation = run_permutation(
            best_model,
            X_test,
            y_test,
        )

    # --------------------------------------------------------
    # TREE
    # --------------------------------------------------------

    tree = pd.DataFrame()

    if best_model is not None:

        tree = run_tree_importance(
            best_model,
            X_train.columns,
        )

    # --------------------------------------------------------
    # FACTOR RANKING
    # --------------------------------------------------------

    ranking = build_factor_ranking(
        correlations,
        ols_table,
        permutation,
        tree,
    )

    # --------------------------------------------------------
    # GROUP
    # --------------------------------------------------------

    groups = group_summary(
        ranking
    )

    # --------------------------------------------------------
    # VIF
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TESTS
    # --------------------------------------------------------

    tests = run_tests(
        X_train,
        y_train,
        ols,
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # QUALITY
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
# GROUPS
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
# FACTORS
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
        column
        for column in columns
        if column in show.columns
    ]

    st.dataframe(
        show[
            columns
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "#### 🎯 5 yếu tố nổi bật nhất"
    )

    for _, row in ranking.head(
        5
    ).iterrows():

        feature = text(
            row.get(
                "Biến"
            )
        )

        group = text(
            row.get(
                "Nhóm"
            )
        )

        score = num(
            row.get(
                "Score"
            )
        )

        direction = text(
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

        pvalue = num(
            row.get(
                "p-value"
            )
        )

        line = (
            f"**{feature}** "
            f"({group})"
        )

        if score is not None:

            line += (
                f" · Score {score:.1f}"
            )

        line += (
            f" · {direction}"
        )

        if spear is not None:

            line += (
                f" · Spearman {spear:+.4f}"
            )

        if pvalue is not None:

            line += (
                f" · p-value {pvalue:.5f}"
            )

        st.markdown(
            "• " + line
        )


# ============================================================
# MODELS
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

    st.dataframe(
        models,
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

    st.info(
        f"Model tốt nhất theo RMSE: "
        f"**{best['Mô hình']}** · "
        f"RMSE {rmse:.6f} · "
        f"R² {r2:.4f}"
        if (
            rmse is not None
            and r2 is not None
        )
        else
        f"Model tốt nhất: **{best['Mô hình']}**"
    )


# ============================================================
# OLS
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
            table,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "OLS dùng tập biến an toàn hơn để tránh ma trận suy hạng; "
        "các biến còn lại vẫn được giữ trong phân tích correlation "
        "và machine learning."
    )


# ============================================================
# VIF
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
            "VIF của tập OLS không cho thấy đa cộng tuyến quá mạnh."
        )


# ============================================================
# TESTS
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
            "ADF — Augmented Dickey-Fuller",
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
                    "Bác bỏ nghiệm đơn vị ở mức 5%."
                )

            else:

                st.info(
                    "Chưa đủ bằng chứng bác bỏ nghiệm đơn vị."
                )

    # --------------------------------------------------------
    # JB
    # --------------------------------------------------------

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

                st.success(
                    "Chưa có bằng chứng mạnh chống lại giả định chuẩn."
                )

    # --------------------------------------------------------
    # BP
    # --------------------------------------------------------

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
                    "Chưa thấy phương sai thay đổi rõ."
                )

    # --------------------------------------------------------
    # WHITE
    # --------------------------------------------------------

    white = tests.get(
        "White"
    )

    if isinstance(
        white,
        dict,
    ):

        p = num(
            white.get(
                "p_value"
            )
        )

        with st.expander(
            "White Test",
            expanded=False,
        ):

            st.write(
                f"Statistic: "
                f"{format_number(white.get('statistic'), 5)}"
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
                    "Chưa thấy phương sai thay đổi rõ."
                )

    # --------------------------------------------------------
    # DW
    # --------------------------------------------------------

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
                        "Chưa thấy tự tương quan mạnh."
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
    # LJUNG BOX
    # --------------------------------------------------------

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
                    "Còn tự tương quan trong phần dư."
                )

            else:

                st.success(
                    "Chưa thấy tự tương quan phần dư rõ."
                )


# ============================================================
# CONCLUSION
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
            "Chưa đủ dữ liệu để kết luận."
        )

        return

    top = ranking.iloc[
        0
    ]

    factor = text(
        top.get(
            "Biến"
        )
    )

    group = text(
        top.get(
            "Nhóm"
        )
    )

    score = num(
        top.get(
            "Score"
        )
    )

    direction = text(
        top.get(
            "Quan hệ"
        ),
        "Không xác định",
    )

    spear = num(
        top.get(
            "Spearman"
        )
    )

    p = num(
        top.get(
            "p-value"
        )
    )

    line = (
        f"**Yếu tố nổi bật nhất:** "
        f"{factor} ({group})"
    )

    if score is not None:

        line += (
            f" · Score {score:.1f}/100"
        )

    line += (
        f" · {direction}"
    )

    if spear is not None:

        line += (
            f" · Spearman {spear:+.4f}"
        )

    if p is not None:

        line += (
            f" · p-value {p:.5f}"
        )

    st.markdown(
        "• " + line
    )

    if (
        p is not None
        and p < 0.05
    ):

        st.markdown(
            f"• **{factor}** có bằng chứng "
            "thống kê ở mức 5% trong OLS."
        )

    else:

        st.markdown(
            f"• **{factor}** có tín hiệu nổi bật "
            "nhưng chưa đủ bằng chứng để xem là quan hệ nhân quả."
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

        model_name = text(
            best.get(
                "Mô hình"
            )
        )

        if (
            rmse is not None
            and r2 is not None
        ):

            st.markdown(
                f"• Model tốt nhất: **{model_name}** "
                f"(RMSE {rmse:.6f}, R² {r2:.4f})."
            )

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

        else:

            st.markdown(
                f"• Model tốt nhất: **{model_name}**."
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
            f"→ giá khoảng "
            f"**{predicted_price:,.0f} đồng**."
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
            f"({int(best['Số horizon'])} horizon)."
        )


# ============================================================
# RESEARCH AI
# ============================================================

CACHE_TTL_AI = 1800

RESEARCH_AI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


def get_gemini_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    if key is None:
        return None

    key = str(key).strip()

    return key if key else None


def _safe_text(value, default=""):
    if value is None:
        return default

    try:
        value = str(value).strip()
    except Exception:
        return default

    return value if value else default


def dataframe_to_ai_text(
    df,
    columns=None,
    limit=15,
):
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
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

    work = work.head(limit).copy()

    for column in work.columns:

        if pd.api.types.is_numeric_dtype(
            work[column]
        ):

            work[column] = pd.to_numeric(
                work[column],
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
        "=== NGHIÊN CỨU ĐỊNH LƯỢNG ==="
    )

    blocks.append(
        f"Mã cổ phiếu: {symbol}"
    )

    blocks.append(
        f"Khoảng dữ liệu: "
        f"{start_date} → {end_date}"
    )

    all_features = result.get(
        "all_features",
        [],
    )

    blocks.append(
        f"Tổng số biến nghiên cứu: "
        f"{len(all_features)}"
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

        # ----------------------------------------------------
        # FACTORS
        # ----------------------------------------------------

        blocks.append(
            "\nTOP YẾU TỐ"
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
            "\nNHÓM YẾU TỐ"
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
                "\nOLS"
            )

            blocks.append(
                f"R²: "
                f"{num(ols.get('r2'))}"
            )

            blocks.append(
                f"Adjusted R²: "
                f"{num(ols.get('adj_r2'))}"
            )

            blocks.append(
                "Biến OLS: "
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
            "\nHỆ SỐ OLS"
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
        # MODELS
        # ----------------------------------------------------

        blocks.append(
            "\nHIỆU SUẤT MODEL"
        )

        blocks.append(
            dataframe_to_ai_text(
                item.get(
                    "models"
                ),
                [
                    "Mô hình",
                    "MAE",
                    "RMSE",
                    "R²",
                    "Baseline_RMSE",
                    "So_baseline",
                ],
                10,
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
                15,
            )
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        tests = item.get(
            "tests",
            {},
        )

        if tests:

            blocks.append(
                "\nKIỂM ĐỊNH"
            )

            for test_name, test_value in tests.items():

                blocks.append(
                    f"{test_name}: "
                    f"{test_value}"
                )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        forecast = item.get(
            "forecast",
            {},
        )

        blocks.append(
            "\nDỰ BÁO"
        )

        blocks.append(
            f"Giá hiện tại: "
            f"{num(forecast.get('current_price'))}"
        )

        blocks.append(
            f"Lợi suất dự báo: "
            f"{num(forecast.get('predicted_return'))}"
        )

        blocks.append(
            f"Giá dự báo: "
            f"{num(forecast.get('predicted_price'))}"
        )

        blocks.append(
            f"Model tốt nhất: "
            f"{item.get('best_model')}"
        )

    return "\n".join(
        blocks
    )


def build_research_ai_prompt(
    context,
    user_question="",
):
    question = _safe_text(
        user_question
    )

    if question:

        question_text = f"""
CÂU HỎI CỦA NGƯỜI DÙNG:
{question}
""".strip()

    else:

        question_text = """
Không có câu hỏi riêng.
Hãy tự đọc nghiên cứu và chỉ ra những điểm quan trọng nhất.
""".strip()

    return f"""
Bạn là Brian AI, chuyên viên đọc nghiên cứu định lượng
cổ phiếu Việt Nam.

Bạn KHÔNG chạy lại mô hình.
Bạn CHỈ đọc các kết quả thống kê đã được cung cấp.

{question_text}

============================================================
MỤC TIÊU
============================================================

Người đọc phải hiểu nhanh:

1. Giá/lợi suất cổ phiếu đang có quan hệ mạnh nhất với yếu tố nào?
2. Yếu tố đó cùng chiều hay ngược chiều?
3. Bằng chứng thống kê mạnh hay yếu?
4. 1D, 5D, 20D có cho kết quả giống nhau không?
5. Model có dự báo tốt hơn benchmark không?
6. Có vấn đề về VIF, heteroskedasticity hoặc autocorrelation không?
7. Nói đơn giản nghiên cứu này đang chỉ ra điều gì?

============================================================
QUY TẮC
============================================================

- Chỉ dùng số liệu đã cung cấp.
- Không bịa thêm số liệu.
- Không bịa thêm biến.
- Không biến correlation thành quan hệ nhân quả.
- Không nói "X làm giá tăng" nếu chỉ có bằng chứng tương quan.
- p-value < 0.05 mới được gọi là có ý nghĩa thống kê ở mức 5%.
- Nếu R² test âm, phải nói rõ model kém benchmark.
- Nếu VIF cao, phải cảnh báo đa cộng tuyến.
- Nếu BP/White có p < 0.05, nói có dấu hiệu phương sai thay đổi.
- Nếu DW/Ljung-Box cho thấy tự tương quan, phải nói.
- Không liệt kê hàng chục biến.
- Chọn 3–5 yếu tố quan trọng nhất.
- Ưu tiên yếu tố được nhiều phương pháp cùng xác nhận.
- Ưu tiên yếu tố ổn định qua nhiều horizon.
- Viết tiếng Việt tự nhiên, ngắn gọn.
- Không viết như giáo trình.
- Không khuyến nghị mua/bán.

============================================================
ĐỊNH DẠNG
============================================================

# 🧠 Brian AI — Đọc nghiên cứu

## Kết luận chính

Viết 3–4 câu.

Phải nói thẳng:
- yếu tố nào nổi bật nhất
- mức độ bằng chứng
- model có đáng tin không

## 🎯 Giá đang phụ thuộc vào gì?

Chọn tối đa 5 biến.

Mỗi biến:

**Tên biến — Nhóm**
- Quan hệ:
- Spearman:
- Beta:
- p-value:
- Ý nghĩa:

## 📈 1D / 5D / 20D

Viết ngắn:
- 1D:
- 5D:
- 20D:
- Yếu tố nhất quán nhất:

## 🤖 Model

Nêu:
- Model tốt nhất
- RMSE
- R²
- So với baseline
- Có đáng tin cho dự báo hay không

## ⚠️ Điểm cần lưu ý

Chỉ nêu các vấn đề thật sự xuất hiện trong số liệu.

## Nói đơn giản

Viết 3–5 câu như đang giải thích cho một người đầu tư
không chuyên thống kê.

============================================================
DỮ LIỆU NGHIÊN CỨU
============================================================

{context}
""".strip()


def _call_research_ai_model(
    api_key,
    model,
    prompt,
):
    try:

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

        output = _safe_text(
            getattr(
                response,
                "text",
                "",
            )
        )

        if not output:

            raise RuntimeError(
                "Model không trả về nội dung."
            )

        return output

    except Exception as error:

        raise error


def _is_retryable_ai_error(
    error,
):
    message = str(
        error
    ).lower()

    retry_tokens = [
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
        token in message
        for token in retry_tokens
    )


def _is_fatal_ai_error(
    error,
):
    message = str(
        error
    ).lower()

    fatal_tokens = [
        "401",
        "403",
        "invalid api key",
        "api key not valid",
        "permission denied",
        "unauthenticated",
        "quota exceeded",
        "billing",
    ]

    return any(
        token in message
        for token in fatal_tokens
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
                "Thiếu GEMINI_API_KEY "
                "trong Streamlit Secrets."
            ),
        }

    prompt = build_research_ai_prompt(
        context,
        question,
    )

    errors = []

    for model in RESEARCH_AI_MODELS:

        try:

            output = _call_research_ai_model(
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

            # Lỗi key/quyền/quota:
            # dừng ngay.
            if _is_fatal_ai_error(
                error
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": error_text,
                }

            # 503 / high demand:
            # thử model kế tiếp.
            if _is_retryable_ai_error(
                error
            ):
                continue

            # Lỗi khác:
            # không spam API.
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
            "Các model AI hiện tại đều không khả dụng.\n\n"
            + "\n".join(
                errors
            )
        ),
    }


@st.cache_data(
    ttl=CACHE_TTL_AI,
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
        "ngắn gọn yếu tố nào đang liên hệ mạnh nhất với lợi suất."
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

    # ========================================================
    # CUSTOM QUESTION
    # ========================================================

    question = st.text_area(
        "Hỏi Brian AI",
        placeholder=(
            "Ví dụ:\n"
            "Biến nào ảnh hưởng giá mạnh nhất?\n"
            "Khối ngoại có ảnh hưởng đáng kể không?\n"
            "Yếu tố thị trường chung có quan trọng không?\n"
            "1D, 5D hay 20D đáng tin hơn?\n"
            "Nói đơn giản nghiên cứu này đang cho thấy gì?\n"
            "Tại sao model có R² âm?"
        ),
        height=120,
        key="research_ai_question",
    ).strip()

    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.markdown(
        "#### ⚡ Hỏi nhanh"
    )

    q1, q2, q3 = st.columns(3)

    quick_question = None

    with q1:

        if st.button(
            "🎯 Yếu tố mạnh nhất",
            key="research_ai_quick_1",
            width="stretch",
        ):

            quick_question = (
                "Yếu tố nào có quan hệ mạnh nhất "
                "với lợi suất cổ phiếu và bằng chứng mạnh tới đâu?"
            )

    with q2:

        if st.button(
            "📈 Giá phụ thuộc vào gì?",
            key="research_ai_quick_2",
            width="stretch",
        ):

            quick_question = (
                "Nói đơn giản giá cổ phiếu đang phụ thuộc "
                "vào 3 đến 5 yếu tố nào nhiều nhất?"
            )

    with q3:

        if st.button(
            "🔬 Độ tin cậy nghiên cứu",
            key="research_ai_quick_3",
            width="stretch",
        ):

            quick_question = (
                "Nghiên cứu này đáng tin đến đâu? "
                "Hãy xem model, R², baseline, VIF và các kiểm định."
            )

    if quick_question:

        question = quick_question

    # ========================================================
    # RUN
    # ========================================================

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
            "Brian AI đang đọc kết quả nghiên cứu..."
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

    # ========================================================
    # RESULT
    # ========================================================

    saved_scope = st.session_state.get(
        "research_ai_scope"
    )

    current_scope = (
        symbol,
        str(start_date),
        str(end_date),
    )

    ai_result = st.session_state.get(
        "research_ai_result"
    )

    if (
        ai_result is None
        or saved_scope != current_scope
    ):

        st.info(
            "Đặt câu hỏi hoặc để trống rồi bấm "
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

        error = ai_result.get(
            "error",
            "",
        )

        if error:

            st.code(
                error
            )

        return

    model = ai_result.get(
        "model"
    )

    if model:

        st.caption(
            f"Model: {model}"
        )

    question_used = st.session_state.get(
        "research_ai_question_used",
        "",
    )

    if question_used:

        with st.container(
            border=True
        ):

            st.caption(
                "Câu hỏi:"
            )

            st.write(
                question_used
            )

    output = _safe_text(
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

    # ========================================================
    # SAMPLE
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
            "Mẫu dưới 60 quan sát: "
            "kết luận định lượng cần rất thận trọng."
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
    # VARIABLES
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
        "🧬 Toàn bộ biến"
    )

    total_groups = len(
        groups
    )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Tổng biến",
            len(all_features),
        )

    with b:

        st.metric(
            "Nhóm yếu tố",
            total_groups,
        )

    with c:

        st.metric(
            "Biến market / sector / flow",
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
                    "Dòng tiền",
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
                        str(
                            item
                        )
                        for item in values
                    )
                )

            else:

                st.caption(
                    "Không có dữ liệu nhóm này."
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
    # FINAL SUMMARY
    # ========================================================

    st.divider()

    st.header(
        "🎯 Tóm tắt kết quả"
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

    st.warning(
        "Các kết quả trên là quan hệ thống kê và khả năng dự báo "
        "trong mẫu đã chọn, không phải bằng chứng nhân quả."
    )

    # ========================================================
    # RESEARCH AI
    # ========================================================

    render_research_ai(
        result=result,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_analysis():
    render_stock_analysis()
