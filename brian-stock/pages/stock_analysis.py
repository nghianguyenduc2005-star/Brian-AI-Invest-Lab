from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import (
    price_volume_chart,
)

from data.market import (
    display_symbol,
    load_market_data,
    load_research_history,
    market_snapshot,
    normalize_symbol,
)


# ============================================================
# CẤU HÌNH
# ============================================================

HORIZONS = {
    "1D": 1,
    "5D": 5,
    "20D": 20,
}

PRESETS = {
    "1 tháng": 31,
    "3 tháng": 93,
    "6 tháng": 186,
    "1 năm": 365,
    "3 năm": 1095,
    "5 năm": 1825,
    "10 năm": 3652,
}

CACHE_TTL = 900

MIN_OBSERVATIONS = 40

TRAIN_RATIO = 0.80


# ============================================================
# TIỆN ÍCH
# ============================================================

def num(
    value: Any,
    default=None,
):
    try:

        value = float(
            value
        )

        if pd.isna(
            value
        ):
            return default

        if not np.isfinite(
            value
        ):
            return default

        return value

    except Exception:

        return default


def format_price(
    value,
):
    value = num(
        value
    )

    if value is None:
        return "—"

    return f"{value:,.0f} đồng"


def format_percent(
    value,
):
    value = num(
        value
    )

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def format_volume(
    value,
):
    value = num(
        value
    )

    if value is None:
        return "—"

    value = abs(
        value
    )

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn"
        )

    return (
        f"{value:,.0f}"
    )


def format_value(
    value,
):
    value = num(
        value
    )

    if value is None:
        return "—"

    if value >= 1_000_000_000_000:

        return (
            f"{value / 1_000_000_000_000:.2f} nghìn tỷ"
        )

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:.2f} tỷ"
        )

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:.2f} triệu"
        )

    if value >= 1_000:

        return (
            f"{value / 1_000:.2f} nghìn"
        )

    return (
        f"{value:,.0f}"
    )


def format_rsi(
    value,
):
    value = num(
        value
    )

    if value is None:
        return "—"

    return f"{value:.1f}"


def format_number(
    value,
    digits=4,
):
    value = num(
        value
    )

    if value is None:
        return "—"

    return f"{value:.{digits}f}"


# ============================================================
# TRẠNG THÁI KỸ THUẬT
# ============================================================

def rsi_status(
    value,
):
    value = num(
        value
    )

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
    price = num(
        price
    )

    sma20 = num(
        sma20
    )

    sma50 = num(
        sma50
    )

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
    value = num(
        value
    )

    if value is None:
        return "Không xác định"

    if value > 0:
        return "MACD dương"

    if value < 0:
        return "MACD âm"

    return "MACD trung tính"


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL,
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
# NGÀY DỮ LIỆU
# ============================================================

def normalize_datetime_index(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return df

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


def get_date_range(
    df,
):
    df = normalize_datetime_index(
        df
    )

    if (
        df is None
        or df.empty
    ):
        return None, None

    start = df.index.min()
    end = df.index.max()

    if (
        pd.isna(start)
        or pd.isna(end)
    ):
        return None, None

    return (
        start.date(),
        end.date(),
    )


def slice_local(
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

    if work.empty:
        return work

    start_ts = pd.Timestamp(
        start_date
    )

    end_ts = (
        pd.Timestamp(
            end_date
        )
        + pd.Timedelta(
            days=1
        )
        - pd.Timedelta(
            microseconds=1
        )
    )

    result = work.loc[
        (
            work.index >= start_ts
        )
        & (
            work.index <= end_ts
        )
    ].copy()

    return result


# ============================================================
# FEATURE DISCOVERY
# ============================================================

def get_all_numeric_features(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return []

    features = []

    for column in df.columns:

        if str(
            column
        ) in {
            "Target_1D",
            "Target_5D",
            "Target_20D",
        }:
            continue

        if str(
            column
        ) in {
            "Time",
            "Date",
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
# BUILD RESEARCH DATASET
# ============================================================

def prepare_research_dataset(
    sample,
):
    if (
        sample is None
        or sample.empty
    ):
        return None

    if "Close" not in sample.columns:
        return None

    df = sample.copy()

    df = normalize_datetime_index(
        df
    )

    if df.empty:
        return None

    # ========================================================
    # CHUẨN HÓA NUMERIC
    # ========================================================

    for column in df.columns:

        try:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        except Exception:
            pass

    # ========================================================
    # RETURN
    # ========================================================

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df[
        "Return"
    ] = (
        close.pct_change()
    )

    df[
        "ReturnPct"
    ] = (
        df[
            "Return"
        ]
        * 100
    )

    # ========================================================
    # TẤT CẢ BIẾN
    #
    # Không tự giới hạn số biến.
    # ========================================================

    features = get_all_numeric_features(
        df
    )

    # ========================================================
    # LOẠI TARGET
    # ========================================================

    features = [
        column
        for column in features
        if column not in {
            "Target_1D",
            "Target_5D",
            "Target_20D",
        }
    ]

    # ========================================================
    # TARGET 1D / 5D / 20D
    # ========================================================

    for horizon, days in HORIZONS.items():

        df[
            f"Target_{horizon}"
        ] = (
            close.shift(-days)
            / close
            - 1
        )

    # ========================================================
    # INF
    # ========================================================

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # ========================================================
    # X
    #
    # Không bỏ các biến có một ít NaN.
    # Median fill để giữ toàn bộ biến.
    # ========================================================

    X = df[
        features
    ].copy()

    for column in features:

        X[
            column
        ] = pd.to_numeric(
            X[
                column
            ],
            errors="coerce",
        )

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

    # ========================================================
    # LOẠI CỘT KHÔNG CÓ BIẾN ĐỘNG
    #
    # Biến vẫn được ghi nhận, nhưng không thể đưa vào
    # mô hình có chuẩn hóa/VIF nếu variance = 0.
    # Ta đánh dấu riêng.
    # ========================================================

    zero_variance = []

    usable_features = []

    for column in features:

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

                usable_features.append(
                    column
                )

            else:

                zero_variance.append(
                    column
                )

        except Exception:

            zero_variance.append(
                column
            )

    datasets = {}

    for horizon in HORIZONS:

        target = df[
            f"Target_{horizon}"
        ]

        valid = (
            target.notna()
            & np.isfinite(
                target
            )
        )

        X_h = X.loc[
            valid,
            usable_features,
        ].copy()

        y_h = target.loc[
            valid
        ].copy()

        if len(
            X_h
        ) >= 20:

            datasets[
                horizon
            ] = {
                "X": X_h.astype(
                    float
                ),
                "y": y_h.astype(
                    float
                ),
            }

    if not datasets:

        return None

    return {
        "features": features,
        "usable_features": usable_features,
        "zero_variance": zero_variance,
        "datasets": datasets,
        "rows_raw": len(df),
    }


# ============================================================
# CORRELATION
# ============================================================

def correlation_analysis(
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

        mask = (
            x.notna()
            & yy.notna()
            & np.isfinite(x)
            & np.isfinite(yy)
        )

        x2 = x.loc[
            mask
        ]

        y2 = yy.loc[
            mask
        ]

        if len(
            x2
        ) < 20:
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
                "Pearson": pearson,
                "Spearman": spearman,
            }
        )

    if not rows:

        return pd.DataFrame()

    table = pd.DataFrame(
        rows
    )

    table[
        "|Spearman|"
    ] = table[
        "Spearman"
    ].abs()

    table[
        "|Pearson|"
    ] = table[
        "Pearson"
    ].abs()

    return (
        table
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

def run_standardized_ols(
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

        for column in X_train.columns:

            beta = num(
                model.params.get(
                    column
                ),
                np.nan,
            )

            p_value = num(
                model.pvalues.get(
                    column
                ),
                np.nan,
            )

            std_error = num(
                model.bse.get(
                    column
                ),
                np.nan,
            )

            rows.append(
                {
                    "Biến": column,
                    "Beta": beta,
                    "p-value": p_value,
                    "Std Error": std_error,
                    "|Beta|": (
                        abs(
                            beta
                        )
                        if np.isfinite(
                            beta
                        )
                        else np.nan
                    ),
                }
            )

        table = pd.DataFrame(
            rows
        )

        table = (
            table
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
                np.nan,
            ),
            "adj_r2": num(
                model.rsquared_adj,
                np.nan,
            ),
        }

    except Exception as error:

        return {
            "model": None,
            "table": pd.DataFrame(),
            "r2": np.nan,
            "adj_r2": np.nan,
            "error": str(
                error
            ),
        }


# ============================================================
# MODEL ENSEMBLE
# ============================================================

def run_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )

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
            "Lasso",
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        Lasso(
                            alpha=0.0001,
                            max_iter=50000,
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
    ]

    rows = []

    fitted = {}

    predictions = {}

    for name, model in models:

        try:

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(
                X_test
            )

            pred = np.asarray(
                pred,
                dtype=float,
            )

            actual = np.asarray(
                y_test,
                dtype=float,
            )

            residual = (
                actual
                - pred
            )

            mae = float(
                np.mean(
                    np.abs(
                        residual
                    )
                )
            )

            mse = float(
                np.mean(
                    residual ** 2
                )
            )

            rmse = float(
                np.sqrt(
                    mse
                )
            )

            denom = float(
                np.sum(
                    (
                        actual
                        - np.mean(
                            actual
                        )
                    )
                    ** 2
                )
            )

            if denom > 0:

                r2 = float(
                    1
                    - (
                        np.sum(
                            residual
                            ** 2
                        )
                        / denom
                    )
                )

            else:

                r2 = np.nan

            rows.append(
                {
                    "Mô hình": name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R²": r2,
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

    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "RMSE",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        result,
        fitted,
        predictions,
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
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )

        table = pd.DataFrame(
            {
                "Biến": X_test.columns,
                "Permutation": (
                    result.importances_mean
                ),
                "Permutation_STD": (
                    result.importances_std
                ),
            }
        )

        table[
            "|Permutation|"
        ] = table[
            "Permutation"
        ].abs()

        return (
            table
            .sort_values(
                "|Permutation|",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    except Exception:

        return pd.DataFrame()


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

        data = X.copy()

        # Xóa cột variance 0.
        keep = []

        for column in data.columns:

            try:

                if float(
                    data[
                        column
                    ].var(
                        ddof=0
                    )
                ) > 0:

                    keep.append(
                        column
                    )

            except Exception:
                pass

        data = data[
            keep
        ]

        if data.empty:

            return pd.DataFrame()

        # Với quá nhiều biến, VIF trực tiếp có thể rất nặng.
        # Nhưng vẫn tính toàn bộ biến theo block.
        rows = []

        values = (
            data
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                0.0
            )
            .astype(float)
        )

        matrix = values.values

        for index, column in enumerate(
            values.columns
        ):

            try:

                vif_value = float(
                    variance_inflation_factor(
                        matrix,
                        index,
                    )
                )

            except Exception:

                vif_value = np.inf

            rows.append(
                {
                    "Biến": column,
                    "VIF": vif_value,
                }
            )

        result = pd.DataFrame(
            rows
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
# TESTS
# ============================================================

def run_statistical_tests(
    X_train,
    y_train,
    ols_result,
):
    tests = {}

    residuals = None

    # ========================================================
    # RESIDUALS
    # ========================================================

    if ols_result is not None:

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
            residuals = None

    # ========================================================
    # ADF ON TARGET
    # ========================================================

    try:

        from statsmodels.tsa.stattools import (
            adfuller,
        )

        if len(
            y_train
        ) >= 20:

            adf = adfuller(
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
                    adf[0]
                ),
                "p_value": float(
                    adf[1]
                ),
            }

    except Exception:
        pass

    if residuals is None:

        return tests

    # ========================================================
    # JARQUE BERA
    # ========================================================

    try:

        from statsmodels.stats.stattools import (
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
            ),
        }

    except Exception:
        pass

    # ========================================================
    # LJUNG-BOX
    # ========================================================

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

    # ========================================================
    # BREUSCH-PAGAN
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

    # ========================================================
    # WHITE
    # ========================================================
    #
    # White test có thể bùng nổ số cột khi có quá nhiều
    # biến. Dùng tối đa 12 biến variance lớn nhất để kiểm định
    # phương sai, nhưng KHÔNG cắt biến khỏi mô hình chính.
    # ========================================================

    try:

        import statsmodels.api as sm

        from statsmodels.stats.diagnostic import (
            het_white,
        )

        selected = (
            X_train
            .var()
            .sort_values(
                ascending=False
            )
            .head(
                min(
                    12,
                    len(
                        X_train.columns
                    ),
                )
            )
            .index
            .tolist()
        )

        if selected:

            X_white = sm.add_constant(
                X_train[
                    selected
                ],
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
                "features_used": selected,
            }

    except Exception:
        pass

    return tests


# ============================================================
# FACTOR RANKING
# ============================================================

def build_factor_ranking(
    correlations,
    ols_table,
    permutation,
):
    if (
        correlations is None
        or correlations.empty
    ):

        return pd.DataFrame()

    result = correlations.copy()

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

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
                    "|Permutation|",
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
            "|Permutation|"
        ] = np.nan

    # --------------------------------------------------------
    # SCORE
    #
    # 3 nguồn:
    # correlation
    # standardized OLS
    # permutation
    # --------------------------------------------------------

    scores = []

    for column in result[
        "Biến"
    ]:

        row = result[
            result[
                "Biến"
            ]
            == column
        ].iloc[
            0
        ]

        values = []

        for field in [
            "|Spearman|",
            "|Beta|",
            "|Permutation|",
        ]:

            value = num(
                row.get(
                    field
                ),
                None,
            )

            if value is not None:

                values.append(
                    value
                )

        if not values:

            scores.append(
                np.nan
            )

        else:

            scores.append(
                np.mean(
                    values
                )
            )

    result[
        "RawImportance"
    ] = scores

    # --------------------------------------------------------
    # Convert percentile.
    # --------------------------------------------------------

    result[
        "Score"
    ] = (
        result[
            "RawImportance"
        ]
        .rank(
            pct=True
        )
        * 100
    )

    # --------------------------------------------------------
    # Quan hệ
    # --------------------------------------------------------

    def get_direction(
        row
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

    # --------------------------------------------------------
    # Ý nghĩa
    # --------------------------------------------------------

    def get_significance(
        row
    ):
        p = num(
            row.get(
                "p-value"
            ),
            None,
        )

        if p is None:
            return "Không có p-value"

        if p < 0.01:
            return "Rất mạnh (<1%)"

        if p < 0.05:
            return "Có ý nghĩa 5%"

        if p < 0.10:
            return "Có ý nghĩa 10%"

        return "Chưa có ý nghĩa"

    result[
        "Ý nghĩa"
    ] = result.apply(
        get_significance,
        axis=1,
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
# KẾT LUẬN
# ============================================================

def build_research_conclusion(
    item,
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

        score = num(
            top.get(
                "Score"
            ),
            None,
        )

        spear = num(
            top.get(
                "Spearman"
            ),
            None,
        )

        beta = num(
            top.get(
                "Beta"
            ),
            None,
        )

        p = num(
            top.get(
                "p-value"
            ),
            None,
        )

        direction = str(
            top.get(
                "Quan hệ",
                "Không xác định",
            )
        )

        text = (
            f"Yếu tố nổi bật nhất trong mẫu là "
            f"**{factor}**"
        )

        if score is not None:

            text += (
                f" với điểm quan trọng tổng hợp "
                f"{score:.1f}/100"
            )

        text += (
            f", quan hệ **{direction.lower()}**"
        )

        if spear is not None:

            text += (
                f", Spearman = {spear:+.4f}"
            )

        if beta is not None:

            text += (
                f", Beta chuẩn hóa = {beta:+.4f}"
            )

        if p is not None:

            text += (
                f", p-value = {p:.5f}"
            )

        text += "."

        conclusions.append(
            text
        )

        # ----------------------------------------------------
        # SIGNIFICANCE
        # ----------------------------------------------------

        if (
            p is not None
            and p < 0.05
        ):

            conclusions.append(
                f"Biến **{factor}** có bằng chứng "
                "thống kê trong OLS ở mức 5%."
            )

        elif (
            p is not None
            and p < 0.10
        ):

            conclusions.append(
                f"Biến **{factor}** có tín hiệu "
                "thống kê ở mức 10%, nhưng chưa đạt 5%."
            )

        else:

            conclusions.append(
                f"Biến **{factor}** nổi bật về "
                "mức độ liên hệ/dự báo, nhưng chưa đủ "
                "bằng chứng thống kê ở mức 5%."
            )

    # ========================================================
    # BEST MODEL
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
            best[
                "Mô hình"
            ]
        )

        rmse = num(
            best[
                "RMSE"
            ],
            None,
        )

        r2 = num(
            best[
                "R²"
            ],
            None,
        )

        conclusions.append(
            f"Mô hình dự báo tốt nhất theo RMSE "
            f"trên tập test là **{model_name}** "
            f"(RMSE = "
            f"{rmse:.6f}"
            if rmse is not None
            else
            f"Mô hình tốt nhất là **{model_name}**"
            )
        )

        if r2 is not None:

            if r2 < 0:

                conclusions.append(
                    "R² âm: mô hình chưa tốt hơn "
                    "benchmark trung bình trên tập test."
                )

            elif r2 < 0.10:

                conclusions.append(
                    f"R² = {r2:.4f}: khả năng giải thích "
                    "biến động lợi suất tương lai còn thấp."
                )

            elif r2 < 0.25:

                conclusions.append(
                    f"R² = {r2:.4f}: có tín hiệu dự báo "
                    "nhưng mức giải thích còn vừa phải."
                )

            else:

                conclusions.append(
                    f"R² = {r2:.4f}: mô hình có mức "
                    "giải thích đáng chú ý trong mẫu."
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
            ),
            None,
        )

        if (
            p is not None
            and p < 0.05
        ):

            conclusions.append(
                "Breusch-Pagan cho thấy có dấu hiệu "
                "phương sai thay đổi ở mức 5%."
            )

        else:

            conclusions.append(
                "Breusch-Pagan chưa cho thấy "
                "phương sai thay đổi rõ ở mức 5%."
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
            ),
            None,
        )

        if value is not None:

            if 1.5 <= value <= 2.5:

                conclusions.append(
                    f"Durbin-Watson = {value:.4f}: "
                    "phần dư không cho thấy tự tương quan "
                    "mạnh theo quy tắc thực hành thông thường."
                )

            elif value < 1.5:

                conclusions.append(
                    f"Durbin-Watson = {value:.4f}: "
                    "có dấu hiệu tự tương quan dương."
                )

            else:

                conclusions.append(
                    f"Durbin-Watson = {value:.4f}: "
                    "có dấu hiệu tự tương quan âm."
                )

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
            ),
            None,
        )

        if (
            p is not None
            and p < 0.05
        ):

            conclusions.append(
                "Ljung-Box phát hiện tự tương quan "
                "còn lại trong phần dư ở mức 5%."
            )

        else:

            conclusions.append(
                "Ljung-Box chưa phát hiện "
                "tự tương quan phần dư rõ ở mức 5%."
            )

    return conclusions


# ============================================================
# RESEARCH ENGINE
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL,
    show_spinner=False,
)
def execute_research(
    sample,
    selected_horizons,
):
    prepared = prepare_research_dataset(
        sample
    )

    if prepared is None:

        return {
            "ok": False,
            "error": (
                "Không đủ dữ liệu để nghiên cứu."
            ),
        }

    result = {
        "ok": True,
        "features": prepared[
            "features"
        ],
        "usable_features": prepared[
            "usable_features"
        ],
        "zero_variance": prepared[
            "zero_variance"
        ],
        "rows_raw": prepared[
            "rows_raw"
        ],
        "horizons": {},
    }

    for horizon in HORIZONS:

        if horizon not in selected_horizons:
            continue

        dataset = prepared[
            "datasets"
        ].get(
            horizon
        )

        if dataset is None:
            continue

        X = dataset[
            "X"
        ]

        y = dataset[
            "y"
        ]

        if len(X) < MIN_OBSERVATIONS:
            continue

        split = int(
            len(X)
            * TRAIN_RATIO
        )

        if (
            split < 20
            or (
                len(X)
                - split
            ) < 10
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
            correlation_analysis(
                X,
                y,
            )
        )

        # ====================================================
        # OLS
        # ====================================================

        ols = (
            run_standardized_ols(
                X_train,
                y_train,
            )
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

        # ====================================================
        # MODELS
        # ====================================================

        (
            models,
            fitted_models,
            predictions,
        ) = run_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # ====================================================
        # PERMUTATION
        # ========================================================

        permutation = (
            pd.DataFrame()
        )

        if (
            isinstance(
                models,
                pd.DataFrame,
            )
            and not models.empty
            and fitted_models
        ):

            best_name = str(
                models.iloc[
                    0
                ][
                    "Mô hình"
                ]
            )

            best_model = fitted_models.get(
                best_name
            )

            if best_model is not None:

                permutation = (
                    run_permutation(
                        best_model,
                        X_test,
                        y_test,
                    )
                )

        # ====================================================
        # FACTOR RANKING
        # ====================================================

        ranking = (
            build_factor_ranking(
                correlations,
                ols_table,
                permutation,
            )
        )

        # ====================================================
        # VIF
        # ====================================================

        vif = run_vif(
            X_train
        )

        # ====================================================
        # TESTS
        # ====================================================

        tests = run_statistical_tests(
            X_train,
            y_train,
            ols,
        )

        result[
            "horizons"
        ][
            horizon
        ] = {
            "observations": len(X),
            "train": len(X_train),
            "test": len(X_test),
            "correlations": correlations,
            "ols": ols,
            "ols_table": ols_table,
            "models": models,
            "fitted_models": fitted_models,
            "predictions": predictions,
            "permutation": permutation,
            "ranking": ranking,
            "vif": vif,
            "tests": tests,
        }

    if not result[
        "horizons"
    ]:

        return {
            "ok": False,
            "error": (
                "Không có horizon nào đủ dữ liệu."
            ),
        }

    return result


# ============================================================
# RENDER RESEARCH
# ============================================================

def render_quant_research(
    symbol,
):
    st.divider()

    st.header(
        "🧪 Nghiên cứu định lượng"
    )

    st.write(
        "Chọn đúng khoảng thời gian mẫu trước khi chạy. "
        "Dữ liệu nghiên cứu được lấy theo từng đoạn lịch sử "
        "và cache; thay đổi preset hoặc ngày không làm gọi "
        "API liên tục nếu cùng khoảng dữ liệu đã có trong cache."
    )

    # ========================================================
    # MODE CHỌN MẪU
    # ========================================================

    mode = st.radio(
        "Cách chọn mẫu",
        [
            "Preset",
            "Khoảng ngày tùy chọn",
        ],
        horizontal=True,
        key="research_mode",
    )

    today = pd.Timestamp.today().date()

    # ========================================================
    # PRESET
    # ========================================================

    if mode == "Preset":

        preset = st.selectbox(
            "Khoảng thời gian",
            list(
                PRESETS.keys()
            ),
            index=3,
            key="research_preset",
        )

        days = PRESETS[
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

    # ========================================================
    # CUSTOM DATE
    # ========================================================

    else:

        default_start = (
            pd.Timestamp(
                today
            )
            - pd.Timedelta(
                days=365
            )
        ).date()

        c1, c2 = st.columns(
            2
        )

        with c1:

            start_date = st.date_input(
                "Từ ngày",
                value=default_start,
                max_value=today,
                key="research_custom_start",
            )

        with c2:

            end_date = st.date_input(
                "Đến ngày",
                value=today,
                max_value=today,
                key="research_custom_end",
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    if start_date > end_date:

        st.error(
            "Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc."
        )

        return

    # ========================================================
    # GET HISTORY
    #
    # Đây là API path.
    # Một lần cho khoảng cần nghiên cứu,
    # sau đó cache.
    # ========================================================

    with st.spinner(
        "Đang chuẩn bị dữ liệu lịch sử..."
    ):

        try:

            history = load_research_history(
                symbol,
                start_date,
                end_date,
            )

        except Exception as error:

            st.error(
                "Không lấy được dữ liệu lịch sử."
            )

            st.code(
                str(error)
            )

            return

    # ========================================================
    # LOCAL CUT — bảo đảm đúng ngày
    # ========================================================

    sample = slice_local(
        history,
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
        get_date_range(
            sample
        )
    )

    # ========================================================
    # SAMPLE INFO
    # ========================================================

    st.subheader(
        "📚 Mẫu thực tế"
    )

    a, b, c, d = st.columns(
        4
    )

    with a:

        st.metric(
            "Khoảng chọn",
            (
                preset
                if mode == "Preset"
                else "Tùy chọn"
            ),
        )

    with b:

        st.metric(
            "Quan sát",
            f"{len(sample):,}",
        )

    with c:

        st.metric(
            "Ngày đầu thực tế",
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
            "Ngày cuối thực tế",
            (
                actual_end.strftime(
                    "%d/%m/%Y"
                )
                if actual_end
                else "—"
            ),
        )

    # ========================================================
    # CẢNH BÁO SAMPLE
    # ========================================================

    if len(sample) < 30:

        st.error(
            "Mẫu quá nhỏ để thực hiện nghiên cứu định lượng đáng tin cậy."
        )

    elif len(sample) < 60:

        st.warning(
            "Mẫu còn nhỏ. Có thể chạy nhưng kết luận thống kê "
            "cần được xem thận trọng."
        )

    elif len(sample) < 120:

        st.info(
            "Mẫu đã đủ để chạy phân tích cơ bản; mẫu dài hơn "
            "thường cho kết luận ổn định hơn."
        )

    else:

        st.success(
            f"Mẫu hiện có {len(sample):,} quan sát."
        )

    # ========================================================
    # HORIZON
    # ========================================================

    selected_horizons = st.multiselect(
        "Horizon lợi suất tương lai",
        list(
            HORIZONS.keys()
        ),
        default=[
            "1D",
            "5D",
            "20D",
        ],
        key="research_selected_horizons",
    )

    if not selected_horizons:

        st.warning(
            "Chọn ít nhất một horizon."
        )

        return

    # ========================================================
    # RUN BUTTON
    # ========================================================

    if st.button(
        "🚀 Chạy toàn bộ nghiên cứu",
        type="primary",
        width="stretch",
        key="research_run_button",
    ):

        with st.spinner(
            "Đang chạy toàn bộ mô hình, biến và kiểm định..."
        ):

            result = execute_research(
                sample,
                tuple(
                    selected_horizons
                ),
            )

        st.session_state[
            "research_result"
        ] = result

        st.session_state[
            "research_symbol"
        ] = symbol

        st.session_state[
            "research_start"
        ] = start_date

        st.session_state[
            "research_end"
        ] = end_date

        st.session_state[
            "research_horizons"
        ] = selected_horizons

    # ========================================================
    # RESULT
    # ========================================================

    result = st.session_state.get(
        "research_result"
    )

    saved_symbol = st.session_state.get(
        "research_symbol"
    )

    if (
        result is None
        or saved_symbol != symbol
    ):

        st.caption(
            "Chọn mẫu dữ liệu và bấm "
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

    saved_start = st.session_state.get(
        "research_start"
    )

    saved_end = st.session_state.get(
        "research_end"
    )

    saved_horizons = st.session_state.get(
        "research_horizons",
        [],
    )

    st.success(
        f"Đã nghiên cứu {display_symbol(symbol)} "
        f"trên mẫu "
        f"{saved_start.strftime('%d/%m/%Y')}"
        f" → "
        f"{saved_end.strftime('%d/%m/%Y')}"
        f" với {len(sample):,} quan sát."
    )

    # ========================================================
    # VARIABLES
    # ========================================================

    features = result.get(
        "features",
        [],
    )

    usable_features = result.get(
        "usable_features",
        [],
    )

    zero_variance = result.get(
        "zero_variance",
        [],
    )

    st.header(
        "🧬 Toàn bộ biến nghiên cứu"
    )

    a, b, c = st.columns(
        3
    )

    with a:

        st.metric(
            "Tổng biến",
            f"{len(features):,}",
        )

    with b:

        st.metric(
            "Biến sử dụng",
            f"{len(usable_features):,}",
        )

    with c:

        st.metric(
            "Biến variance = 0",
            f"{len(zero_variance):,}",
        )

    with st.expander(
        "Xem toàn bộ danh sách biến",
        expanded=False,
    ):

        rows = []

        for index, feature in enumerate(
            features,
            start=1,
        ):

            rows.append(
                {
                    "STT": index,
                    "Biến": feature,
                    "Đưa vào mô hình": (
                        "Có"
                        if feature
                        in usable_features
                        else "Không"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                rows
            ),
            width="stretch",
            hide_index=True,
        )

    if zero_variance:

        st.caption(
            "Các biến variance = 0 vẫn được giữ trong "
            "báo cáo danh sách, nhưng không thể cung cấp "
            "thông tin dự báo độc lập cho mô hình."
        )

    # ========================================================
    # EACH HORIZON
    # ========================================================

    cross_factor_rows = []

    for horizon in HORIZONS:

        if horizon not in saved_horizons:
            continue

        item = result[
            "horizons"
        ].get(
            horizon
        )

        if item is None:
            continue

        st.divider()

        label = {
            "1D": "1 ngày",
            "5D": "5 ngày",
            "20D": "20 ngày",
        }[
            horizon
        ]

        st.header(
            f"📈 Horizon {horizon} — lợi suất sau {label}"
        )

        # ====================================================
        # OBS
        # ====================================================

        a, b, c = st.columns(
            3
        )

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

        # ====================================================
        # FACTOR
        # ====================================================

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

            factor = str(
                top[
                    "Biến"
                ]
            )

            score = num(
                top.get(
                    "Score"
                ),
                None,
            )

            direction = str(
                top.get(
                    "Quan hệ"
                )
            )

            st.subheader(
                "🎯 Yếu tố nổi bật nhất"
            )

            st.success(
                (
                    f"**{factor}**"
                    + (
                        f" · Score {score:.1f}/100"
                        if score is not None
                        else ""
                    )
                    + f" · {direction}"
                )
            )

            # Save cross-horizon ranking.

            for rank, (_, row) in enumerate(
                ranking.head(
                    20
                ).iterrows(),
                start=1,
            ):

                cross_factor_rows.append(
                    {
                        "Horizon": horizon,
                        "Biến": row[
                            "Biến"
                        ],
                        "Rank": rank,
                        "Score": num(
                            row.get(
                                "Score"
                            ),
                            np.nan,
                        ),
                    }
                )

            # =================================================
            # TABLE
            # =================================================

            st.markdown(
                "#### 🏆 Xếp hạng yếu tố"
            )

            show = ranking.head(
                25
            ).copy()

            for column in [
                "Score",
                "Pearson",
                "Spearman",
                "Beta",
                "p-value",
                "Permutation",
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

            if "Score" in show.columns:
                show[
                    "Score"
                ] = show[
                    "Score"
                ].round(1)

            if "Pearson" in show.columns:
                show[
                    "Pearson"
                ] = show[
                    "Pearson"
                ].round(4)

            if "Spearman" in show.columns:
                show[
                    "Spearman"
                ] = show[
                    "Spearman"
                ].round(4)

            if "Beta" in show.columns:
                show[
                    "Beta"
                ] = show[
                    "Beta"
                ].round(4)

            if "p-value" in show.columns:
                show[
                    "p-value"
                ] = show[
                    "p-value"
                ].round(5)

            if "Permutation" in show.columns:
                show[
                    "Permutation"
                ] = show[
                    "Permutation"
                ].round(6)

            display_columns = [
                "Biến",
                "Score",
                "Quan hệ",
                "Pearson",
                "Spearman",
                "Beta",
                "p-value",
                "Permutation",
                "Ý nghĩa",
            ]

            display_columns = [
                column
                for column in display_columns
                if column in show.columns
            ]

            st.dataframe(
                show[
                    display_columns
                ],
                width="stretch",
                hide_index=True,
            )

            # =================================================
            # EXPLANATION
            # =================================================

            st.markdown(
                "#### 🧠 Ý nghĩa các yếu tố"
            )

            for _, row in ranking.head(
                5
            ).iterrows():

                factor_name = str(
                    row.get(
                        "Biến"
                    )
                )

                spear = num(
                    row.get(
                        "Spearman"
                    ),
                    None,
                )

                beta = num(
                    row.get(
                        "Beta"
                    ),
                    None,
                )

                p = num(
                    row.get(
                        "p-value"
                    ),
                    None,
                )

                direction = str(
                    row.get(
                        "Quan hệ",
                        "Không xác định",
                    )
                )

                explanation = (
                    f"**{factor_name}**: "
                    f"{direction.lower()}"
                )

                if spear is not None:

                    explanation += (
                        f", Spearman "
                        f"{spear:+.4f}"
                    )

                if beta is not None:

                    explanation += (
                        f", Beta "
                        f"{beta:+.4f}"
                    )

                if p is not None:

                    explanation += (
                        f", p-value "
                        f"{p:.5f}"
                    )

                st.markdown(
                    "• "
                    + explanation
                    + "."
                )

        # ====================================================
        # MODEL RESULTS
        # ====================================================

        models = item.get(
            "models"
        )

        st.subheader(
            "🤖 So sánh mô hình"
        )

        if (
            isinstance(
                models,
                pd.DataFrame,
            )
            and not models.empty
        ):

            model_show = models.copy()

            for column in [
                "MAE",
                "MSE",
                "RMSE",
                "R²",
            ]:

                if column in model_show.columns:

                    model_show[
                        column
                    ] = pd.to_numeric(
                        model_show[
                            column
                        ],
                        errors="coerce",
                    ).round(
                        6
                    )

            st.dataframe(
                model_show,
                width="stretch",
                hide_index=True,
            )

            best = models.iloc[
                0
            ]

            st.info(
                f"**Model tốt nhất:** "
                f"{best['Mô hình']} · "
                f"RMSE "
                f"{best['RMSE']:.6f}"
            )

        # ====================================================
        # OLS
        # ====================================================

        ols = item.get(
            "ols"
        )

        ols_table = item.get(
            "ols_table"
        )

        st.subheader(
            "📐 OLS chuẩn hóa"
        )

        if (
            isinstance(
                ols,
                dict,
            )
            and ols.get(
                "model"
            ) is not None
        ):

            a, b = st.columns(
                2
            )

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

            if (
                isinstance(
                    ols_table,
                    pd.DataFrame,
                )
                and not ols_table.empty
            ):

                ols_show = ols_table.head(
                    25
                ).copy()

                for column in [
                    "Beta",
                    "p-value",
                    "Std Error",
                ]:

                    if column in ols_show.columns:

                        ols_show[
                            column
                        ] = pd.to_numeric(
                            ols_show[
                                column
                            ],
                            errors="coerce",
                        ).round(
                            6
                        )

                st.dataframe(
                    ols_show[
                        [
                            "Biến",
                            "Beta",
                            "p-value",
                            "Std Error",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Beta được chuẩn hóa nên có thể so sánh "
                    "độ lớn tương đối giữa các biến."
                )

        else:

            st.warning(
                "OLS không chạy được trên mẫu này."
            )

        # ====================================================
        # VIF
        # ====================================================

        vif = item.get(
            "vif"
        )

        st.subheader(
            "🔗 Đa cộng tuyến — VIF"
        )

        if (
            isinstance(
                vif,
                pd.DataFrame,
            )
            and not vif.empty
        ):

            vif_show = vif.head(
                30
            ).copy()

            vif_show[
                "VIF"
            ] = pd.to_numeric(
                vif_show[
                    "VIF"
                ],
                errors="coerce",
            ).round(
                3
            )

            st.dataframe(
                vif_show,
                width="stretch",
                hide_index=True,
            )

            max_vif = num(
                vif[
                    "VIF"
                ].replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                ).max(),
                None,
            )

            if (
                max_vif is not None
                and max_vif >= 10
            ):

                st.warning(
                    "Có đa cộng tuyến mạnh. "
                    "Không nên diễn giải hệ số OLS riêng lẻ "
                    "như bằng chứng nhân quả."
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
                    "VIF không cho thấy đa cộng tuyến quá mạnh."
                )

        # ====================================================
        # TESTS
        # ====================================================

        tests = item.get(
            "tests",
            {},
        )

        st.subheader(
            "🧪 Kiểm định thống kê"
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
                ),
                None,
            )

            with st.expander(
                "ADF — Augmented Dickey-Fuller",
                expanded=False,
            ):

                a, b = st.columns(
                    2
                )

                with a:

                    st.metric(
                        "Statistic",
                        format_number(
                            adf.get(
                                "statistic"
                            ),
                            5,
                        ),
                    )

                with b:

                    st.metric(
                        "p-value",
                        format_number(
                            p,
                            6,
                        ),
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
                        "Chưa đủ bằng chứng bác bỏ nghiệm đơn vị ở mức 5%."
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
                ),
                None,
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
                        "Phần dư không phù hợp tốt với phân phối chuẩn."
                    )

                else:

                    st.success(
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

            p = num(
                bp.get(
                    "p_value"
                ),
                None,
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
                        "Chưa có bằng chứng rõ về phương sai thay đổi."
                    )

        # White
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
                ),
                None,
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

            value = num(
                dw.get(
                    "statistic"
                ),
                None,
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
                            "Tương đối gần 2 — chưa thấy tự tương quan mạnh."
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
                ),
                None,
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
                        "Có bằng chứng tự tương quan còn lại trong phần dư."
                    )

                else:

                    st.success(
                        "Chưa có bằng chứng mạnh về tự tương quan phần dư."
                    )

        # ====================================================
        # CONCLUSION
        # ====================================================

        st.subheader(
            "🧠 Kết luận nghiên cứu"
        )

        conclusions = build_research_conclusion(
            item
        )

        if conclusions:

            for conclusion in conclusions:

                st.markdown(
                    "• "
                    + conclusion
                )

        st.warning(
            "Lưu ý: đây là quan hệ thống kê/khả năng dự báo "
            "trong mẫu, không phải bằng chứng nhân quả."
        )

    # ========================================================
    # CROSS HORIZON
    # ========================================================

    st.divider()

    st.header(
        "🏆 Yếu tố nhất quán giữa 1D / 5D / 20D"
    )

    if cross_factor_rows:

        cross = pd.DataFrame(
            cross_factor_rows
        )

        stable = (
            cross
            .groupby(
                "Biến",
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
            "Số horizon",
            "Rank TB",
            "Score TB",
        ]

        st.dataframe(
            stable.head(
                25
            ),
            width="stretch",
            hide_index=True,
        )

        if not stable.empty:

            strongest = stable.iloc[
                0
            ]

            st.success(
                f"Yếu tố nhất quán nhất: "
                f"**{strongest['Biến']}** "
                f"({int(strongest['Số horizon'])} horizon)."
            )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    st.divider()

    st.header(
        "🎯 Tóm tắt cho nhà đầu tư"
    )

    summary_rows = []

    for horizon, item in result[
        "horizons"
    ].items():

        if horizon not in saved_horizons:
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

        top = ranking.iloc[
            0
        ]

        summary_rows.append(
            {
                "Horizon": horizon,
                "Yếu tố nổi bật": top[
                    "Biến"
                ],
                "Score": num(
                    top.get(
                        "Score"
                    ),
                    np.nan,
                ),
                "Quan hệ": top.get(
                    "Quan hệ"
                ),
                "Spearman": num(
                    top.get(
                        "Spearman"
                    ),
                    np.nan,
                ),
                "Beta": num(
                    top.get(
                        "Beta"
                    ),
                    np.nan,
                ),
                "p-value": num(
                    top.get(
                        "p-value"
                    ),
                    np.nan,
                ),
            }
        )

    if summary_rows:

        summary = pd.DataFrame(
            summary_rows
        )

        for column in [
            "Score",
            "Spearman",
            "Beta",
            "p-value",
        ]:

            summary[
                column
            ] = summary[
                column
            ].round(
                5
            )

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

        st.info(
            "Đọc bảng này theo thứ tự: "
            "yếu tố nào đứng đầu nhiều horizon, "
            "có Spearman/Beta ổn định, và p-value có ý nghĩa. "
            "Không nên kết luận chỉ từ một model duy nhất."
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
        "Phân tích kỹ thuật, dữ liệu lịch sử và nghiên cứu "
        "định lượng trên khoảng thời gian do người dùng chọn."
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

        # Clear research after changing symbol.
        for key in [
            "research_result",
            "research_symbol",
            "research_source_data",
            "research_source_symbol",
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

        data = load_display_data(
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
    # STOCK HEADER
    # ========================================================

    st.subheader(
        f"📈 {display_symbol(symbol)}"
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    a, b, c, d = st.columns(
        4
    )

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
            format_rsi(
                rsi_value
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

    a, b, c, d = st.columns(
        4
    )

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

    a, b, c, d = st.columns(
        4
    )

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

    a, b, c, d = st.columns(
        4
    )

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
    # QUANT RESEARCH
    # ========================================================

    render_quant_research(
        symbol
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_analysis():
    render_stock_analysis()
