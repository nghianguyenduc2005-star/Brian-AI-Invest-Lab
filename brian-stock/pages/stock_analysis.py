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

#
# Chỉ API một lần với khoảng đủ rộng.
# Sau đó sample được cắt LOCAL.
#
RESEARCH_SOURCE_PERIOD = "10y"

RESEARCH_PRESETS = {
    "1 tháng": 31,
    "3 tháng": 93,
    "6 tháng": 186,
    "1 năm": 365,
    "3 năm": 1095,
    "5 năm": 1825,
    "10 năm": 3650,
}

HORIZONS = {
    "1D": 1,
    "5D": 5,
    "20D": 20,
}

CACHE_TTL = 900


# ============================================================
# BASIC
# ============================================================

def num(value: Any, default=None):
    try:
        value = float(value)

        if pd.isna(value):
            return default

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


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


def format_volume(value):
    value = num(value)

    if value is None:
        return "—"

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f} nghìn"

    return f"{value:,.0f}"


def format_number(value, digits=4):
    value = num(value)

    if value is None:
        return "—"

    return f"{value:.{digits}f}"


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


def ma_status(price, sma20, sma50):
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
# CACHE DATA
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL,
    show_spinner=False,
)
def load_display_data(symbol):
    return load_market_data(
        symbol,
        DISPLAY_PERIOD,
    )


@st.cache_data(
    ttl=CACHE_TTL,
    show_spinner=False,
)
def load_research_source_data(symbol):
    """
    API chỉ được gọi ở đây.

    Tải 10 năm một lần.
    Các preset và ngày tùy chỉnh phía dưới
    chỉ cắt DataFrame local.
    """
    return load_market_data(
        symbol,
        RESEARCH_SOURCE_PERIOD,
    )


# ============================================================
# DATE HELPERS
# ============================================================

def normalize_datetime_index(df):
    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):
        return df

    work = df.copy()

    try:

        work.index = pd.to_datetime(
            work.index,
            errors="coerce",
        )

        work = work[
            ~work.index.isna()
        ]

        work = work.sort_index()

    except Exception:
        pass

    return work


def get_available_date_range(df):
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

    if pd.isna(start) or pd.isna(end):
        return None, None

    return (
        start.date(),
        end.date(),
    )


def slice_sample(
    df,
    start_date,
    end_date,
):
    """
    Cắt sample hoàn toàn LOCAL.
    Không gọi API.
    """

    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    work = normalize_datetime_index(
        df
    )

    if start_date is None:
        start_date = work.index.min().date()

    if end_date is None:
        end_date = work.index.max().date()

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
# FULL NUMERIC FEATURES
# ============================================================

def get_numeric_features(df):
    if (
        df is None
        or df.empty
    ):
        return []

    features = []

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            features.append(
                column
            )

    return features


# ============================================================
# BUILD RESEARCH DATA
# ============================================================

def prepare_research_dataset(
    data,
):
    if (
        data is None
        or data.empty
        or "Close" not in data.columns
    ):
        return None

    df = data.copy()

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    if "Return" not in df.columns:

        df["Return"] = (
            close.pct_change()
        )

    # --------------------------------------------------------
    # ALL NUMERIC VARIABLES
    # --------------------------------------------------------

    features = get_numeric_features(
        df
    )

    forbidden = {
        "Target_1D",
        "Target_5D",
        "Target_20D",
        "ReturnPct",
    }

    features = [
        column
        for column in features
        if column not in forbidden
    ]

    if "Return" not in features:

        features.append(
            "Return"
        )

    # --------------------------------------------------------
    # numeric conversion
    # --------------------------------------------------------

    for column in features:

        df[
            column
        ] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # FUTURE TARGETS
    # --------------------------------------------------------

    for horizon, days in HORIZONS.items():

        df[
            f"Target_{horizon}"
        ] = (
            close.shift(-days)
            / close
            - 1
        )

    # --------------------------------------------------------
    # REPLACE INF
    # --------------------------------------------------------

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # FILL FEATURES
    #
    # Không bỏ biến.
    # --------------------------------------------------------

    X = df[
        features
    ].copy()

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
            valid
        ].copy()

        y_h = target.loc[
            valid
        ].copy()

        if len(X_h) >= 20:

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

def run_ols(
    X_train,
    y_train,
):
    try:

        import statsmodels.api as sm
        from sklearn.preprocessing import StandardScaler

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

            rows.append(
                {
                    "Biến": column,
                    "Beta": beta,
                    "p-value": p_value,
                    "|Beta|": (
                        abs(beta)
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

    except Exception:
        return None


# ============================================================
# MODELS
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

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import (
        StandardScaler,
    )

    model_list = [
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

    for name, model in model_list:

        try:

            model.fit(
                X_train,
                y_train,
            )

            prediction = model.predict(
                X_test
            )

            residual = (
                y_test.values
                - prediction
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

            total = np.sum(
                (
                    y_test.values
                    - y_test.mean()
                ) ** 2
            )

            if total > 0:

                r2 = float(
                    1
                    - np.sum(
                        residual ** 2
                    )
                    / total
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

        except Exception:
            pass

    if not rows:

        return (
            pd.DataFrame(),
            {},
        )

    table = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "RMSE"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        table,
        fitted,
    )


# ============================================================
# PERMUTATION
# ============================================================

def permutation_importance(
    model,
    X_test,
    y_test,
):
    try:

        from sklearn.inspection import (
            permutation_importance as sklearn_permutation,
        )

        result = sklearn_permutation(
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
                "STD": (
                    result.importances_std
                ),
            }
        )

        return (
            table
            .sort_values(
                "Permutation",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# FACTOR RANK
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
                ]
            ],
            on="Biến",
            how="left",
        )

    else:

        result[
            "Permutation"
        ] = np.nan

    # --------------------------------------------------------
    # Ranking normalized 0 → 100
    # --------------------------------------------------------

    rank_corr = (
        result[
            "|Spearman|"
        ]
        .rank(
            ascending=True,
            pct=True,
        )
    )

    rank_beta = (
        result[
            "|Beta|"
        ]
        .rank(
            ascending=True,
            pct=True,
        )
    )

    rank_perm = (
        result[
            "Permutation"
        ]
        .rank(
            ascending=True,
            pct=True,
        )
    )

    result[
        "Score"
    ] = (
        (
            rank_corr
            + rank_beta
            + rank_perm
        )
        / 3
        * 100
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    def direction(row):

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
        direction,
        axis=1,
    )

    # --------------------------------------------------------
    # Significance
    # --------------------------------------------------------

    def significance(row):

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
        "Ý nghĩa"
    ] = result.apply(
        significance,
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
# TESTS
# ============================================================

def run_tests(
    X_train,
    y_train,
    ols_result,
):
    tests = {}

    residuals = None

    if ols_result is not None:

        try:

            residuals = np.asarray(
                ols_result[
                    "model"
                ].resid,
                dtype=float,
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    try:

        from statsmodels.tsa.stattools import (
            adfuller,
        )

        if len(y_train) >= 20:

            result = adfuller(
                y_train,
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

    if residuals is None:
        return tests

    # --------------------------------------------------------
    # JB
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
    # DW
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
    # Ljung Box
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

    # --------------------------------------------------------
    # BP / WHITE
    # --------------------------------------------------------

    try:

        import statsmodels.api as sm

        from statsmodels.stats.diagnostic import (
            het_breuschpagan,
            het_white,
        )

        # BP — all variables
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

        # White:
        # giới hạn interaction để tránh bùng nổ ma trận
        selected = (
            X_train.var()
            .sort_values(
                ascending=False
            )
            .head(
                min(
                    8,
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
# FULL RESEARCH
# ============================================================

@st.cache_data(
    ttl=CACHE_TTL,
    show_spinner=False,
)
def run_research(
    sample_data,
):
    prepared = prepare_research_dataset(
        sample_data
    )

    if prepared is None:

        return {
            "ok": False,
            "error": (
                "Không đủ quan sát để nghiên cứu."
            ),
        }

    output = {
        "ok": True,
        "features": prepared[
            "features"
        ],
        "rows_raw": prepared[
            "rows_raw"
        ],
        "horizons": {},
    }

    for horizon in HORIZONS:

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
        ]

        y = prepared[
            "datasets"
        ][
            horizon
        ][
            "y"
        ]

        if len(X) < 20:
            continue

        split = int(
            len(X) * 0.8
        )

        if (
            split < 15
            or len(X) - split < 5
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

        correlations = (
            correlation_analysis(
                X,
                y,
            )
        )

        ols = run_ols(
            X_train,
            y_train,
        )

        ols_table = (
            ols["table"]
            if ols is not None
            else pd.DataFrame()
        )

        models, fitted = run_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        permutation = pd.DataFrame()

        if (
            not models.empty
            and fitted
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

            if best_model is not None:

                permutation = (
                    permutation_importance(
                        best_model,
                        X_test,
                        y_test,
                    )
                )

        ranking = build_factor_ranking(
            correlations,
            ols_table,
            permutation,
        )

        tests = run_tests(
            X_train,
            y_train,
            ols,
        )

        output[
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
            "permutation": permutation,
            "ranking": ranking,
            "tests": tests,
        }

    if not output[
        "horizons"
    ]:

        return {
            "ok": False,
            "error": (
                "Không có horizon nào đủ số quan sát."
            ),
        }

    return output


# ============================================================
# RESEARCH CONCLUSION
# ============================================================

def build_conclusion(
    result,
):
    conclusions = []

    ranking = result.get(
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

        variable = str(
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

        relation = str(
            top.get(
                "Quan hệ",
                "Không xác định",
            )
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

        text = (
            f"Yếu tố nổi bật nhất là **{variable}**"
        )

        if score is not None:

            text += (
                f" với điểm tổng hợp {score:.1f}/100"
            )

        text += (
            f", quan hệ {relation.lower()}"
        )

        if spear is not None:

            text += (
                f", Spearman = {spear:+.4f}"
            )

        if beta is not None:

            text += (
                f", Beta = {beta:+.4f}"
            )

        if p is not None:

            text += (
                f", p-value = {p:.5f}"
            )

        text += "."

        conclusions.append(
            text
        )

        if (
            p is not None
            and p < 0.05
        ):

            conclusions.append(
                f"**{variable}** có bằng chứng thống kê "
                "ở mức 5% trong mô hình OLS."
            )

        else:

            conclusions.append(
                f"**{variable}** nổi bật về quan hệ/dự báo, "
                "nhưng chưa đủ bằng chứng thống kê ở mức 5%."
            )

    models = result.get(
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

        conclusions.append(
            f"Mô hình có RMSE thấp nhất là "
            f"**{best['Mô hình']}**, "
            f"RMSE = {best['RMSE']:.6f}, "
            f"R² = {best['R²']:.4f}."
        )

    return conclusions


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
        "Chọn chính xác mẫu dữ liệu trước khi chạy. "
        "Sau khi dữ liệu nguồn được tải, việc đổi ngày "
        "chỉ cắt DataFrame local và không gọi API lại."
    )

    # ========================================================
    # SOURCE DATA
    # ========================================================

    with st.spinner(
        "Chuẩn bị dữ liệu lịch sử..."
    ):

        source_data = (
            st.session_state.get(
                "research_source_data"
            )
        )

        source_symbol = (
            st.session_state.get(
                "research_source_symbol"
            )
        )

        if (
            source_data is None
            or source_symbol != symbol
        ):

            try:

                source_data = (
                    load_research_source_data(
                        symbol
                    )
                )

                st.session_state[
                    "research_source_data"
                ] = source_data

                st.session_state[
                    "research_source_symbol"
                ] = symbol

            except Exception as error:

                st.error(
                    "Không thể tải dữ liệu lịch sử."
                )

                st.caption(
                    str(error)
                )

                return

    available_start, available_end = (
        get_available_date_range(
            source_data
        )
    )

    if (
        available_start is None
        or available_end is None
    ):

        st.error(
            "Không xác định được khoảng dữ liệu."
        )

        return

    # ========================================================
    # SAMPLE MODE
    # ========================================================

    st.subheader(
        "📅 Chọn mẫu dữ liệu"
    )

    mode = st.radio(
        "Kiểu chọn mẫu",
        [
            "Preset",
            "Khoảng ngày tùy chọn",
        ],
        horizontal=True,
        key="research_sample_mode",
    )

    if mode == "Preset":

        preset = st.selectbox(
            "Khoảng thời gian",
            list(
                RESEARCH_PRESETS.keys()
            ),
            index=2,
            key="research_preset",
        )

        days = RESEARCH_PRESETS[
            preset
        ]

        end_date = available_end

        start_candidate = (
            pd.Timestamp(
                end_date
            )
            - pd.Timedelta(
                days=days
            )
        ).date()

        start_date = max(
            start_candidate,
            available_start,
        )

        sample_label = preset

    else:

        col1, col2 = st.columns(
            2
        )

        with col1:

            start_date = st.date_input(
                "Từ ngày",
                value=available_start,
                min_value=available_start,
                max_value=available_end,
                key="research_start_date",
            )

        with col2:

            end_date = st.date_input(
                "Đến ngày",
                value=available_end,
                min_value=available_start,
                max_value=available_end,
                key="research_end_date",
            )

        sample_label = (
            f"{start_date.strftime('%d/%m/%Y')} "
            f"→ "
            f"{end_date.strftime('%d/%m/%Y')}"
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    if start_date > end_date:

        st.error(
            "Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc."
        )

        return

    sample = slice_sample(
        source_data,
        start_date,
        end_date,
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
            "Mẫu chọn",
            sample_label,
        )

    with b:

        st.metric(
            "Quan sát",
            f"{len(sample):,}",
        )

    with c:

        actual_start, actual_end = (
            get_available_date_range(
                sample
            )
        )

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
    # HORIZONS
    # ========================================================

    selected_horizons = st.multiselect(
        "Horizon lợi suất tương lai",
        list(
            HORIZONS.keys()
        ),
        default=list(
            HORIZONS.keys()
        ),
        key="research_horizons",
    )

    # ========================================================
    # SAMPLE WARNING
    # ========================================================

    if len(sample) < 30:

        st.error(
            "Mẫu quá nhỏ. Nên tăng khoảng thời gian."
        )

    elif len(sample) < 60:

        st.warning(
            "Mẫu còn nhỏ; kết luận thống kê có thể kém ổn định."
        )

    elif len(sample) < 120:

        st.info(
            "Mẫu đủ để chạy thử, nhưng nên ưu tiên mẫu dài hơn "
            "nếu cần kết luận nghiên cứu chắc hơn."
        )

    else:

        st.success(
            "Quy mô mẫu tương đối tốt cho nghiên cứu."
        )

    # ========================================================
    # RUN
    # ========================================================

    if st.button(
        "🚀 Chạy nghiên cứu trên đúng mẫu này",
        type="primary",
        width="stretch",
        key="run_research_button",
    ):

        if not selected_horizons:

            st.warning(
                "Chọn ít nhất một horizon."
            )

            return

        with st.spinner(
            "Đang chạy mô hình và kiểm định..."
        ):

            result = run_research(
                sample
            )

        st.session_state[
            "research_result"
        ] = result

        st.session_state[
            "research_symbol"
        ] = symbol

        st.session_state[
            "research_start_saved"
        ] = start_date

        st.session_state[
            "research_end_saved"
        ] = end_date

        st.session_state[
            "research_horizons_saved"
        ] = selected_horizons

    # ========================================================
    # RESULTS
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
            "Chọn mẫu rồi bấm chạy nghiên cứu."
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
        "research_start_saved"
    )

    saved_end = st.session_state.get(
        "research_end_saved"
    )

    saved_horizons = st.session_state.get(
        "research_horizons_saved",
        [],
    )

    st.success(
        f"Nghiên cứu đã chạy trên mẫu "
        f"**{saved_start.strftime('%d/%m/%Y')} → "
        f"{saved_end.strftime('%d/%m/%Y')}** · "
        f"{len(sample):,} quan sát."
    )

    # ========================================================
    # ALL VARIABLES
    # ========================================================

    features = result.get(
        "features",
        [],
    )

    st.subheader(
        "🔢 Biến được đưa vào nghiên cứu"
    )

    st.metric(
        "Tổng biến",
        f"{len(features):,}",
    )

    with st.expander(
        "Xem toàn bộ biến",
        expanded=False,
    ):

        st.dataframe(
            pd.DataFrame(
                {
                    "STT": range(
                        1,
                        len(features)
                        + 1,
                    ),
                    "Biến": features,
                }
            ),
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # HORIZONS
    # ========================================================

    for horizon in HORIZONS:

        if horizon not in saved_horizons:
            continue

        if horizon not in result[
            "horizons"
        ]:
            continue

        item = result[
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

        st.header(
            f"📈 Kết quả lợi suất tương lai {label}"
        )

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

        # ----------------------------------------------------
        # TOP FACTOR
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

            st.subheader(
                "🎯 Yếu tố ảnh hưởng nổi bật nhất"
            )

            top_factor = str(
                top[
                    "Biến"
                ]
            )

            top_score = num(
                top[
                    "Score"
                ],
                0,
            )

            top_relation = str(
                top[
                    "Quan hệ"
                ]
            )

            st.success(
                f"**{top_factor}** · "
                f"Score {top_score:.1f}/100 · "
                f"{top_relation}"
            )

            st.markdown(
                "#### 🏆 Xếp hạng yếu tố"
            )

            show = ranking.head(
                20
            ).copy()

            show[
                "Score"
            ] = show[
                "Score"
            ].round(1)

            show[
                "Spearman"
            ] = show[
                "Spearman"
            ].round(4)

            show[
                "Beta"
            ] = show[
                "Beta"
            ].round(4)

            show[
                "p-value"
            ] = show[
                "p-value"
            ].round(5)

            show[
                "Permutation"
            ] = show[
                "Permutation"
            ].round(6)

            st.dataframe(
                show[
                    [
                        "Biến",
                        "Score",
                        "Quan hệ",
                        "Spearman",
                        "Beta",
                        "p-value",
                        "Permutation",
                        "Ý nghĩa",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            # ------------------------------------------------
            # FACTOR EXPLANATION
            # ------------------------------------------------

            st.markdown(
                "#### 🧠 Diễn giải"
            )

            for _, row in ranking.head(
                5
            ).iterrows():

                factor = str(
                    row.get(
                        "Biến"
                    )
                )

                relation = str(
                    row.get(
                        "Quan hệ"
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

                p = num(
                    row.get(
                        "p-value"
                    )
                )

                text = (
                    f"**{factor}** có quan hệ "
                    f"{relation.lower()}"
                )

                if spear is not None:

                    text += (
                        f", Spearman = {spear:+.4f}"
                    )

                if beta is not None:

                    text += (
                        f", Beta = {beta:+.4f}"
                    )

                if p is not None:

                    text += (
                        f", p-value = {p:.5f}"
                    )

                st.markdown(
                    "• " + text + "."
                )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

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

            st.dataframe(
                models.round(6),
                width="stretch",
                hide_index=True,
            )

            best = models.iloc[
                0
            ]

            st.info(
                f"Model tốt nhất theo RMSE: "
                f"**{best['Mô hình']}** · "
                f"RMSE = {best['RMSE']:.6f} · "
                f"R² = {best['R²']:.4f}"
            )

            r2 = num(
                best["R²"]
            )

            if r2 is not None:

                if r2 < 0:

                    st.warning(
                        "R² âm: mô hình chưa dự báo tốt hơn benchmark trung bình."
                    )

                elif r2 < 0.10:

                    st.info(
                        "R² thấp: tín hiệu giải thích còn yếu."
                    )

                elif r2 < 0.25:

                    st.success(
                        "Có tín hiệu dự báo nhưng mức giải thích vẫn vừa phải."
                    )

                else:

                    st.success(
                        "Mức giải thích tương đối đáng chú ý trong mẫu."
                    )

        # ----------------------------------------------------
        # TESTS
        # ----------------------------------------------------

        tests = item.get(
            "tests",
            {},
        )

        st.subheader(
            "🧪 Kiểm định"
        )

        # ADF
        if "ADF" in tests:

            t = tests[
                "ADF"
            ]

            p = num(
                t.get(
                    "p_value"
                )
            )

            a, b = st.columns(2)

            with a:

                st.metric(
                    "ADF",
                    format_number(
                        t.get(
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

            if p is not None and p < 0.05:

                st.success(
                    "ADF: bác bỏ nghiệm đơn vị ở mức 5%."
                )

            else:

                st.warning(
                    "ADF: chưa đủ bằng chứng bác bỏ nghiệm đơn vị."
                )

        # JB
        if "Jarque-Bera" in tests:

            t = tests[
                "Jarque-Bera"
            ]

            p = num(
                t.get(
                    "p_value"
                )
            )

            if p is not None and p < 0.05:

                st.warning(
                    "Jarque-Bera: phần dư không gần phân phối chuẩn."
                )

            else:

                st.success(
                    "Jarque-Bera: chưa thấy lệch chuẩn mạnh."
                )

        # BP
        if "Breusch-Pagan" in tests:

            t = tests[
                "Breusch-Pagan"
            ]

            p = num(
                t.get(
                    "p_value"
                )
            )

            if p is not None and p < 0.05:

                st.warning(
                    "Breusch-Pagan: có dấu hiệu phương sai thay đổi."
                )

            else:

                st.success(
                    "Breusch-Pagan: chưa thấy phương sai thay đổi rõ."
                )

        # WHITE
        if "White" in tests:

            t = tests[
                "White"
            ]

            p = num(
                t.get(
                    "p_value"
                )
            )

            if p is not None and p < 0.05:

                st.warning(
                    "White: có dấu hiệu phương sai thay đổi."
                )

            else:

                st.success(
                    "White: chưa thấy phương sai thay đổi rõ."
                )

        # DW
        if "Durbin-Watson" in tests:

            value = num(
                tests[
                    "Durbin-Watson"
                ].get(
                    "statistic"
                )
            )

            st.metric(
                "Durbin-Watson",
                format_number(
                    value,
                    4,
                ),
            )

            if value is not None:

                if 1.5 <= value <= 2.5:

                    st.success(
                        "DW tương đối gần 2."
                    )

                elif value < 1.5:

                    st.warning(
                        "DW có dấu hiệu tự tương quan dương."
                    )

                else:

                    st.warning(
                        "DW có dấu hiệu tự tương quan âm."
                    )

        # LJUNG BOX
        if "Ljung-Box" in tests:

            t = tests[
                "Ljung-Box"
            ]

            p = num(
                t.get(
                    "p_value"
                )
            )

            if p is not None and p < 0.05:

                st.warning(
                    "Ljung-Box: còn tự tương quan trong phần dư."
                )

            else:

                st.success(
                    "Ljung-Box: chưa phát hiện tự tương quan phần dư rõ."
                )

        # ----------------------------------------------------
        # CONCLUSION
        # ----------------------------------------------------

        st.subheader(
            "🧠 Kết luận nghiên cứu"
        )

        conclusions = build_conclusion(
            item
        )

        if conclusions:

            for conclusion in conclusions:

                st.markdown(
                    "• " + conclusion
                )

        st.warning(
            "Kết quả mô hình biểu thị quan hệ và khả năng dự báo "
            "trong mẫu đã chọn, không phải bằng chứng nhân quả."
        )

    # ========================================================
    # CROSS HORIZON
    # ========================================================

    st.divider()

    st.header(
        "🏆 Yếu tố nhất quán giữa các horizon"
    )

    cross_rows = []

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

        for rank, (_, row) in enumerate(
            ranking.head(
                20
            ).iterrows(),
            start=1,
        ):

            cross_rows.append(
                {
                    "Horizon": horizon,
                    "Biến": row[
                        "Biến"
                    ],
                    "Rank": rank,
                    "Score": num(
                        row[
                            "Score"
                        ],
                        0,
                    ),
                }
            )

    if cross_rows:

        cross = pd.DataFrame(
            cross_rows
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

        stable = (
            stable
            .sort_values(
                [
                    "So_horizon",
                    "Score_TB",
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
            "Số horizon",
            "Rank TB",
            "Score TB",
        ]

        st.dataframe(
            stable.head(
                20
            ),
            width="stretch",
            hide_index=True,
        )

        if not stable.empty:

            st.success(
                f"Yếu tố nổi bật nhất xuyên horizon: "
                f"**{stable.iloc[0]['Biến']}**."
            )


# ============================================================
# MAIN PAGE
# ============================================================

def render_stock_analysis():

    st.caption(
        "BRIAN STOCK · STOCK RESEARCH"
    )

    st.title(
        "Phân tích cổ phiếu"
    )

    st.write(
        "Phân tích kỹ thuật và nghiên cứu định lượng "
        "trên mẫu thời gian do người dùng lựa chọn."
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

        clean = normalize_symbol(
            symbol_input
        )

        if clean:

            st.session_state[
                "stock_analysis_symbol"
            ] = clean

            # reset research
            for key in [
                "research_source_data",
                "research_source_symbol",
                "research_result",
                "research_symbol",
            ]:

                st.session_state.pop(
                    key,
                    None,
                )

            st.rerun()

        else:

            st.warning(
                "Vui lòng nhập mã cổ phiếu."
            )

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
            f"Không thể tải dữ liệu {display_symbol(symbol)}."
        )

        st.caption(
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
    # METRICS
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
            (
                f"{rsi_value:.1f}"
                if rsi_value is not None
                else "—"
            ),
        )

    with d:

        st.metric(
            "Khối lượng",
            format_volume(
                volume
            ),
        )

    e, f, g, h = st.columns(
        4
    )

    with e:

        st.metric(
            "MA20",
            format_price(
                sma20
            ),
        )

    with f:

        st.metric(
            "MA50",
            format_price(
                sma50
            ),
        )

    with g:

        st.metric(
            "MACD",
            format_number(
                macd,
                3,
            ),
        )

    with h:

        st.metric(
            "Biến động 20 phiên",
            (
                f"{volatility:.2f}%"
                if volatility is not None
                else "—"
            ),
        )

    # ========================================================
    # TECHNICAL
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
    # QUANT
    # ========================================================

    render_quant_research(
        symbol
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def render_analysis():
    render_stock_analysis()
