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

# Mục tiêu: giữ nhiều biến để nghiên cứu nhưng không để OLS/VIF
# chạy trên ma trận khổng lồ gây rank-deficient / treo CPU.
MAX_MODEL_FEATURES = 40
MAX_OLS_FEATURES = 15
MAX_VIF_FEATURES = 12
MAX_DISPLAY_FEATURES = 80

TRAIN_RATIO = 0.80
MIN_OBSERVATIONS = 60

# Các biến cấp độ giá trực tiếp thường gây đa cộng tuyến cực mạnh.
PRICE_LEVEL_PREFIXES = (
    "ema",
    "sma",
    "bollinger_",
    "high",
    "low",
)

# ============================================================
# EXPECTED FACTOR UNIVERSE
# ============================================================

EXPECTED_TECHNICAL = [
    "Return",
    "ReturnPct",
    "LogReturn",
    "RSI",
    "EMA9",
    "EMA12",
    "EMA20",
    "EMA26",
    "EMA50",
    "EMA100",
    "EMA200",
    "SMA5",
    "SMA10",
    "SMA20",
    "SMA50",
    "SMA100",
    "SMA200",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "Price_vs_SMA100",
    "Price_vs_SMA200",
    "Price_vs_EMA20",
    "Price_vs_EMA50",
    "Price_vs_EMA100",
    "Price_vs_EMA200",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Bollinger_Mid",
    "Bollinger_Upper",
    "Bollinger_Lower",
    "Bollinger_Width",
    "Bollinger_Position",
    "Volatility5",
    "Volatility20",
    "Volatility60",
    "Volatility_20D",
    "Range",
    "Range_Percent",
    "Body",
    "Body_Percent",
    "TrueRange",
    "ATR14",
    "ATR_Percent",
    "Momentum1",
    "Momentum5",
    "Momentum10",
    "Momentum20",
    "Momentum60",
    "Momentum1Pct",
    "Momentum5Pct",
    "Momentum10Pct",
    "Momentum20Pct",
    "Momentum60Pct",
    "High20",
    "High50",
    "High252",
    "Low20",
    "Low50",
    "Low252",
    "Distance_From_High20",
    "Distance_From_High50",
    "Distance_From_High252",
    "Distance_From_Low20",
    "Distance_From_Low50",
    "Distance_From_Low252",
    "Stochastic_K",
    "Stochastic_D",
    "ROC5",
    "ROC10",
    "ROC20",
    "OBV_Proxy",
    "Dollar_Volume",
    "Volume",
    "Volume_SMA5",
    "Volume_SMA20",
    "Volume_SMA50",
    "Volume_Change",
    "Relative_Volume",
    "Value",
    "Trading_Value",
    "Trading_Value_Change",
    "Trading_Value_SMA20",
]

EXPECTED_FLOW = [
    "foreign_net_val_calc",
    "foreign_net_vol_calc",
    "foreign_net_value_to_trading_value",
    "foreign_buy_val",
    "foreign_sell_val",
    "foreign_buy_vol",
    "foreign_sell_vol",
    "foreign_buy_sell_value_ratio",
    "foreign_buy_sell_volume_ratio",
    "proprietary_net_val_calc",
    "proprietary_net_vol_calc",
    "proprietary_net_value_to_trading_value",
    "proprietary_buy_val",
    "proprietary_sell_val",
    "proprietary_buy_vol",
    "proprietary_sell_vol",
    "proprietary_buy_sell_value_ratio",
    "proprietary_buy_sell_volume_ratio",
]

EXPECTED_MARKET = [
    "market_vnindex_close",
    "market_vnindex_return",
    "market_vnindex_return_pct",
    "market_vnindex_momentum20",
    "market_vnindex_volatility20",
    "market_vnindex_volume",
    "market_vnindex_value",
    "market_vn30_close",
    "market_vn30_return",
    "market_vn30_return_pct",
    "market_vn30_momentum20",
    "market_vn30_volatility20",
    "market_vn30_volume",
    "market_vn30_value",
    "market_hnx_close",
    "market_hnx_return",
    "market_hnx_return_pct",
    "market_hnx_momentum20",
    "market_hnx_volatility20",
    "market_hnx_volume",
    "market_hnx_value",
    "market_positive_index_count",
    "market_negative_index_count",
    "market_average_return",
    "stock_minus_market_1d",
    "stock_minus_market_momentum20",
]

EXPECTED_SECTOR = [
    "sector_return",
    "sector_return_pct",
    "sector_momentum20",
    "sector_momentum60",
    "sector_volatility20",
    "sector_positive_ratio",
    "sector_negative_ratio",
    "sector_peer_count",
    "stock_minus_sector_1d",
    "stock_minus_sector_momentum20",
]

EXPECTED_OTHER = [
    "Open",
    "High",
    "Low",
    "Close",
]

TARGET_COLUMNS = {
    "Target_1D",
    "Target_5D",
    "Target_20D",
    "Target_1D_Pct",
    "Target_5D_Pct",
    "Target_20D_Pct",
}

# ============================================================
# BASIC HELPERS
# ============================================================

def num(value: Any, default=None):
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def format_price(value):
    value = num(value)
    return "—" if value is None else f"{value:,.0f} đồng"


def format_percent(value):
    value = num(value)
    return "—" if value is None else f"{value:+.2f}%"


def format_number(value, digits=4):
    value = num(value)
    return "—" if value is None else f"{value:.{digits}f}"


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
        return "Trên MA20" if price > sma20 else "Dưới MA20"
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
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work.index = pd.to_datetime(work.index, errors="coerce")
    work = work[~work.index.isna()].copy()
    work = work.sort_index()
    work = work[~work.index.duplicated(keep="last")].copy()
    return work


def slice_date_range(df, start_date, end_date):
    work = normalize_index(df)
    if work.empty:
        return work
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return work.loc[(work.index >= start_ts) & (work.index <= end_ts)].copy()


def actual_range(df):
    work = normalize_index(df)
    if work.empty:
        return None, None
    return work.index.min().date(), work.index.max().date()

# ============================================================
# FEATURE GROUPING
# ============================================================

def feature_group(feature):
    low = str(feature).lower()
    if low.startswith("foreign_"):
        return "Khối ngoại"
    if low.startswith("proprietary_"):
        return "Tự doanh"
    if low.startswith("sector_") or low.startswith("stock_minus_sector"):
        return "Nhóm ngành"
    if low.startswith("market_") or low.startswith("stock_minus_market"):
        return "Thị trường chung"
    if low.startswith("flow_"):
        return "Dòng tiền"
    if any(x in low for x in ["volume", "trading_value", "dollar_volume", "obv", "value"]):
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
        groups.setdefault(feature_group(feature), []).append(feature)
    return groups

# ============================================================
# DATASET COMPLETION / TARGETS
# ============================================================

def ensure_expected_columns(df):
    """Giữ universe biến cố định để UI luôn cho thấy đầy đủ nhóm nghiên cứu.
    Cột nào nguồn không cung cấp sẽ là NaN và không được giả mạo thành số.
    """
    work = normalize_index(df)
    if work.empty:
        return work

    expected = []
    expected.extend(EXPECTED_OTHER)
    expected.extend(EXPECTED_TECHNICAL)
    expected.extend(EXPECTED_FLOW)
    expected.extend(EXPECTED_MARKET)
    expected.extend(EXPECTED_SECTOR)

    for col in dict.fromkeys(expected):
        if col not in work.columns:
            work[col] = np.nan

    return work


def ensure_targets(df):
    work = ensure_expected_columns(df)
    if work.empty or "Close" not in work.columns:
        return work

    close = pd.to_numeric(work["Close"], errors="coerce")
    for horizon, periods in HORIZONS.items():
        target = close.shift(-periods) / close - 1
        work[f"Target_{horizon}"] = target
        work[f"Target_{horizon}_Pct"] = target * 100
    return work.replace([np.inf, -np.inf], np.nan)


def get_all_features(df):
    if df is None or df.empty:
        return []
    features = []
    for col in df.columns:
        name = str(col)
        if name in TARGET_COLUMNS or name.lower().startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return list(dict.fromkeys(features))

# ============================================================
# FEATURE ENGINEERING EXTRA
# ============================================================

def add_research_extras(df):
    """Bổ sung biến chéo và rolling mà data.market có thể chưa có."""
    work = ensure_expected_columns(df).copy()
    if work.empty:
        return work

    close = pd.to_numeric(work["Close"], errors="coerce")
    ret = close.pct_change()

    # Lag / acceleration
    work["Return_Lag1"] = ret.shift(1)
    work["Return_Lag5"] = ret.shift(5)
    work["Return_MA5"] = ret.rolling(5).mean()
    work["Return_MA20"] = ret.rolling(20).mean()
    work["Return_Std20"] = ret.rolling(20).std()
    work["Return_Skew20"] = ret.rolling(20).skew()

    # Trend slope proxy via normalized price distance
    if "SMA20" in work.columns:
        work["Trend_Slope_20"] = work["SMA20"].pct_change(5)
    if "SMA50" in work.columns:
        work["Trend_Slope_50"] = work["SMA50"].pct_change(10)

    # Beta / residual-style market factors
    if "market_vnindex_return" in work.columns:
        mkt = pd.to_numeric(work["market_vnindex_return"], errors="coerce")
        cov = ret.rolling(60).cov(mkt)
        var = mkt.rolling(60).var().replace(0, np.nan)
        work["Market_Beta_60"] = cov / var
        work["Relative_Return_vs_VNINDEX"] = ret - mkt

    # Sector relative strength
    if "sector_return" in work.columns:
        sec = pd.to_numeric(work["sector_return"], errors="coerce")
        work["Relative_Return_vs_Sector"] = ret - sec
        work["Sector_Relative_Strength_20"] = (
            (1 + ret).rolling(20).apply(np.prod, raw=True)
            / (1 + sec).rolling(20).apply(np.prod, raw=True)
            - 1
        )

    # Flow normalized by trading value
    if "Trading_Value" in work.columns:
        tv = pd.to_numeric(work["Trading_Value"], errors="coerce").replace(0, np.nan)
        for src, out in [
            ("foreign_net_val_calc", "Foreign_Flow_Intensity"),
            ("proprietary_net_val_calc", "Proprietary_Flow_Intensity"),
        ]:
            if src in work.columns:
                work[out] = pd.to_numeric(work[src], errors="coerce") / tv

    # Flow persistence
    for col in ["foreign_net_val_calc", "proprietary_net_val_calc"]:
        if col in work.columns:
            s = pd.to_numeric(work[col], errors="coerce")
            work[f"{col}_sma5"] = s.rolling(5).mean()
            work[f"{col}_sma20"] = s.rolling(20).mean()
            work[f"{col}_cum20"] = s.rolling(20).sum()

    return work.replace([np.inf, -np.inf], np.nan)

# ============================================================
# PREPARE MODEL DATA
# ============================================================

def prepare_xy(df, features, target_col):
    if df is None or df.empty or target_col not in df.columns:
        return None

    cols = [x for x in features if x in df.columns]
    if not cols:
        return None

    X = df[cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median = X[col].median()
        X[col] = X[col].fillna(0.0 if pd.isna(median) else median)

    valid = y.notna() & np.isfinite(y)
    X = X.loc[valid]
    y = y.loc[valid]

    # Loại variance=0 cho matrix model nhưng không xóa khỏi dataset.
    usable = []
    for col in X.columns:
        variance = float(X[col].var(ddof=0)) if len(X) else 0.0
        if np.isfinite(variance) and variance > 0:
            usable.append(col)

    if not usable:
        return None

    return X[usable].astype(float), y.astype(float)

# ============================================================
# CORRELATION ALL FEATURES
# ============================================================

def correlation_table(X, y):
    rows = []
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce")
        yy = pd.to_numeric(y, errors="coerce")
        mask = x.notna() & yy.notna() & np.isfinite(x) & np.isfinite(yy)
        x2 = x.loc[mask]
        y2 = yy.loc[mask]
        if len(x2) < 20 or x2.nunique() <= 1 or y2.nunique() <= 1:
            continue
        try:
            pearson = float(x2.corr(y2, method="pearson"))
        except Exception:
            pearson = np.nan
        try:
            spearman = float(x2.corr(y2, method="spearman"))
        except Exception:
            spearman = np.nan
        rows.append({
            "Biến": col,
            "Nhóm": feature_group(col),
            "Pearson": pearson,
            "Spearman": spearman,
            "AbsSpearman": abs(spearman) if np.isfinite(spearman) else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("AbsSpearman", ascending=False, na_position="last").reset_index(drop=True)

# ============================================================
# FAST FEATURE SELECTION
# ============================================================

def select_model_features(X, y, max_features=MAX_MODEL_FEATURES):
    """Giữ toàn bộ biến trong dataset; chỉ giảm số chiều lúc fit model.
    Ưu tiên mỗi nhóm có đại diện để flow/market/sector không bị biến kỹ thuật lấn át hết.
    """
    if X is None or X.empty:
        return []

    corr = correlation_table(X, y)
    if corr.empty:
        return list(X.columns[:max_features])

    selected = []
    by_group = {g: [] for g in [
        "Kỹ thuật", "Dòng tiền", "Khối ngoại", "Tự doanh", "Thị trường chung", "Nhóm ngành"
    ]}

    for _, row in corr.iterrows():
        by_group.setdefault(row["Nhóm"], []).append(row["Biến"])

    # Vòng 1: ít nhất vài biến cho mỗi nhóm nếu có.
    for group in list(by_group.keys()):
        for col in by_group[group][:3]:
            if col not in selected and len(selected) < max_features:
                selected.append(col)

    # Vòng 2: lấy theo correlation rank.
    for col in corr["Biến"].tolist():
        if len(selected) >= max_features:
            break
        if col not in selected:
            selected.append(col)

    return selected[:max_features]


def select_ols_features(X, y, max_features=MAX_OLS_FEATURES):
    if X is None or X.empty:
        return []

    corr = correlation_table(X, y)
    ranked = corr["Biến"].tolist() if not corr.empty else list(X.columns)

    selected = []
    for col in ranked:
        if len(selected) >= max_features:
            break
        if not selected:
            selected.append(col)
            continue
        okay = True
        for existing in selected:
            try:
                pair = abs(float(X[col].corr(X[existing])))
            except Exception:
                pair = 0.0
            if pair >= 0.90:
                okay = False
                break
        if okay:
            selected.append(col)
    return selected[:max_features]

# ============================================================
# OLS
# ============================================================

def run_safe_ols(X_train, y_train):
    try:
        import statsmodels.api as sm
        from sklearn.preprocessing import StandardScaler

        selected = select_ols_features(X_train, y_train)
        if not selected:
            return {"model": None, "table": pd.DataFrame(), "features": []}

        X = X_train[selected].copy()
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=selected, index=X.index
        )
        model = sm.OLS(y_train, sm.add_constant(scaled, has_constant="add")).fit(cov_type="HC3")

        rows = []
        for col in selected:
            beta = num(model.params.get(col))
            pvalue = num(model.pvalues.get(col))
            stderr = num(model.bse.get(col))
            rows.append({
                "Biến": col,
                "Nhóm": feature_group(col),
                "Beta": beta,
                "p-value": pvalue,
                "Std Error": stderr,
                "AbsBeta": abs(beta) if beta is not None else np.nan,
            })
        table = pd.DataFrame(rows).sort_values("AbsBeta", ascending=False, na_position="last").reset_index(drop=True)
        return {
            "model": model,
            "table": table,
            "features": selected,
            "r2": num(model.rsquared),
            "adj_r2": num(model.rsquared_adj),
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
# FAST MODELS
# ============================================================

def run_models(X_train, X_test, y_train, y_test):
    try:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return pd.DataFrame(), {}

    models = [
        ("Ridge", Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])),
        ("Random Forest", RandomForestRegressor(
            n_estimators=180, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )),
        ("Extra Trees", ExtraTreesRegressor(
            n_estimators=180, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )),
    ]

    actual = np.asarray(y_test, dtype=float)
    baseline = np.repeat(float(np.mean(np.asarray(y_train, dtype=float))), len(actual))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline)))

    rows = []
    fitted = {}
    for name, model in models:
        try:
            model.fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test), dtype=float)
            mse = float(mean_squared_error(actual, pred))
            rmse = float(np.sqrt(mse))
            rows.append({
                "Mô hình": name,
                "MAE": float(mean_absolute_error(actual, pred)),
                "RMSE": rmse,
                "R²": float(r2_score(actual, pred)),
                "Baseline_RMSE": baseline_rmse,
                "So_baseline": "Tốt hơn" if rmse < baseline_rmse else "Kém hơn",
            })
            fitted[name] = model
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(), {}

    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True), fitted

# ============================================================
# IMPORTANCE
# ============================================================

def run_permutation(model, X_test, y_test):
    try:
        from sklearn.inspection import permutation_importance
        out = permutation_importance(
            model, X_test, y_test,
            n_repeats=3, random_state=42, n_jobs=-1,
            scoring="neg_mean_squared_error",
        )
        table = pd.DataFrame({
            "Biến": X_test.columns,
            "Permutation": out.importances_mean,
            "Permutation_STD": out.importances_std,
        })
        table["AbsPermutation"] = table["Permutation"].abs()
        table["Nhóm"] = table["Biến"].apply(feature_group)
        return table.sort_values("AbsPermutation", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def run_tree_importance(model, features):
    try:
        raw = model
        if hasattr(model, "named_steps"):
            raw = model.named_steps.get("model", model)
        if not hasattr(raw, "feature_importances_"):
            return pd.DataFrame()
        table = pd.DataFrame({
            "Biến": features,
            "TreeImportance": np.asarray(raw.feature_importances_, dtype=float),
        })
        table["Nhóm"] = table["Biến"].apply(feature_group)
        return table.sort_values("TreeImportance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# ============================================================
# FACTOR RANKING
# ============================================================

def percentile_score(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(np.where(s.notna(), 100.0, 0.0), index=s.index)
    return s.rank(pct=True, na_option="keep") * 100


def build_factor_ranking(correlation, ols, permutation, tree):
    if correlation is None or correlation.empty:
        return pd.DataFrame()

    result = correlation.copy()

    if isinstance(ols, pd.DataFrame) and not ols.empty:
        result = result.merge(ols[["Biến", "Beta", "p-value", "AbsBeta"]], on="Biến", how="left")
    else:
        result["Beta"] = np.nan
        result["p-value"] = np.nan
        result["AbsBeta"] = np.nan

    if isinstance(permutation, pd.DataFrame) and not permutation.empty:
        result = result.merge(permutation[["Biến", "Permutation", "AbsPermutation"]], on="Biến", how="left")
    else:
        result["Permutation"] = np.nan
        result["AbsPermutation"] = np.nan

    if isinstance(tree, pd.DataFrame) and not tree.empty:
        result = result.merge(tree[["Biến", "TreeImportance"]], on="Biến", how="left")
    else:
        result["TreeImportance"] = np.nan

    result["CorrScore"] = percentile_score(result["AbsSpearman"])
    result["BetaScore"] = percentile_score(result["AbsBeta"])
    result["PermutationScore"] = percentile_score(result["AbsPermutation"])
    result["TreeScore"] = percentile_score(result["TreeImportance"])

    # Có score thì mới đóng góp; không có dữ liệu thì không giả mạo thành 0 hoàn toàn.
    score_parts = pd.concat([
        result["CorrScore"],
        result["BetaScore"],
        result["PermutationScore"],
        result["TreeScore"],
    ], axis=1)
    result["Score"] = score_parts.mean(axis=1, skipna=True)

    def direction(row):
        beta = num(row.get("Beta"))
        spear = num(row.get("Spearman"))
        value = beta if beta is not None else spear
        if value is None:
            return "Không xác định"
        if value > 0:
            return "Cùng chiều"
        if value < 0:
            return "Ngược chiều"
        return "Trung tính"

    def significance(p):
        p = num(p)
        if p is None:
            return "Không có p-value"
        if p < 0.01:
            return "Rất mạnh"
        if p < 0.05:
            return "Có ý nghĩa 5%"
        if p < 0.10:
            return "Có tín hiệu 10%"
        return "Chưa rõ"

    result["Quan hệ"] = result.apply(direction, axis=1)
    result["Ý nghĩa"] = result["p-value"].apply(significance)
    return result.sort_values(["Score", "AbsSpearman"], ascending=False, na_position="last").reset_index(drop=True)


def group_summary(ranking):
    if ranking is None or ranking.empty:
        return pd.DataFrame()
    return (
        ranking.groupby("Nhóm", as_index=False)
        .agg(
            Score_TB=("Score", "mean"),
            Score_Max=("Score", "max"),
            So_bien=("Biến", "count"),
        )
        .sort_values(["Score_TB", "Score_Max"], ascending=False)
        .reset_index(drop=True)
    )

# ============================================================
# TESTS - LIGHTWEIGHT
# ============================================================

def run_tests(X_train, y_train, ols):
    tests = {}
    try:
        from statsmodels.tsa.stattools import adfuller
        out = adfuller(np.asarray(y_train, dtype=float), autolag="AIC")
        tests["ADF"] = {"statistic": float(out[0]), "p_value": float(out[1])}
    except Exception:
        pass

    residuals = None
    try:
        model = ols.get("model")
        if model is not None:
            residuals = np.asarray(model.resid, dtype=float)
    except Exception:
        pass

    if residuals is None or len(residuals) < 20:
        return tests

    try:
        from statsmodels.stats.stattools import durbin_watson, jarque_bera
        tests["Jarque-Bera"] = {
            "statistic": float(jarque_bera(residuals)[0]),
            "p_value": float(jarque_bera(residuals)[1]),
        }
        tests["Durbin-Watson"] = {"statistic": float(durbin_watson(residuals))}
    except Exception:
        pass

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lag = min(10, max(1, len(residuals) // 10))
        lb = acorr_ljungbox(residuals, lags=[lag], return_df=True)
        tests["Ljung-Box"] = {
            "lag": lag,
            "statistic": float(lb["lb_stat"].iloc[-1]),
            "p_value": float(lb["lb_pvalue"].iloc[-1]),
        }
    except Exception:
        pass

    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
        features = ols.get("features", [])
        xbp = X_train[[c for c in features if c in X_train.columns]]
        if not xbp.empty:
            bp = het_breuschpagan(residuals, sm.add_constant(xbp, has_constant="add"))
            tests["Breusch-Pagan"] = {"statistic": float(bp[0]), "p_value": float(bp[1])}
    except Exception:
        pass

    return tests

# ============================================================
# ONE HORIZON
# ============================================================

def execute_one_horizon(df, all_features, horizon):
    xy = prepare_xy(df, all_features, f"Target_{horizon}")
    if xy is None:
        return None
    X_all, y = xy
    if len(X_all) < MIN_OBSERVATIONS:
        return None

    split = int(len(X_all) * TRAIN_RATIO)
    if split < 30 or len(X_all) - split < 10:
        return None

    # ALL FEATURES -> correlation, nhưng chỉ feature subset -> model.
    selected = select_model_features(X_all, y, MAX_MODEL_FEATURES)
    X = X_all[selected].copy()

    X_train = X.iloc[:split].copy()
    X_test = X.iloc[split:].copy()
    y_train = y.iloc[:split].copy()
    y_test = y.iloc[split:].copy()

    # Correlation vẫn dùng TOÀN BỘ biến của dataset.
    correlations = correlation_table(X_all, y)

    ols = run_safe_ols(X_train, y_train)
    ols_table = ols.get("table", pd.DataFrame())

    models, fitted = run_models(X_train, X_test, y_train, y_test)

    best_name = None
    best_model = None
    if not models.empty:
        best_name = str(models.iloc[0]["Mô hình"])
        best_model = fitted.get(best_name)

    permutation = run_permutation(best_model, X_test, y_test) if best_model is not None else pd.DataFrame()
    tree = run_tree_importance(best_model, list(X.columns)) if best_model is not None else pd.DataFrame()

    ranking = build_factor_ranking(correlations, ols_table, permutation, tree)
    groups = group_summary(ranking)

    # VIF chỉ trên OLS subset để không treo.
    vif = pd.DataFrame()
    ols_features = ols.get("features", [])
    if ols_features:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            data_vif = X_train[ols_features].copy()
            rows = []
            vals = data_vif.values.astype(float)
            for i, col in enumerate(data_vif.columns):
                try:
                    v = float(variance_inflation_factor(vals, i))
                except Exception:
                    v = np.inf
                rows.append({"Biến": col, "Nhóm": feature_group(col), "VIF": v})
            vif = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
        except Exception:
            vif = pd.DataFrame()

    tests = run_tests(X_train, y_train, ols)

    current_price = num(df["Close"].iloc[-1]) if "Close" in df.columns else None
    predicted_return = None
    predicted_price = None

    if best_model is not None:
        try:
            latest = df[selected].copy()
            for col in latest.columns:
                latest[col] = pd.to_numeric(latest[col], errors="coerce")
                med = latest[col].median()
                latest[col] = latest[col].fillna(0.0 if pd.isna(med) else med)
            predicted_return = float(best_model.predict(latest.iloc[[-1]])[0])
            if current_price is not None:
                predicted_price = current_price * (1 + predicted_return)
        except Exception:
            pass

    quality = "Chưa đánh giá"
    if not models.empty:
        r2 = num(models.iloc[0].get("R²"))
        if r2 is not None:
            quality = "Yếu" if r2 < 0 else "Thấp" if r2 < 0.10 else "Trung bình" if r2 < 0.25 else "Khá"

    return {
        "horizon": horizon,
        "observations": len(X_all),
        "train": len(X_train),
        "test": len(X_test),
        "all_features": list(X_all.columns),
        "model_features": list(X.columns),
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
# FULL RESEARCH ENGINE
# ============================================================

def execute_full_research(sample):
    prepared = add_research_extras(ensure_targets(sample))
    if prepared.empty:
        return {"ok": False, "error": "Không có dữ liệu nghiên cứu."}

    all_features = get_all_features(prepared)
    if not all_features:
        return {"ok": False, "error": "Không tìm thấy biến định lượng."}

    horizon_results = {}
    for horizon in HORIZONS:
        item = execute_one_horizon(prepared, all_features, horizon)
        if item is not None:
            horizon_results[horizon] = item

    if not horizon_results:
        return {"ok": False, "error": "Không có horizon nào đủ dữ liệu."}

    return {
        "ok": True,
        "data": prepared,
        "all_features": all_features,
        "groups": group_features(all_features),
        "horizons": horizon_results,
    }

# ============================================================
# DISPLAY DATA
# ============================================================

@st.cache_data(ttl=CACHE_TTL_DISPLAY, show_spinner=False)
def load_display_data(symbol):
    return load_market_data(symbol, "1y")

# ============================================================
# RENDERERS
# ============================================================

def render_forecast(item):
    f = item.get("forecast", {})
    current = num(f.get("current_price"))
    pred = num(f.get("predicted_return"))
    price = num(f.get("predicted_price"))

    st.subheader("🎯 Dự báo")
    a, b, c, d = st.columns(4)
    with a: st.metric("Giá hiện tại", format_price(current))
    with b: st.metric("Lợi suất dự báo", format_percent(pred * 100 if pred is not None else None))
    with c: st.metric("Giá dự báo", format_price(price))
    with d: st.metric("Model", str(item.get("best_model", "—")))

    if pred is not None:
        if pred > 0:
            st.success(f"Mô hình nghiêng về **TĂNG** {pred * 100:+.2f}% trong {item['horizon']}.")
        elif pred < 0:
            st.warning(f"Mô hình nghiêng về **GIẢM** {pred * 100:+.2f}% trong {item['horizon']}.")
        else:
            st.info("Mô hình nghiêng về đi ngang.")


def render_groups(item):
    groups = item.get("groups")
    if not isinstance(groups, pd.DataFrame) or groups.empty:
        return
    st.subheader("🧩 Nhóm yếu tố")
    show = groups.rename(columns={"Nhóm": "Nhóm yếu tố", "Score_TB": "Score TB", "Score_Max": "Score cao nhất", "So_bien": "Số biến"})
    st.dataframe(show, width="stretch", hide_index=True)
    top = groups.iloc[0]
    st.info(f"Nhóm đang nổi bật nhất: **{top['Nhóm']}** · Score TB {top['Score_TB']:.1f}/100.")


def render_factors(item):
    ranking = item.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        st.info("Chưa có xếp hạng yếu tố.")
        return

    st.subheader("🏆 Yếu tố ảnh hưởng / dự báo")
    show = ranking.head(MAX_DISPLAY_FEATURES).copy()
    display_cols = ["Biến", "Nhóm", "Score", "Quan hệ", "Pearson", "Spearman", "Beta", "p-value", "Permutation", "TreeImportance", "Ý nghĩa"]
    display_cols = [c for c in display_cols if c in show.columns]
    for col in ["Score", "Pearson", "Spearman", "Beta", "p-value", "Permutation", "TreeImportance"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce")
    st.dataframe(show[display_cols], width="stretch", hide_index=True)

    st.markdown("#### 🔥 5 yếu tố nổi bật nhất")
    for _, row in ranking.head(5).iterrows():
        feature = str(row.get("Biến"))
        group = str(row.get("Nhóm"))
        score = num(row.get("Score"))
        spear = num(row.get("Spearman"))
        beta = num(row.get("Beta"))
        p = num(row.get("p-value"))
        direction = str(row.get("Quan hệ", "Không xác định"))
        bits = [f"**{feature}** ({group})"]
        if score is not None: bits.append(f"Score {score:.1f}")
        bits.append(direction)
        if spear is not None: bits.append(f"Spearman {spear:+.4f}")
        if beta is not None: bits.append(f"Beta {beta:+.4f}")
        if p is not None: bits.append(f"p {p:.5f}")
        st.markdown("• " + " · ".join(bits))


def render_models(item):
    models = item.get("models")
    st.subheader("🤖 So sánh mô hình")
    if not isinstance(models, pd.DataFrame) or models.empty:
        st.warning("Không có kết quả model.")
        return
    show = models.copy()
    for col in ["MAE", "RMSE", "R²", "Baseline_RMSE"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(6)
    st.dataframe(show, width="stretch", hide_index=True)
    best = models.iloc[0]
    st.info(f"Model tốt nhất: **{best['Mô hình']}** · RMSE {best['RMSE']:.6f} · R² {best['R²']:.4f}")


def render_ols(item):
    st.subheader("📐 OLS chuẩn hóa")
    ols = item.get("ols")
    table = item.get("ols_table")
    if not isinstance(ols, dict) or ols.get("model") is None:
        st.info("OLS không chạy được trên tập biến an toàn này.")
        return
    a, b, c = st.columns(3)
    with a: st.metric("R²", format_number(ols.get("r2"), 4))
    with b: st.metric("Adjusted R²", format_number(ols.get("adj_r2"), 4))
    with c: st.metric("Biến OLS", len(ols.get("features", [])))
    if isinstance(table, pd.DataFrame) and not table.empty:
        st.dataframe(table.head(30), width="stretch", hide_index=True)
    st.caption("Toàn bộ biến vẫn được nghiên cứu; OLS chỉ dùng một tập con ít đa cộng tuyến để tránh ma trận suy biến.")


def render_vif(item):
    vif = item.get("vif")
    if not isinstance(vif, pd.DataFrame) or vif.empty:
        return
    st.subheader("🔗 VIF")
    st.dataframe(vif.head(MAX_VIF_FEATURES), width="stretch", hide_index=True)
    max_vif = num(vif["VIF"].replace([np.inf, -np.inf], np.nan).max())
    if max_vif is None:
        return
    if max_vif >= 10:
        st.warning(f"VIF cao nhất {max_vif:.2f}: đa cộng tuyến mạnh trong tập OLS.")
    elif max_vif >= 5:
        st.info(f"VIF cao nhất {max_vif:.2f}: có đa cộng tuyến đáng chú ý.")
    else:
        st.success("VIF của tập OLS tương đối an toàn.")


def render_tests(item):
    tests = item.get("tests", {})
    if not tests:
        return
    st.subheader("🧪 Kiểm định")

    adf = tests.get("ADF")
    if isinstance(adf, dict):
        p = num(adf.get("p_value"))
        with st.expander("ADF", expanded=False):
            st.write(f"Statistic: {format_number(adf.get('statistic'), 5)}")
            st.write(f"p-value: {format_number(p, 6)}")
            st.info("ADF kiểm tra tính dừng của chuỗi mục tiêu; không phải kiểm định nhân quả.")

    jb = tests.get("Jarque-Bera")
    if isinstance(jb, dict):
        p = num(jb.get("p_value"))
        with st.expander("Jarque-Bera", expanded=False):
            st.write(f"Statistic: {format_number(jb.get('statistic'), 5)}")
            st.write(f"p-value: {format_number(p, 6)}")

    dw = tests.get("Durbin-Watson")
    if isinstance(dw, dict):
        value = num(dw.get("statistic"))
        with st.expander("Durbin-Watson", expanded=False):
            st.metric("Statistic", format_number(value, 4))
            if value is not None:
                st.info("Giá trị càng gần 2 thì tự tương quan bậc một càng thấp theo quy tắc thực hành.")

    lb = tests.get("Ljung-Box")
    if isinstance(lb, dict):
        p = num(lb.get("p_value"))
        with st.expander("Ljung-Box", expanded=False):
            st.write(f"Lag: {lb.get('lag', '—')}")
            st.write(f"Statistic: {format_number(lb.get('statistic'), 5)}")
            st.write(f"p-value: {format_number(p, 6)}")

    bp = tests.get("Breusch-Pagan")
    if isinstance(bp, dict):
        p = num(bp.get("p_value"))
        with st.expander("Breusch-Pagan", expanded=False):
            st.write(f"Statistic: {format_number(bp.get('statistic'), 5)}")
            st.write(f"p-value: {format_number(p, 6)}")


def render_conclusion(item):
    ranking = item.get("ranking")
    st.subheader("🧠 Kết luận")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        st.info("Chưa đủ kết quả.")
        return
    top = ranking.iloc[0]
    factor = str(top.get("Biến"))
    group = str(top.get("Nhóm"))
    score = num(top.get("Score"))
    direction = str(top.get("Quan hệ", "Không xác định"))
    p = num(top.get("p-value"))
    spear = num(top.get("Spearman"))

    text = f"**{factor}** ({group}) đang là tín hiệu nổi bật nhất"
    if score is not None:
        text += f" với Score {score:.1f}/100"
    text += f", quan hệ {direction.lower()}"
    if spear is not None:
        text += f", Spearman {spear:+.4f}"
    if p is not None:
        text += f", p-value {p:.5f}"
    text += "."
    st.markdown("• " + text)

    if p is not None and p < 0.05:
        st.markdown(f"• Có bằng chứng thống kê ở mức 5% cho **{factor}** trong mô hình OLS.")
    else:
        st.markdown(f"• **{factor}** nổi bật về liên hệ/dự báo trong mẫu, nhưng chưa đủ để gọi là quan hệ nhân quả.")

    models = item.get("models")
    if isinstance(models, pd.DataFrame) and not models.empty:
        best = models.iloc[0]
        r2 = num(best.get("R²"))
        rmse = num(best.get("RMSE"))
        model_name = str(best.get("Mô hình"))
        if rmse is not None and r2 is not None:
            st.markdown(f"• Model tốt nhất: **{model_name}** · RMSE {rmse:.6f} · R² {r2:.4f}.")
            if r2 < 0:
                st.warning("Model chưa vượt benchmark trên tập test.")
            elif r2 < 0.10:
                st.info("Khả năng dự báo còn thấp.")
            elif r2 < 0.25:
                st.info("Có tín hiệu dự báo nhưng sức giải thích vừa phải.")
            else:
                st.success("Model có sức giải thích đáng chú ý trên mẫu test.")

    forecast = item.get("forecast", {})
    pred = num(forecast.get("predicted_return"))
    pred_price = num(forecast.get("predicted_price"))
    if pred is not None and pred_price is not None:
        st.markdown(f"• Ước lượng theo model cho {item['horizon']}: **{pred * 100:+.2f}%** → khoảng **{pred_price:,.0f} đồng**.")


def render_cross_horizon(result):
    st.divider()
    st.header("🏆 Yếu tố nhất quán 1D / 5D / 20D")
    rows = []
    for horizon in HORIZONS:
        item = result["horizons"].get(horizon)
        if not item:
            continue
        ranking = item.get("ranking")
        if not isinstance(ranking, pd.DataFrame) or ranking.empty:
            continue
        for rank, (_, row) in enumerate(ranking.head(20).iterrows(), start=1):
            rows.append({"Horizon": horizon, "Biến": row.get("Biến"), "Nhóm": row.get("Nhóm"), "Rank": rank, "Score": num(row.get("Score"))})
    if not rows:
        st.info("Chưa đủ dữ liệu để so sánh giữa các horizon.")
        return
    work = pd.DataFrame(rows)
    stable = (
        work.groupby(["Biến", "Nhóm"], as_index=False)
        .agg(So_horizon=("Horizon", "nunique"), Rank_TB=("Rank", "mean"), Score_TB=("Score", "mean"))
        .sort_values(["So_horizon", "Score_TB", "Rank_TB"], ascending=[False, False, True])
    )
    stable["Rank_TB"] = stable["Rank_TB"].round(2)
    stable["Score_TB"] = stable["Score_TB"].round(1)
    stable.columns = ["Biến", "Nhóm", "Số horizon", "Rank TB", "Score TB"]
    st.dataframe(stable.head(30), width="stretch", hide_index=True)
    if not stable.empty:
        best = stable.iloc[0]
        st.success(f"Yếu tố nhất quán nhất: **{best['Biến']}** · {int(best['Số horizon'])}/3 horizon.")

# ============================================================
# MAIN PAGE
# ============================================================

def render_stock_analysis():
    st.caption("BRIAN STOCK · STOCK RESEARCH")
    st.title("Phân tích cổ phiếu")
    st.write("Phân tích kỹ thuật, dòng tiền, khối ngoại, tự doanh, thị trường chung, nhóm ngành và nghiên cứu định lượng.")

    current_symbol = st.session_state.get("stock_analysis_symbol", "HPG")
    symbol_input = st.text_input(
        "Mã cổ phiếu",
        value=current_symbol,
        placeholder="Ví dụ HPG, FPT, VNM...",
        key="stock_analysis_symbol_input",
    )

    if st.button("🔄 Tải dữ liệu", type="primary", key="stock_analysis_load_button"):
        clean = normalize_symbol(symbol_input)
        if not clean:
            st.warning("Vui lòng nhập mã cổ phiếu.")
            return
        st.session_state["stock_analysis_symbol"] = clean
        for key in ["stock_research_result", "stock_research_symbol", "stock_research_dates"]:
            st.session_state.pop(key, None)
        st.rerun()

    symbol = normalize_symbol(st.session_state.get("stock_analysis_symbol", symbol_input))

    # --------------------------------------------------------
    # DISPLAY DATA ONLY — không research API ở đây
    # --------------------------------------------------------
    try:
        display_data = load_display_data(symbol)
    except Exception as error:
        st.error(f"Không thể tải dữ liệu {display_symbol(symbol)}.")
        st.caption(str(error))
        return

    if display_data is None or display_data.empty:
        st.warning("Không có dữ liệu.")
        return

    snapshot = market_snapshot(display_data)
    price = num(snapshot.get("price"))
    change = num(snapshot.get("change_1d"))
    rsi_value = num(snapshot.get("rsi"))
    volume = num(snapshot.get("volume"))
    sma20 = num(snapshot.get("sma20"))
    sma50 = num(snapshot.get("sma50"))
    macd = num(snapshot.get("macd"))
    volatility = num(snapshot.get("volatility20"))
    last = display_data.iloc[-1]
    atr14 = num(last.get("ATR14"))
    volume_sma20 = num(last.get("Volume_SMA20"))
    relative_volume = (volume / volume_sma20) if volume is not None and volume_sma20 not in (None, 0) else None

    st.subheader(f"📈 {display_symbol(symbol)}")
    a, b, c, d = st.columns(4)
    with a: st.metric("Giá", format_price(price))
    with b: st.metric("Thay đổi 1D", format_percent(change))
    with c: st.metric("RSI", format_number(rsi_value, 1))
    with d: st.metric("Khối lượng", format_volume(volume))

    a, b, c, d = st.columns(4)
    with a: st.metric("MA20", format_price(sma20))
    with b: st.metric("MA50", format_price(sma50))
    with c: st.metric("MACD", format_number(macd, 3))
    with d: st.metric("Biến động 20 phiên", f"{volatility:.2f}%" if volatility is not None else "—")

    st.subheader("🧭 Trạng thái kỹ thuật")
    a, b, c, d = st.columns(4)
    with a: st.metric("Xu hướng", ma_status(price, sma20, sma50))
    with b: st.metric("RSI", rsi_status(rsi_value))
    with c: st.metric("MACD", macd_status(macd))
    with d: st.metric("Thanh khoản", f"{relative_volume:.2f}x TB20" if relative_volume is not None else "—")

    st.subheader("📋 Chỉ báo bổ sung")
    a, b, c, d = st.columns(4)
    with a: st.metric("ATR14", format_price(atr14))
    with b: st.metric("Volume TB20", format_volume(volume_sma20))
    with c: st.metric("Giá mở cửa", format_price(num(last.get("Open"))))
    with d:
        high, low, close = num(last.get("High")), num(last.get("Low")), num(last.get("Close"))
        position = ((close - low) / (high - low) * 100) if high is not None and low is not None and close is not None and high != low else None
        st.metric("Vị trí biên ngày", f"{position:.1f}%" if position is not None else "—")

    st.subheader("📊 Biểu đồ kỹ thuật")
    try:
        chart = price_volume_chart(display_data)
        if chart is not None:
            st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
    except Exception as error:
        st.warning(f"Không thể hiển thị biểu đồ: {error}")

    # --------------------------------------------------------
    # RESEARCH CONTROL
    # --------------------------------------------------------
    st.divider()
    st.header("🧪 Nghiên cứu định lượng")
    st.write("Giữ toàn bộ universe biến trong dataset; khi chạy model chỉ giảm chiều ở bước OLS/ML để tránh treo vì đa cộng tuyến.")

    mode = st.radio("Kiểu chọn mẫu", ["Preset", "Khoảng ngày tùy chọn"], horizontal=True, key="research_mode")
    today = pd.Timestamp.today().date()

    if mode == "Preset":
        preset = st.selectbox("Khoảng thời gian", list(PERIOD_PRESETS.keys()), index=3, key="research_preset")
        start_date = (pd.Timestamp(today) - pd.Timedelta(days=PERIOD_PRESETS[preset])).date()
        end_date = today
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Từ ngày", value=(pd.Timestamp(today) - pd.Timedelta(days=365)).date(), max_value=today, key="research_start_date")
        with c2:
            end_date = st.date_input("Đến ngày", value=today, max_value=today, key="research_end_date")
        preset = "Tùy chọn"

    if start_date > end_date:
        st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
        return

    st.info(f"Mẫu yêu cầu: {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")

    run_research = st.button("🚀 Chạy toàn bộ nghiên cứu", type="primary", width="stretch", key="run_stock_research")

    if run_research:
        st.session_state.pop("stock_research_result", None)
        with st.status("Đang nghiên cứu...", expanded=True) as status:
            try:
                st.write("1/3 Đang tải dữ liệu kỹ thuật, dòng tiền, khối ngoại, tự doanh, thị trường và nhóm ngành...")
                research_data = load_multifactor_research_history(symbol, start_date, end_date)

                st.write("2/3 Đang chuẩn hóa và tạo toàn bộ biến nghiên cứu...")
                sample = slice_date_range(research_data, start_date, end_date)
                if sample.empty:
                    status.update(label="Không có dữ liệu", state="error")
                    st.error("Không có phiên giao dịch trong khoảng chọn.")
                    return

                st.write(f"Đã có {len(sample):,} phiên. Đang chạy 1D / 5D / 20D...")
                result = execute_full_research(sample)
                status.update(label="Đã hoàn tất nghiên cứu", state="complete")

            except Exception as error:
                status.update(label="Nghiên cứu thất bại", state="error")
                st.error("Không thể chạy nghiên cứu.")
                st.caption(str(error))
                return

        st.session_state["stock_research_result"] = result
        st.session_state["stock_research_symbol"] = symbol
        st.session_state["stock_research_dates"] = (start_date, end_date)

    result = st.session_state.get("stock_research_result")
    saved_symbol = st.session_state.get("stock_research_symbol")
    saved_dates = st.session_state.get("stock_research_dates")

    if result is None or saved_symbol != symbol or saved_dates != (start_date, end_date):
        if result is not None:
            st.caption("Đang chọn mẫu mới. Bấm “Chạy toàn bộ nghiên cứu” để chạy lại.")
        return

    if not result.get("ok", False):
        st.error(result.get("error", "Nghiên cứu thất bại."))
        return

    research_data = result["data"]
    actual_start, actual_end = actual_range(research_data)

    st.divider()
    st.header("📚 Mẫu nghiên cứu")
    a, b, c, d = st.columns(4)
    with a: st.metric("Khoảng chọn", preset)
    with b: st.metric("Quan sát", f"{len(research_data):,}")
    with c: st.metric("Ngày đầu", actual_start.strftime("%d/%m/%Y") if actual_start else "—")
    with d: st.metric("Ngày cuối", actual_end.strftime("%d/%m/%Y") if actual_end else "—")

    if len(research_data) < 60:
        st.warning("Mẫu dưới 60 quan sát: kết luận cần rất thận trọng.")
    elif len(research_data) < 120:
        st.info("Mẫu đủ cho nghiên cứu cơ bản, nhưng chưa phải mẫu dài.")
    else:
        st.success(f"Mẫu có {len(research_data):,} quan sát thực tế.")

    # --------------------------------------------------------
    # ALL VARIABLES
    # --------------------------------------------------------
    all_features = result.get("all_features", [])
    groups = result.get("groups", {})

    st.header("🧬 Toàn bộ biến nghiên cứu")
    counts = {k: len(v) for k, v in groups.items()}
    a, b, c, d, e, f = st.columns(6)
    with a: st.metric("Tổng biến", len(all_features))
    with b: st.metric("Kỹ thuật", counts.get("Kỹ thuật", 0))
    with c: st.metric("Dòng tiền", counts.get("Dòng tiền", 0))
    with d: st.metric("Khối ngoại", counts.get("Khối ngoại", 0))
    with e: st.metric("Tự doanh", counts.get("Tự doanh", 0))
    with f: st.metric("Market + Sector", counts.get("Thị trường chung", 0) + counts.get("Nhóm ngành", 0))

    with st.expander("Xem toàn bộ biến theo nhóm", expanded=False):
        for group_name, values in groups.items():
            st.markdown(f"**{group_name} ({len(values)})**")
            st.write(", ".join(str(x) for x in values) if values else "Không có dữ liệu từ nguồn.")

    # --------------------------------------------------------
    # HORIZONS
    # --------------------------------------------------------
    for horizon in HORIZONS:
        item = result["horizons"].get(horizon)
        if item is None:
            continue

        st.divider()
        st.header(f"📈 {horizon} — {HORIZON_LABELS[horizon]}")
        a, b, c, d = st.columns(4)
        with a: st.metric("Quan sát", f"{item['observations']:,}")
        with b: st.metric("Train", f"{item['train']:,}")
        with c: st.metric("Test", f"{item['test']:,}")
        with d: st.metric("Biến dataset", f"{len(item['all_features']):,}")

        st.caption(f"Model dùng {len(item['model_features']):,} biến được chọn từ toàn bộ universe; correlation vẫn quét toàn bộ biến.")

        render_forecast(item)
        render_groups(item)
        render_factors(item)
        render_models(item)
        render_ols(item)
        render_vif(item)
        render_tests(item)
        render_conclusion(item)

    render_cross_horizon(result)

    st.divider()
    st.header("🎯 Tóm tắt cuối cùng")
    rows = []
    for horizon in HORIZONS:
        item = result["horizons"].get(horizon)
        if item is None:
            continue
        ranking = item.get("ranking")
        forecast = item.get("forecast", {})
        if not isinstance(ranking, pd.DataFrame) or ranking.empty:
            continue
        top = ranking.iloc[0]
        pred = num(forecast.get("predicted_return"))
        rows.append({
            "Horizon": horizon,
            "Yếu tố nổi bật": top.get("Biến"),
            "Nhóm": top.get("Nhóm"),
            "Quan hệ": top.get("Quan hệ"),
            "Score": num(top.get("Score")),
            "Spearman": num(top.get("Spearman")),
            "Beta": num(top.get("Beta")),
            "p-value": num(top.get("p-value")),
            "Dự báo %": pred * 100 if pred is not None else np.nan,
        })

    if rows:
        summary = pd.DataFrame(rows)
        st.dataframe(summary.round(5), width="stretch", hide_index=True)
        counts_factor = summary["Yếu tố nổi bật"].value_counts()
        if not counts_factor.empty:
            common = str(counts_factor.index[0])
            n = int(counts_factor.iloc[0])
            st.success(f"Yếu tố đứng đầu nhiều horizon nhất: **{common}** · {n}/{len(summary)} horizon.")

    st.warning("Đây là quan hệ thống kê và khả năng dự báo trong mẫu, không phải bằng chứng nhân quả hay cam kết giá tương lai.")


def render_analysis():
    render_stock_analysis()
