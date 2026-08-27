# data/market.py
# ============================================================
# BRIAN AI INVEST LAB
# MARKET DATA ENGINE
# ============================================================
#
# Nguồn:
#   - Yahoo Finance / yfinance: dữ liệu giá thị trường thật
#   - Google News RSS: tin tức thật
#
# Không dùng dữ liệu random/mock.
# Tự xử lý:
#   - Mã Việt Nam: HPG -> HPG.VN
#   - Mã quốc tế: AAPL -> AAPL
#   - MultiIndex từ yfinance
#   - period dạng string
#   - timedelta days dạng string
#   - lỗi Yahoo / mã không tồn tại
#
# ============================================================

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import feedparser
except Exception:
    feedparser = None


# ============================================================
# SYMBOL HELPERS
# ============================================================

def normalize_symbol(symbol: str) -> str:
    """
    Chuẩn hóa mã cổ phiếu.

    Ví dụ:
        HPG       -> HPG
        hpg       -> HPG
        HPG.VN    -> HPG.VN
        VNM.VN    -> VNM.VN
        AAPL      -> AAPL
    """

    if symbol is None:
        return "HPG"

    s = str(symbol).strip().upper()

    # Xóa khoảng trắng
    s = re.sub(r"\s+", "", s)

    if not s:
        return "HPG"

    return s


def display_symbol(symbol: str) -> str:
    """
    Mã hiển thị trên giao diện.

    HPG.VN -> HPG
    HPG   -> HPG
    """

    if symbol is None:
        return ""

    s = str(symbol).strip().upper()

    if s.endswith(".VN"):
        s = s[:-3]

    return s


def _ticker_candidates(symbol: str) -> list[str]:
    """
    Tạo danh sách ticker để thử.

    Với mã Việt Nam:
        HPG -> HPG.VN -> HPG

    Với mã đã có .VN:
        HPG.VN -> HPG

    Với mã quốc tế:
        AAPL -> AAPL
    """

    s = normalize_symbol(symbol)

    candidates: list[str] = []

    # Nếu người dùng đã nhập .VN
    if s.endswith(".VN"):
        base = s[:-3]

        if s not in candidates:
            candidates.append(s)

        if base and base not in candidates:
            candidates.append(base)

        return candidates

    # Mã 2-5 ký tự có khả năng là cổ phiếu Việt Nam.
    # Thử .VN trước, sau đó thử ticker gốc.
    if re.fullmatch(r"[A-Z0-9]{2,5}", s):
        candidates.append(f"{s}.VN")

    candidates.append(s)

    # Loại duplicate
    return list(dict.fromkeys(candidates))


# ============================================================
# FORMAT HELPERS
# ============================================================

def fmt_price(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.0f}"
    except Exception:
        return "—"


def fmt_pct(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):+.2f}%"
    except Exception:
        return "—"


# ============================================================
# RSI
# ============================================================

def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    RSI theo phương pháp Wilder/EMA.
    """

    period = int(period)

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    # Nếu giá tăng liên tục
    result = result.where(~((avg_loss == 0) & (avg_gain > 0)), 100)

    # Nếu không có biến động
    result = result.where(
        ~((avg_loss == 0) & (avg_gain == 0)),
        50,
    )

    return result


# ============================================================
# DATAFRAME NORMALIZATION
# ============================================================

def _normalize_yfinance_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chuẩn hóa DataFrame từ yfinance.

    Xử lý cả:
        Open High Low Close Volume
    và MultiIndex:
        ('Close', 'HPG.VN')
    """

    if df is None:
        raise ValueError("Yahoo Finance không trả về dữ liệu.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dữ liệu thị trường không phải DataFrame.")

    if df.empty:
        raise ValueError("Yahoo Finance trả về DataFrame rỗng.")

    df = df.copy()

    # --------------------------------------------------------
    # MultiIndex
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):

        # Tìm level chứa các cột OHLCV
        selected_level = None

        for level in range(df.columns.nlevels):
            values = {
                str(x).strip().lower()
                for x in df.columns.get_level_values(level)
            }

            if "close" in values:
                selected_level = level
                break

        if selected_level is not None:
            df.columns = df.columns.get_level_values(selected_level)
        else:
            df.columns = [
                str(col[-1] if isinstance(col, tuple) else col)
                for col in df.columns
            ]

    # --------------------------------------------------------
    # Column names
    # --------------------------------------------------------

    rename_map = {}

    for col in df.columns:
        name = str(col).strip().lower()

        if name == "open":
            rename_map[col] = "Open"

        elif name == "high":
            rename_map[col] = "High"

        elif name == "low":
            rename_map[col] = "Low"

        elif name == "close":
            rename_map[col] = "Close"

        elif name in {
            "adj close",
            "adj_close",
            "adjusted close",
        }:
            rename_map[col] = "Adj Close"

        elif name == "volume":
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dữ liệu Yahoo thiếu cột: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Datetime index
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    df = df[
        ~df.index.isna()
    ].copy()

    # Sort thời gian
    df = df.sort_index()

    # Xóa duplicate timestamp
    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ].copy()

    # Xóa dòng không có Close
    df = df.dropna(
        subset=["Close"]
    )

    if df.empty:
        raise ValueError(
            "Không còn dữ liệu giá hợp lệ sau khi xử lý."
        )

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Thêm toàn bộ chỉ báo dùng bởi Dashboard.
    """

    df = _normalize_yfinance_dataframe(df)

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    df["Return"] = (
        df["Close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    df["RSI"] = rsi(
        df["Close"],
        period=14,
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["MACD_Hist"] = (
        df["MACD"]
        - df["MACD_Signal"]
    )

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    df["SMA20"] = (
        df["Close"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    df["SMA50"] = (
        df["Close"]
        .rolling(
            window=50,
            min_periods=50,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Volatility
    #
    # Lưu dưới dạng %
    # Ví dụ 25.3 = 25.3%
    # --------------------------------------------------------

    df["Volatility20"] = (
        df["Return"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .std()
        * np.sqrt(252)
        * 100
    )

    # Tương thích code cũ
    df["Volatility_20D"] = (
        df["Volatility20"] / 100
    )

    # --------------------------------------------------------
    # Volume change
    # --------------------------------------------------------

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 1D return %
    # --------------------------------------------------------

    df["ReturnPct"] = (
        df["Return"] * 100
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Không drop toàn bộ NaN ở đầu dataframe.
    # Giữ dữ liệu giá nguyên vẹn.
    return df.copy()


# ============================================================
# RAW YAHOO DOWNLOAD
# ============================================================

def _download_yahoo(
    ticker: str,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Tải dữ liệu thật từ Yahoo Finance.

    Có 2 cách:
        1. Ticker.history()
        2. yf.download()

    Không tạo dữ liệu giả.
    """

    period = str(period).strip()

    allowed_periods = {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }

    if period not in allowed_periods:
        period = "1y"

    last_error = None

    # --------------------------------------------------------
    # Method 1: Ticker.history
    # --------------------------------------------------------

    try:

        obj = yf.Ticker(ticker)

        df = obj.history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
        ):
            return _normalize_yfinance_dataframe(
                df
            )

    except Exception as exc:
        last_error = exc

    # --------------------------------------------------------
    # Method 2: yf.download
    # --------------------------------------------------------

    try:

        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
        ):
            return _normalize_yfinance_dataframe(
                df
            )

    except Exception as exc:
        last_error = exc

    if last_error:
        raise ValueError(
            f"Yahoo Finance lỗi với {ticker}: "
            f"{last_error}"
        )

    raise ValueError(
        f"Không tìm thấy dữ liệu Yahoo Finance cho {ticker}."
    )


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def load_market_data(
    symbol: str,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Hàm chính được Dashboard sử dụng.

    Tự động thử:

        HPG
        ↓
        HPG.VN
        ↓
        HPG

    Nhờ vậy không cần hard-code danh sách hàng nghìn mã.
    """

    original = normalize_symbol(symbol)

    candidates = _ticker_candidates(
        original
    )

    errors = []

    for ticker in candidates:

        try:

            raw = _download_yahoo(
                ticker=ticker,
                period=period,
            )

            data = add_indicators(
                raw
            )

            if data is None or data.empty:
                continue

            # Đảm bảo có đủ dữ liệu cơ bản
            if len(data) < 2:
                continue

            # Gắn ticker vào attrs để các module khác
            # có thể biết dữ liệu lấy từ đâu.
            data.attrs["symbol"] = ticker
            data.attrs["display_symbol"] = display_symbol(
                ticker
            )
            data.attrs["source"] = "Yahoo Finance"

            return data

        except Exception as exc:
            errors.append(
                f"{ticker}: {exc}"
            )

    error_text = " | ".join(errors)

    raise ValueError(
        f"Không lấy được dữ liệu cho "
        f"{display_symbol(original)}. "
        f"Đã thử: {', '.join(candidates)}. "
        f"{error_text}"
    )


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def market_data(
    symbol: str,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Alias tương thích với code v2.
    """

    return load_market_data(
        symbol,
        period,
    )


# ============================================================
# COMPANY INFO
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def load_company_info(
    symbol: str,
) -> dict:
    """
    Lấy thông tin cơ bản từ Yahoo Finance.
    """

    candidates = _ticker_candidates(
        symbol
    )

    for ticker in candidates:

        try:

            obj = yf.Ticker(
                ticker
            )

            # fast_info ít request hơn info
            try:
                info = dict(
                    obj.fast_info
                )

                if info:
                    return info

            except Exception:
                pass

            # fallback
            try:
                info = obj.info

                if isinstance(
                    info,
                    dict,
                ):
                    return info

            except Exception:
                pass

        except Exception:
            continue

    return {}


# ============================================================
# NEWS
# ============================================================

def _clean_html(text: str) -> str:
    if text is None:
        return ""

    text = html.unescape(
        str(text)
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _safe_int(
    value,
    default: int,
) -> int:
    """
    Quan trọng:
    tránh lỗi:
        unsupported type for timedelta days component: str
    """

    try:
        return int(
            float(value)
        )
    except Exception:
        return int(default)


def _parse_news_datetime(
    value,
):
    """
    Parse thời gian RSS an toàn.
    """

    if not value:
        return None

    # feedparser có thể trả time struct
    try:
        if hasattr(
            value,
            "tm_year",
        ):
            return datetime(
                value.tm_year,
                value.tm_mon,
                value.tm_mday,
                value.tm_hour,
                value.tm_min,
                value.tm_sec,
                tzinfo=timezone.utc,
            )
    except Exception:
        pass

    # Email/RFC date
    try:

        from email.utils import (
            parsedate_to_datetime,
        )

        dt = parsedate_to_datetime(
            str(value)
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt

    except Exception:
        pass

    # Pandas fallback
    try:

        dt = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )

        if pd.isna(dt):
            return None

        return dt.to_pydatetime()

    except Exception:
        return None


@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def fetch_news(
    symbol: str,
    limit: int = 8,
) -> list[dict]:
    """
    Google News RSS.

    Trả về:
        title
        link
        published
        summary
        source
    """

    limit = _safe_int(
        limit,
        8,
    )

    limit = max(
        1,
        min(
            limit,
            50,
        ),
    )

    base = display_symbol(
        symbol
    )

    query = quote(
        f'"{base}" cổ phiếu OR '
        f'"{base}" chứng khoán'
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={query}"
        "&hl=vi"
        "&gl=VN"
        "&ceid=VN:vi"
    )

    results: list[dict] = []

    # --------------------------------------------------------
    # feedparser
    # --------------------------------------------------------

    if feedparser is not None:

        try:

            feed = feedparser.parse(
                url
            )

            entries = getattr(
                feed,
                "entries",
                [],
            )

            for entry in entries:

                title = _clean_html(
                    entry.get(
                        "title",
                        "",
                    )
                )

                link = str(
                    entry.get(
                        "link",
                        "",
                    )
                ).strip()

                if not title:
                    continue

                published_raw = (
                    entry.get(
                        "published",
                        "",
                    )
                    or entry.get(
                        "updated",
                        "",
                    )
                )

                dt = _parse_news_datetime(
                    published_raw
                )

                if dt is not None:
                    published_text = (
                        dt.astimezone()
                        .strftime(
                            "%d/%m/%Y %H:%M"
                        )
                    )
                else:
                    published_text = str(
                        published_raw
                    )

                source = "Google News"

                source_obj = entry.get(
                    "source"
                )

                if isinstance(
                    source_obj,
                    dict,
                ):
                    source = (
                        source_obj.get(
                            "title"
                        )
                        or "Google News"
                    )

                summary = _clean_html(
                    entry.get(
                        "summary",
                        "",
                    )
                )

                results.append(
                    {
                        "title": title,
                        "link": link,
                        "published": published_text,
                        "summary": summary,
                        "source": source,
                    }
                )

                if len(results) >= limit:
                    break

            return results

        except Exception:
            pass

    # Không có feedparser / feedparser lỗi
    return []


# ============================================================
# OLD API COMPATIBILITY
# ============================================================

def get_news(
    ticker: str,
    days=7,
) -> list[dict]:
    """
    Compatibility với code cũ.

    days có thể là:
        7
        "7"
        7.0

    Không còn lỗi timedelta days component: str.
    """

    days = _safe_int(
        days,
        7,
    )

    days = max(
        1,
        min(
            days,
            365,
        ),
    )

    news = fetch_news(
        ticker,
        limit=20,
    )

    if not news:
        return []

    # Lọc theo số ngày
    cutoff = (
        datetime.now(
            timezone.utc
        )
        - timedelta(
            days=days
        )
    )

    filtered = []

    for item in news:

        raw = item.get(
            "published"
        )

        dt = _parse_news_datetime(
            raw
        )

        # Nếu không parse được thì vẫn giữ tin
        if dt is None:
            filtered.append(
                item
            )
            continue

        if dt >= cutoff:
            filtered.append(
                item
            )

    return filtered[:20]


# ============================================================
# NEWS SENTIMENT
# ============================================================

def classify_news(
    title: str,
) -> str:
    """
    Sentiment đơn giản từ headline.
    """

    positive_words = [
        "tăng",
        "tích cực",
        "lợi nhuận",
        "kỷ lục",
        "tăng trưởng",
        "khởi sắc",
        "bứt phá",
        "mua",
        "triển vọng",
        "hưởng lợi",
        "doanh thu tăng",
        "cải thiện",
        "vượt kế hoạch",
        "vượt kỳ vọng",
    ]

    negative_words = [
        "giảm",
        "tiêu cực",
        "thua lỗ",
        "rủi ro",
        "sụt giảm",
        "áp lực",
        "bán",
        "khó khăn",
        "suy giảm",
        "nợ xấu",
        "cảnh báo",
        "điều tra",
        "vi phạm",
        "sa thải",
    ]

    text = str(
        title or ""
    ).lower()

    positive = sum(
        word in text
        for word in positive_words
    )

    negative = sum(
        word in text
        for word in negative_words
    )

    if positive > negative:
        return "positive"

    if negative > positive:
        return "negative"

    return "neutral"


# ============================================================
# QUANT / MACHINE LEARNING DATA
# ============================================================

def build_quant(
    df: pd.DataFrame,
):
    """
    Xây dựng OLS + Random Forest.

    Return:
        (
            ols_model,
            rf_model,
            metrics,
            next_return,
            feature_importance
        )

    Nếu dữ liệu chưa đủ:
        None
    """

    if df is None:
        return None

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        return None

    if df.empty:
        return None

    required = [
        "Return",
        "RSI",
        "MACD",
        "MACD_Hist",
        "Volatility20",
        "Volume_Change",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        return None

    try:
        from sklearn.ensemble import (
            RandomForestRegressor,
        )
        from sklearn.metrics import (
            mean_absolute_error,
            r2_score,
        )
        import statsmodels.api as sm

    except Exception:
        return None

    work = df.copy()

    # Target = return phiên kế tiếp
    work["Target"] = (
        work["Return"]
        .shift(-1)
    )

    features = [
        "RSI",
        "MACD",
        "MACD_Hist",
        "Volatility20",
        "Volume_Change",
        "Return",
    ]

    work = work.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    work = work.dropna(
        subset=features
        + ["Target"]
    )

    # Không đủ dữ liệu để ML
    if len(work) < 60:
        return None

    X = work[
        features
    ].astype(float)

    y = work[
        "Target"
    ].astype(float)

    split = int(
        len(work) * 0.8
    )

    # Đảm bảo train/test đều có dữ liệu
    if split < 30:
        return None

    if split >= len(work):
        return None

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_train = y.iloc[
        :split
    ]

    y_test = y.iloc[
        split:
    ]

    try:

        # ----------------------------------------------------
        # OLS
        # ----------------------------------------------------

        X_train_ols = sm.add_constant(
            X_train,
            has_constant="add",
        )

        ols = sm.OLS(
            y_train,
            X_train_ols,
        ).fit(
            cov_type="HC3"
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        rf.fit(
            X_train,
            y_train,
        )

        prediction_test = (
            rf.predict(
                X_test
            )
        )

        mae = float(
            mean_absolute_error(
                y_test,
                prediction_test,
            )
        )

        try:
            r2 = float(
                r2_score(
                    y_test,
                    prediction_test,
                )
            )
        except Exception:
            r2 = float("nan")

        metrics = {
            "MAE": mae,
            "R2": r2,
        }

        # ----------------------------------------------------
        # Next-day prediction
        # ----------------------------------------------------

        latest_x = X.iloc[
            [-1]
        ]

        next_return = float(
            rf.predict(
                latest_x
            )[0]
        )

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        importance = pd.Series(
            rf.feature_importances_,
            index=features,
        ).sort_values(
            ascending=False
        )

        return (
            ols,
            rf,
            metrics,
            next_return,
            importance,
        )

    except Exception:
        return None


# ============================================================
# COMPATIBILITY: RUN OLS
# ============================================================

def run_ols(
    data: pd.DataFrame,
):
    """
    API tương thích code cũ.
    """

    if data is None or data.empty:
        return None

    try:
        import statsmodels.api as sm
    except Exception:
        return None

    features = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility20",
    ]

    if any(
        col not in data.columns
        for col in features
    ):
        return None

    clean = data[
        features + ["Return"]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(clean) < 50:
        return None

    try:

        X = sm.add_constant(
            clean[features],
            has_constant="add",
        )

        y = clean["Return"]

        model = sm.OLS(
            y,
            X,
        ).fit()

        return model

    except Exception:
        return None


# ============================================================
# COMPATIBILITY: RANDOM FOREST
# ============================================================

def run_random_forest(
    data: pd.DataFrame,
):
    """
    API tương thích code cũ.
    """

    if data is None or data.empty:
        return None

    try:
        from sklearn.ensemble import (
            RandomForestRegressor,
        )
    except Exception:
        return None

    features = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility20",
    ]

    if any(
        col not in data.columns
        for col in features
    ):
        return None

    clean = data[
        features + ["Return"]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    clean["Target"] = (
        clean["Return"]
        .shift(-1)
    )

    clean = clean.dropna()

    if len(clean) < 80:
        return None

    try:

        X = clean[
            features
        ].astype(float)

        y = clean[
            "Target"
        ].astype(float)

        split = int(
            len(clean) * 0.8
        )

        if split < 40:
            return None

        X_train = X.iloc[
            :split
        ]

        X_test = X.iloc[
            split:
        ]

        y_train = y.iloc[
            :split
        ]

        y_test = y.iloc[
            split:
        ]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        prediction = float(
            model.predict(
                X.iloc[[-1]]
            )[0]
        )

        importance = dict(
            zip(
                features,
                model.feature_importances_,
            )
        )

        return {
            "model": model,
            "prediction": prediction,
            "importance": importance,
        }

    except Exception:
        return None


# ============================================================
# MARKET SUMMARY
# ============================================================

def market_snapshot(
    df: pd.DataFrame,
) -> dict:
    """
    Lấy snapshot mới nhất.
    """

    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]

    if len(df) > 1:
        previous = df.iloc[-2]
    else:
        previous = latest

    try:
        change = (
            float(latest["Close"])
            / float(previous["Close"])
            - 1
        ) * 100
    except Exception:
        change = np.nan

    result = {
        "price": float(
            latest["Close"]
        )
        if not pd.isna(
            latest["Close"]
        )
        else np.nan,

        "change_1d": change,

        "return_1d": (
            float(
                latest["Return"]
            ) * 100
            if not pd.isna(
                latest["Return"]
            )
            else np.nan
        ),

        "rsi": (
            float(
                latest["RSI"]
            )
            if not pd.isna(
                latest["RSI"]
            )
            else np.nan
        ),

        "macd": (
            float(
                latest["MACD"]
            )
            if not pd.isna(
                latest["MACD"]
            )
            else np.nan
        ),

        "sma20": (
            float(
                latest["SMA20"]
            )
            if not pd.isna(
                latest["SMA20"]
            )
            else np.nan
        ),

        "sma50": (
            float(
                latest["SMA50"]
            )
            if not pd.isna(
                latest["SMA50"]
            )
            else np.nan
        ),

        "volatility20": (
            float(
                latest["Volatility20"]
            )
            if not pd.isna(
                latest["Volatility20"]
            )
            else np.nan
        ),

        "volume": (
            float(
                latest["Volume"]
            )
            if not pd.isna(
                latest["Volume"]
            )
            else np.nan
        ),
    }

    return result


# ============================================================
# END
# ============================================================
