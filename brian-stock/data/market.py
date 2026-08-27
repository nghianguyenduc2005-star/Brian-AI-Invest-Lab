import re
import numpy as np
import pandas as pd
import streamlit as st

from config.settings import VIETNAM_TICKERS

# vnstock là nguồn chính cho cổ phiếu Việt Nam
try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except Exception:
    VNSTOCK_AVAILABLE = False


def normalize_symbol(s):
    s = (s or "").strip().upper().replace(" ", "")

    if not s:
        return "HPG.VN"

    # Người dùng có thể nhập HPG.VN
    if s.endswith(".VN"):
        return s

    # Mã Việt Nam
    if re.fullmatch(r"[A-Z0-9]{2,5}", s):
        return s + ".VN"

    return s


def display_symbol(s):
    return s.upper().replace(".VN", "")


def rsi(series, period=14):
    d = series.diff()

    gain = d.clip(lower=0).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    loss = (-d.clip(upper=0)).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - 100 / (1 + rs)


def add_indicators(df):
    df = df.copy()

    # Chuẩn hóa tên cột
    df.columns = [str(c).strip().title() for c in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]

    for c in required:
        if c not in df.columns:
            raise ValueError(
                f"Thiếu cột {c}. Các cột nhận được: {list(df.columns)}"
            )

    # Đảm bảo numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])

    # Return
    df["Return"] = df["Close"].pct_change()

    # RSI
    df["RSI"] = rsi(df["Close"])

    # MACD
    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # Volatility
    df["Volatility20"] = (
        df["Return"].rolling(20).std()
        * np.sqrt(252)
        * 100
    )

    # Không drop toàn bộ dữ liệu quá sớm.
    # Giữ lại dữ liệu giá để chart vẫn hoạt động.
    return df


def _period_to_dates(period):
    """
    Chuyển period kiểu yfinance sang khoảng ngày.
    """
    end = pd.Timestamp.today().normalize()

    mapping = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }

    days = mapping.get(period, 365)

    start = end - pd.Timedelta(days=days)

    return (
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


def _load_vnstock(symbol, period):
    if not VNSTOCK_AVAILABLE:
        raise RuntimeError(
            "Chưa cài vnstock. Hãy thêm vnstock vào requirements.txt."
        )

    # Bỏ .VN trước khi gửi sang nguồn Việt Nam
    ticker = display_symbol(symbol)

    start, end = _period_to_dates(period)

    stock = Vnstock().stock(
        symbol=ticker,
        source="VCI",
    )

    df = stock.quote.history(
        start=start,
        end=end,
        interval="1D",
    )

    if df is None or df.empty:
        raise ValueError(
            f"Không có dữ liệu thị trường cho {ticker}"
        )

    df = df.copy()

    # vnstock dùng time thay vì Date
    if "time" in df.columns:
        df["Date"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])

    # Chuẩn hóa tên
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    df.rename(columns=rename_map, inplace=True)

    # Một số phiên bản có thể trả tên viết hoa
    df.columns = [
        str(c).strip().title()
        for c in df.columns
    ]

    # Đặt Date làm index nếu có
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df.set_index("Date")

    return df


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def load_market_data(symbol, period="1y"):
    """
    Nguồn dữ liệu chính:
        vnstock / VCI

    Không dùng Yahoo Finance cho cổ phiếu Việt Nam,
    tránh lỗi Yahoo HTTP 429 / crumb.
    """

    symbol = normalize_symbol(symbol)

    # =========================
    # VIETNAM STOCK
    # =========================
    if symbol.endswith(".VN"):
        df = _load_vnstock(symbol, period)

        if df is None or df.empty:
            raise ValueError(
                f"Không lấy được dữ liệu cho {display_symbol(symbol)}"
            )

        return add_indicators(df)

    # =========================
    # NON-VIETNAM
    # =========================
    # Chỉ dùng Yahoo cho mã quốc tế.
    try:
        import yfinance as yf

        t = yf.Ticker(symbol)

        df = t.history(
            period=period,
            interval="1d",
            auto_adjust=False,
        )

        if df is None or df.empty:
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

        if df is None or df.empty:
            raise ValueError(
                f"Không có dữ liệu cho {symbol}"
            )

        return add_indicators(df)

    except Exception as e:
        raise ValueError(
            f"Không tải được dữ liệu {symbol}: {e}"
        )
