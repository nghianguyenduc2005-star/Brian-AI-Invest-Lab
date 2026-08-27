import os
import re
import html
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import statsmodels.api as sm

from datetime import datetime, timedelta
from urllib.parse import quote
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. CẤU HÌNH APP
# =========================================================

st.set_page_config(
    page_title="Brian AI Invest Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CSS — DARK LUXURY
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 0%,
                rgba(37, 56, 88, 0.20),
                transparent 35%
            ),
            #080b10;
        color: #F4F7FA;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0a0e14;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ---------- TYPOGRAPHY ---------- */

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    .muted {
        color: #8D98A8;
        font-size: 0.85rem;
    }

    .gold {
        color: #D6B36A;
    }

    /* ---------- BRAND ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 4px 0 20px 0;
    }

    .brand-mark {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background:
            linear-gradient(
                135deg,
                #D6B36A,
                #8E6B32
            );
        display: flex;
        align-items: center;
        justify-content: center;
        color: #080b10;
        font-size: 23px;
        font-weight: 900;
        box-shadow:
            0 10px 30px rgba(214,179,106,0.18);
    }

    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .brand-subtitle {
        color: #8994A4;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(20,28,40,0.98),
                rgba(10,14,20,0.98)
            );
        border: 1px solid rgba(214,179,106,0.16);
        box-shadow:
            0 25px 70px rgba(0,0,0,0.28);
        margin-bottom: 25px;
    }

    .hero-label {
        color: #D6B36A;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #9BA6B6;
        font-size: 0.95rem;
        max-width: 900px;
        line-height: 1.6;
    }

    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        background: #0E131B;
        border: 1px solid rgba(255,255,255,0.065);
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 115px;
    }

    .kpi-label {
        color: #8F9AAA;
        font-size: 0.78rem;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: #F5F7FA;
        font-size: 1.65rem;
        font-weight: 800;
    }

    .kpi-sub {
        color: #727E8E;
        font-size: 0.75rem;
        margin-top: 5px;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin: 30px 0 12px 0;
    }

    .section-desc {
        color: #7F8B9B;
        font-size: 0.84rem;
        margin-bottom: 15px;
    }

    /* ---------- AI REPORT ---------- */

    .ai-box {
        background:
            linear-gradient(
                145deg,
                rgba(20,29,43,0.98),
                rgba(11,15,22,0.98)
            );
        border: 1px solid rgba(91,141,214,0.20);
        border-radius: 20px;
        padding: 26px 30px;
        box-shadow:
            0 20px 60px rgba(0,0,0,0.20);
    }

    .ai-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(91,141,214,0.12);
        color: #75A8EA;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* ---------- NEWS ---------- */

    .news-card {
        padding: 16px 18px;
        margin-bottom: 10px;
        border-radius: 14px;
        background: #0D1219;
        border: 1px solid rgba(255,255,255,0.055);
    }

    .news-title {
        color: #E9EDF2;
        font-weight: 700;
        line-height: 1.45;
    }

    .news-meta {
        color: #6F7B8B;
        font-size: 0.72rem;
        margin-top: 7px;
    }

    /* ---------- SIGNAL ---------- */

    .signal {
        border-radius: 16px;
        padding: 20px;
        background: #0E131B;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .signal-title {
        color: #8994A4;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .signal-value {
        font-size: 1.45rem;
        font-weight: 850;
        margin-top: 7px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #596575;
        font-size: 0.72rem;
        padding-top: 35px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 45px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(214,179,106,0.28);
        background: #151B24;
        color: #F2F4F7;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #D6B36A;
        color: #D6B36A;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. HÀM TIỆN ÍCH
# =========================================================

def get_secret(name):
    """Lấy secret từ Streamlit hoặc environment."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name)


def fmt_number(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def fmt_percent(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}%"


def clean_symbol(symbol):
    symbol = symbol.upper().strip()
    symbol = re.sub(r"[^A-Z0-9.\-]", "", symbol)
    return symbol


# =========================================================
# 4. LOGO THƯƠNG HIỆU
# =========================================================

def render_brand():
    logo_path = "logo.png"

    if os.path.exists(logo_path):

        st.sidebar.image(
            logo_path,
            width=180
        )

    else:

        st.sidebar.markdown(
            """
            <div class="brand">
                <div class="brand-mark">B</div>
                <div>
                    <div class="brand-title">
                        BRIAN AI
                    </div>
                    <div class="brand-subtitle">
                        INVESTMENT RESEARCH LAB
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


render_brand()


# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.markdown(
    "### ⚙️ Thiết lập phân tích"
)

ticker = st.sidebar.text_input(
    "Mã cổ phiếu",
    value="HPG",
    help="Ví dụ: HPG, FPT, VCB, VIC..."
)

ticker = clean_symbol(ticker)

period = st.sidebar.selectbox(
    "Khoảng dữ liệu",
    [
        "3 tháng",
        "6 tháng",
        "1 năm",
        "2 năm",
        "5 năm",
    ],
    index=2
)

news_days = st.sidebar.slider(
    "Tin tức gần nhất",
    min_value=3,
    max_value=14,
    value=7
)

st.sidebar.markdown("---")

run_analysis = st.sidebar.button(
    "🚀 Chạy phân tích",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Brian AI Invest Lab"
)

st.sidebar.caption(
    "Quant • Machine Learning • News Intelligence"
)


# =========================================================
# 6. MAPPING PERIOD
# =========================================================

period_map = {
    "3 tháng": "3mo",
    "6 tháng": "6mo",
    "1 năm": "1y",
    "2 năm": "2y",
    "5 năm": "5y",
}


# =========================================================
# 7. TÍNH CHỈ BÁO KỸ THUẬT
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]

    # Return
    df["Return"] = close.pct_change()

    # SMA
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()

    # EMA
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    # MACD
    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # RSI
    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # Volatility annualized
    df["Volatility"] = (
        df["Return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    return df


# =========================================================
# 8. LẤY DỮ LIỆU THỊ TRƯỜNG
# =========================================================

@st.cache_data(ttl=900)
def load_market_data(symbol, selected_period):

    stock = yf.Ticker(symbol)

    df = stock.history(
        period=selected_period,
        auto_adjust=False
    )

    if df is None or df.empty:
        raise ValueError(
            f"Không tìm thấy dữ liệu cho mã {symbol}."
        )

    df = df.reset_index()

    if "Date" not in df.columns:
        df.rename(
            columns={
                df.columns[0]: "Date"
            },
            inplace=True
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.set_index(
        "Date"
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    df = df[
        [
            c for c in required
            if c in df.columns
        ]
    ].copy()

    df = calculate_indicators(
        df
    )

    return df


# =========================================================
# 9. THÔNG TIN CƠ BẢN CÔNG TY
# =========================================================

@st.cache_data(ttl=3600)
def load_company_info(symbol):

    try:

        info = yf.Ticker(
            symbol
        ).info

        return {
            "name": info.get(
                "longName",
                symbol
            ),
            "sector": info.get(
                "sector",
                "N/A"
            ),
            "industry": info.get(
                "industry",
                "N/A"
            ),
            "market_cap": info.get(
                "marketCap"
            ),
            "pe": info.get(
                "trailingPE"
            ),
            "pb": info.get(
                "priceToBook"
            ),
            "dividend": info.get(
                "dividendYield"
            ),
        }

    except Exception:

        return {
            "name": symbol,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": None,
            "pe": None,
            "pb": None,
            "dividend": None,
        }


# =========================================================
# 10. LẤY TIN TỨC
# =========================================================

@st.cache_data(ttl=900)
def load_news(symbol, days=7):

    query = quote(
        f"{symbol} cổ phiếu"
    )

    url = (
        "https://news.google.com/rss/search"
        f"?q={query}"
        "&hl=vi"
        "&gl=VN"
        "&ceid=VN:vi"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64)"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        from xml.etree import ElementTree

        root = ElementTree.fromstring(
            response.content
        )

        items = []

        cutoff = datetime.utcnow() - timedelta(
            days=days
        )

        for item in root.findall(
            ".//item"
        ):

            title = item.findtext(
                "title",
                default=""
            )

            link = item.findtext(
                "link",
                default=""
            )

            pub_date = item.findtext(
                "pubDate",
                default=""
            )

            source_node = item.find(
                "source"
            )

            source = (
                source_node.text
                if source_node is not None
                else "Google News"
            )

            try:

                published = pd.to_datetime(
                    pub_date,
                    utc=True
                ).tz_convert(None)

            except Exception:

                published = pd.Timestamp.now()

            if published.to_pydatetime() < cutoff:
                continue

            items.append(
                {
                    "title": title,
                    "link": link,
                    "source": source,
                    "published": published.strftime(
                        "%d/%m/%Y %H:%M"
                    ),
                }
            )

            if len(items) >= 15:
                break

        return items

    except Exception:

        return []


# =========================================================
# 11. OLS REGRESSION
# =========================================================

def run_ols(df):

    data = df[
        [
            "Return",
            "Volume",
            "RSI",
            "MACD",
            "Volatility",
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(data) < 40:
        return None

    y = data["Return"]

    x = data[
        [
            "Volume",
            "RSI",
            "MACD",
            "Volatility",
        ]
    ]

    x = sm.add_constant(x)

    model = sm.OLS(
        y,
        x
    ).fit()

    return model


# =========================================================
# 12. RANDOM FOREST
# =========================================================

def run_random_forest(df):

    data = df[
        [
            "Return",
            "Volume",
            "RSI",
            "MACD",
            "Volatility",
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(data) < 50:
        return None

    features = [
        "Volume",
        "RSI",
        "MACD",
        "Volatility",
    ]

    X = data[features]
    y = data["Return"]

    split = int(
        len(data) * 0.8
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(
        ascending=False
    )

    return {
        "model": model,
        "importance": importance,
        "mae": mae,
        "r2": r2,
        "actual": y_test,
        "predicted": predictions,
    }


# =========================================================
# 13. XÁC ĐỊNH TÍN HIỆU
# =========================================================

def determine_signal(df):

    last = df.iloc[-1]

    score = 0
    reasons = []

    # RSI
    rsi = last["RSI"]

    if rsi < 30:
        score += 1
        reasons.append(
            "RSI đang ở vùng quá bán."
        )

    elif rsi > 70:
        score -= 1
        reasons.append(
            "RSI đang ở vùng quá mua."
        )

    # MACD
    if last["MACD"] > last["MACD_Signal"]:
        score += 1
        reasons.append(
            "MACD nằm trên đường tín hiệu."
        )

    else:
        score -= 1
        reasons.append(
            "MACD nằm dưới đường tín hiệu."
        )

    # SMA
    if last["Close"] > last["SMA20"]:
        score += 1
        reasons.append(
            "Giá nằm trên SMA20."
        )

    else:
        score -= 1
        reasons.append(
            "Giá nằm dưới SMA20."
        )

    if last["SMA20"] > last["SMA50"]:
        score += 1
        reasons.append(
            "SMA20 nằm trên SMA50."
        )

    else:
        score -= 1
        reasons.append(
            "SMA20 nằm dưới SMA50."
        )

    if score >= 2:
        label = "TÍCH CỰC"
    elif score <= -2:
        label = "TIÊU CỰC"
    else:
        label = "TRUNG LẬP"

    return label, score, reasons


# =========================================================
# 14. TẠO DATA SUMMARY CHO GEMINI
# =========================================================

def build_market_summary(
    symbol,
    df,
    company,
    ols,
    rf
):

    last = df.iloc[-1]

    price = last["Close"]

    first_price = df["Close"].iloc[0]

    period_return = (
        (price / first_price) - 1
    ) * 100

    signal, score, reasons = (
        determine_signal(df)
    )

    summary = f"""
MÃ CỔ PHIẾU: {symbol}

TÊN CÔNG TY:
{company.get("name", "N/A")}

NGÀNH:
{company.get("sector", "N/A")}

GIÁ HIỆN TẠI:
{price:.2f}

THAY ĐỔI TRONG KHOẢNG DỮ LIỆU:
{period_return:.2f}%

RSI:
{last["RSI"]:.2f}

MACD:
{last["MACD"]:.4f}

MACD SIGNAL:
{last["MACD_Signal"]:.4f}

SMA20:
{last["SMA20"]:.2f}

SMA50:
{last["SMA50"]:.2f}

VOLATILITY 20 NGÀY:
{last["Volatility"]:.2f}%

TÍN HIỆU KỸ THUẬT:
{signal}

ĐIỂM TÍN HIỆU:
{score}

LÝ DO:
{"; ".join(reasons)}
"""

    if ols is not None:

        summary += f"""

OLS R-SQUARED:
{ols.rsquared:.4f}

OLS P-VALUE F-STATISTIC:
{ols.f_pvalue:.4f}
"""

        for name, value in ols.params.items():

            summary += (
                f"\nOLS COEFFICIENT "
                f"{name}: {value:.6f}"
            )

    if rf is not None:

        summary += f"""

RANDOM FOREST R2 TEST:
{rf["r2"]:.4f}

RANDOM FOREST MAE TEST:
{rf["mae"]:.6f}

FEATURE IMPORTANCE:
"""

        for feature, value in (
            rf["importance"].items()
        ):

            summary += (
                f"\n{feature}: "
                f"{value:.4f}"
            )

    return summary


# =========================================================
# 15. GEMINI
# =========================================================

def ask_gemini(
    prompt,
    model_name="gemini-2.5-flash"
):

    api_key = get_secret(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return (
            "⚠️ Chưa cấu hình `GEMINI_API_KEY`."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        response = (
            client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        )

        return response.text

    except Exception as e:

        return (
            "❌ Gemini gặp lỗi khi xử lý.\n\n"
            f"`{str(e)}`"
        )


# =========================================================
# 16. AI INVESTMENT REPORT
# =========================================================

def generate_investment_report(
    symbol,
    days,
    market_summary,
    news
):

    news_text = "\n".join(
        [
            (
                f"- {n['title']} | "
                f"{n['source']} | "
                f"{n['published']} | "
                f"{n['link']}"
            )
            for n in news
        ]
    )

    if not news_text:
        news_text = (
            "Không thu thập được tin tức "
            "trong khoảng thời gian yêu cầu."
        )

    prompt = f"""
Bạn là Brian AI — chuyên gia phân tích
đầu tư định lượng.

Hãy lập một báo cáo nghiên cứu bằng
TIẾNG VIỆT dựa trên dữ liệu bên dưới.

=========================
DỮ LIỆU THỊ TRƯỜNG
=========================

{market_summary}

=========================
TIN TỨC {days} NGÀY GẦN NHẤT
=========================

{news_text}

=========================
CẤU TRÚC BÁO CÁO
=========================

# 🧠 Investment Intelligence — {symbol}

## 1. Tóm tắt điều hành

Tóm tắt tình hình cổ phiếu trong
5-7 câu dễ hiểu.

## 2. Diễn biến giá

Phân tích:
- Giá hiện tại
- Xu hướng
- Return
- SMA20
- SMA50

## 3. Phân tích kỹ thuật

Phân tích:
- RSI
- MACD
- Volatility

Giải thích ý nghĩa của từng chỉ báo.

## 4. Phân tích định lượng

Giải thích kết quả OLS.

Đặc biệt:
- R-squared
- F-test
- coefficient
- ý nghĩa thống kê

Nếu p-value không đủ mạnh,
hãy nói rõ rằng chưa có bằng chứng
thống kê đủ thuyết phục.

## 5. Machine Learning

Phân tích:
- Random Forest R²
- MAE
- Feature importance

Không được biến feature importance
thành kết luận nhân quả.

## 6. Tin tức

Tổng hợp các tin quan trọng nhất.

Với mỗi tin:
- Nội dung chính
- Tác động có thể có
- Tích cực / tiêu cực / trung lập

Không được bịa tin.

## 7. Catalyst

Các yếu tố có thể hỗ trợ giá.

## 8. Risk

Các rủi ro chính.

## 9. Góc nhìn tổng hợp

Đưa ra:
- Tích cực
- Trung lập
- Tiêu cực

Dựa trên toàn bộ dữ liệu.

## 10. Kết luận

Tóm tắt trong 5 dòng.

=========================
QUY TẮC
=========================

1. Chỉ sử dụng dữ liệu được cung cấp.
2. Không bịa số liệu.
3. Không bịa tin tức.
4. Không nói "chắc chắn tăng" hoặc
   "chắc chắn giảm".
5. Không biến tương quan thành nhân quả.
6. Phân biệt dữ liệu thực tế và
   nhận định của AI.
7. Nếu thiếu dữ liệu phải nói:
   "Chưa đủ dữ liệu để kết luận."
8. Không đưa ra lời khuyên mua/bán
   cá nhân hóa.
9. Viết hoàn toàn bằng tiếng Việt.
10. Viết theo phong cách research desk
    chuyên nghiệp, súc tích.
"""

    return ask_gemini(
        prompt
    )


# =========================================================
# 17. HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-label">
            BRIAN AI INVEST LAB
        </div>

        <div class="hero-title">
            Investment Intelligence Platform
        </div>

        <div class="hero-text">
            Nền tảng tổng hợp dữ liệu thị trường,
            phân tích định lượng, Machine Learning
            và tin tức thành một báo cáo nghiên cứu
            đầu tư duy nhất.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 18. LOAD DATA
# =========================================================

try:

    df = load_market_data(
        ticker,
        period_map[period]
    )

    company = load_company_info(
        ticker
    )

except Exception as e:

    st.error(
        f"Không thể tải dữ liệu cho {ticker}: {e}"
    )

    st.stop()


# =========================================================
# 19. CALCULATIONS
# =========================================================

last = df.iloc[-1]

current_price = last["Close"]

previous_price = (
    df["Close"].iloc[-2]
    if len(df) > 1
    else current_price
)

daily_return = (
    (current_price / previous_price) - 1
) * 100

period_return = (
    (current_price / df["Close"].iloc[0]) - 1
) * 100

rsi = last["RSI"]

macd = last["MACD"]

volatility = last["Volatility"]

signal, signal_score, signal_reasons = (
    determine_signal(df)
)


# =========================================================
# 20. COMPANY HEADER
# =========================================================

st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:end;
        margin-bottom:18px;
    ">

        <div>
            <div class="muted">
                ĐANG PHÂN TÍCH
            </div>

            <div style="
                font-size:2rem;
                font-weight:850;
            ">
                {ticker}
            </div>

            <div class="muted">
                {html.escape(
                    company.get("name", ticker)
                )}
            </div>
        </div>

        <div style="
            text-align:right;
        ">
            <div class="muted">
                Cập nhật
            </div>

            <div>
                {datetime.now().strftime(
                    "%d/%m/%Y %H:%M"
                )}
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 21. KPI
# =========================================================

k1, k2, k3, k4, k5 = st.columns(5)

with k1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                GIÁ HIỆN TẠI
            </div>
            <div class="kpi-value">
                {fmt_number(current_price)}
            </div>
            <div class="kpi-sub">
                {ticker}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                1D RETURN
            </div>
            <div class="kpi-value">
                {fmt_percent(daily_return)}
            </div>
            <div class="kpi-sub">
                So với phiên trước
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                RSI 14
            </div>
            <div class="kpi-value">
                {fmt_number(rsi, 1)}
            </div>
            <div class="kpi-sub">
                Động lượng
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                MACD
            </div>
            <div class="kpi-value">
                {fmt_number(macd, 3)}
            </div>
            <div class="kpi-sub">
                MACD 12/26/9
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k5:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                VOLATILITY
            </div>
            <div class="kpi-value">
                {fmt_percent(volatility)}
            </div>
            <div class="kpi-sub">
                Annualized 20D
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 22. SIGNAL
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Tín hiệu tổng hợp</div>',
    unsafe_allow_html=True
)

s1, s2, s3 = st.columns(3)

with s1:

    st.markdown(
        f"""
        <div class="signal">
            <div class="signal-title">
                Technical Signal
            </div>
            <div class="signal-value">
                {signal}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s2:

    st.markdown(
        f"""
        <div class="signal">
            <div class="signal-title">
                Return {period}
            </div>
            <div class="signal-value">
                {period_return:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s3:

    st.markdown(
        f"""
        <div class="signal">
            <div class="signal-title">
                Signal Score
            </div>
            <div class="signal-value">
                {signal_score:+d}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 23. PRICE CHART
# =========================================================

st.markdown(
    '<div class="section-title">📈 Diễn biến giá</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="section-desc">
        Giá đóng cửa và đường trung bình động
        của {ticker} trong {period}.
    </div>
    """,
    unsafe_allow_html=True
)

chart_df = df[
    [
        "Close",
        "SMA20",
        "SMA50",
    ]
].rename(
    columns={
        "Close": "Giá",
        "SMA20": "SMA 20",
        "SMA50": "SMA 50",
    }
)

st.line_chart(
    chart_df,
    height=420
)


# =========================================================
# 24. TECHNICAL DATA
# =========================================================

st.markdown(
    '<div class="section-title">📐 Phân tích kỹ thuật</div>',
    unsafe_allow_html=True
)

t1, t2 = st.columns(2)

with t1:

    technical_df = pd.DataFrame(
        {
            "Chỉ báo": [
                "RSI 14",
                "MACD",
                "MACD Signal",
                "SMA20",
                "SMA50",
                "Volatility",
            ],
            "Giá trị": [
                fmt_number(rsi),
                fmt_number(macd, 4),
                fmt_number(
                    last["MACD_Signal"],
                    4
                ),
                fmt_number(
                    last["SMA20"]
                ),
                fmt_number(
                    last["SMA50"]
                ),
                fmt_percent(
                    volatility
                ),
            ],
        }
    )

    st.dataframe(
        technical_df,
        use_container_width=True,
        hide_index=True
    )

with t2:

    st.markdown(
        '<div class="signal">',
        unsafe_allow_html=True
    )

    st.markdown(
        "#### Nhận định kỹ thuật"
    )

    for reason in signal_reasons:

        st.write(
            f"• {reason}"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 25. QUANT + ML
# =========================================================

st.markdown(
    '<div class="section-title">🤖 Quant & Machine Learning</div>',
    unsafe_allow_html=True
)

ols_model = run_ols(df)

rf_result = run_random_forest(df)

q1, q2 = st.columns(2)


with q1:

    st.markdown(
        '<div class="ai-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-badge">'
        'QUANTITATIVE MODEL'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### OLS Regression"
    )

    if ols_model is not None:

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "R²",
                f"{ols_model.rsquared:.4f}"
            )

        with c2:
            st.metric(
                "F-test p-value",
                f"{ols_model.f_pvalue:.4f}"
            )

        coef_df = pd.DataFrame(
            {
                "Biến": ols_model.params.index,
                "Coefficient": ols_model.params.values,
                "P-value": ols_model.pvalues.values,
            }
        )

        st.dataframe(
            coef_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Chưa đủ dữ liệu để chạy OLS."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


with q2:

    st.markdown(
        '<div class="ai-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-badge">'
        'MACHINE LEARNING'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Random Forest"
    )

    if rf_result is not None:

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Test R²",
                f"{rf_result['r2']:.4f}"
            )

        with c2:

            st.metric(
                "MAE",
                f"{rf_result['mae']:.6f}"
            )

        importance_df = (
            rf_result["importance"]
            .rename("Importance")
            .reset_index()
            .rename(
                columns={
                    "index": "Biến"
                }
            )
        )

        st.bar_chart(
            importance_df.set_index(
                "Biến"
            ),
            height=220
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Chưa đủ dữ liệu để huấn luyện ML."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 26. NEWS
# =========================================================

news = load_news(
    ticker,
    news_days
)

st.markdown(
    '<div class="section-title">📰 Tin tức & Market Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="section-desc">
        Các tin tức được thu thập trong
        {news_days} ngày gần nhất.
    </div>
    """,
    unsafe_allow_html=True
)

if news:

    for item in news:

        st.markdown(
            f"""
            <div class="news-card">

                <div class="news-title">
                    {html.escape(
                        item["title"]
                    )}
                </div>

                <div class="news-meta">
                    {html.escape(
                        item["source"]
                    )}
                    &nbsp; • &nbsp;
                    {item["published"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"[Đọc nguồn →]({item['link']})"
        )

else:

    st.info(
        "Không tìm thấy tin tức phù hợp "
        "trong khoảng thời gian đã chọn."
    )


# =========================================================
# 27. AI REPORT
# =========================================================

st.markdown(
    '<div class="section-title">🧠 AI Investment Research</div>',
    unsafe_allow_html=True
)

market_summary = build_market_summary(
    ticker,
    df,
    company,
    ols_model,
    rf_result
)

if run_analysis:

    with st.spinner(
        "Gemini đang tổng hợp dữ liệu và viết báo cáo..."
    ):

        report = generate_investment_report(
            ticker,
            news_days,
            market_summary,
            news
        )

    st.markdown(
        '<div class="ai-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ai-badge">'
        'BRIAN AI RESEARCH ENGINE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        report
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

else:

    st.info(
        "Bấm **🚀 Chạy phân tích** ở thanh bên "
        "để Gemini tổng hợp dữ liệu thị trường, "
        "Quant, Machine Learning và tin tức."
    )


# =========================================================
# 28. CHAT VỚI GEMINI
# =========================================================

st.markdown(
    '<div class="section-title">💬 Hỏi Brian AI</div>',
    unsafe_allow_html=True
)

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


question = st.chat_input(
    f"Hỏi Brian AI về {ticker}..."
)

if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    st.chat_message(
        "user"
    ).markdown(
        question
    )

    context = f"""
Bạn đang phân tích mã {ticker}.

DỮ LIỆU:
{market_summary}

TIN TỨC:
{chr(10).join(
    [
        n["title"]
        for n in news
    ]
)}

CÂU HỎI:
{question}

Hãy trả lời bằng tiếng Việt.

Chỉ sử dụng dữ liệu ở trên.
Nếu dữ liệu không đủ thì nói rõ.

Không đưa ra lời khuyên mua bán
cá nhân hóa.
"""

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Brian AI đang phân tích..."
        ):

            answer = ask_gemini(
                context
            )

        st.markdown(
            answer
        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# =========================================================
# 29. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        BRIAN AI INVEST LAB
        &nbsp;•&nbsp;
        Quantitative Research
        &nbsp;•&nbsp;
        Machine Learning
        &nbsp;•&nbsp;
        Market Intelligence
        <br><br>
        Dữ liệu mang tính nghiên cứu.
        Không phải khuyến nghị đầu tư cá nhân.
    </div>
    """,
    unsafe_allow_html=True
)
