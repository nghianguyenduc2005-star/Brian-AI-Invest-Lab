import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import html
import re
from datetime import datetime, timedelta

import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

gemini_key = st.secrets["GEMINI_API_KEY"]

# Gemini
from google import genai


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="BRIAN AI — Investment Research Lab",
    page_icon="🅱️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

    /* ===== GLOBAL ===== */

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, rgba(212,175,55,0.08), transparent 28%),
            linear-gradient(180deg, #070a0f 0%, #0a0d12 100%);
        color: #f4f4f5;
    }

    section[data-testid="stSidebar"] {
        background: #080b10;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ===== SIDEBAR BRAND ===== */

    .brand {
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 8px 0 25px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 25px;
    }

    .brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #d8b45a, #8e6a25);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #080a0e;
        font-size: 25px;
        font-weight: 900;
        box-shadow: 0 8px 30px rgba(212,175,55,0.18);
    }

    .brand-name {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 0.4px;
    }

    .brand-sub {
        font-size: 10px;
        color: #9ca3af;
        letter-spacing: 1.2px;
        margin-top: 2px;
    }

    /* ===== HERO ===== */

    .hero {
        padding: 34px 38px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(22,30,42,0.95),
                rgba(10,13,18,0.95)
            );
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 25px 70px rgba(0,0,0,0.28);
        margin-bottom: 28px;
    }

    .hero-label {
        color: #d8b45a;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 850;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    .hero-text {
        color: #aeb7c5;
        font-size: 15px;
        line-height: 1.7;
        max-width: 850px;
    }

    /* ===== SECTION ===== */

    .section-title {
        font-size: 22px;
        font-weight: 800;
        margin: 30px 0 16px 0;
    }

    /* ===== METRICS ===== */

    .metric-card {
        background: linear-gradient(145deg, #121821, #0d1118);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 105px;
    }

    .metric-label {
        color: #8993a2;
        font-size: 12px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #f5f5f5;
        font-size: 27px;
        font-weight: 800;
    }

    .metric-sub {
        color: #7f8998;
        font-size: 11px;
        margin-top: 5px;
    }

    /* ===== AI REPORT ===== */

    .ai-box {
        background:
            linear-gradient(
                145deg,
                rgba(26,33,45,0.96),
                rgba(12,16,23,0.96)
            );
        border: 1px solid rgba(216,180,90,0.20);
        border-radius: 18px;
        padding: 25px 28px;
        line-height: 1.75;
        box-shadow: 0 20px 50px rgba(0,0,0,0.20);
    }

    .ai-title {
        color: #d8b45a;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    /* ===== NEWS ===== */

    .news-card {
        background: #10151d;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }

    .news-title {
        font-size: 14px;
        font-weight: 700;
        line-height: 1.45;
    }

    .news-meta {
        color: #7f8998;
        font-size: 11px;
        margin-top: 7px;
    }

    /* ===== TABLE ===== */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* ===== BUTTON ===== */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(216,180,90,0.35);
        background: linear-gradient(135deg, #171d26, #10151c);
        color: #f4f4f5;
        font-weight: 700;
        min-height: 45px;
    }

    .stButton > button:hover {
        border-color: #d8b45a;
        color: #d8b45a;
    }

    /* ===== DIVIDER ===== */

    hr {
        border-color: rgba(255,255,255,0.07);
    }

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def normalize_symbol(symbol: str) -> str:
    """
    Người dùng nhập:
        HPG
        FPT
        VCB

    Yahoo Finance:
        HPG.VN
        FPT.VN
        VCB.VN
    """

    symbol = symbol.strip().upper()

    if "." in symbol:
        return symbol

    return f"{symbol}.VN"


def format_vnd(value):
    if value is None or pd.isna(value):
        return "—"

    return f"{value:,.0f} ₫"


def format_percent(value):
    if value is None or pd.isna(value):
        return "—"

    return f"{value:.2f}%"


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal


# =========================================================
# MARKET DATA
# =========================================================

@st.cache_data(ttl=900)
def get_market_data(symbol, period):

    yahoo_symbol = normalize_symbol(symbol)

    try:

        ticker = yf.Ticker(yahoo_symbol)

        data = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

        if data is None or data.empty:
            return None, yahoo_symbol, "Không tìm thấy dữ liệu."

        data = data.copy()

        # Clean columns
        data = data[[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]]

        # Remove timezone
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        # Indicators
        data["Return"] = data["Close"].pct_change() * 100

        data["RSI"] = calculate_rsi(data["Close"])

        data["MACD"], data["MACD_Signal"] = calculate_macd(
            data["Close"]
        )

        data["Volatility"] = (
            data["Return"]
            .rolling(20)
            .std()
            * np.sqrt(252)
        )

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        data.dropna(subset=["Close"], inplace=True)

        return data, yahoo_symbol, None

    except Exception as e:

        return None, yahoo_symbol, str(e)


# =========================================================
# NEWS
# =========================================================

@st.cache_data(ttl=900)
def get_news(symbol, limit=7):

    clean_symbol = symbol.upper().replace(".VN", "")

    query = f"{clean_symbol} chứng khoán"

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={query}&hl=vi&gl=VN&ceid=VN:vi"
    )

    try:

        feed = feedparser.parse(rss_url)

        news = []

        for entry in feed.entries[:limit]:

            title = html.unescape(
                re.sub("<.*?>", "", entry.get("title", ""))
            )

            link = entry.get("link", "")

            published = entry.get(
                "published",
                ""
            )

            news.append({
                "title": title,
                "link": link,
                "published": published
            })

        return news

    except Exception:

        return []


# =========================================================
# QUANT ANALYSIS
# =========================================================

def run_quant_analysis(data):

    df = data.copy()

    features = [
        "Volume",
        "RSI",
        "MACD",
        "Volatility",
        "MA20",
        "MA50"
    ]

    available = [
        x for x in features
        if x in df.columns
    ]

    model_df = df[
        ["Return"] + available
    ].dropna()

    if len(model_df) < 30:

        return {
            "ols": None,
            "importance": {},
            "r2": None
        }

    y = model_df["Return"]

    X = model_df[available]

    X = sm.add_constant(X)

    try:

        ols = sm.OLS(
            y,
            X
        ).fit()

        r2 = float(ols.rsquared)

    except Exception:

        ols = None
        r2 = None

    importance = {}

    try:

        rf = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            min_samples_leaf=3
        )

        rf.fit(
            model_df[available],
            y
        )

        importance = dict(
            zip(
                available,
                rf.feature_importances_
            )
        )

    except Exception:
        importance = {}

    return {
        "ols": ols,
        "importance": importance,
        "r2": r2
    }


# =========================================================
# GEMINI
# =========================================================

def ask_gemini(
    symbol,
    data,
    news,
    quant
):

    try:

        if "GEMINI_API_KEY" not in st.secrets:

            return (
                "⚠️ Chưa cấu hình GEMINI_API_KEY trong "
                "Streamlit Secrets."
            )

        api_key = st.secrets["GEMINI_API_KEY"]

        client = genai.Client(
            api_key=api_key
        )

        latest = data.iloc[-1]

        price = float(latest["Close"])

        rsi = float(latest["RSI"])

        macd = float(latest["MACD"])

        volatility = float(
            latest["Volatility"]
        ) if not pd.isna(
            latest["Volatility"]
        ) else None

        return_1d = float(
            latest["Return"]
        ) if not pd.isna(
            latest["Return"]
        ) else 0

        return_1m = (
            data["Close"].iloc[-1]
            /
            data["Close"].iloc[-22]
            - 1
        ) * 100 if len(data) >= 22 else None

        return_3m = (
            data["Close"].iloc[-1]
            /
            data["Close"].iloc[-64]
            - 1
        ) * 100 if len(data) >= 64 else None

        news_text = ""

        for item in news:

            news_text += (
                f"- {item['title']}\n"
            )

        if not news_text:

            news_text = "Không lấy được tin tức."

        importance_text = ""

        for key, value in sorted(
            quant["importance"].items(),
            key=lambda x: x[1],
            reverse=True
        ):

            importance_text += (
                f"{key}: {value:.4f}\n"
            )

        prompt = f"""
Bạn là Brian AI, một trợ lý nghiên cứu đầu tư chuyên nghiệp.

Hãy viết một báo cáo phân tích chứng khoán bằng TIẾNG VIỆT,
ngắn gọn nhưng có chiều sâu.

Mã cổ phiếu: {symbol}

DỮ LIỆU THỊ TRƯỜNG:
- Giá hiện tại: {price:,.0f}
- Return 1 ngày: {return_1d:.2f}%
- Return khoảng 1 tháng: {return_1m:.2f}% nếu có
- Return khoảng 3 tháng: {return_3m:.2f}% nếu có
- RSI: {rsi:.2f}
- MACD: {macd:.4f}
- Volatility năm hóa: {volatility:.2f}% nếu có

OLS:
- R-squared: {quant["r2"] if quant["r2"] is not None else "N/A"}

RANDOM FOREST FEATURE IMPORTANCE:
{importance_text}

TIN TỨC GẦN ĐÂY:
{news_text}

YÊU CẦU BÁO CÁO:

1. Tóm tắt tình hình hiện tại.
2. Phân tích xu hướng giá.
3. Phân tích RSI và MACD.
4. Phân tích biến động/rủi ro.
5. Tổng hợp các tin tức đáng chú ý.
6. Đánh giá yếu tố nào đang ảnh hưởng mạnh nhất.
7. Nêu các điểm tích cực.
8. Nêu các rủi ro/cảnh báo.
9. Đưa ra kịch bản:
   - Tích cực
   - Cơ sở
   - Tiêu cực
10. Kết luận cuối cùng thật rõ ràng.

Không được bịa số liệu hoặc tin tức.
Nếu dữ liệu không đủ để kết luận thì nói rõ.
Không khẳng định chắc chắn giá sẽ tăng/giảm.
Đây là báo cáo nghiên cứu, không phải lời khuyên đầu tư cá nhân.

Định dạng đẹp bằng Markdown.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            "⚠️ Gemini chưa thể tạo báo cáo.\n\n"
            f"Chi tiết lỗi: `{str(e)}`"
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">B</div>
            <div>
                <div class="brand-name">BRIAN AI</div>
                <div class="brand-sub">
                    INVESTMENT RESEARCH LAB
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### ⚙️ Thiết lập phân tích"
    )

    symbol = st.text_input(
        "Mã cổ phiếu",
        value="HPG",
        max_chars=10
    )

    period_label = st.selectbox(
        "Khoảng dữ liệu",
        [
            "6 tháng",
            "1 năm",
            "2 năm",
            "5 năm"
        ],
        index=1
    )

    period_map = {
        "6 tháng": "6mo",
        "1 năm": "1y",
        "2 năm": "2y",
        "5 năm": "5y"
    }

    news_limit = st.slider(
        "Tin tức gần nhất",
        min_value=3,
        max_value=15,
        value=7
    )

    st.markdown("---")

    run_analysis = st.button(
        "🚀 Chạy phân tích",
        use_container_width=True
    )

    st.markdown("---")

    st.caption(
        "Brian AI Invest Lab"
    )

    st.caption(
        "Quant • Machine Learning • News Intelligence"
    )


# =========================================================
# HERO
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
            Tổng hợp dữ liệu thị trường, phân tích định lượng,
            Machine Learning và tin tức thành một báo cáo
            nghiên cứu đầu tư thống nhất.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INITIAL STATE
# =========================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# =========================================================
# RUN
# =========================================================

if run_analysis:

    if not symbol.strip():

        st.error(
            "Vui lòng nhập mã cổ phiếu."
        )

        st.stop()

    with st.spinner(
        f"Đang thu thập dữ liệu {symbol.upper()}..."
    ):

        data, yahoo_symbol, error = get_market_data(
            symbol,
            period_map[period_label]
        )

    if data is None or data.empty:

        st.error(
            f"Không thể tải dữ liệu cho "
            f"{symbol.upper()} ({yahoo_symbol})."
        )

        st.warning(
            "Kiểm tra lại mã cổ phiếu hoặc thử lại sau."
        )

        st.stop()

    # News

    with st.spinner("Đang thu thập tin tức..."):

        news = get_news(
            symbol,
            news_limit
        )

    # Quant

    with st.spinner(
        "Đang chạy mô hình định lượng..."
    ):

        quant = run_quant_analysis(
            data
        )

    # Gemini

    with st.spinner(
        "Gemini đang tổng hợp báo cáo..."
    ):

        ai_report = ask_gemini(
            symbol.upper(),
            data,
            news,
            quant
        )

    st.session_state.analysis_done = True

    st.session_state.data = data

    st.session_state.news = news

    st.session_state.quant = quant

    st.session_state.ai_report = ai_report

    st.session_state.symbol = symbol.upper()

    st.session_state.yahoo_symbol = yahoo_symbol


# =========================================================
# DISPLAY RESULTS
# =========================================================

if st.session_state.analysis_done:

    data = st.session_state.data

    news = st.session_state.news

    quant = st.session_state.quant

    ai_report = st.session_state.ai_report

    display_symbol = st.session_state.symbol

    latest = data.iloc[-1]

    price = float(latest["Close"])

    day_return = float(
        latest["Return"]
    )

    rsi = float(
        latest["RSI"]
    )

    macd = float(
        latest["MACD"]
    )

    volatility = float(
        latest["Volatility"]
    ) if not pd.isna(
        latest["Volatility"]
    ) else np.nan

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        f"""
        <div class="section-title">
            📊 {display_symbol} — Market Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"Dữ liệu: {st.session_state.yahoo_symbol} • "
        f"{len(data)} phiên • Cập nhật {data.index[-1].strftime('%d/%m/%Y')}"
    )

    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    GIÁ HIỆN TẠI
                </div>
                <div class="metric-value">
                    {format_vnd(price)}
                </div>
                <div class="metric-sub">
                    Close
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    1D RETURN
                </div>
                <div class="metric-value">
                    {format_percent(day_return)}
                </div>
                <div class="metric-sub">
                    Phiên gần nhất
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    RSI
                </div>
                <div class="metric-value">
                    {rsi:.1f}
                </div>
                <div class="metric-sub">
                    14-period
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    MACD
                </div>
                <div class="metric-value">
                    {macd:.2f}
                </div>
                <div class="metric-sub">
                    Momentum
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:

        volatility_text = (
            f"{volatility:.1f}%"
            if not pd.isna(volatility)
            else "—"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    VOLATILITY
                </div>
                <div class="metric-value">
                    {volatility_text}
                </div>
                <div class="metric-sub">
                    Annualized
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # PRICE CHART
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Giá & xu hướng</div>',
        unsafe_allow_html=True
    )

    chart_df = data[
        ["Close", "MA20", "MA50"]
    ].rename(
        columns={
            "Close": "Giá",
            "MA20": "MA20",
            "MA50": "MA50"
        }
    )

    st.line_chart(
        chart_df,
        height=420
    )

    # =====================================================
    # INDICATORS
    # =====================================================

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown(
            '<div class="section-title">📐 RSI</div>',
            unsafe_allow_html=True
        )

        st.line_chart(
            data[["RSI"]],
            height=260
        )

    with col_b:

        st.markdown(
            '<div class="section-title">〽️ MACD</div>',
            unsafe_allow_html=True
        )

        st.line_chart(
            data[
                ["MACD", "MACD_Signal"]
            ],
            height=260
        )

    # =====================================================
    # AI REPORT
    # =====================================================

    st.markdown(
        '<div class="section-title">🧠 Brian AI Research Report</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="ai-box">
            <div class="ai-title">
                AI INVESTMENT INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(ai_report)

    # =====================================================
    # NEWS
    # =====================================================

    st.markdown(
        '<div class="section-title">📰 Tin tức gần đây</div>',
        unsafe_allow_html=True
    )

    if news:

        for item in news:

            title = html.escape(
                item["title"]
            )

            published = html.escape(
                item["published"]
            )

            link = item["link"]

            st.markdown(
                f"""
                <div class="news-card">
                    <div class="news-title">
                        {title}
                    </div>
                    <div class="news-meta">
                        {published}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"[Đọc bài viết ↗]({link})"
            )

    else:

        st.info(
            "Chưa lấy được tin tức từ nguồn RSS."
        )

    # =====================================================
    # QUANT
    # =====================================================

    st.markdown(
        '<div class="section-title">🔬 Phân tích định lượng</div>',
        unsafe_allow_html=True
    )

    q1, q2 = st.columns(2)

    with q1:

        st.markdown("#### OLS Regression")

        if quant["ols"] is not None:

            ols = quant["ols"]

            ols_table = pd.DataFrame({
                "Biến": ols.params.index,
                "Coefficient": ols.params.values,
                "P-value": ols.pvalues.values
            })

            st.dataframe(
                ols_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"R² = {ols.rsquared:.4f} • "
                f"Adj. R² = {ols.rsquared_adj:.4f}"
            )

        else:

            st.info(
                "Không đủ dữ liệu để chạy OLS."
            )

    with q2:

        st.markdown(
            "#### Random Forest — Feature Importance"
        )

        if quant["importance"]:

            importance_df = pd.DataFrame(
                list(
                    quant["importance"].items()
                ),
                columns=[
                    "Biến",
                    "Importance"
                ]
            ).sort_values(
                "Importance",
                ascending=False
            )

            st.bar_chart(
                importance_df.set_index("Biến")
            )

        else:

            st.info(
                "Không đủ dữ liệu để chạy Random Forest."
            )

    # =====================================================
    # RAW DATA
    # =====================================================

    with st.expander(
        "📋 Xem dữ liệu thị trường"
    ):

        st.dataframe(
            data.tail(100).sort_index(
                ascending=False
            ),
            use_container_width=True
        )

else:

    st.info(
        "👈 Nhập mã cổ phiếu bên trái rồi bấm "
        "**🚀 Chạy phân tích** để bắt đầu."
    )
