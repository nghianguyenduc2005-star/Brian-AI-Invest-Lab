import re
import html
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import statsmodels.api as sm

from urllib.parse import quote
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Brian AI Invest Lab",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }

    .news-card {
        background: #171a21;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #292d36;
    }

    .metric-card {
        background: #171a21;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #292d36;
    }

    .source {
        color: #7db7ff;
        font-size: 13px;
    }

    .positive {
        color: #3ddc84;
    }

    .negative {
        color: #ff6b6b;
    }

    .neutral {
        color: #cccccc;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.title("⚡ Brian AI Invest Lab")
st.caption("Quant & ML Copilot — Market Data + News + AI Synthesis")


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Chào Nghĩa! 👋\n\n"
                "Tao có thể tổng hợp tin tức + dữ liệu thị trường + "
                "Quant/ML cho cổ phiếu.\n\n"
                "Ví dụ:\n"
                "- Tổng hợp HPG trong 7 ngày gần nhất\n"
                "- Phân tích FPT 30 ngày\n"
                "- Phân tích VCB và các tin tức mới nhất"
            )
        }
    ]


# =========================================================
# PARSE USER REQUEST
# =========================================================

def parse_request(prompt):
    """
    Trích xuất mã cổ phiếu và số ngày từ câu hỏi.
    Ví dụ:
    'tổng hợp thông tin HPG trong 7 ngày gần nhất'
    -> HPG, 7
    """

    prompt_upper = prompt.upper()

    # Tìm ticker Việt Nam phổ biến: 2-4 ký tự
    blacklist = {
        "TỔNG", "HỢP", "THÔNG", "TIN", "TRONG",
        "NGÀY", "GẦN", "NHẤT", "PHÂN", "TÍCH",
        "CỔ", "PHIẾU", "CHO", "MÃ", "TỪ", "ĐẾN"
    }

    words = re.findall(r"\b[A-Z]{2,4}\b", prompt_upper)

    ticker = None

    for word in words:
        if word not in blacklist:
            ticker = word
            break

    if ticker is None:
        ticker = "HPG"

    # Tìm số ngày
    day_match = re.search(r"(\d+)\s*ngày", prompt.lower())

    if day_match:
        days = int(day_match.group(1))
    else:
        days = 7

    days = max(1, min(days, 30))

    return ticker, days


# =========================================================
# MARKET DATA
# =========================================================

@st.cache_data(ttl=900)
def get_market_data(ticker):
    """
    Lấy dữ liệu giá thật từ Yahoo Finance.
    Mã Việt Nam dùng hậu tố .VN
    """

    yahoo_ticker = ticker.upper()

    if not yahoo_ticker.endswith(".VN"):
        yahoo_ticker = yahoo_ticker + ".VN"

    try:
        df = yf.download(
            yahoo_ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            return None, yahoo_ticker

        # Xử lý MultiIndex nếu Yahoo trả về
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        return df, yahoo_ticker

    except Exception as e:
        return None, yahoo_ticker


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    data = df.copy()

    data["Return"] = data["Close"].pct_change()

    # RSI 14
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Volume change
    data["Volume_Change"] = data["Volume"].pct_change()

    # Volatility 20 ngày
    data["Volatility_20D"] = (
        data["Return"].rolling(20).std() * np.sqrt(252)
    )

    return data


# =========================================================
# OLS
# =========================================================

def run_ols(data):

    features = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility_20D"
    ]

    clean = data[features + ["Return"]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(clean) < 50:
        return None

    X = sm.add_constant(clean[features])
    y = clean["Return"]

    model = sm.OLS(y, X).fit()

    return model


# =========================================================
# RANDOM FOREST
# =========================================================

def run_random_forest(data):

    features = [
        "Volume_Change",
        "RSI",
        "MACD",
        "Volatility_20D"
    ]

    clean = data[features + ["Return"]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    # Dự báo return ngày tiếp theo
    clean["Target"] = clean["Return"].shift(-1)

    clean = clean.dropna()

    if len(clean) < 80:
        return None

    X = clean[features]
    y = clean["Target"]

    split = int(len(clean) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        min_samples_leaf=3
    )

    model.fit(X_train, y_train)

    latest = X.iloc[[-1]]

    prediction = model.predict(latest)[0]

    importance = dict(
        zip(
            features,
            model.feature_importances_
        )
    )

    return {
        "model": model,
        "prediction": prediction,
        "importance": importance
    }


# =========================================================
# NEWS
# =========================================================

@st.cache_data(ttl=900)
def get_news(ticker, days=7):

    query = quote(
        f"{ticker} cổ phiếu"
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={query}&hl=vi&gl=VN&ceid=VN:vi"
    )

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        text = response.text

        items = re.findall(
            r"<item>(.*?)</item>",
            text,
            flags=re.DOTALL
        )

        news = []

        cutoff = datetime.utcnow() - timedelta(days=days)

        for item in items:

            title_match = re.search(
                r"<title>(.*?)</title>",
                item,
                flags=re.DOTALL
            )

            link_match = re.search(
                r"<link>(.*?)</link>",
                item,
                flags=re.DOTALL
            )

            pub_match = re.search(
                r"<pubDate>(.*?)</pubDate>",
                item,
                flags=re.DOTALL
            )

            source_match = re.search(
                r"<source[^>]*>(.*?)</source>",
                item,
                flags=re.DOTALL
            )

            if not title_match or not link_match:
                continue

            title = html.unescape(
                re.sub(
                    "<.*?>",
                    "",
                    title_match.group(1)
                )
            )

            link = html.unescape(
                link_match.group(1).strip()
            )

            source = (
                html.unescape(
                    re.sub(
                        "<.*?>",
                        "",
                        source_match.group(1)
                    )
                )
                if source_match
                else "Google News"
            )

            published = None

            if pub_match:

                try:
                    published = pd.to_datetime(
                        pub_match.group(1),
                        utc=True
                    )
                except Exception:
                    published = None

            if published is not None:

                published_naive = published.tz_convert(
                    None
                ).to_pydatetime()

                if published_naive < cutoff:
                    continue

            news.append(
                {
                    "title": title,
                    "link": link,
                    "source": source,
                    "published": published
                }
            )

        return news[:20]

    except Exception as e:

        return []


# =========================================================
# SIMPLE NEWS SENTIMENT
# =========================================================

def classify_news(title):

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
        "hưởng lợi"
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
        "suy giảm"
    ]

    text = title.lower()

    pos = sum(
        word in text
        for word in positive_words
    )

    neg = sum(
        word in text
        for word in negative_words
    )

    if pos > neg:
        return "positive"

    if neg > pos:
        return "negative"

    return "neutral"


# =========================================================
# OPENAI AI SYNTHESIS
# =========================================================

def generate_ai_summary(
    ticker,
    days,
    market_summary,
    news
):

    api_key = st.secrets.get(
        "OPENAI_API_KEY",
        None
    )

    if not api_key:
        return None

    try:

        from openai import OpenAI

        client = OpenAI(
            api_key=api_key
        )

        news_text = "\n".join(
            [
                f"- {n['title']} | {n['source']}"
                for n in news
            ]
        )

        prompt = f"""
Bạn là Brian AI Invest Lab, một trợ lý phân tích
thị trường chứng khoán.

Hãy tổng hợp thông tin về cổ phiếu {ticker}
trong {days} ngày gần nhất.

DỮ LIỆU THỊ TRƯỜNG:
{market_summary}

TIN TỨC:
{news_text}

Yêu cầu:

1. Tóm tắt tình hình hiện tại.
2. Những tin tức quan trọng nhất.
3. Yếu tố tích cực.
4. Yếu tố tiêu cực.
5. Rủi ro cần chú ý.
6. Phân tích kỹ thuật dựa trên dữ liệu được cung cấp.
7. Nhận xét ML nếu có.
8. Kết luận ngắn gọn.

Không được bịa số liệu hoặc tin tức.

Nếu dữ liệu không đủ để kết luận,
phải nói rõ "chưa đủ dữ liệu".

Không đưa ra lời khuyên mua/bán tuyệt đối.
Đây là phân tích thông tin, không phải khuyến nghị đầu tư.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return response.output_text

    except Exception as e:

        return (
            f"Không gọi được AI API: {str(e)}"
        )


# =========================================================
# DISPLAY NEWS
# =========================================================

def display_news(news):

    if not news:

        st.warning(
            "Không lấy được tin tức trong khoảng thời gian yêu cầu."
        )

        return

    for item in news:

        sentiment = classify_news(
            item["title"]
        )

        if sentiment == "positive":
            label = "🟢 Tích cực"

        elif sentiment == "negative":
            label = "🔴 Tiêu cực"

        else:
            label = "⚪ Trung lập"

        published = item["published"]

        if published is not None:
            date_text = published.strftime(
                "%d/%m/%Y %H:%M"
            )
        else:
            date_text = ""

        st.markdown(
            f"""
            <div class="news-card">

            <b>{item["title"]}</b>

            <br><br>

            {label}

            <br>

            <span class="source">
            {item["source"]} · {date_text}
            </span>

            <br><br>

            <a href="{item["link"]}" target="_blank">
            🔗 Đọc nguồn
            </a>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# MARKET DASHBOARD
# =========================================================

def display_market_dashboard(
    ticker,
    data,
    rf_result,
    ols_model
):

    latest = data.iloc[-1]

    price = float(latest["Close"])

    return_1d = float(
        latest["Return"] * 100
    )

    rsi = float(latest["RSI"])

    macd = float(latest["MACD"])

    volatility = float(
        latest["Volatility_20D"] * 100
    )

    cols = st.columns(5)

    cols[0].metric(
        "Giá",
        f"{price:,.0f}"
    )

    cols[1].metric(
        "1D Return",
        f"{return_1d:.2f}%"
    )

    cols[2].metric(
        "RSI",
        f"{rsi:.1f}"
    )

    cols[3].metric(
        "MACD",
        f"{macd:.3f}"
    )

    cols[4].metric(
        "Volatility",
        f"{volatility:.1f}%"
    )

    st.subheader("📈 Giá thực tế")

    chart_data = data[
        ["Close"]
    ].tail(180)

    st.line_chart(
        chart_data
    )

    if rf_result:

        prediction = (
            rf_result["prediction"] * 100
        )

        st.subheader(
            "🤖 Random Forest — dự báo Return ngày kế tiếp"
        )

        if prediction > 0:

            st.success(
                f"Mô hình dự báo return ngày kế tiếp: "
                f"**+{prediction:.2f}%**"
            )

        else:

            st.error(
                f"Mô hình dự báo return ngày kế tiếp: "
                f"**{prediction:.2f}%**"
            )

        st.write(
            "**Feature importance:**"
        )

        importance_df = pd.DataFrame(
            {
                "Feature":
                    list(
                        rf_result[
                            "importance"
                        ].keys()
                    ),
                "Importance":
                    list(
                        rf_result[
                            "importance"
                        ].values()
                    )
            }
        ).sort_values(
            "Importance",
            ascending=False
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    if ols_model:

        with st.expander(
            "📐 Xem kết quả OLS"
        ):

            st.text(
                ols_model.summary().as_text()
            )


# =========================================================
# CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )


# =========================================================
# MAIN CHAT
# =========================================================

prompt = st.chat_input(
    "Ví dụ: Tổng hợp thông tin HPG trong 7 ngày gần nhất"
)


if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        ticker, days = parse_request(
            prompt
        )

        st.write(
            f"🔎 Đang phân tích **{ticker}** "
            f"trong **{days} ngày gần nhất**..."
        )

        # -------------------------------------------------
        # MARKET
        # -------------------------------------------------

        data, yahoo_ticker = get_market_data(
            ticker
        )

        if data is None:

            st.error(
                f"Không lấy được dữ liệu cho {ticker}."
            )

            st.info(
                f"Đã thử mã Yahoo Finance: {yahoo_ticker}"
            )

            st.stop()

        data = calculate_indicators(
            data
        )

        # -------------------------------------------------
        # NEWS
        # -------------------------------------------------

        news = get_news(
            ticker,
            days
        )

        # -------------------------------------------------
        # MODELS
        # -------------------------------------------------

        ols_model = run_ols(
            data
        )

        rf_result = run_random_forest(
            data
        )

        # -------------------------------------------------
        # MARKET SUMMARY FOR AI
        # -------------------------------------------------

        latest = data.iloc[-1]

        market_summary = f"""
Ticker: {ticker}

Latest price:
{latest["Close"]}

1-day return:
{latest["Return"] * 100:.2f}%

RSI:
{latest["RSI"]:.2f}

MACD:
{latest["MACD"]:.4f}

20-day annualized volatility:
{latest["Volatility_20D"] * 100:.2f}%
"""

        if rf_result:

            market_summary += f"""

Random Forest next-day predicted return:
{rf_result["prediction"] * 100:.2f}%
"""

        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        ai_summary = generate_ai_summary(
            ticker,
            days,
            market_summary,
            news
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.success(
            f"✅ Đã tổng hợp dữ liệu cho {ticker}"
        )

        st.subheader(
            "🧠 AI Investment Brief"
        )

        if ai_summary:

            st.markdown(
                ai_summary
            )

        else:

            st.info(
                "Chưa cấu hình OPENAI_API_KEY. "
                "Dưới đây là dữ liệu thị trường và tin tức "
                "thật được thu thập."
            )

        # -------------------------------------------------
        # MARKET
        # -------------------------------------------------

        st.subheader(
            "📊 Market & Quant"
        )

        display_market_dashboard(
            ticker,
            data,
            rf_result,
            ols_model
        )

        # -------------------------------------------------
        # NEWS
        # -------------------------------------------------

        st.subheader(
            f"📰 Tin tức {ticker} "
            f"trong {days} ngày gần nhất"
        )

        display_news(
            news
        )

        # -------------------------------------------------
        # SAVE RESPONSE
        # -------------------------------------------------

        response_for_history = (
            f"Đã phân tích {ticker} "
            f"trong {days} ngày gần nhất."
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_for_history
            }
        )
