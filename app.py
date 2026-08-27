import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Brian AI Invest Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM DARK UI
# ============================================================
st.markdown(
    """
    <style>
    :root { --gold:#c9a45c; --gold2:#e8d39a; --bg:#080b10; --panel:#10151d; --panel2:#151b24; --muted:#8f9aaa; --line:#27303c; }
    .stApp { background: radial-gradient(circle at 80% 0%, #18202c 0%, #080b10 40%); color:#f4f5f7; }
    [data-testid="stSidebar"] { background:#090d13; border-right:1px solid #202833; }
    [data-testid="stSidebar"] * { color:#e9edf3; }
    .block-container { padding-top:1.6rem; max-width:1450px; }
    .brand { display:flex; align-items:center; gap:14px; margin-bottom:28px; }
    .brand-mark { width:48px; height:48px; border-radius:14px; background:linear-gradient(145deg,#e4c67b,#a97b2f); color:#101010; display:flex; align-items:center; justify-content:center; font-weight:900; font-size:24px; box-shadow:0 10px 30px rgba(201,164,92,.15); }
    .brand-title { font-size:20px; font-weight:800; letter-spacing:.4px; }
    .brand-sub { font-size:11px; color:#9ca6b5; letter-spacing:1.4px; text-transform:uppercase; }
    .hero { border:1px solid #2a3441; background:linear-gradient(135deg,rgba(22,29,39,.96),rgba(11,15,21,.96)); border-radius:24px; padding:34px; margin-bottom:22px; box-shadow:0 18px 60px rgba(0,0,0,.25); }
    .eyebrow { color:var(--gold2); font-size:12px; letter-spacing:2px; font-weight:800; text-transform:uppercase; }
    .hero h1 { font-size:42px; margin:8px 0 10px; letter-spacing:-1px; }
    .hero p { color:#aeb7c5; max-width:850px; font-size:15px; line-height:1.7; margin:0; }
    .section-title { font-size:22px; font-weight:800; margin:28px 0 14px; }
    .metric-card { background:linear-gradient(145deg,#111721,#0d1219); border:1px solid #26303d; border-radius:16px; padding:16px 18px; min-height:104px; }
    .metric-label { color:#929dac; font-size:12px; margin-bottom:9px; }
    .metric-value { color:#f7f8fa; font-size:25px; font-weight:800; }
    .metric-delta { font-size:12px; margin-top:5px; color:#9fa9b8; }
    .gold { color:var(--gold2); }
    .news-card { background:#111720; border:1px solid #252e3a; border-radius:14px; padding:15px; margin-bottom:10px; }
    .news-title { font-weight:700; line-height:1.45; }
    .news-meta { color:#7f8998; font-size:11px; margin-top:7px; }
    .signal { border-radius:16px; padding:18px; border:1px solid #2b3441; background:#111720; }
    .signal-title { color:#aab4c3; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
    .signal-value { font-size:30px; font-weight:900; margin-top:4px; }
    .small-note { color:#7f8998; font-size:12px; line-height:1.6; }
    div[data-testid="stMetric"] { background:#111720; border:1px solid #26303d; padding:12px 14px; border-radius:14px; }
    .stButton > button { border:1px solid #705d36; background:linear-gradient(135deg,#c7a45e,#a9803d); color:#111; font-weight:800; border-radius:10px; }
    .stTabs [data-baseweb="tab"] { font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value).strip() if value is not None else default
    except Exception:
        return default


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace(" ", "")
    if not s:
        return "HPG.VN"
    # Vietnamese tickers: HPG -> HPG.VN. Already suffixed stays unchanged.
    if re.fullmatch(r"[A-Z0-9]{2,5}", s) and not s.endswith((".VN", ".HK", ".NS", ".L")):
        if s in {"HPG", "FPT", "VCB", "VHM", "VIC", "MWG", "SSI", "VND", "HPG", "TCB", "MBB", "ACB", "GAS", "MSN", "VRE", "BID", "CTG", "PLX", "VJC", "POW", "SAB", "NVL", "DIG", "PDR"}:
            return s + ".VN"
    return s


def display_symbol(symbol: str) -> str:
    return symbol.upper().replace(".VN", "")


def fmt_price(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}"


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x:+.2f}%"


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột {col}")
    df = df.dropna(subset=["Close"]).copy()
    df["Return"] = df["Close"].pct_change()
    df["RSI"] = rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Volatility20"] = df["Return"].rolling(20).std() * np.sqrt(252) * 100
    df["Volume_Change"] = df["Volume"].pct_change()
    return df.dropna().copy()


@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(symbol: str, period: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        # Fallback to download, which can sometimes be more reliable.
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho {symbol}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return add_indicators(df)


@st.cache_data(ttl=900, show_spinner=False)
def load_company_info(symbol: str) -> dict:
    try:
        info = yf.Ticker(symbol).fast_info
        return dict(info)
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(symbol: str, limit: int = 8) -> list[dict]:
    base = display_symbol(symbol)
    query = quote(f"{base} stock OR {base} chứng khoán")
    url = f"https://news.google.com/rss/search?q={query}&hl=vi&gl=VN&ceid=VN:vi"
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries[:limit]:
        published = entry.get("published", "")
        dt_text = published
        try:
            dt = parsedate_to_datetime(published)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_text = dt.astimezone().strftime("%d/%m/%Y %H:%M")
        except Exception:
            pass
        results.append({
            "title": html.unescape(entry.get("title", "")).strip(),
            "link": entry.get("link", ""),
            "published": dt_text,
            "summary": re.sub("<[^>]+>", " ", entry.get("summary", "")).strip(),
            "source": entry.get("source", {}).get("title", "Google News") if isinstance(entry.get("source"), dict) else "Google News",
        })
    return results


def build_quant(df: pd.DataFrame):
    work = df.copy()
    work["Target"] = work["Return"].shift(-1)
    features = ["RSI", "MACD", "MACD_Hist", "Volatility20", "Volume_Change", "Return"]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["Target"])
    if len(work) < 60:
        return None

    X = work[features]
    y = work["Target"]
    split = int(len(work) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # OLS
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit(cov_type="HC3")

    # Random Forest — prediction is next-session return, not a guarantee of price.
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }
    latest_x = X.iloc[[-1]]
    next_return = float(rf.predict(latest_x)[0])
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return ols, rf, metrics, next_return, importance


def make_chart(df: pd.DataFrame):
    recent = df.tail(180).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["Open"], high=recent["High"], low=recent["Low"], close=recent["Close"],
            name="Giá"
        ), row=1, col=1
    )
    fig.add_trace(go.Scatter(x=recent.index, y=recent["SMA20"], name="SMA20", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["SMA50"], name="SMA50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=recent.index, y=recent["Volume"], name="Volume", opacity=.35), row=2, col=1)
    fig.update_layout(
        height=560, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="#0e131a", plot_bgcolor="#0e131a", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0), font=dict(color="#dbe1e9"),
    )
    return fig


def sentiment_label(rsi_value, macd_value, next_return):
    score = 0
    if rsi_value < 35: score += 1
    elif rsi_value > 70: score -= 1
    if macd_value > 0: score += 1
    else: score -= 1
    if next_return > 0.005: score += 1
    elif next_return < -0.005: score -= 1
    if score >= 2: return "TÍCH CỰC", "Bullish"
    if score <= -2: return "THẬN TRỌNG", "Bearish"
    return "TRUNG TÍNH", "Neutral"


def gemini_analysis(api_key, symbol, df, news, quant):
    if not api_key or genai is None:
        return None, "Gemini chưa được cấu hình."
    try:
        client = genai.Client(api_key=api_key)
        latest = df.iloc[-1]
        ols, rf, metrics, next_return, importance = quant
        news_text = "\n".join([f"- {n['title']} ({n['published']})" for n in news]) or "Không có tin tức phù hợp."
        prompt = f"""
Bạn là chuyên gia phân tích đầu tư của BRIAN AI INVEST LAB. Hãy viết báo cáo bằng TIẾNG VIỆT, rõ ràng và không bịa dữ liệu.
Mã: {display_symbol(symbol)}
Ngày dữ liệu cuối: {df.index[-1].strftime('%d/%m/%Y')}
Giá đóng cửa: {latest['Close']:.2f}
1D return: {latest['Return']*100:.2f}%
RSI: {latest['RSI']:.2f}
MACD: {latest['MACD']:.4f}
Volatility 20 ngày: {latest['Volatility20']:.2f}%/năm
SMA20: {latest['SMA20']:.2f}
SMA50: {latest['SMA50']:.2f}
Random Forest dự báo return phiên kế tiếp: {next_return*100:.2f}%
RF MAE: {metrics['MAE']*100:.3f} điểm % return; RF R2: {metrics['R2']:.3f}
Top feature importance: {importance.head(5).to_dict()}
OLS R-squared: {ols.rsquared:.3f}; OLS F-test p-value: {ols.f_pvalue:.4f}

TIN TỨC THẬT ĐÃ THU THẬP:
{news_text}

Yêu cầu output:
1. Tóm tắt điều gì đang xảy ra.
2. Phân tích kỹ thuật.
3. Phân tích dữ liệu/ML và nói rõ độ tin cậy, không biến dự báo thành cam kết.
4. Tổng hợp tác động của tin tức.
5. 3 yếu tố tích cực.
6. 3 rủi ro.
7. Kết luận ngắn: TÍCH CỰC / TRUNG TÍNH / THẬN TRỌNG và vì sao.
8. Nêu rõ đây là thông tin hỗ trợ nghiên cứu, không phải khuyến nghị mua/bán.
"""
        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt,
        )
        return response.text, None
    except Exception as exc:
        return None, f"Gemini lỗi: {exc}"


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-mark">B</div>
      <div><div class="brand-title">BRIAN AI</div><div class="brand-sub">Investment Research Lab</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### ⚙️ Thiết lập phân tích")
symbol_input = st.sidebar.text_input("Mã cổ phiếu", value="HPG", help="Ví dụ HPG, FPT, VCB. Với Việt Nam app tự thêm .VN")
period = st.sidebar.selectbox("Khoảng dữ liệu", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
news_limit = st.sidebar.slider("Số tin tức", 3, 15, 8)
run = st.sidebar.button("🚀 Chạy phân tích", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Nguồn dữ liệu**")
st.sidebar.markdown("Yahoo Finance · Google News RSS · Gemini")
st.sidebar.markdown('<div class="small-note">Dữ liệu bên ngoài có thể trễ, thiếu hoặc bị giới hạn bởi nhà cung cấp.</div>', unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
symbol = normalize_symbol(symbol_input)
gemini_key = secret("GEMINI_API_KEY")

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">BRIAN AI INVEST LAB</div>
      <h1>Investment Intelligence Platform</h1>
      <p>Thu thập dữ liệu thị trường thật, chỉ báo kỹ thuật, Machine Learning và tin tức để tạo một Investment Brief bằng tiếng Việt. Không dùng dữ liệu random giả.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run and "analysis_symbol" not in st.session_state:
    st.session_state.analysis_symbol = symbol

if run:
    st.session_state.analysis_symbol = symbol

active_symbol = st.session_state.get("analysis_symbol", symbol)

try:
    with st.spinner(f"Đang tải dữ liệu thật cho {display_symbol(active_symbol)}..."):
        df = load_market_data(active_symbol, period)
        news = fetch_news(active_symbol, news_limit)
        quant = build_quant(df)
except Exception as exc:
    st.error(f"Không thể tải dữ liệu cho {display_symbol(active_symbol)}: {exc}")
    st.info("Nếu mã Việt Nam, hãy thử HPG, FPT, VCB. Nếu mã quốc tế, nhập ticker đúng theo Yahoo Finance.")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
change = (latest["Close"] / prev["Close"] - 1) * 100
next_return = quant[3] if quant else np.nan
signal, signal_en = sentiment_label(latest["RSI"], latest["MACD"], next_return if not pd.isna(next_return) else 0)

# Header metrics
st.markdown(f'<div class="section-title">📊 {display_symbol(active_symbol)} · Market Snapshot</div>', unsafe_allow_html=True)
cols = st.columns(6)
metrics = [
    ("Giá đóng cửa", fmt_price(latest["Close"]), f"{fmt_pct(change)} phiên gần nhất"),
    ("1D Return", fmt_pct(latest["Return"] * 100), "so với phiên trước"),
    ("RSI (14)", f"{latest['RSI']:.1f}", "<30 quá bán · >70 quá mua"),
    ("MACD", f"{latest['MACD']:.3f}", "động lượng"),
    ("Volatility", f"{latest['Volatility20']:.1f}%", "annualized 20D"),
    ("ML Next Return", fmt_pct(next_return * 100) if not pd.isna(next_return) else "—", "dự báo thống kê"),
]
for c, (label, value, delta) in zip(cols, metrics):
    with c:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-delta">{delta}</div></div>', unsafe_allow_html=True)

# Signal
st.markdown('<div class="section-title">🎯 Tín hiệu tổng hợp</div>', unsafe_allow_html=True)
a, b, c = st.columns([1, 1, 2])
with a:
    st.markdown(f'<div class="signal"><div class="signal-title">Trạng thái</div><div class="signal-value gold">{signal}</div><div class="small-note">Technical + ML heuristic</div></div>', unsafe_allow_html=True)
with b:
    trend = "Trên SMA50" if latest["Close"] >= latest["SMA50"] else "Dưới SMA50"
    st.markdown(f'<div class="signal"><div class="signal-title">Xu hướng</div><div class="signal-value">{trend}</div><div class="small-note">SMA20: {latest["SMA20"]:.2f} · SMA50: {latest["SMA50"]:.2f}</div></div>', unsafe_allow_html=True)
with c:
    st.markdown('<div class="signal"><div class="signal-title">Lưu ý</div><div class="small-note">Random Forest chỉ dự báo return dựa trên dữ liệu lịch sử và các biến kỹ thuật. Đây không phải dự báo chắc chắn về giá tương lai.</div></div>', unsafe_allow_html=True)

# Chart
st.markdown('<div class="section-title">📈 Giá & Volume thực tế</div>', unsafe_allow_html=True)
st.plotly_chart(make_chart(df), use_container_width=True, config={"displaylogo": False})

# Tabs
st.markdown('<div class="section-title">🧠 Quant · ML · News · Gemini</div>', unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["📐 Quant", "🤖 Machine Learning", "📰 Tin tức", "✨ Gemini Brief"])

with t1:
    if quant:
        ols, rf, metrics, next_return, importance = quant
        left, right = st.columns(2)
        with left:
            st.subheader("OLS Regression")
            st.write(f"R²: **{ols.rsquared:.3f}** · F-test p-value: **{ols.f_pvalue:.4f}**")
            coef = pd.DataFrame({"Hệ số": ols.params, "p-value": ols.pvalues}).round(5)
            st.dataframe(coef, use_container_width=True)
        with right:
            st.subheader("Chỉ báo hiện tại")
            technical = pd.DataFrame({
                "Chỉ báo": ["RSI", "MACD", "MACD Signal", "Volatility 20D", "SMA20", "SMA50"],
                "Giá trị": [latest["RSI"], latest["MACD"], latest["MACD_Signal"], latest["Volatility20"], latest["SMA20"], latest["SMA50"]],
            })
            st.dataframe(technical.round(4), hide_index=True, use_container_width=True)
    else:
        st.warning("Không đủ dữ liệu để chạy OLS/ML. Hãy chọn khoảng 1 năm trở lên.")

with t2:
    if quant:
        ols, rf, metrics, next_return, importance = quant
        st.subheader("Random Forest — Feature Importance")
        imp = importance.rename("Importance").to_frame()
        st.bar_chart(imp)
        m1, m2 = st.columns(2)
        m1.metric("MAE test", f"{metrics['MAE']*100:.3f}%")
        m2.metric("R² test", f"{metrics['R2']:.3f}")
        st.caption("Test set được giữ theo thứ tự thời gian để giảm rò rỉ dữ liệu. ML chỉ mang tính nghiên cứu.")
    else:
        st.warning("Chưa đủ dữ liệu cho ML.")

with t3:
    if news:
        for n in news:
            title = html.escape(n["title"])
            source = html.escape(n["source"])
            published = html.escape(n["published"])
            st.markdown(f'<div class="news-card"><div class="news-title">{title}</div><div class="news-meta">{source} · {published}</div></div>', unsafe_allow_html=True)
            if n["link"]:
                st.markdown(f'[Đọc bài viết ↗]({n["link"]})')
    else:
        st.info("Chưa tìm thấy tin phù hợp cho mã này.")

with t4:
    if not gemini_key:
        st.warning("Chưa cấu hình GEMINI_API_KEY. Phần Market/Quant/News vẫn chạy, nhưng Gemini Brief chưa hoạt động.")
        st.code('GEMINI_API_KEY = "AIza..."', language="toml")
    else:
        if st.button("✨ Tạo Investment Brief bằng Gemini", type="primary"):
            with st.spinner("Gemini đang tổng hợp dữ liệu thị trường + tin tức..."):
                brief, err = gemini_analysis(gemini_key, active_symbol, df, news, quant)
            if err:
                st.error(err)
            else:
                st.markdown(brief)
        else:
            st.info("Bấm nút để Gemini tổng hợp dữ liệu thật và tin tức thành báo cáo tiếng Việt.")

# Footer
st.markdown("---")
st.markdown('<div class="small-note">BRIAN AI INVEST LAB · Research dashboard · Không phải lời khuyên đầu tư. Dữ liệu và tin tức phụ thuộc nguồn bên ngoài.</div>', unsafe_allow_html=True)
