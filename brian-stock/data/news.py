import html, re
from urllib.parse import quote
import feedparser
import streamlit as st

SOURCES = ["DNSE", "TCBS", "Vietstock"]

@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(symbol="", limit=12, query_extra=""):
    base = symbol.replace(".VN","").upper()
    q = quote(f"{base} chứng khoán {query_extra}".strip())
    url = f"https://news.google.com/rss/search?q={q}&hl=vi&gl=VN&ceid=VN:vi"
    feed = feedparser.parse(url)
    out=[]
    for e in feed.entries[:limit]:
        title = html.unescape(e.get("title","")).strip()
        source = e.get("source",{}).get("title","Google News") if isinstance(e.get("source"),dict) else "Google News"
        out.append({
            "title": title,
            "source": source,
            "link": e.get("link",""),
            "published": e.get("published",""),
            "is_vietnam_source": any(s.lower() in title.lower()+" "+source.lower() for s in SOURCES),
        })
    return out

def fetch_market_news(limit=20):
    return fetch_news("", limit, "Việt Nam thị trường chứng khoán VN-Index DNSE TCBS Vietstock")
