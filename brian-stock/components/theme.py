import base64
from pathlib import Path
import streamlit as st
from config.settings import LOGO_PATH, BACKGROUND_PATH

def _data_uri(path: Path):
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

def inject_theme():
    logo = _data_uri(LOGO_PATH)
    bg = _data_uri(BACKGROUND_PATH)
    bg_css = f'url("{bg}")' if bg else "none"

    st.markdown(f"""
    <style>
    :root {{
      --bg:#071019; --panel:#0d1822; --panel2:#101e2a;
      --line:#203140; --text:#edf3f7; --muted:#91a2b0;
      --red:#e53935; --red2:#ff5b57; --green:#24c78e;
    }}
    .stApp {{
      background:
        linear-gradient(90deg, rgba(5,12,18,.97) 0%, rgba(5,12,18,.88) 48%, rgba(5,12,18,.94) 100%),
        {bg_css};
      background-size: cover;
      background-attachment: fixed;
      color:var(--text);
    }}
    [data-testid="stSidebar"] {{
      background:rgba(5,12,18,.97);
      border-right:1px solid var(--line);
    }}
    [data-testid="stSidebar"] * {{ color:#e8eef2; }}
    .block-container {{ max-width:1500px; padding-top:1.25rem; }}
    .brand-wrap {{
      display:flex; align-items:center; gap:12px; padding:4px 0 22px;
      border-bottom:1px solid var(--line); margin-bottom:18px;
    }}
    .brand-logo {{
      width:54px; height:54px; object-fit:contain; border-radius:12px;
      background:#fff; padding:4px;
    }}
    .brand-name {{ font-size:21px; font-weight:900; letter-spacing:.5px; }}
    .brand-sub {{ color:#8fa0ad; font-size:10px; letter-spacing:1.5px; }}
    .hero {{
      position:relative; overflow:hidden; min-height:230px;
      border:1px solid #253847; border-radius:22px; padding:34px;
      background:linear-gradient(90deg, rgba(8,17,25,.98), rgba(8,17,25,.72), rgba(8,17,25,.90));
      box-shadow:0 20px 70px rgba(0,0,0,.28);
    }}
    .hero h1 {{ font-size:40px; margin:4px 0 8px; letter-spacing:-1.3px; }}
    .hero p {{ max-width:760px; color:#aab8c3; line-height:1.7; margin:0; }}
    .eyebrow {{ color:#ff625e; font-size:11px; letter-spacing:2px; font-weight:900; }}
    .section-title {{ font-size:21px; font-weight:900; margin:26px 0 13px; }}
    .card {{
      background:linear-gradient(145deg,rgba(16,30,42,.98),rgba(10,21,30,.98));
      border:1px solid var(--line); border-radius:16px; padding:16px;
    }}
    .metric-label {{ color:#8fa0ad; font-size:11px; text-transform:uppercase; letter-spacing:.8px; }}
    .metric-value {{ font-size:25px; font-weight:900; margin-top:6px; }}
    .metric-sub {{ color:#8798a6; font-size:11px; margin-top:4px; }}
    .positive {{ color:var(--green); }}
    .negative {{ color:#ff6864; }}
    .neutral {{ color:#c5d0d8; }}
    .news-card {{ padding:14px 16px; border:1px solid var(--line); border-radius:13px; background:#0c1822; margin-bottom:9px; }}
    .news-title {{ font-weight:750; line-height:1.45; }}
    .news-meta {{ color:#748895; font-size:10px; margin-top:6px; }}
    .chat-bubble {{
      border:1px solid var(--line); border-radius:15px; padding:14px 16px;
      background:#0d1a24; margin:7px 0;
    }}
    .chat-user {{ border-color:#4d2a2a; background:#1a1214; }}
    .tiny {{ color:#82939f; font-size:11px; }}
    div[data-testid="stMetric"] {{
      background:#0c1822; border:1px solid var(--line); border-radius:14px;
    }}
    .stButton > button {{
      border:1px solid #71302e; border-radius:10px;
      background:linear-gradient(135deg,#e64541,#b92e2b);
      color:#fff; font-weight:800;
    }}
    </style>
    """, unsafe_allow_html=True)
