from pathlib import Path

APP_NAME = "BRIAN STOCK"
BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
BACKGROUND_PATH = ASSETS_DIR / "background.png"

VIETNAM_TICKERS = {
    "HPG","FPT","VCB","VHM","VIC","MWG","SSI","VND","TCB","MBB","ACB",
    "GAS","MSN","VRE","BID","CTG","PLX","VJC","POW","SAB","NVL","DIG","PDR",
    "STB","HDB","TPB","VPB","SHB","VIX","HDC","REE","GMD","DGC","NLG"
}

DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]
