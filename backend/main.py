import os
import requests
import pandas as pd
import numpy as np
import pyreadstat
from fastapi import FastAPI, UploadFile, File, Form
from google import genai

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- HÀM TÍNH TOÁN CHỈ BÁO KỸ THUẬT NỘI HÀM (KHÔNG CẦN THƯ VIỆN NGOÀI) ---
def add_technical_indicators(df: pd.DataFrame):
    """Tính toán thêm các biến số kỹ thuật để tránh bỏ sót biến"""
    # 1. Đường trung bình động
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. Biến động giá (Volatility - Độ lệch chuẩn 20 phiên)
    df['Volatility_20D'] = df['Close'].pct_change().rolling(window=20).std()
    
    return df

# --- HÀM LẤY DỮ LIỆU ĐA BIẾN THỊ TRƯỜNG ---
def fetch_comprehensive_market_data(symbol: str):
    """Lấy dữ liệu cổ phiếu + VN-Index + Khối ngoại + Chỉ số Kỹ thuật"""
    # 1. Dữ liệu giá cổ phiếu từ DNSE
    url_stock = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from=0&to=9999999999&symbol={symbol.upper()}&resolution=1D"
    res_stock = requests.get(url_stock, headers={"User-Agent": "Mozilla/5.0"})
    
    # 2. Dữ liệu VN-Index làm biến Benchmark
    url_vni = "https://services.entrade.com.vn/chart-api/v2/ohlcs/index?from=0&to=9999999999&symbol=VNINDEX&resolution=1D"
    res_vni = requests.get(url_vni, headers={"User-Agent": "Mozilla/5.0"})
    
    # 3. Chỉ số tài chính & Giao dịch Khối ngoại từ TCBS
    url_tcbs = f"https://apipub.tcbs.com.vn/stock-insight/v1/finance/{symbol.upper()}/financial-indicators"
    res_tcbs = requests.get(url_tcbs, headers={"User-Agent": "Mozilla/5.0"})
    
    df_stock = None
    if res_stock.status_code == 200:
        data = res_stock.json()
        df_stock = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s').dt.strftime('%Y-%m-%d'),
            'Open': data['o'], 'High': data['h'], 'Low': data['l'],
            'Close': data['c'], 'Volume': data['v']
        })
        # Thêm biến kỹ thuật
        df_stock = add_technical_indicators(df_stock)

    df_vni = None
    if res_vni.status_code == 200:
        data_vni = res_vni.json()
        df_vni = pd.DataFrame({
            'Date': pd.to_datetime(data_vni['t'], unit='s').dt.strftime('%Y-%m-%d'),
            'VNINDEX_Close': data_vni['c'],
            'VNINDEX_Volume': data_vni['v']
        })

    # Merge dữ liệu Cổ phiếu và VN-Index theo Ngày để so sánh tương quan
    if df_stock is not None and df_vni is not None:
        merged_df = pd.merge(df_stock, df_vni, on='Date', how='inner')
        return merged_df.tail(20), res_tcbs.json() if res_tcbs.status_code == 200 else {}
    
    return None, {}

@app.get("/api/stock/{symbol}")
def analyze_stock_multivariable(symbol: str):
    df_market, ratios = fetch_comprehensive_market_data(symbol)
    
    if df_market is None:
        return {"error": "Không lấy được dữ liệu thị trường."}

    # Bổ sung Prompt yêu cầu AI chạy ma trận tương quan ngầm
    prompt = f"""
    Bạn là AI Copilot Phân tích Định lượng & Đầu tư Chứng khoán.
    Mã cổ phiếu phân tích: {symbol.upper()}

    [MA TRẬN DỮ LIỆU ĐA BIẾN MỞ RỘNG (20 PHIÊN GẦN NHẤT)]:
    Biến số gồm:
    - Cổ phiếu {symbol.upper()}: Open, High, Low, Close, Volume
    - Biến Kỹ thuật: MA20, MA50, RSI_14, Volatility_20D (Độ lệch chuẩn)
    - Biến Thị trường chung: VNINDEX_Close, VNINDEX_Volume

    {df_market.to_string(index=False)}

    [CHỈ SỐ TÀI CHÍNH (TCBS)]:
    {ratios}

    YÊU CẦU PHÂN TÍCH:
    1. **Phân tích Tương quan & Hệ số Beta:** Đánh giá mức độ biến động của {symbol.upper()} so với VN-Index. Cổ phiếu này đang mạnh hơn hay yếu hơn thị trường chung (Relative Strength)?
    2. **Động lượng & Biến động:** Nhận diện tín hiệu từ RSI, vị thế so với MA20/MA50 và mức độ rủi ro biến động (Volatility).
    3. **Tài chính & Định giá:** Đánh giá các chỉ số tài chính nền tảng.
    4. **Kết luận:** Tổng hợp các biến tác động để đưa ra kịch bản xu hướng và khuyến nghị rủi ro.
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    return {
        "symbol": symbol.upper(),
        "analysis": response.text,
        "prices": df_market.to_dict(orient="records")
    }
