def snapshot(df):
    x = df.iloc[-1]
    return {
        "close": float(x["Close"]),
        "rsi": float(x["RSI"]),
        "macd": float(x["MACD"]),
        "sma20": float(x["SMA20"]),
        "sma50": float(x["SMA50"]),
        "volume": float(x["Volume"]),
        "return": float(x["Return"]),
    }
