def signal(rsi, macd):
    score = (1 if rsi < 35 else -1 if rsi > 70 else 0) + (1 if macd > 0 else -1)
    return "TÍCH CỰC" if score >= 1 else "THẬN TRỌNG" if score <= -1 else "TRUNG TÍNH"
