def client_message(action, symbol, price, extra=""):
    verb = "tham khảo MUA" if action == "MUA" else "cân nhắc BÁN"
    return f"""Soạn một tin nhắn tiếng Việt thật ngắn để gửi khách hàng.
Mã: {symbol}
Giá tham chiếu: {price}
Hành động: {verb}
Thông tin thêm: {extra}
Chỉ viết 1-3 câu, dễ hiểu, lịch sự, không dùng thuật ngữ ML/OLS/R².
Không hứa chắc lợi nhuận. Không tự bịa dữ liệu."""
