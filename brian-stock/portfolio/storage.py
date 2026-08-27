import pandas as pd

COLUMNS=["Ngày mua","Mã cổ phiếu","Ngành","Giá mua","Giá hiện tại","Số lượng","Giá bán","Ngày bán"]

def empty():
    return pd.DataFrame(columns=COLUMNS)
