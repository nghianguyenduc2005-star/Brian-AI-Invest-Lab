import pandas as pd

def calculate(df):
    df=df.copy()
    for c in ["Giá mua","Giá hiện tại","Giá bán","Số lượng"]:
        if c not in df: df[c]=0.0
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    df["Giá trị mua"]=df["Giá mua"]*df["Số lượng"]
    df["Giá trị hiện tại"]=df["Giá hiện tại"]*df["Số lượng"]
    df["Lãi/Lỗ"]=df["Giá trị hiện tại"]-df["Giá trị mua"]
    df["Lãi/Lỗ %"]=df["Lãi/Lỗ"].div(df["Giá trị mua"].replace(0,pd.NA))*100
    return df
