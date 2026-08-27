import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def run_quant(df, independent_vars, dependent_var):
    vars_ = [v for v in independent_vars if v in df.columns]
    if dependent_var not in df.columns:
        raise ValueError(f"Biến phụ thuộc không tồn tại: {dependent_var}")
    work = df[vars_ + [dependent_var]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(work) < 60:
        raise ValueError("Cần ít nhất 60 quan sát sau khi loại dữ liệu thiếu.")
    split = int(len(work)*.8)
    X_train, X_test = work[vars_].iloc[:split], work[vars_].iloc[split:]
    y_train, y_test = work[dependent_var].iloc[:split], work[dependent_var].iloc[split:]
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit(cov_type="HC3")
    rf = RandomForestRegressor(n_estimators=250,max_depth=7,min_samples_leaf=3,random_state=42,n_jobs=-1)
    rf.fit(X_train,y_train)
    pred=rf.predict(X_test)
    return {
        "ols": ols,
        "rf": rf,
        "mae": mean_absolute_error(y_test,pred),
        "r2": r2_score(y_test,pred),
        "importance": pd.Series(rf.feature_importances_,index=vars_).sort_values(ascending=False),
    }
