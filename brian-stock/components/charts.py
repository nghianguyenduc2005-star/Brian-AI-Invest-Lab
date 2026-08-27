```python
# brian-stock/components/charts.py

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots


# ============================================================
# HÀM PHỤ
# ============================================================

def _co(du_lieu, ten_cot):
    return ten_cot in du_lieu.columns


def _them_duong(
    bieu_do,
    du_lieu,
    ten_cot,
    ten_hien_thi,
    mau,
    hang=1,
    do_rong=1.4,
    kieu_nét="solid",
):
    if not _co(du_lieu, ten_cot):
        return

    bieu_do.add_trace(
        go.Scatter(
            x=du_lieu.index,
            y=du_lieu[ten_cot],
            name=ten_hien_thi,
            mode="lines",
            line={
                "color": mau,
                "width": do_rong,
                "dash": kieu_nét,
            },
            hovertemplate=(
                f"{ten_hien_thi}: "
                "%{y:,.2f}"
                "<extra></extra>"
            ),
        ),
        row=hang,
        col=1,
    )


def _lay_danh_sach_chi_bao():
    """
    Danh sách chỉ báo cho người dùng bật/tắt.
    """

    return [
        "SMA20",
        "SMA50",
        "EMA20",
        "EMA50",
        "Bollinger",
        "RSI",
        "MACD",
        "Khoảng trống giá",
    ]


def _tao_mau_khoi_luong(du_lieu):
    """
    Volume:
        Xanh = giá đóng > giá mở
        Đỏ   = giá đóng < giá mở
        Xám  = bằng nhau
    """

    ket_qua = []

    for _, dong in du_lieu.iterrows():

        gia_mo = float(dong["Open"])
        gia_dong = float(dong["Close"])

        if gia_dong > gia_mo:
            ket_qua.append("#00d4a8")

        elif gia_dong < gia_mo:
            ket_qua.append("#ff4d5a")

        else:
            ket_qua.append("#94a3b8")

    return ket_qua


def _tinh_khoang_trong_gia(du_lieu):
    """
    Xác định gap tăng / gap giảm giữa phiên trước
    và giá mở cửa phiên hiện tại.

    Gap tăng:
        Open hôm nay > High hôm qua

    Gap giảm:
        Open hôm nay < Low hôm qua
    """

    du_lieu = du_lieu.copy()

    cao_hom_truoc = du_lieu["High"].shift(1)
    thap_hom_truoc = du_lieu["Low"].shift(1)

    du_lieu["Gap_Tang"] = (
        du_lieu["Open"] > cao_hom_truoc
    )

    du_lieu["Gap_Giam"] = (
        du_lieu["Open"] < thap_hom_truoc
    )

    du_lieu["Gap_Percent"] = np.nan

    dieu_kien_tang = (
        du_lieu["Gap_Tang"]
        & cao_hom_truoc.notna()
        & (cao_hom_truoc != 0)
    )

    dieu_kien_giam = (
        du_lieu["Gap_Giam"]
        & thap_hom_truoc.notna()
        & (thap_hom_truoc != 0)
    )

    du_lieu.loc[
        dieu_kien_tang,
        "Gap_Percent",
    ] = (
        (
            du_lieu.loc[
                dieu_kien_tang,
                "Open",
            ]
            / cao_hom_truoc.loc[
                dieu_kien_tang
            ]
            - 1
        )
        * 100
    )

    du_lieu.loc[
        dieu_kien_giam,
        "Gap_Percent",
    ] = (
        (
            du_lieu.loc[
                dieu_kien_giam,
                "Open",
            ]
            / thap_hom_truoc.loc[
                dieu_kien_giam
            ]
            - 1
        )
        * 100
    )

    return du_lieu


# ============================================================
# BIỂU ĐỒ CHÍNH
# ============================================================

def price_volume_chart(
    dataframe: pd.DataFrame,
):
    """
    Biểu đồ nghiên cứu kỹ thuật:

    1. Giá
    2. Khối lượng
    3. RSI
    4. MACD

    Có thể bật/tắt:
        SMA20
        SMA50
        EMA20
        EMA50
        Bollinger
        RSI
        MACD
        Khoảng trống giá
    """

    if dataframe is None:
        return None

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return None

    if dataframe.empty:
        return None

    du_lieu = dataframe.copy()

    # ========================================================
    # KIỂM TRA OHLCV
    # ========================================================

    cac_cot_bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for ten_cot in cac_cot_bat_buoc:

        if ten_cot not in du_lieu.columns:
            return None

        du_lieu[ten_cot] = pd.to_numeric(
            du_lieu[ten_cot],
            errors="coerce",
        )

    du_lieu = du_lieu.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    if du_lieu.empty:
        return None

    # ========================================================
    # ĐIỀU KHIỂN CHỈ BÁO
    # ========================================================

    danh_sach_chi_bao = _lay_danh_sach_chi_bao()

    mac_dinh = [
        "SMA20",
        "SMA50",
        "Bollinger",
        "RSI",
        "MACD",
        "Khoảng trống giá",
    ]

    lua_chon = st.multiselect(
        "Chỉ báo hiển thị",
        options=danh_sach_chi_bao,
        default=mac_dinh,
        key="chon_chi_bao_bieu_do",
    )

    # ========================================================
    # XÁC ĐỊNH PANEL
    # ========================================================

    co_rsi = "RSI" in lua_chon
    co_macd = "MACD" in lua_chon

    so_panel = 2

    if co_rsi:
        so_panel += 1

    if co_macd:
        so_panel += 1

    # ========================================================
    # TỶ LỆ PANEL
    #
    # Giá -> Volume sát nhau.
    # ========================================================

    ty_le = []

    ty_le.append(
        0.58
    )

    ty_le.append(
        0.17
    )

    if co_rsi:
        ty_le.append(
            0.125
        )

    if co_macd:
        ty_le.append(
            0.125
        )

    # Chuẩn hóa tỷ lệ
    tong = sum(
        ty_le
    )

    ty_le = [
        x / tong
        for x in ty_le
    ]

    # ========================================================
    # TẠO BIỂU ĐỒ
    # ========================================================

    bieu_do = make_subplots(
        rows=so_panel,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.008,
        row_heights=ty_le,
    )

    # ========================================================
    # PANEL 1 — GIÁ
    # ========================================================

    bieu_do.add_trace(
        go.Candlestick(
            x=du_lieu.index,
            open=du_lieu["Open"],
            high=du_lieu["High"],
            low=du_lieu["Low"],
            close=du_lieu["Close"],
            name="Giá",
            increasing={
                "line": {
                    "color": "#00d4a8",
                    "width": 1,
                },
                "fillcolor": "#00d4a8",
            },
            decreasing={
                "line": {
                    "color": "#ff4d5a",
                    "width": 1,
                },
                "fillcolor": "#ff4d5a",
            },
            whiskerwidth=0.5,
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>Mở: %{open:,.0f}"
                "<br>Cao: %{high:,.0f}"
                "<br>Thấp: %{low:,.0f}"
                "<br>Đóng: %{close:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # ========================================================
    # SMA20
    # ========================================================

    if "SMA20" in lua_chon:
        _them_duong(
            bieu_do,
            du_lieu,
            "SMA20",
            "SMA20",
            "#ff784f",
            1,
            1.6,
        )

    # ========================================================
    # SMA50
    # ========================================================

    if "SMA50" in lua_chon:
        _them_duong(
            bieu_do,
            du_lieu,
            "SMA50",
            "SMA50",
            "#19d3ae",
            1,
            1.6,
        )

    # ========================================================
    # EMA20
    # ========================================================

    if "EMA20" in lua_chon:
        _them_duong(
            bieu_do,
            du_lieu,
            "EMA20",
            "EMA20",
            "#ffd166",
            1,
            1.2,
        )

    # ========================================================
    # EMA50
    # ========================================================

    if "EMA50" in lua_chon:
        _them_duong(
            bieu_do,
            du_lieu,
            "EMA50",
            "EMA50",
            "#9b8cff",
            1,
            1.2,
        )

    # ========================================================
    # BOLLINGER
    # ========================================================

    if (
        "Bollinger" in lua_chon
        and _co(
            du_lieu,
            "Bollinger_Upper",
        )
        and _co(
            du_lieu,
            "Bollinger_Lower",
        )
    ):

        bieu_do.add_trace(
            go.Scatter(
                x=du_lieu.index,
                y=du_lieu[
                    "Bollinger_Upper"
                ],
                name="Dải trên",
                mode="lines",
                line={
                    "color": "#94a3b8",
                    "width": 1,
                    "dash": "dot",
                },
                hovertemplate=(
                    "Dải trên: "
                    "%{y:,.0f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        bieu_do.add_trace(
            go.Scatter(
                x=du_lieu.index,
                y=du_lieu[
                    "Bollinger_Lower"
                ],
                name="Dải dưới",
                mode="lines",
                line={
                    "color": "#94a3b8",
                    "width": 1,
                    "dash": "dot",
                },
                fill="tonexty",
                fillcolor=(
                    "rgba(148,163,184,0.06)"
                ),
                hovertemplate=(
                    "Dải dưới: "
                    "%{y:,.0f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ========================================================
    # KHOẢNG TRỐNG GIÁ
    # ========================================================

    if "Khoảng trống giá" in lua_chon:

        du_lieu = _tinh_khoang_trong_gia(
            du_lieu
        )

        gap_tang = du_lieu[
            du_lieu["Gap_Tang"]
        ]

        gap_giam = du_lieu[
            du_lieu["Gap_Giam"]
        ]

        # Gap tăng
        if not gap_tang.empty:

            bieu_do.add_trace(
                go.Scatter(
                    x=gap_tang.index,
                    y=gap_tang["Open"],
                    name="Gap tăng",
                    mode="markers",
                    marker={
                        "symbol": "triangle-up",
                        "size": 8,
                        "color": "#00d4a8",
                    },
                    customdata=np.column_stack(
                        [
                            gap_tang[
                                "Gap_Percent"
                            ].fillna(0)
                        ]
                    ),
                    hovertemplate=(
                        "Gap tăng"
                        "<br>Giá mở: %{y:,.0f}"
                        "<br>Mức gap: %{customdata[0]:+.2f}%"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        # Gap giảm
        if not gap_giam.empty:

            bieu_do.add_trace(
                go.Scatter(
                    x=gap_giam.index,
                    y=gap_giam["Open"],
                    name="Gap giảm",
                    mode="markers",
                    marker={
                        "symbol": "triangle-down",
                        "size": 8,
                        "color": "#ff4d5a",
                    },
                    customdata=np.column_stack(
                        [
                            gap_giam[
                                "Gap_Percent"
                            ].fillna(0)
                        ]
                    ),
                    hovertemplate=(
                        "Gap giảm"
                        "<br>Giá mở: %{y:,.0f}"
                        "<br>Mức gap: %{customdata[0]:+.2f}%"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    # ========================================================
    # PANEL 2 — KHỐI LƯỢNG
    # ========================================================

    mau_khoi_luong = (
        _tao_mau_khoi_luong(
            du_lieu
        )
    )

    bieu_do.add_trace(
        go.Bar(
            x=du_lieu.index,
            y=du_lieu["Volume"],
            name="Khối lượng",
            marker={
                "color": mau_khoi_luong,
                "line": {
                    "width": 0,
                },
            },
            opacity=0.78,
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>Khối lượng: %{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # ========================================================
    # TRUNG BÌNH KHỐI LƯỢNG
    # ========================================================

    if _co(
        du_lieu,
        "Volume_SMA20",
    ):

        _them_duong(
            bieu_do,
            du_lieu,
            "Volume_SMA20",
            "KLTB20",
            "#f59e0b",
            2,
            1.2,
        )

    # ========================================================
    # PANEL RSI
    # ========================================================

    hang_hien_tai = 3

    if co_rsi:

        if _co(
            du_lieu,
            "RSI",
        ):

            bieu_do.add_trace(
                go.Scatter(
                    x=du_lieu.index,
                    y=du_lieu["RSI"],
                    name="RSI",
                    mode="lines",
                    line={
                        "color": "#38bdf8",
                        "width": 1.7,
                    },
                    hovertemplate=(
                        "RSI: %{y:.1f}"
                        "<extra></extra>"
                    ),
                ),
                row=hang_hien_tai,
                col=1,
            )

            bieu_do.add_hline(
                y=70,
                line_dash="dot",
                line_width=1,
                line_color="#ff4d5a",
                row=hang_hien_tai,
                col=1,
            )

            bieu_do.add_hline(
                y=50,
                line_dash="dot",
                line_width=1,
                line_color="#64748b",
                row=hang_hien_tai,
                col=1,
            )

            bieu_do.add_hline(
                y=30,
                line_dash="dot",
                line_width=1,
                line_color="#00d4a8",
                row=hang_hien_tai,
                col=1,
            )

        hang_hien_tai += 1

    # ========================================================
    # PANEL MACD
    # ========================================================

    if co_macd:

        if _co(
            du_lieu,
            "MACD",
        ):

            if _co(
                du_lieu,
                "MACD_Hist",
            ):

                gia_tri_hist = (
                    du_lieu[
                        "MACD_Hist"
                    ]
                    .fillna(0)
                    .astype(float)
                )

                mau_hist = [
                    (
                        "#00d4a8"
                        if gia_tri >= 0
                        else "#ff4d5a"
                    )
                    for gia_tri
                    in gia_tri_hist
                ]

                bieu_do.add_trace(
                    go.Bar(
                        x=du_lieu.index,
                        y=gia_tri_hist,
                        name="MACD Histogram",
                        marker={
                            "color": mau_hist,
                            "line": {
                                "width": 0,
                            },
                        },
                        opacity=0.55,
                        hovertemplate=(
                            "MACD Histogram: "
                            "%{y:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=hang_hien_tai,
                    col=1,
                )

            bieu_do.add_trace(
                go.Scatter(
                    x=du_lieu.index,
                    y=du_lieu["MACD"],
                    name="MACD",
                    mode="lines",
                    line={
                        "color": "#f59e0b",
                        "width": 1.5,
                    },
                    hovertemplate=(
                        "MACD: %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=hang_hien_tai,
                col=1,
            )

            if _co(
                du_lieu,
                "MACD_Signal",
            ):

                bieu_do.add_trace(
                    go.Scatter(
                        x=du_lieu.index,
                        y=du_lieu[
                            "MACD_Signal"
                        ],
                        name="Tín hiệu MACD",
                        mode="lines",
                        line={
                            "color": "#a78bfa",
                            "width": 1.3,
                        },
                        hovertemplate=(
                            "Tín hiệu: "
                            "%{y:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=hang_hien_tai,
                    col=1,
                )

            bieu_do.add_hline(
                y=0,
                line_dash="dot",
                line_width=1,
                line_color="#64748b",
                row=hang_hien_tai,
                col=1,
            )

    # ========================================================
    # NHÃN TRỤC
    # ========================================================

    bieu_do.update_yaxes(
        title_text="Giá",
        row=1,
        col=1,
    )

    bieu_do.update_yaxes(
        title_text="KL",
        row=2,
        col=1,
    )

    hang_nhan = 3

    if co_rsi:

        bieu_do.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            row=hang_nhan,
            col=1,
        )

        hang_nhan += 1

    if co_macd:

        bieu_do.update_yaxes(
            title_text="MACD",
            row=hang_nhan,
            col=1,
        )

    # ========================================================
    # GIAO DIỆN
    # ========================================================

    bieu_do.update_layout(
        template="plotly_dark",
        height=900,
        margin={
            "l": 12,
            "r": 18,
            "t": 52,
            "b": 20,
        },
        hovermode="x unified",
        dragmode="zoom",
        xaxis_rangeslider_visible=False,
        bargap=0.08,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {
                "size": 10,
            },
        },
        font={
            "size": 10,
        },
        paper_bgcolor="#07131d",
        plot_bgcolor="#07131d",
    )

    # ========================================================
    # GRID
    # ========================================================

    for hang in range(
        1,
        so_panel + 1,
    ):

        bieu_do.update_xaxes(
            showgrid=False,
            zeroline=False,
            row=hang,
            col=1,
        )

        bieu_do.update_yaxes(
            showgrid=True,
            gridcolor=(
                "rgba(120,140,160,0.16)"
            ),
            zeroline=False,
            row=hang,
            col=1,
        )

    # ========================================================
    # ẨN RANGE SLIDER
    # ========================================================

    bieu_do.update_xaxes(
        rangeslider_visible=False,
        row=1,
        col=1,
    )

    return bieu_do


# ============================================================
# BIỂU ĐỒ VN-INDEX
# ============================================================

def vnindex_chart(
    dataframe: pd.DataFrame,
):

    return price_volume_chart(
        dataframe
    )
```
