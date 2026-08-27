# brian-stock/components/charts.py

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots


# ============================================================
# HÀM PHỤ
# ============================================================

def _co(dataframe, ten_cot):
    return ten_cot in dataframe.columns


def _them_duong(
    bieu_do,
    dataframe,
    ten_cot,
    ten_hien_thi,
    mau,
    hang=1,
    truc=1,
    do_rong=1.5,
):
    if not _co(dataframe, ten_cot):
        return

    bieu_do.add_trace(
        go.Scatter(
            x=dataframe.index,
            y=dataframe[ten_cot],
            name=ten_hien_thi,
            mode="lines",
            line={
                "color": mau,
                "width": do_rong,
            },
            hovertemplate=(
                f"{ten_hien_thi}: %{{y:,.2f}}"
                "<extra></extra>"
            ),
        ),
        row=hang,
        col=1,
    )


# ============================================================
# BIỂU ĐỒ GIÁ + KHỐI LƯỢNG + RSI + MACD
# ============================================================

def price_volume_chart(
    dataframe: pd.DataFrame,
):
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
    # KIỂM TRA CỘT CƠ BẢN
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

    # ========================================================
    # CHỈ GIỮ DỮ LIỆU HỢP LỆ
    # ========================================================

    du_lieu = du_lieu.copy()

    for ten_cot in cac_cot_bat_buoc:

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
    # TẠO 4 KHU VỰC
    #
    # 1. Giá
    # 2. Khối lượng
    # 3. RSI
    # 4. MACD
    # ========================================================

    bieu_do = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[
            0.52,
            0.18,
            0.15,
            0.15,
        ],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # ========================================================
    # 1. NẾN OHLC
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
                }
            },
            decreasing={
                "line": {
                    "color": "#ff4d5a",
                    "width": 1,
                }
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
    # 2. SMA
    # ========================================================

    _them_duong(
        bieu_do,
        du_lieu,
        "SMA20",
        "SMA20",
        "#ff784f",
        1,
        1,
        1.6,
    )

    _them_duong(
        bieu_do,
        du_lieu,
        "SMA50",
        "SMA50",
        "#19d3ae",
        1,
        1,
        1.6,
    )

    # ========================================================
    # 3. EMA
    # ========================================================

    _them_duong(
        bieu_do,
        du_lieu,
        "EMA20",
        "EMA20",
        "#ffd166",
        1,
        1,
        1.2,
    )

    _them_duong(
        bieu_do,
        du_lieu,
        "EMA50",
        "EMA50",
        "#9b8cff",
        1,
        1,
        1.2,
    )

    # ========================================================
    # 4. BOLLINGER
    # ========================================================

    if (
        _co(
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
                    "color": "#8aa4c4",
                    "width": 1,
                    "dash": "dot",
                },
                hovertemplate=(
                    "Dải trên: %{y:,.0f}"
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
                    "color": "#8aa4c4",
                    "width": 1,
                    "dash": "dot",
                },
                fill="tonexty",
                fillcolor=(
                    "rgba(120,150,180,0.08)"
                ),
                hovertemplate=(
                    "Dải dưới: %{y:,.0f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ========================================================
    # 5. KHỐI LƯỢNG
    # ========================================================

    mau_khoi_luong = []

    for _, dong in du_lieu.iterrows():

        gia_mo = dong["Open"]
        gia_dong = dong["Close"]

        if gia_dong >= gia_mo:
            mau_khoi_luong.append(
                "#8b5cf6"
            )
        else:
            mau_khoi_luong.append(
                "#6b7280"
            )

    bieu_do.add_trace(
        go.Bar(
            x=du_lieu.index,
            y=du_lieu["Volume"],
            name="Khối lượng",
            marker={
                "color": mau_khoi_luong
            },
            opacity=0.72,
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
    # 6. TRUNG BÌNH KHỐI LƯỢNG
    # ========================================================

    _them_duong(
        bieu_do,
        du_lieu,
        "Volume_SMA20",
        "KLTB20",
        "#f59e0b",
        2,
        1,
        1.3,
    )

    # ========================================================
    # 7. RSI
    # ========================================================

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
            row=3,
            col=1,
        )

        bieu_do.add_hline(
            y=70,
            line_dash="dot",
            line_width=1,
            line_color="#ff4d5a",
            row=3,
            col=1,
        )

        bieu_do.add_hline(
            y=30,
            line_dash="dot",
            line_width=1,
            line_color="#00d4a8",
            row=3,
            col=1,
        )

        bieu_do.add_hline(
            y=50,
            line_dash="dot",
            line_width=1,
            line_color="#64748b",
            row=3,
            col=1,
        )

    # ========================================================
    # 8. MACD
    # ========================================================

    if _co(
        du_lieu,
        "MACD",
    ):

        bieu_do.add_trace(
            go.Bar(
                x=du_lieu.index,
                y=(
                    du_lieu["MACD_Hist"]
                    if _co(
                        du_lieu,
                        "MACD_Hist",
                    )
                    else 0
                ),
                name="MACD Histogram",
                marker={
                    "color": [
                        (
                            "#00d4a8"
                            if gia_tri >= 0
                            else "#ff4d5a"
                        )
                        for gia_tri
                        in (
                            du_lieu[
                                "MACD_Hist"
                            ]
                            if _co(
                                du_lieu,
                                "MACD_Hist",
                            )
                            else pd.Series(
                                0,
                                index=du_lieu.index,
                            )
                        )
                    ]
                },
                opacity=0.55,
                hovertemplate=(
                    "MACD Histogram: %{y:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=4,
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
            row=4,
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
                        "Tín hiệu: %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=4,
                col=1,
            )

        bieu_do.add_hline(
            y=0,
            line_dash="dot",
            line_width=1,
            line_color="#64748b",
            row=4,
            col=1,
        )

    # ========================================================
    # TIÊU ĐỀ TRỤC
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

    bieu_do.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        row=3,
        col=1,
    )

    bieu_do.update_yaxes(
        title_text="MACD",
        row=4,
        col=1,
    )

    # ========================================================
    # GIAO DIỆN
    # ========================================================

    bieu_do.update_layout(
        template="plotly_dark",
        height=980,
        margin={
            "l": 10,
            "r": 20,
            "t": 35,
            "b": 10,
        },
        hovermode="x unified",
        dragmode="zoom",
        xaxis_rangeslider_visible=False,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.015,
            "xanchor": "left",
            "x": 0,
            "font": {
                "size": 11
            },
        },
        font={
            "size": 11
        },
        paper_bgcolor="#07131d",
        plot_bgcolor="#07131d",
    )

    # ========================================================
    # GRID
    # ========================================================

    for hang in [
        1,
        2,
        3,
        4,
    ]:

        bieu_do.update_xaxes(
            showgrid=False,
            zeroline=False,
            row=hang,
            col=1,
        )

        bieu_do.update_yaxes(
            showgrid=True,
            gridcolor=(
                "rgba(120,140,160,0.18)"
            ),
            zeroline=False,
            row=hang,
            col=1,
        )

    # ========================================================
    # NÚT CÔNG CỤ
    # ========================================================

    bieu_do.update_layout(
        modebar={
            "remove": [
                "lasso2d",
                "select2d",
            ]
        }
    )

    return bieu_do


# ============================================================
# BIỂU ĐỒ VN-INDEX
# ============================================================

def vnindex_chart(
    dataframe: pd.DataFrame,
):

    if dataframe is None:
        return None

    if dataframe.empty:
        return None

    return price_volume_chart(
        dataframe
    )
