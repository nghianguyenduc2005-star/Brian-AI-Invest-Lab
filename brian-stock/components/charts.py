from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots


# ============================================================
# CẤU HÌNH MÀU
# ============================================================

MAU_GIA_TANG = "#00d4a8"
MAU_GIA_GIAM = "#ff4d5a"

MAU_SMA20 = "#ff784f"
MAU_SMA50 = "#19d3ae"

MAU_EMA20 = "#ffd166"
MAU_EMA50 = "#9b8cff"

MAU_BOLLINGER = "#94a3b8"
MAU_KHOI_LUONG_TRUNG_BINH = "#f59e0b"

MAU_RSI = "#38bdf8"

MAU_MACD = "#f59e0b"
MAU_MACD_TIN_HIEU = "#a78bfa"

MAU_TRUNG_TINH = "#64748b"


# ============================================================
# HÀM KIỂM TRA CỘT
# ============================================================

def _co(du_lieu, ten_cot):
    return ten_cot in du_lieu.columns


# ============================================================
# THÊM ĐƯỜNG CHỈ BÁO
# ============================================================

def _them_duong(
    bieu_do,
    du_lieu,
    ten_cot,
    ten_hien_thi,
    mau,
    hang,
    do_rong=1.5,
    kieu_net="solid",
):
    if not _co(
        du_lieu,
        ten_cot,
    ):
        return

    bieu_do.add_trace(
        go.Scatter(
            x=du_lieu.index,
            y=du_lieu[ten_cot],
            name=ten_hien_thi,
            mode="lines",
            line=dict(
                color=mau,
                width=do_rong,
                dash=kieu_net,
            ),
            hovertemplate=(
                f"{ten_hien_thi}: "
                "%{y:,.2f}"
                "<extra></extra>"
            ),
        ),
        row=hang,
        col=1,
    )


# ============================================================
# MÀU KHỐI LƯỢNG
# ============================================================

def _mau_khoi_luong(
    du_lieu,
):
    ket_qua = []

    for _, dong in du_lieu.iterrows():

        gia_mo = float(
            dong["Open"]
        )

        gia_dong = float(
            dong["Close"]
        )

        if gia_dong > gia_mo:
            ket_qua.append(
                MAU_GIA_TANG
            )

        elif gia_dong < gia_mo:
            ket_qua.append(
                MAU_GIA_GIAM
            )

        else:
            ket_qua.append(
                "#94a3b8"
            )

    return ket_qua


# ============================================================
# TÍNH GAP
# ============================================================

def _them_gap(
    du_lieu,
    bieu_do,
):
    du_lieu = du_lieu.copy()

    cao_phien_truoc = (
        du_lieu["High"].shift(1)
    )

    thap_phien_truoc = (
        du_lieu["Low"].shift(1)
    )

    du_lieu["GapTang"] = (
        du_lieu["Open"]
        > cao_phien_truoc
    )

    du_lieu["GapGiam"] = (
        du_lieu["Open"]
        < thap_phien_truoc
    )

    du_lieu["GapPhanTram"] = np.nan

    dieu_kien_tang = (
        du_lieu["GapTang"]
        & cao_phien_truoc.notna()
        & (cao_phien_truoc != 0)
    )

    dieu_kien_giam = (
        du_lieu["GapGiam"]
        & thap_phien_truoc.notna()
        & (thap_phien_truoc != 0)
    )

    du_lieu.loc[
        dieu_kien_tang,
        "GapPhanTram",
    ] = (
        (
            du_lieu.loc[
                dieu_kien_tang,
                "Open",
            ]
            / cao_phien_truoc.loc[
                dieu_kien_tang
            ]
            - 1
        )
        * 100
    )

    du_lieu.loc[
        dieu_kien_giam,
        "GapPhanTram",
    ] = (
        (
            du_lieu.loc[
                dieu_kien_giam,
                "Open",
            ]
            / thap_phien_truoc.loc[
                dieu_kien_giam
            ]
            - 1
        )
        * 100
    )

    gap_tang = du_lieu[
        du_lieu["GapTang"]
    ]

    gap_giam = du_lieu[
        du_lieu["GapGiam"]
    ]

    # --------------------------------------------------------
    # GAP TĂNG
    # --------------------------------------------------------

    if not gap_tang.empty:

        bieu_do.add_trace(
            go.Scatter(
                x=gap_tang.index,
                y=gap_tang["Open"],
                name="Gap tăng",
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=9,
                    color=MAU_GIA_TANG,
                ),
                customdata=np.column_stack(
                    [
                        gap_tang[
                            "GapPhanTram"
                        ].fillna(0)
                    ]
                ),
                hovertemplate=(
                    "Gap tăng"
                    "<br>Giá mở: %{y:,.0f}"
                    "<br>Khoảng trống: "
                    "%{customdata[0]:+.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # --------------------------------------------------------
    # GAP GIẢM
    # --------------------------------------------------------

    if not gap_giam.empty:

        bieu_do.add_trace(
            go.Scatter(
                x=gap_giam.index,
                y=gap_giam["Open"],
                name="Gap giảm",
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=9,
                    color=MAU_GIA_GIAM,
                ),
                customdata=np.column_stack(
                    [
                        gap_giam[
                            "GapPhanTram"
                        ].fillna(0)
                    ]
                ),
                hovertemplate=(
                    "Gap giảm"
                    "<br>Giá mở: %{y:,.0f}"
                    "<br>Khoảng trống: "
                    "%{customdata[0]:+.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


# ============================================================
# TÌM NGÀY KHÔNG GIAO DỊCH
# ============================================================

def _tao_ngay_khong_giao_dich(
    chi_so,
):
    if len(chi_so) < 2:
        return []

    ngay = pd.to_datetime(
        chi_so,
        errors="coerce",
    )

    ngay = pd.DatetimeIndex(
        ngay
    ).dropna()

    if len(ngay) < 2:
        return []

    try:
        ngay = ngay.tz_localize(None)
    except Exception:
        pass

    ngay_da_co = set(
        ngay.normalize()
    )

    ngay_dau = (
        ngay.min().normalize()
    )

    ngay_cuoi = (
        ngay.max().normalize()
    )

    tat_ca_ngay = pd.date_range(
        start=ngay_dau,
        end=ngay_cuoi,
        freq="D",
    )

    ngay_thieu = []

    for mot_ngay in tat_ca_ngay:

        # Bỏ thứ bảy / chủ nhật.
        if mot_ngay.weekday() >= 5:
            continue

        if (
            mot_ngay.normalize()
            not in ngay_da_co
        ):
            ngay_thieu.append(
                mot_ngay.strftime(
                    "%Y-%m-%d"
                )
            )

    return ngay_thieu


# ============================================================
# TẠO NHÃN THỜI GIAN TIẾNG VIỆT
# ============================================================

def _tao_nhan_thoi_gian_tieng_viet(
    chi_so,
):
    """
    Tạo nhãn:

        Thg 1 2026
        Thg 2 2026
        Thg 3 2026

    thay cho:

        Jan 2026
        Feb 2026
        Mar 2026
    """

    if len(chi_so) == 0:
        return [], []

    ngay = pd.to_datetime(
        chi_so,
        errors="coerce",
    )

    ngay = pd.DatetimeIndex(
        ngay
    ).dropna()

    if len(ngay) == 0:
        return [], []

    try:
        ngay = ngay.tz_localize(None)
    except Exception:
        pass

    ngay_dau = (
        ngay.min().normalize()
    )

    ngay_cuoi = (
        ngay.max().normalize()
    )

    # Tạo các mốc đầu tháng.
    moc_thang = pd.date_range(
        start=ngay_dau.replace(
            day=1
        ),
        end=ngay_cuoi,
        freq="MS",
    )

    # Nếu quá nhiều tháng thì giảm số nhãn.
    so_thang = len(moc_thang)

    if so_thang > 24:

        buoc = max(
            1,
            int(
                np.ceil(
                    so_thang / 18
                )
            ),
        )

        moc_thang = moc_thang[
            ::buoc
        ]

    thang_viet = [
        f"Thg {ngay_thang.month} "
        f"{ngay_thang.year}"
        for ngay_thang
        in moc_thang
    ]

    return (
        list(moc_thang),
        thang_viet,
    )


# ============================================================
# BIỂU ĐỒ CHÍNH
# ============================================================

def price_volume_chart(
    du_lieu: pd.DataFrame,
):

    if du_lieu is None:
        return None

    if not isinstance(
        du_lieu,
        pd.DataFrame,
    ):
        return None

    if du_lieu.empty:
        return None

    du_lieu = du_lieu.copy()

    # ========================================================
    # KIỂM TRA CỘT
    # ========================================================

    cot_bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for cot in cot_bat_buoc:

        if cot not in du_lieu.columns:
            return None

        du_lieu[cot] = pd.to_numeric(
            du_lieu[cot],
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
    # CHỌN CHỈ BÁO
    # ========================================================

    lua_chon = st.multiselect(
        "Chỉ báo hiển thị",
        [
            "SMA20",
            "SMA50",
            "EMA20",
            "EMA50",
            "Bollinger",
            "RSI",
            "MACD",
            "Khoảng trống giá",
        ],
        default=[
            "SMA20",
            "SMA50",
            "Bollinger",
            "RSI",
            "MACD",
            "Khoảng trống giá",
        ],
        key="bo_chon_chi_bao",
    )

    co_rsi = (
        "RSI"
        in lua_chon
    )

    co_macd = (
        "MACD"
        in lua_chon
    )

    # ========================================================
    # SỐ PANEL
    # ========================================================

    so_hang = 2

    if co_rsi:
        so_hang += 1

    if co_macd:
        so_hang += 1

    # ========================================================
    # TỶ LỆ PANEL
    # ========================================================

    ty_le = [
        0.60,
        0.16,
    ]

    if co_rsi:
        ty_le.append(
            0.12
        )

    if co_macd:
        ty_le.append(
            0.12
        )

    tong = sum(
        ty_le
    )

    ty_le = [
        gia_tri / tong
        for gia_tri
        in ty_le
    ]

    # ========================================================
    # TẠO SUBPLOT
    # ========================================================

    bieu_do = make_subplots(
        rows=so_hang,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.006,
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
            increasing=dict(
                line=dict(
                    color=MAU_GIA_TANG,
                    width=1,
                ),
                fillcolor=MAU_GIA_TANG,
            ),
            decreasing=dict(
                line=dict(
                    color=MAU_GIA_GIAM,
                    width=1,
                ),
                fillcolor=MAU_GIA_GIAM,
            ),
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
            MAU_SMA20,
            1,
            1.7,
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
            MAU_SMA50,
            1,
            1.7,
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
            MAU_EMA20,
            1,
            1.3,
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
            MAU_EMA50,
            1,
            1.3,
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
                line=dict(
                    color=MAU_BOLLINGER,
                    width=1,
                    dash="dot",
                ),
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
                line=dict(
                    color=MAU_BOLLINGER,
                    width=1,
                    dash="dot",
                ),
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
    # GAP
    # ========================================================

    if (
        "Khoảng trống giá"
        in lua_chon
    ):

        _them_gap(
            du_lieu,
            bieu_do,
        )

    # ========================================================
    # PANEL 2 — KHỐI LƯỢNG
    # ========================================================

    bieu_do.add_trace(
        go.Bar(
            x=du_lieu.index,
            y=du_lieu[
                "Volume"
            ],
            name="Khối lượng",
            marker=dict(
                color=_mau_khoi_luong(
                    du_lieu
                ),
                line=dict(
                    width=0,
                ),
            ),
            opacity=0.78,
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>Khối lượng: "
                "%{y:,.0f}"
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
            MAU_KHOI_LUONG_TRUNG_BINH,
            2,
            1.3,
        )

    # ========================================================
    # PANEL RSI
    # ========================================================

    hang = 3

    if co_rsi:

        if _co(
            du_lieu,
            "RSI",
        ):

            bieu_do.add_trace(
                go.Scatter(
                    x=du_lieu.index,
                    y=du_lieu[
                        "RSI"
                    ],
                    name="RSI",
                    mode="lines",
                    line=dict(
                        color=MAU_RSI,
                        width=1.7,
                    ),
                    hovertemplate=(
                        "RSI: %{y:.1f}"
                        "<extra></extra>"
                    ),
                ),
                row=hang,
                col=1,
            )

            bieu_do.add_hline(
                y=70,
                line_dash="dot",
                line_width=1,
                line_color=MAU_GIA_GIAM,
                row=hang,
                col=1,
            )

            bieu_do.add_hline(
                y=50,
                line_dash="dot",
                line_width=1,
                line_color=MAU_TRUNG_TINH,
                row=hang,
                col=1,
            )

            bieu_do.add_hline(
                y=30,
                line_dash="dot",
                line_width=1,
                line_color=MAU_GIA_TANG,
                row=hang,
                col=1,
            )

        hang += 1

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

                histogram = (
                    du_lieu[
                        "MACD_Hist"
                    ]
                    .fillna(0)
                    .astype(float)
                )

                mau_histogram = [
                    (
                        MAU_GIA_TANG
                        if gia_tri >= 0
                        else MAU_GIA_GIAM
                    )
                    for gia_tri
                    in histogram
                ]

                bieu_do.add_trace(
                    go.Bar(
                        x=du_lieu.index,
                        y=histogram,
                        name="MACD Histogram",
                        marker=dict(
                            color=mau_histogram,
                            line=dict(
                                width=0,
                            ),
                        ),
                        opacity=0.55,
                        hovertemplate=(
                            "Histogram: "
                            "%{y:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=hang,
                    col=1,
                )

            bieu_do.add_trace(
                go.Scatter(
                    x=du_lieu.index,
                    y=du_lieu[
                        "MACD"
                    ],
                    name="MACD",
                    mode="lines",
                    line=dict(
                        color=MAU_MACD,
                        width=1.5,
                    ),
                    hovertemplate=(
                        "MACD: %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=hang,
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
                        line=dict(
                            color=MAU_MACD_TIN_HIEU,
                            width=1.3,
                        ),
                        hovertemplate=(
                            "Tín hiệu: "
                            "%{y:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=hang,
                    col=1,
                )

            bieu_do.add_hline(
                y=0,
                line_dash="dot",
                line_width=1,
                line_color=MAU_TRUNG_TINH,
                row=hang,
                col=1,
            )

    # ========================================================
    # NHÃN TRỤC Y
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

    hang_truc = 3

    if co_rsi:

        bieu_do.update_yaxes(
            title_text="RSI",
            range=[
                0,
                100,
            ],
            row=hang_truc,
            col=1,
        )

        hang_truc += 1

    if co_macd:

        bieu_do.update_yaxes(
            title_text="MACD",
            row=hang_truc,
            col=1,
        )

    # ========================================================
    # TRỤC THỜI GIAN TIẾNG VIỆT
    # ========================================================

    moc_thoi_gian, nhan_tieng_viet = (
        _tao_nhan_thoi_gian_tieng_viet(
            du_lieu.index
        )
    )

    # ========================================================
    # BỎ CUỐI TUẦN + NGÀY NGHỈ
    # ========================================================

    khoang_trong = [
        dict(
            bounds=[
                "sat",
                "mon",
            ]
        )
    ]

    ngay_nghi = (
        _tao_ngay_khong_giao_dich(
            du_lieu.index
        )
    )

    if ngay_nghi:

        khoang_trong.append(
            dict(
                values=ngay_nghi
            )
        )

    bieu_do.update_xaxes(
        rangebreaks=khoang_trong
    )

    # ========================================================
    # ĐẶT NHÃN THÁNG TIẾNG VIỆT
    # ========================================================

    if moc_thoi_gian:

        bieu_do.update_xaxes(
            tickmode="array",
            tickvals=moc_thoi_gian,
            ticktext=nhan_tieng_viet,
        )

    # ========================================================
    # GIAO DIỆN
    # ========================================================

    bieu_do.update_layout(
        template="plotly_dark",
        height=920,
        margin=dict(
            l=12,
            r=20,
            t=58,
            b=28,
        ),
        hovermode="x unified",
        dragmode="zoom",
        bargap=0.04,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(
                size=10,
            ),
        ),
        font=dict(
            size=10,
        ),
        paper_bgcolor="#07131d",
        plot_bgcolor="#07131d",
    )

    # ========================================================
    # LƯỚI
    # ========================================================

    for hang_luoi in range(
        1,
        so_hang + 1,
    ):

        bieu_do.update_xaxes(
            showgrid=False,
            zeroline=False,
            row=hang_luoi,
            col=1,
        )

        bieu_do.update_yaxes(
            showgrid=True,
            gridcolor=(
                "rgba(120,140,160,0.16)"
            ),
            zeroline=False,
            row=hang_luoi,
            col=1,
        )

    # ========================================================
    # CẤU HÌNH TRỤC THỜI GIAN CUỐI
    # ========================================================

    bieu_do.update_xaxes(
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=False,
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
