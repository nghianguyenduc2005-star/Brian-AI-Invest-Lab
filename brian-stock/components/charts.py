from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots


# ============================================================
# MÀU SẮC
# ============================================================

MAU_TANG = "#00d4a8"
MAU_GIAM = "#ff4d5a"
MAU_SMA20 = "#ff784f"
MAU_SMA50 = "#19d3ae"
MAU_EMA20 = "#ffd166"
MAU_EMA50 = "#9b8cff"
MAU_BOLLINGER = "#94a3b8"
MAU_TRUNG_BINH_KL = "#f59e0b"
MAU_RSI = "#38bdf8"
MAU_MACD = "#f59e0b"
MAU_TIN_HIEU = "#a78bfa"
MAU_TRUNG_TINH = "#64748b"


# ============================================================
# KIỂM TRA CỘT
# ============================================================

def _co_cot(
    du_lieu,
    ten_cot,
):
    return (
        isinstance(
            du_lieu,
            pd.DataFrame,
        )
        and ten_cot in du_lieu.columns
    )


# ============================================================
# THÊM ĐƯỜNG
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
    if not _co_cot(
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
    mau = []

    for _, dong in du_lieu.iterrows():

        gia_mo = dong.get(
            "Open",
            np.nan,
        )

        gia_dong = dong.get(
            "Close",
            np.nan,
        )

        try:
            gia_mo = float(gia_mo)
            gia_dong = float(gia_dong)
        except Exception:
            mau.append(MAU_TRUNG_TINH)
            continue

        if gia_dong > gia_mo:
            mau.append(MAU_TANG)

        elif gia_dong < gia_mo:
            mau.append(MAU_GIAM)

        else:
            mau.append(MAU_TRUNG_TINH)

    return mau


# ============================================================
# TÍNH GAP
# ============================================================

def _them_khoang_trong_gia(
    du_lieu,
    bieu_do,
):
    """
    Chỉ tính khoảng trống giá giữa hai phiên
    thực tế liền nhau trong dữ liệu.

    Không bị ảnh hưởng bởi:
    - thứ bảy
    - chủ nhật
    - ngày nghỉ lễ
    """

    if len(du_lieu) < 2:
        return

    du_lieu = du_lieu.copy()

    cao_phien_truoc = (
        du_lieu["High"].shift(1)
    )

    thap_phien_truoc = (
        du_lieu["Low"].shift(1)
    )

    gia_mo = du_lieu["Open"]

    dieu_kien_gap_tang = (
        cao_phien_truoc.notna()
        & (gia_mo > cao_phien_truoc)
    )

    dieu_kien_gap_giam = (
        thap_phien_truoc.notna()
        & (gia_mo < thap_phien_truoc)
    )

    gap_tang = du_lieu[
        dieu_kien_gap_tang
    ].copy()

    gap_giam = du_lieu[
        dieu_kien_gap_giam
    ].copy()

    if not gap_tang.empty:

        phan_tram = (
            (
                gap_tang["Open"]
                / cao_phien_truoc.loc[
                    gap_tang.index
                ]
                - 1
            )
            * 100
        )

        bieu_do.add_trace(
            go.Scatter(
                x=gap_tang.index,
                y=gap_tang["Open"],
                name="Khoảng trống tăng",
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=9,
                    color=MAU_TANG,
                ),
                customdata=(
                    phan_tram
                    .fillna(0)
                    .to_numpy()
                ),
                hovertemplate=(
                    "<b>Khoảng trống tăng</b>"
                    "<br>Giá mở: %{y:,.0f}"
                    "<br>Mức gap: %{customdata:+.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if not gap_giam.empty:

        phan_tram = (
            (
                gap_giam["Open"]
                / thap_phien_truoc.loc[
                    gap_giam.index
                ]
                - 1
            )
            * 100
        )

        bieu_do.add_trace(
            go.Scatter(
                x=gap_giam.index,
                y=gap_giam["Open"],
                name="Khoảng trống giảm",
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=9,
                    color=MAU_GIAM,
                ),
                customdata=(
                    phan_tram
                    .fillna(0)
                    .to_numpy()
                ),
                hovertemplate=(
                    "<b>Khoảng trống giảm</b>"
                    "<br>Giá mở: %{y:,.0f}"
                    "<br>Mức gap: %{customdata:+.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


# ============================================================
# NGÀY KHÔNG GIAO DỊCH
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
        if ngay.tz is not None:
            ngay = ngay.tz_localize(None)
    except Exception:
        pass

    ngay_da_co = set(
        ngay.normalize()
    )

    ngay_dau = (
        ngay.min()
        .normalize()
    )

    ngay_cuoi = (
        ngay.max()
        .normalize()
    )

    tat_ca_ngay = pd.date_range(
        start=ngay_dau,
        end=ngay_cuoi,
        freq="D",
    )

    ngay_thieu = []

    for mot_ngay in tat_ca_ngay:

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
# NHÃN THỜI GIAN TIẾNG VIỆT
# ============================================================

def _tao_nhan_thoi_gian(
    chi_so,
):
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
        if ngay.tz is not None:
            ngay = ngay.tz_localize(None)
    except Exception:
        pass

    ngay_dau = (
        ngay.min()
        .normalize()
    )

    ngay_cuoi = (
        ngay.max()
        .normalize()
    )

    moc_thang = pd.date_range(
        start=ngay_dau.replace(day=1),
        end=ngay_cuoi,
        freq="MS",
    )

    if len(moc_thang) > 18:

        buoc = int(
            np.ceil(
                len(moc_thang)
                / 18
            )
        )

        moc_thang = (
            moc_thang[::buoc]
        )

    nhan = []

    for moc in moc_thang:

        nhan.append(
            f"Thg {moc.month} "
            f"{moc.year}"
        )

    return (
        list(moc_thang),
        nhan,
    )


# ============================================================
# NHÃN HOVER NGÀY VIỆT NAM
# ============================================================

def _ngay_hover(
    gia_tri,
):
    try:
        return pd.Timestamp(
            gia_tri
        ).strftime(
            "%d/%m/%Y"
        )
    except Exception:
        return str(gia_tri)


# ============================================================
# CHỈ BÁO ĐƯỢC CHỌN
# ============================================================

def _bo_chon_chi_bao():
    return st.multiselect(
        "Chỉ báo hiển thị",
        [
            "Trung bình 20 phiên",
            "Trung bình 50 phiên",
            "Trung bình lũy thừa 20 phiên",
            "Trung bình lũy thừa 50 phiên",
            "Dải Bollinger",
            "Sức mạnh tương đối RSI",
            "MACD",
            "Khoảng trống giá",
        ],
        default=[
            "Trung bình 20 phiên",
            "Trung bình 50 phiên",
            "Dải Bollinger",
            "Sức mạnh tương đối RSI",
            "MACD",
        ],
        key="bo_chon_chi_bao",
    )


# ============================================================
# BIỂU ĐỒ GIÁ + KHỐI LƯỢNG
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
    # CHUẨN HÓA OHLCV
    # ========================================================

    for cot in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

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

    du_lieu = du_lieu.sort_index()

    # ========================================================
    # CHỌN CHỈ BÁO
    # ========================================================

    lua_chon = _bo_chon_chi_bao()

    co_rsi = (
        "Sức mạnh tương đối RSI"
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
        0.58,
        0.18,
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
    # TẠO BIỂU ĐỒ
    # ========================================================

    bieu_do = make_subplots(
        rows=so_hang,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.006,
        row_heights=ty_le,
    )

    # ========================================================
    # GIÁ
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
                    color=MAU_TANG,
                    width=1,
                ),
                fillcolor=MAU_TANG,
            ),
            decreasing=dict(
                line=dict(
                    color=MAU_GIAM,
                    width=1,
                ),
                fillcolor=MAU_GIAM,
            ),
            whiskerwidth=0.5,
            hovertemplate=(
                "Ngày: %{x|%d/%m/%Y}"
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

    if (
        "Trung bình 20 phiên"
        in lua_chon
    ):

        _them_duong(
            bieu_do,
            du_lieu,
            "SMA20",
            "TB20",
            MAU_SMA20,
            1,
            1.7,
        )

    # ========================================================
    # SMA50
    # ========================================================

    if (
        "Trung bình 50 phiên"
        in lua_chon
    ):

        _them_duong(
            bieu_do,
            du_lieu,
            "SMA50",
            "TB50",
            MAU_SMA50,
            1,
            1.7,
        )

    # ========================================================
    # EMA20
    # ========================================================

    if (
        "Trung bình lũy thừa 20 phiên"
        in lua_chon
    ):

        _them_duong(
            bieu_do,
            du_lieu,
            "EMA20",
            "TLT20",
            MAU_EMA20,
            1,
            1.4,
        )

    # ========================================================
    # EMA50
    # ========================================================

    if (
        "Trung bình lũy thừa 50 phiên"
        in lua_chon
    ):

        _them_duong(
            bieu_do,
            du_lieu,
            "EMA50",
            "TLT50",
            MAU_EMA50,
            1,
            1.4,
        )

    # ========================================================
    # BOLLINGER
    # ========================================================

    if (
        "Dải Bollinger"
        in lua_chon
        and _co_cot(
            du_lieu,
            "Bollinger_Upper",
        )
        and _co_cot(
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
                line=dict(
                    color=MAU_BOLLINGER,
                    width=1,
                    dash="dot",
                ),
                fill="tonexty",
                fillcolor=(
                    "rgba(148,163,184,0.05)"
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
    # GAP
    # ========================================================

    if (
        "Khoảng trống giá"
        in lua_chon
    ):

        _them_khoang_trong_gia(
            du_lieu,
            bieu_do,
        )

    # ========================================================
    # KHỐI LƯỢNG
    # ========================================================

    bieu_do.add_trace(
        go.Bar(
            x=du_lieu.index,
            y=du_lieu["Volume"],
            name="Khối lượng",
            marker=dict(
                color=_mau_khoi_luong(
                    du_lieu
                ),
                line=dict(
                    width=0,
                ),
            ),
            opacity=0.82,
            hovertemplate=(
                "Ngày: %{x|%d/%m/%Y}"
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

    ten_cot_kltb = (
        "Volume_SMA20"
    )

    if _co_cot(
        du_lieu,
        ten_cot_kltb,
    ):

        _them_duong(
            bieu_do,
            du_lieu,
            ten_cot_kltb,
            "KLTB20",
            MAU_TRUNG_BINH_KL,
            2,
            1.4,
        )

    # ========================================================
    # RSI
    # ========================================================

    hang_hien_tai = 3

    if co_rsi:

        if _co_cot(
            du_lieu,
            "RSI",
        ):

            bieu_do.add_trace(
                go.Scatter(
                    x=du_lieu.index,
                    y=du_lieu["RSI"],
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
                row=hang_hien_tai,
                col=1,
            )

            bieu_do.add_hline(
                y=70,
                line_dash="dot",
                line_width=1,
                line_color=MAU_GIAM,
                row=hang_hien_tai,
                col=1,
            )

            bieu_do.add_hline(
                y=50,
                line_dash="dot",
                line_width=1,
                line_color=MAU_TRUNG_TINH,
                row=hang_hien_tai,
                col=1,
            )

            bieu_do.add_hline(
                y=30,
                line_dash="dot",
                line_width=1,
                line_color=MAU_TANG,
                row=hang_hien_tai,
                col=1,
            )

        hang_hien_tai += 1

    # ========================================================
    # MACD
    # ========================================================

    if co_macd:

        if _co_cot(
            du_lieu,
            "MACD",
        ):

            if _co_cot(
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
                        MAU_TANG
                        if gia_tri >= 0
                        else MAU_GIAM
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
                            "Histogram: %{y:.3f}"
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
                    line=dict(
                        color=MAU_MACD,
                        width=1.5,
                    ),
                    hovertemplate=(
                        "MACD: %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=hang_hien_tai,
                col=1,
            )

            if _co_cot(
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
                            color=MAU_TIN_HIEU,
                            width=1.3,
                        ),
                        hovertemplate=(
                            "Tín hiệu: %{y:.3f}"
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
                line_color=MAU_TRUNG_TINH,
                row=hang_hien_tai,
                col=1,
            )

    # ========================================================
    # TRỤC Y
    # ========================================================

    bieu_do.update_yaxes(
        title_text="Giá",
        row=1,
        col=1,
    )

    bieu_do.update_yaxes(
        title_text="Khối lượng",
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
    # NGÀY NGHỈ / CUỐI TUẦN
    # ========================================================

    rangebreaks = [
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

        rangebreaks.append(
            dict(
                values=ngay_nghi
            )
        )

    bieu_do.update_xaxes(
        rangebreaks=rangebreaks
    )

    # ========================================================
    # NHÃN THÁNG TIẾNG VIỆT
    # ========================================================

    moc_thang, nhan_thang = (
        _tao_nhan_thoi_gian(
            du_lieu.index
        )
    )

    if moc_thang:

        bieu_do.update_xaxes(
            tickmode="array",
            tickvals=moc_thang,
            ticktext=nhan_thang,
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
            b=30,
        ),
        hovermode="x unified",
        dragmode="zoom",
        bargap=0.03,
        paper_bgcolor="#07131d",
        plot_bgcolor="#07131d",
        font=dict(
            size=10,
        ),
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
    )

    # ========================================================
    # LƯỚI
    # ========================================================

    for hang in range(
        1,
        so_hang + 1,
    ):

        bieu_do.update_xaxes(
            showgrid=False,
            zeroline=False,
            row=hang,
            col=1,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
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
    # TẮT RANGE SLIDER
    # ========================================================

    bieu_do.update_xaxes(
        rangeslider_visible=False
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
