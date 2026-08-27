from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots


# ============================================================
# MÀU
# ============================================================

MAU_TANG = "#00d4a8"
MAU_GIAM = "#ff4d5a"
MAU_SMA20 = "#ff784f"
MAU_SMA50 = "#19d3ae"
MAU_EMA20 = "#ffd166"
MAU_EMA50 = "#9b8cff"
MAU_BOLLINGER = "#94a3b8"
MAU_KLTB = "#f59e0b"
MAU_RSI = "#38bdf8"
MAU_MACD = "#f59e0b"
MAU_MACD_TIN_HIEU = "#a78bfa"
MAU_TRUNG_TINH = "#64748b"


# ============================================================
# TIỆN ÍCH
# ============================================================

def _co_cot(
    du_lieu: pd.DataFrame,
    ten_cot: str,
) -> bool:

    return (
        isinstance(
            du_lieu,
            pd.DataFrame,
        )
        and ten_cot in du_lieu.columns
    )


def _to_float_series(
    series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _dinh_dang_don_vi(
    gia_tri,
) -> str:

    try:

        gia_tri = float(
            gia_tri
        )

        if not np.isfinite(
            gia_tri
        ):
            return "0"

    except Exception:

        return "0"

    gia_tri_abs = abs(
        gia_tri
    )

    if gia_tri_abs >= 1_000_000_000:

        return (
            f"{gia_tri / 1_000_000_000:.1f} tỷ"
        )

    if gia_tri_abs >= 1_000_000:

        return (
            f"{gia_tri / 1_000_000:.1f} triệu"
        )

    if gia_tri_abs >= 1_000:

        return (
            f"{gia_tri / 1_000:.1f} nghìn"
        )

    return f"{gia_tri:,.0f}"


def _tao_tick_khoi_luong(
    series: pd.Series,
):

    series = _to_float_series(
        series
    ).dropna()

    if series.empty:
        return [], []

    gia_tri_lon_nhat = float(
        series.max()
    )

    if (
        not np.isfinite(
            gia_tri_lon_nhat
        )
        or gia_tri_lon_nhat <= 0
    ):
        return [], []

    so_tick = 5

    tickvals = np.linspace(
        0,
        gia_tri_lon_nhat,
        so_tick,
    )

    ticktext = [
        _dinh_dang_don_vi(
            gia_tri
        )
        for gia_tri
        in tickvals
    ]

    return (
        list(tickvals),
        ticktext,
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

    gia_tri = _to_float_series(
        du_lieu[ten_cot]
    )

    bieu_do.add_trace(
        go.Scatter(
            x=du_lieu.index,
            y=gia_tri,
            name=ten_hien_thi,
            mode="lines",
            line=dict(
                color=mau,
                width=do_rong,
                dash=kieu_net,
            ),
            connectgaps=False,
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

            gia_mo = float(
                gia_mo
            )

            gia_dong = float(
                gia_dong
            )

        except Exception:

            mau.append(
                MAU_TRUNG_TINH
            )

            continue

        if gia_dong >= gia_mo:

            mau.append(
                MAU_TANG
            )

        else:

            mau.append(
                MAU_GIAM
            )

    return mau


# ============================================================
# KHOẢNG TRỐNG GIÁ
# ============================================================

def _them_khoang_trong_gia(
    du_lieu,
    bieu_do,
):

    if len(du_lieu) < 2:
        return

    cao_phien_truoc = (
        _to_float_series(
            du_lieu["High"]
        )
        .shift(1)
    )

    thap_phien_truoc = (
        _to_float_series(
            du_lieu["Low"]
        )
        .shift(1)
    )

    gia_mo = _to_float_series(
        du_lieu["Open"]
    )

    # --------------------------------------------------------
    # GAP TĂNG
    # --------------------------------------------------------

    dieu_kien_gap_tang = (
        cao_phien_truoc.notna()
        & gia_mo.notna()
        & (
            gia_mo
            > cao_phien_truoc
        )
    )

    # --------------------------------------------------------
    # GAP GIẢM
    # --------------------------------------------------------

    dieu_kien_gap_giam = (
        thap_phien_truoc.notna()
        & gia_mo.notna()
        & (
            gia_mo
            < thap_phien_truoc
        )
    )

    gap_tang = du_lieu[
        dieu_kien_gap_tang
    ].copy()

    gap_giam = du_lieu[
        dieu_kien_gap_giam
    ].copy()

    # ========================================================
    # GAP TĂNG
    # ========================================================

    if not gap_tang.empty:

        muc_truoc = (
            cao_phien_truoc.loc[
                gap_tang.index
            ]
        )

        phan_tram = (
            (
                gap_tang["Open"]
                / muc_truoc
                - 1
            )
            * 100
        )

        bieu_do.add_trace(
            go.Scatter(
                x=gap_tang.index,
                y=gap_tang["Open"],
                name="Gap tăng",
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=9,
                    color=MAU_TANG,
                    line=dict(
                        width=0,
                    ),
                ),
                customdata=np.column_stack(
                    [
                        muc_truoc.to_numpy(),
                        phan_tram
                        .fillna(0)
                        .to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "<b>Khoảng trống tăng</b>"
                    "<br>Giá mở: %{y:,.0f}"
                    "<br>Đỉnh phiên trước: %{customdata[0]:,.0f}"
                    "<br>Mức gap: %{customdata[1]:+.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ========================================================
    # GAP GIẢM
    # ========================================================

    if not gap_giam.empty:

        muc_truoc = (
            thap_phien_truoc.loc[
                gap_giam.index
            ]
        )

        phan_tram = (
            (
                gap_giam["Open"]
                / muc_truoc
                - 1
            )
            * 100
        )

        bieu_do.add_trace(
            go.Scatter(
                x=gap_giam.index,
                y=gap_giam["Open"],
                name="Gap giảm",
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=9,
                    color=MAU_GIAM,
                    line=dict(
                        width=0,
                    ),
                ),
                customdata=np.column_stack(
                    [
                        muc_truoc.to_numpy(),
                        phan_tram
                        .fillna(0)
                        .to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "<b>Khoảng trống giảm</b>"
                    "<br>Giá mở: %{y:,.0f}"
                    "<br>Đáy phiên trước: %{customdata[0]:,.0f}"
                    "<br>Mức gap: %{customdata[1]:+.2f}%"
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

            ngay = ngay.tz_localize(
                None
            )

    except Exception:
        pass

    ngay = ngay.normalize()

    ngay_da_co = set(
        ngay
    )

    ngay_dau = ngay.min()
    ngay_cuoi = ngay.max()

    tat_ca_ngay = pd.date_range(
        start=ngay_dau,
        end=ngay_cuoi,
        freq="D",
    )

    ngay_thieu = []

    for mot_ngay in tat_ca_ngay:

        # ----------------------------------------------------
        # Cuối tuần
        # ----------------------------------------------------

        if mot_ngay.weekday() >= 5:
            continue

        # ----------------------------------------------------
        # Ngày trong tuần nhưng không có dữ liệu
        # thường là ngày nghỉ / không có phiên
        # ----------------------------------------------------

        if mot_ngay not in ngay_da_co:

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

            ngay = ngay.tz_localize(
                None
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Lấy phiên giao dịch đầu tiên của mỗi tháng.
    # Không dùng ngày 01 vì ngày 01 có thể là cuối tuần/ngày nghỉ.
    # --------------------------------------------------------

    bang = pd.DataFrame(
        {
            "ngay": ngay
        }
    )

    bang["thang"] = (
        bang["ngay"]
        .dt.to_period("M")
    )

    cac_moc = (
        bang
        .groupby(
            "thang",
            sort=True,
        )["ngay"]
        .min()
    )

    moc = list(
        cac_moc.to_numpy()
    )

    # --------------------------------------------------------
    # Giới hạn số nhãn
    # --------------------------------------------------------

    if len(moc) > 18:

        buoc = int(
            np.ceil(
                len(moc)
                / 18
            )
        )

        moc = moc[
            ::buoc
        ]

    nhan = []

    for gia_tri in moc:

        gia_tri = pd.Timestamp(
            gia_tri
        )

        nhan.append(
            f"Thg {gia_tri.month} "
            f"{gia_tri.year}"
        )

    return (
        moc,
        nhan,
    )


# ============================================================
# CHỌN CHỈ BÁO
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
        placeholder="Chọn các chỉ báo cần hiển thị",
    )


# ============================================================
# BIỂU ĐỒ CHÍNH
# ============================================================

def price_volume_chart(
    du_lieu: pd.DataFrame,
):

    # ========================================================
    # KIỂM TRA
    # ========================================================

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

    cac_cot_bat_buoc = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for cot in cac_cot_bat_buoc:

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
    ).copy()

    if du_lieu.empty:
        return None

    # ========================================================
    # THỜI GIAN
    # ========================================================

    try:

        du_lieu.index = pd.to_datetime(
            du_lieu.index,
            errors="coerce",
        )

        du_lieu = du_lieu[
            ~du_lieu.index.isna()
        ]

        if getattr(
            du_lieu.index,
            "tz",
            None,
        ) is not None:

            du_lieu.index = (
                du_lieu.index.tz_localize(
                    None
                )
            )

    except Exception:
        pass

    du_lieu = (
        du_lieu
        .sort_index()
        .loc[
            ~du_lieu.index.duplicated(
                keep="last"
            )
        ]
        .copy()
    )

    if du_lieu.empty:
        return None

    # ========================================================
    # CHỌN CHỈ BÁO
    # ========================================================

    lua_chon = (
        _bo_chon_chi_bao()
    )

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

    if so_hang == 2:

        ty_le = [
            0.78,
            0.22,
        ]

    elif so_hang == 3:

        ty_le = [
            0.58,
            0.22,
            0.20,
        ]

    else:

        ty_le = [
            0.52,
            0.20,
            0.14,
            0.14,
        ]

    tong = sum(
        ty_le
    )

    ty_le = [
        x / tong
        for x
        in ty_le
    ]

    # ========================================================
    # TẠO SUBPLOT
    # ========================================================

    bieu_do = make_subplots(
        rows=so_hang,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.008,
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
                "<b>%{x|%d/%m/%Y}</b>"
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
            1.8,
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
            1.8,
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
    ):

        if (
            _co_cot(
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
                    y=_to_float_series(
                        du_lieu[
                            "Bollinger_Upper"
                        ]
                    ),
                    name="Dải trên",
                    mode="lines",
                    line=dict(
                        color=MAU_BOLLINGER,
                        width=1,
                        dash="dot",
                    ),
                    connectgaps=False,
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
                    y=_to_float_series(
                        du_lieu[
                            "Bollinger_Lower"
                        ]
                    ),
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
                    connectgaps=False,
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
                "<b>%{x|%d/%m/%Y}</b>"
                "<br>Khối lượng: %{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # ========================================================
    # KHỐI LƯỢNG TRUNG BÌNH 20
    # ========================================================

    if _co_cot(
        du_lieu,
        "Volume_SMA20",
    ):

        _them_duong(
            bieu_do,
            du_lieu,
            "Volume_SMA20",
            "KLTB20",
            MAU_KLTB,
            2,
            1.4,
        )

    # ========================================================
    # RSI
    # ========================================================

    hang_rsi = 3

    if co_rsi:

        if _co_cot(
            du_lieu,
            "RSI",
        ):

            bieu_do.add_trace(
                go.Scatter(
                    x=du_lieu.index,
                    y=_to_float_series(
                        du_lieu["RSI"]
                    ),
                    name="RSI",
                    mode="lines",
                    line=dict(
                        color=MAU_RSI,
                        width=1.8,
                    ),
                    connectgaps=False,
                    hovertemplate=(
                        "RSI: %{y:.1f}"
                        "<extra></extra>"
                    ),
                ),
                row=hang_rsi,
                col=1,
            )

            bieu_do.add_hline(
                y=70,
                line_dash="dot",
                line_width=1,
                line_color=MAU_GIAM,
                row=hang_rsi,
                col=1,
            )

            bieu_do.add_hline(
                y=50,
                line_dash="dot",
                line_width=1,
                line_color=MAU_TRUNG_TINH,
                row=hang_rsi,
                col=1,
            )

            bieu_do.add_hline(
                y=30,
                line_dash="dot",
                line_width=1,
                line_color=MAU_TANG,
                row=hang_rsi,
                col=1,
            )

        hang_rsi += 1

    # ========================================================
    # MACD
    # ========================================================

    if co_macd:

        hang_macd = (
            3
            if not co_rsi
            else 4
        )

        if _co_cot(
            du_lieu,
            "MACD",
        ):

            # ------------------------------------------------
            # Histogram
            # ------------------------------------------------

            if _co_cot(
                du_lieu,
                "MACD_Hist",
            ):

                histogram = (
                    _to_float_series(
                        du_lieu[
                            "MACD_Hist"
                        ]
                    )
                    .fillna(0)
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
                        opacity=0.50,
                        hovertemplate=(
                            "Histogram: %{y:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=hang_macd,
                    col=1,
                )

            # ------------------------------------------------
            # MACD
            # ------------------------------------------------

            bieu_do.add_trace(
                go.Scatter(
                    x=du_lieu.index,
                    y=_to_float_series(
                        du_lieu["MACD"]
                    ),
                    name="MACD",
                    mode="lines",
                    line=dict(
                        color=MAU_MACD,
                        width=1.6,
                    ),
                    connectgaps=False,
                    hovertemplate=(
                        "MACD: %{y:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=hang_macd,
                col=1,
            )

            # ------------------------------------------------
            # Tín hiệu MACD
            # ------------------------------------------------

            if _co_cot(
                du_lieu,
                "MACD_Signal",
            ):

                bieu_do.add_trace(
                    go.Scatter(
                        x=du_lieu.index,
                        y=_to_float_series(
                            du_lieu[
                                "MACD_Signal"
                            ]
                        ),
                        name="Tín hiệu MACD",
                        mode="lines",
                        line=dict(
                            color=MAU_MACD_TIN_HIEU,
                            width=1.4,
                        ),
                        connectgaps=False,
                        hovertemplate=(
                            "Tín hiệu MACD: %{y:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=hang_macd,
                    col=1,
                )

            bieu_do.add_hline(
                y=0,
                line_dash="dot",
                line_width=1,
                line_color=MAU_TRUNG_TINH,
                row=hang_macd,
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

    # --------------------------------------------------------
    # Đơn vị khối lượng tiếng Việt
    # --------------------------------------------------------

    tickvals_khoi_luong, ticktext_khoi_luong = (
        _tao_tick_khoi_luong(
            du_lieu["Volume"]
        )
    )

    if tickvals_khoi_luong:

        bieu_do.update_yaxes(
            tickmode="array",
            tickvals=tickvals_khoi_luong,
            ticktext=ticktext_khoi_luong,
            row=2,
            col=1,
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if co_macd:

        bieu_do.update_yaxes(
            title_text="MACD",
            row=hang_truc,
            col=1,
        )

    # ========================================================
    # RANGE BREAK
    # ========================================================

    rangebreaks = [
        dict(
            bounds=[
                "sat",
                "mon",
            ]
        )
    ]

    ngay_khong_giao_dich = (
        _tao_ngay_khong_giao_dich(
            du_lieu.index
        )
    )

    if ngay_khong_giao_dich:

        rangebreaks.append(
            dict(
                values=ngay_khong_giao_dich
            )
        )

    bieu_do.update_xaxes(
        rangebreaks=rangebreaks
    )

    # ========================================================
    # NHÃN THỜI GIAN TIẾNG VIỆT
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
        height=900,
        margin=dict(
            l=12,
            r=22,
            t=58,
            b=36,
        ),
        hovermode="x unified",
        dragmode="zoom",
        bargap=0.04,
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
    # LƯỚI + SPIKE
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
            spikecolor=(
                "rgba(148,163,184,0.35)"
            ),
            spikethickness=1,
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

    # ========================================================
    # ẨN NÚT PHÂN TÍCH LỖI / LOGO PLOTLY
    # ========================================================

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
