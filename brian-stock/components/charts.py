import plotly.graph_objects as go
from plotly.subplots import make_subplots

def price_volume_chart(df):
    recent = df.tail(180).copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.025, row_heights=[0.76, 0.24]
    )
    fig.add_trace(
        go.Candlestick(
            x=recent.index, open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"], name="Giá",
            increasing_line_color="#27c88e", decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    if "SMA20" in recent:
        fig.add_trace(go.Scatter(x=recent.index, y=recent["SMA20"], name="SMA20", line=dict(width=1.4)), row=1, col=1)
    if "SMA50" in recent:
        fig.add_trace(go.Scatter(x=recent.index, y=recent["SMA50"], name="SMA50", line=dict(width=1.4)), row=1, col=1)

    fig.add_trace(
        go.Bar(x=recent.index, y=recent["Volume"], name="Khối lượng", opacity=.65),
        row=2, col=1
    )
    fig.update_layout(
        template="plotly_dark", height=600,
        margin=dict(l=8,r=8,t=12,b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a151e",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        font=dict(color="#d9e2e8"),
        hovermode="x unified",
    )
    fig.update_yaxes(side="right", row=1, col=1, title_text="Giá")
    fig.update_yaxes(side="right", row=2, col=1, title_text="KL")
    return fig
